
"""Standalone multi-backbone softmax token-classifier runner for PAP_NER on Linux GPUs."""
import argparse
import hashlib
import itertools
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report, f1_score
from seqeval.metrics import classification_report as entity_report
from seqeval.metrics import f1_score as entity_f1
from seqeval.metrics import precision_score as entity_precision
from seqeval.metrics import recall_score as entity_recall
from seqeval.scheme import IOB2
from tqdm.auto import tqdm

LABELS = ['O', 'B-CQ', 'I-CQ', 'B-ĐT', 'I-ĐT', 'B-VBPL', 'I-VBPL',
          'B-NG', 'I-NG', 'B-SL', 'I-SL']


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def read_conll(path):
    """PAP_NER NER is column 4; column 5 is NOT the target."""
    words, tags = [], []
    with Path(path).open(encoding='utf-8-sig') as stream:
        for line_no, line in enumerate(stream, 1):
            fields = line.split()
            if not fields:
                if words:
                    yield words, tags
                    words, tags = [], []
                continue
            if len(fields) >= 4:
                word, tag = fields[0], fields[3]
            elif len(fields) == 2:
                word, tag = fields
            else:
                raise ValueError(f'{path}:{line_no}: expected 2 or >=4 columns')
            if tag not in LABELS:
                raise ValueError(f'{path}:{line_no}: unknown label {tag!r}')
            words.append(word)
            tags.append(LABELS.index(tag))
    if words:
        yield words, tags


def encode_words(words, tags, tokenizer, max_length):
    """Keep only complete words fitting the BPE budget; align first subword."""
    ids, positions, kept_tags = [], [], []
    for word, tag in zip(words, tags):
        pieces = tokenizer.encode(word, add_special_tokens=False)
        if not pieces:
            pieces = [tokenizer.unk_token_id]
        if len(ids) + len(pieces) > max_length - 2:
            break
        positions.append(len(ids) + 1)  # +1 for <s>
        ids.extend(pieces)
        kept_tags.append(tag)
    if not positions:
        raise ValueError(f'First word exceeds BPE budget: {words[:1]}')
    return {'input_ids': [tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id],
            'valid_ids': positions, 'labels': kept_tags}


class Features(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, examples):
        # Dynamic padding on CPU; GPU transfer happens in the parent process.
        length = math.ceil(max(len(x['input_ids']) for x in examples) / 8) * 8
        words = max(len(x['valid_ids']) for x in examples)
        result = {'input_ids': torch.full((len(examples), length), self.pad_id, dtype=torch.long),
                  'attention_mask': torch.zeros(len(examples), length, dtype=torch.long),
                  'valid_ids': torch.zeros(len(examples), words, dtype=torch.long),
                  'labels': torch.zeros(len(examples), words, dtype=torch.long),
                  'label_masks': torch.zeros(len(examples), words, dtype=torch.bool)}
        for i, ex in enumerate(examples):
            n, w = len(ex['input_ids']), len(ex['valid_ids'])
            result['input_ids'][i, :n] = torch.tensor(ex['input_ids'])
            result['attention_mask'][i, :n] = 1
            result['valid_ids'][i, :w] = torch.tensor(ex['valid_ids'])
            result['labels'][i, :w] = torch.tensor(ex['labels'])
            result['label_masks'][i, :w] = True
        return result

class BackboneTokenClassifier(nn.Module):
    def __init__(self, config, model_name_or_path=None, pretrained=True, dropout=0.1):
        super().__init__()
        self.config = config
        self.encoder = (
            AutoModel.from_pretrained(model_name_or_path, config=config)
            if pretrained else AutoModel.from_config(config)
        )
        hidden_size = getattr(config, 'hidden_size', None)
        if hidden_size is None:
            raise ValueError(f'Unsupported config without hidden_size: {config.model_type}')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, config.num_labels)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=config.initializer_range)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask, valid_ids, label_masks, labels=None,
                decode=False):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        rows = torch.arange(hidden.shape[0], device=hidden.device).unsqueeze(1)
        word_hidden = hidden[rows, valid_ids]
        logits = self.classifier(self.dropout(word_hidden))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.float()[label_masks], labels[label_masks], reduction='sum')
        tags = None
        if decode:
            tags = [row[mask].tolist() for row, mask in zip(logits.argmax(dim=-1), label_masks)]
        return SimpleNamespace(loss=loss, tags=tags)


def feature_dataset(cfg, split, tokenizer):
    source = Path(cfg['data_files'][split])
    fingerprint = hashlib.sha256(source.read_bytes()).hexdigest()
    limit = cfg.get(f'{split}_limit')
    key = hashlib.sha256(f'word-prefix-v2:{fingerprint}:{cfg["model_name_or_path"]}:{cfg["max_seq_length"]}:{limit}'.encode()).hexdigest()[:20]
    cache = Path(cfg['cache_dir']) / f'{split}-{key}.pt'
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        packed = torch.load(cache, map_location='cpu', weights_only=True)
    else:
        examples, truncated, discarded = [], 0, 0
        sentences = read_conll(source)
        if limit is not None:
            sentences = itertools.islice(sentences, limit)
        for words, tags in tqdm(sentences, desc=f'Tokenize {split}'):
            ex = encode_words(words, tags, tokenizer, cfg['max_seq_length'])
            truncated += int(len(ex['labels']) < len(tags))
            discarded += len(tags) - len(ex['labels'])
            examples.append(ex)
        packed = {'examples': examples, 'truncated_sentences': truncated,
                  'discarded_words': discarded, 'sha256': fingerprint, 'limit': limit}
        temp = cache.with_suffix('.tmp')
        torch.save(packed, temp)
        os.replace(temp, cache)
    if not packed['examples']:
        raise ValueError(f'{source} contains no sentences')
    print(f'{split}: {len(packed["examples"]):,} sentences (limit={limit}); '
          f'{packed["truncated_sentences"]:,} truncated; {packed["discarded_words"]:,} words excluded', flush=True)
    return Features(packed['examples']), {k: v for k, v in packed.items() if k != 'examples'}


def loader(dataset, cfg, tokenizer, train=False, seed=None):
    generator = torch.Generator().manual_seed(cfg['seed'] if seed is None else seed)
    return DataLoader(dataset, batch_size=cfg['train_batch_size'] if train else cfg['eval_batch_size'],
                      shuffle=train, generator=generator, num_workers=cfg['num_workers'],
                      pin_memory=cfg['device'] == 'cuda', collate_fn=Collator(tokenizer.pad_token_id),
                      persistent_workers=cfg['num_workers'] > 0)


def gpu_batch(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def amp(cfg, device):
    dtype = torch.bfloat16 if cfg['precision'] == 'bf16' else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == 'cuda' and cfg['precision'] != 'fp32')


@torch.inference_mode()
def evaluate(model, iterator, cfg, device):
    model.eval()
    golds, preds, total_loss = [], [], 0.0
    for batch in tqdm(iterator, desc='Evaluate'):
        batch = gpu_batch(batch, device)
        with amp(cfg, device):
            output = model(**batch, decode=True)
        total_loss += output.loss.item()
        for gold, mask, predicted in zip(batch['labels'].cpu(), batch['label_masks'].cpu(), output.tags):
            actual = gold[mask].tolist()
            assert len(actual) == len(predicted), 'Token-classifier alignment mismatch'
            golds.append([LABELS[i] for i in actual])
            preds.append([LABELS[i] for i in predicted])
    flat_gold = [x for sent in golds for x in sent]
    flat_pred = [x for sent in preds for x in sent]
    strict_precision = entity_precision(
        golds, preds, average='micro', mode='strict', scheme=IOB2, zero_division=0)
    strict_recall = entity_recall(
        golds, preds, average='micro', mode='strict', scheme=IOB2, zero_division=0)
    strict_micro_f1 = entity_f1(
        golds, preds, average='micro', mode='strict', scheme=IOB2, zero_division=0)
    strict_macro_f1 = entity_f1(
        golds, preds, average='macro', mode='strict', scheme=IOB2, zero_division=0)
    return {'loss_per_sentence': total_loss / len(iterator.dataset),
            'bio_accuracy': accuracy_score(flat_gold, flat_pred),
            'bio_macro_f1': f1_score(flat_gold, flat_pred, average='macro', labels=LABELS, zero_division=0),
            'entity_strict_precision_micro': strict_precision,
            'entity_strict_recall_micro': strict_recall,
            'entity_strict_micro_f1': strict_micro_f1,
            'entity_strict_macro_f1': strict_macro_f1}, golds, preds

def save_checkpoint(path, model, cfg, **extra):
    # CPU tensors keep saved models portable and avoid a second VRAM copy.
    args = SimpleNamespace(**cfg, task='pap_ner', model_arch='softmax', label2id=LABELS,
                           id2label=dict(enumerate(LABELS)))
    data = {'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
            'classes': LABELS, 'args': args, 'config': cfg, **extra}
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    torch.save(data, tmp)
    os.replace(tmp, path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup(cfg, pretrained=True):
    device = torch.device('cuda:0' if cfg['device'] == 'cuda' else 'cpu')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Check nvidia-smi or use local_smoke.')
    if device.type == 'cpu' and cfg['precision'] != 'fp32':
        raise ValueError('CPU mode requires fp32 precision.')
    if device.type == 'cuda' and cfg['precision'] == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 unsupported; change precision to fp16.')
    seed_all(cfg['seed'])
    if device.type == 'cpu':
        torch.set_num_threads(cfg.get('cpu_threads', 4))
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(
        cfg['model_name_or_path'], use_fast=cfg['use_fast_tokenizer'])
    if tokenizer.pad_token_id is None:
        raise ValueError(f"Tokenizer has no pad token: {cfg['model_name_or_path']}")
    config = AutoConfig.from_pretrained(cfg['model_name_or_path'], num_labels=len(LABELS),
                                       id2label=dict(enumerate(LABELS)),
                                       label2id={v: k for k, v in enumerate(LABELS)})
    for attr in ('hidden_dropout_prob', 'attention_probs_dropout_prob', 'classifier_dropout'):
        if hasattr(config, attr):
            setattr(config, attr, cfg['dropout'])
    config._attn_implementation = 'eager'
    model = BackboneTokenClassifier(
        config, model_name_or_path=cfg['model_name_or_path'], pretrained=pretrained,
        dropout=cfg['dropout'])
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise FloatingPointError(f'Non-finite model parameter before training: {name}')
    if cfg['gradient_checkpointing']:
        if not hasattr(model.encoder, 'gradient_checkpointing_enable'):
            raise RuntimeError(f"{cfg['model_choice']} does not support gradient checkpointing")
        model.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False})
    model.to(device)
    if device.type == 'cuda':
        print(f'GPU={torch.cuda.get_device_name(0)}; VRAM={torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB; precision={cfg["precision"]}', flush=True)
    else:
        print('Device=CPU; precision=fp32', flush=True)
    return model, tokenizer, device

def optimizer_for(model, cfg):
    encoder = list(model.encoder.named_parameters())
    def is_no_decay(name):
        lowered = name.lower()
        return lowered.endswith('bias') or 'layernorm.weight' in lowered or 'layer_norm.weight' in lowered
    return torch.optim.AdamW([
        {'params': [p for n, p in encoder if not is_no_decay(n)], 'weight_decay': cfg['weight_decay']},
        {'params': [p for n, p in encoder if is_no_decay(n)], 'weight_decay': 0.0},
        {'params': list(model.classifier.parameters()),
         'lr': cfg['classifier_learning_rate'], 'weight_decay': cfg['weight_decay']}],
        lr=cfg['learning_rate'], eps=cfg['adam_epsilon'])


def train(cfg, smoke=False):
    out = Path(cfg['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    last = out / 'last_checkpoint.pt'
    if not smoke and (last.exists() or (out / 'best_model.pt').exists()) and not cfg['resume']:
        raise RuntimeError('Run already exists: set resume=True or choose a new run_name.')
    if not smoke and cfg['resume'] and not last.exists():
        raise FileNotFoundError(f'Resume requested but missing {last}')
    model, tokenizer, device = setup(cfg)
    train_set, train_stats = feature_dataset(cfg, 'train', tokenizer)
    dev_set, dev_stats = feature_dataset(cfg, 'dev', tokenizer)
    if smoke:
        # Longest examples exercise near-worst-case padding/VRAM, not just short sentences.
        longest = sorted(range(len(train_set)), key=lambda i: len(train_set[i]['input_ids']), reverse=True)
        train_set = Subset(train_set, longest[:cfg['train_batch_size'] * cfg['smoke_batches']])
        dev_set = Subset(dev_set, range(min(len(dev_set), cfg['eval_batch_size'])))
    iterator = loader(train_set, cfg, tokenizer, train=True)
    steps_per_epoch = math.ceil(len(iterator) / cfg['gradient_accumulation_steps'])
    optimizer = optimizer_for(model, cfg)
    total_steps = steps_per_epoch * cfg['epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * cfg['warmup_proportion']), total_steps)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg['precision'] == 'fp16')
    first_epoch, best_score, bad_epochs, history = 0, -1.0, 0, []
    if cfg['resume'] and not smoke:
        # Load only the checkpoint created by this run; it contains a Namespace + RNG state.
        saved = torch.load(last, map_location='cpu', weights_only=False)
        keys = ['model_name_or_path', 'model_choice', 'use_fast_tokenizer', 'max_seq_length', 'train_batch_size', 'eval_batch_size',
                'gradient_accumulation_steps', 'learning_rate', 'classifier_learning_rate',
                'weight_decay', 'adam_epsilon', 'max_grad_norm', 'warmup_proportion', 'epochs',
                'precision', 'seed', 'selection_metric', 'gradient_checkpointing', 'early_stop',
                'dropout', 'train_limit', 'dev_limit', 'dataset_repo', 'dataset_revision']
        for key in keys:
            if saved['config'][key] != cfg[key]:
                raise ValueError(f'Cannot resume after changing {key}; use a new run_name.')
        if saved['dataset_stats'] != {'train': train_stats, 'dev': dev_stats}:
            raise ValueError('Training/dev data changed; use a new run_name.')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        scaler.load_state_dict(saved['scaler'])
        first_epoch = saved['epoch'] + 1
        best_score, bad_epochs, history = saved['best_score'], saved['bad_epochs'], saved['history']
        random.setstate(saved['rng']['python'])
        np.random.set_state(saved['rng']['numpy'])
        torch.set_rng_state(saved['rng']['torch'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(saved['rng']['cuda'])
        del saved
        print(f'Resume from epoch {first_epoch + 1}; best dev score={best_score:.6f}', flush=True)
    tokenizer.save_pretrained(out / 'tokenizer')
    model.config.save_pretrained(out / 'model_config')
    write_json(out / 'run_config.json', cfg)
    write_json(out / 'dataset_stats.json', {'train': train_stats, 'dev': dev_stats})
    writer = SummaryWriter(str(out / 'tensorboard'))
    for epoch in range(first_epoch, 1 if smoke else cfg['epochs']):
        if bad_epochs >= cfg['early_stop']:
            print('Early stopping reached.', flush=True)
            break
        iterator = loader(train_set, cfg, tokenizer, train=True, seed=cfg['seed'] + epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss, started = 0.0, time.time()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        bar = tqdm(iterator, desc=f'Epoch {epoch + 1}/{cfg["epochs"]}')
        for step, batch in enumerate(bar):
            batch = gpu_batch(batch, device)
            # Sum word-level cross-entropy losses. With accumulation, sum microbatch
            # losses to match a larger physical batch, including the last partial batch.
            with amp(cfg, device):
                output = model(**batch)
            if not torch.isfinite(output.loss):
                raise FloatingPointError(
                    f'Non-finite loss at epoch={epoch + 1}, batch={step + 1}, '
                    f'precision={cfg["precision"]}; inspect model parameters and batch data.')
            total_loss += output.loss.item()
            scaler.scale(output.loss).backward()
            if (step + 1) % cfg['gradient_accumulation_steps'] == 0 or step + 1 == len(iterator):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['max_grad_norm'],
                                               error_if_nonfinite=cfg['precision'] != 'fp16')
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            bar.set_postfix(loss=f'{output.loss.item():.2f}', lr=f'{optimizer.param_groups[0]["lr"]:.2e}')
        metrics, _, _ = evaluate(model, loader(dev_set, cfg, tokenizer), cfg, device)
        record = {'epoch': epoch + 1, 'train_loss_per_sentence': total_loss / len(train_set),
                  **metrics, 'seconds': time.time() - started,
                  'peak_vram_gib': torch.cuda.max_memory_allocated() / 2**30 if device.type == 'cuda' else 0.0}
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if smoke:
            write_json(out / 'smoke_metrics.json', record)
            writer.close()
            return
        history.append(record)
        for key, val in record.items():
            if key != 'epoch':
                writer.add_scalar(key, val, epoch + 1)
        writer.flush()
        score = metrics[cfg['selection_metric']]
        if score > best_score:
            best_score, bad_epochs = score, 0
            save_checkpoint(out / 'best_model.pt', model, cfg, epoch=epoch, metrics=metrics)
        else:
            bad_epochs += 1
        write_json(out / 'history.json', history)
        save_checkpoint(last, model, cfg, epoch=epoch, best_score=best_score,
                        bad_epochs=bad_epochs, history=history,
                        dataset_stats={'train': train_stats, 'dev': dev_stats},
                        optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                        scaler=scaler.state_dict(), rng={'python': random.getstate(),
                        'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
                        'cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []})
    writer.close()
    print(f'Training finished. Best dev {cfg["selection_metric"]}={best_score:.6f}', flush=True)

def test(cfg):
    out = Path(cfg['output_dir'])
    saved = torch.load(out / 'best_model.pt', map_location='cpu', weights_only=False)
    actual_cfg = dict(saved['config'])
    if (actual_cfg['dataset_repo'], actual_cfg['dataset_revision']) != (cfg['dataset_repo'], cfg['dataset_revision']):
        raise ValueError('Test dataset revision differs from the training checkpoint.')
    actual_cfg.update({k: cfg[k] for k in ['data_files', 'cache_dir', 'output_dir', 'eval_batch_size', 'num_workers']})
    model, tokenizer, device = setup(actual_cfg, pretrained=False)
    model.load_state_dict(saved['model'])
    del saved
    dataset, stats = feature_dataset(actual_cfg, 'test', tokenizer)
    metrics, gold, pred = evaluate(model, loader(dataset, actual_cfg, tokenizer), actual_cfg, device)
    write_json(out / 'test_metrics.json', {'dataset_repo': actual_cfg['dataset_repo'],
                                           'dataset_revision': actual_cfg['dataset_revision'],
                                           'metrics': metrics, 'dataset': stats})
    write_json(out / 'test_predictions.json', {'gold': gold, 'pred': pred})
    report = entity_report(gold, pred, mode='strict', scheme=IOB2, digits=4, zero_division=0)
    report += '\nBIO token report\n' + classification_report(
        [x for s in gold for x in s], [x for s in pred for x in s], labels=LABELS, digits=4, zero_division=0)
    (out / 'test_report.txt').write_text(report, encoding='utf-8')
    print(json.dumps(metrics, indent=2), flush=True)
    print(report, flush=True)


def predict(cfg, text):
    # Input must already be word-segmented (underscores), as in PAP_NER.
    out = Path(cfg['output_dir'])
    saved = torch.load(out / 'best_model.pt', map_location='cpu', weights_only=False)
    actual_cfg = saved['config']
    model, tokenizer, device = setup(actual_cfg, pretrained=False)
    model.load_state_dict(saved['model'])
    model.eval()
    words = text.split()
    if not words:
        raise ValueError('Empty input')
    ex = encode_words(words, [0] * len(words), tokenizer, actual_cfg['max_seq_length'])
    if len(ex['labels']) < len(words):
        print(f'Input truncated: keeping {len(ex["labels"])} / {len(words)} words.', flush=True)
    batch = gpu_batch(Collator(tokenizer.pad_token_id)([ex]), device)
    with torch.inference_mode(), amp(actual_cfg, device):
        tags = model(**batch, decode=True).tags[0]
    for word, tag in zip(words, tags):
        print(f'{word:45} {LABELS[tag]}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'smoke', 'test', 'predict'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--text', default='')
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    status = Path(cfg['output_dir']) / f'{args.mode}_status.json'
    status.parent.mkdir(parents=True, exist_ok=True)
    write_json(status, {'state': 'running', 'pid': os.getpid()})
    try:
        if args.mode in ['train', 'smoke']:
            train(cfg, smoke=args.mode == 'smoke')
        elif args.mode == 'test':
            test(cfg)
        else:
            predict(cfg, args.text)
        write_json(status, {'state': 'completed', 'pid': os.getpid()})
    except BaseException as exc:
        write_json(status, {'state': 'failed', 'pid': os.getpid(), 'error': str(exc)})
        traceback.print_exc()
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            print('OOM: giảm physical train batch nhưng giữ effective batch = 32 bằng gradient accumulation; '
                  'ví dụ batch=4, accumulation=8. Giảm eval_batch_size nếu cần. Use a new run_name.', flush=True)
        raise


if __name__ == '__main__':
    main()
