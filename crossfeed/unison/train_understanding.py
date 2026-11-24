import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import get_scheduler

from crossfeed.mcq import answer_text, qa_prompt
from crossfeed.unison.omni import DEFAULT_MODEL, SR, chat_messages, load_processor, load_thinker, rating_token_ids
from crossfeed.utils import read_jsonl, rng_for, seed_for

LORA_TARGETS = (r'^model\.layers\.\d+\.'
                r'(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$')
ANCHORS = (1.0, 2.0, 3.0, 4.0, 5.0)


def load_wav(path):
    wav, sr = sf.read(path, dtype='float32', always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    return np.clip(wav * 32768.0, -32768, 32767).astype(np.int16)


class UnderstandingDataset(Dataset):
    def __init__(self, rows, seed):
        self.rows, self.seed = rows, seed
        paths = sorted({row['wav'] for row in rows})
        with ThreadPoolExecutor(max_workers=32) as pool:
            self.audio = dict(zip(paths, pool.map(load_wav, paths)))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        index, group_end = item
        row = self.rows[index]
        wav = self.audio[row['wav']].astype(np.float32) / 32768.0
        if row['kind'] == 'score':
            return dict(audio=wav, user_text=row['prompt'], answer=None, target=row['target'], group_end=group_end)
        rng = np.random.default_rng(seed_for(self.seed, 'mcq_options', index))
        options, answer_idx = row['options'], row['answer_idx']
        position = int(rng.integers(len(options)))
        distractors = [o for j, o in enumerate(options) if j != answer_idx]
        distractors = [distractors[j] for j in rng.permutation(len(distractors)).tolist()]
        options = distractors[:position] + [options[answer_idx]] + distractors[position:]
        return dict(audio=wav, user_text=qa_prompt(row['question'], options), answer=answer_text(options, position),
                    target=None, group_end=group_end)


class GroupedSampler(Sampler):
    def __init__(self, rows, regular_batch, dialogue_batch, effective_batch, seed):
        self.sizes = dict(regular=regular_batch, dialogue=dialogue_batch)
        self.effective, self.seed, self.epoch = effective_batch, seed, 0
        self.indices = dict(regular=[], dialogue=[])
        for index, row in enumerate(rows):
            kind = 'dialogue' if row.get('source') == 'dialogue' and row['kind'] == 'mcq' else 'regular'
            self.indices[kind].append(index)
        self.groups = sum(math.ceil(len(v) / effective_batch) for v in self.indices.values())
        self.micro_batches = 0
        for kind, indices in self.indices.items():
            full, rest = divmod(len(indices), effective_batch)
            self.micro_batches += full * math.ceil(effective_batch / self.sizes[kind])
            self.micro_batches += math.ceil(rest / self.sizes[kind])

    def __iter__(self):
        rng = rng_for(self.seed, 'batches', self.epoch)
        groups = []
        for kind, indices in self.indices.items():
            indices = list(indices)
            rng.shuffle(indices)
            for start in range(0, len(indices), self.effective):
                examples = indices[start:start + self.effective]
                size = self.sizes[kind]
                groups.append([examples[i:i + size] for i in range(0, len(examples), size)])
        rng.shuffle(groups)
        for micro_batches in groups:
            for i, batch in enumerate(micro_batches):
                yield [(index, i + 1 == len(micro_batches)) for index in batch]

    def __len__(self):
        return self.micro_batches


class Collator:
    def __init__(self, processor):
        self.processor, self.tokenizer = processor, processor.tokenizer

    def chat(self, user_text, answer=None):
        text = self.processor.apply_chat_template(chat_messages(user_text, True), tokenize=False,
                                                  add_generation_prompt=True)
        return text if answer is None else text + answer + '<|im_end|>'

    def __call__(self, batch):
        full = [self.chat(b['user_text'], b['answer']) for b in batch]
        prompts = [self.chat(b['user_text']) for b in batch]
        old = self.tokenizer.padding_side
        self.tokenizer.padding_side = 'right'
        try:
            inputs = self.processor(text=full, audio=[b['audio'] for b in batch], return_tensors='pt', padding=True,
                                    padding_side='right', sampling_rate=SR)
        finally:
            self.tokenizer.padding_side = old
        n_prompt = [len(x) for x in self.tokenizer(prompts, add_special_tokens=False)['input_ids']]
        n_full = [len(x) for x in self.tokenizer(full, add_special_tokens=False)['input_ids']]
        labels = inputs['input_ids'].clone()
        labels[inputs['attention_mask'] == 0] = -100
        lengths = inputs['attention_mask'].sum(dim=1)
        for i, b in enumerate(batch):
            if b['answer'] is None:
                labels[i, :] = -100
            else:
                labels[i, :int(lengths[i]) - (n_full[i] - n_prompt[i])] = -100
        out = dict(inputs)
        out['labels'] = labels
        out['targets'] = torch.tensor([b['target'] or 0.0 for b in batch], dtype=torch.float32)
        out['score_mask'] = torch.tensor([b['target'] is not None for b in batch], dtype=torch.bool)
        out['group_end'] = batch[0]['group_end']
        return out


def raft_loss(first_logits, targets, anchor_ids):
    probs = torch.softmax(first_logits.float(), dim=-1)[:, anchor_ids]
    predictions = probs @ first_logits.new_tensor(ANCHORS, dtype=torch.float32)
    return torch.nn.functional.mse_loss(predictions, targets.float(), reduction='sum')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--model_path', default=DEFAULT_MODEL)
    parser.add_argument('--lora_r', type=int, default=64)
    parser.add_argument('--lora_alpha', type=int, default=128)
    parser.add_argument('--lora_dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--regular_batch_size', type=int, default=16)
    parser.add_argument('--dialogue_batch_size', type=int, default=8)
    parser.add_argument('--effective_batch_size', type=int, default=32)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--score_weight', type=float, default=1.0)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=531)
    args = parser.parse_args()
    if (Path(args.output_dir) / 'checkpoint-final' / 'training_summary.json').is_file():
        print(f'[train_understanding] {args.output_dir} already complete', flush=True)
        return

    accelerator = Accelerator(mixed_precision='bf16')
    set_seed(args.seed + accelerator.process_index)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    processor = load_processor(args.model_path)
    anchor_ids = torch.tensor(rating_token_ids(processor.tokenizer), device=accelerator.device)
    model = load_thinker(args.model_path, device='cpu')
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                                             target_modules=LORA_TARGETS, bias='none'))
    trainable = [p for p in model.parameters() if p.requires_grad]
    for parameter in trainable:
        parameter.data = parameter.data.float()
    base = model.get_base_model()
    base.enable_input_require_grads()
    base.gradient_checkpointing_enable()
    model.train()

    rows = read_jsonl(args.data)
    dataset = UnderstandingDataset(rows, args.seed)
    sampler = GroupedSampler(rows, args.regular_batch_size, args.dialogue_batch_size, args.effective_batch_size,
                             args.seed)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.num_workers, collate_fn=Collator(processor),
                        pin_memory=True, persistent_workers=args.num_workers > 0,
                        prefetch_factor=4 if args.num_workers > 0 else None)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = sampler.groups * args.epochs
    scheduler = get_scheduler('constant_with_warmup', optimizer=optimizer,
                              num_warmup_steps=int(total_steps * args.warmup_ratio), num_training_steps=total_steps)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    print(f'[train_understanding] {len(rows)} rows, {total_steps} steps', flush=True)

    step, window, running, count = 0, 0, 0.0, 0
    for epoch in range(args.epochs):
        sampler.epoch = epoch
        for batch in loader:
            group_end = batch.pop('group_end')
            labels, targets = batch.pop('labels'), batch.pop('targets')
            score_mask = batch.pop('score_mask')
            batch = {k: v.to(accelerator.device) if hasattr(v, 'to') else v for k, v in batch.items()}
            labels, targets, score_mask = (x.to(accelerator.device) for x in (labels, targets, score_mask))
            with accelerator.autocast():
                logits = model(**batch).logits
                ce = torch.nn.functional.cross_entropy(logits[..., :-1, :].float().reshape(-1, logits.size(-1)),
                                                       labels[..., 1:].reshape(-1), ignore_index=-100, reduction='sum')
                if score_mask.any():
                    positions = batch['attention_mask'].sum(dim=1) - 1
                    first = logits[torch.arange(len(labels), device=labels.device), positions]
                    regression = raft_loss(first[score_mask], targets[score_mask], anchor_ids)
                else:
                    regression = ce.new_zeros(())
            norm = int((labels[..., 1:] != -100).sum()) + args.score_weight * int(score_mask.sum())
            if norm == 0:
                continue
            loss = ce + args.score_weight * regression
            accelerator.backward(loss)
            window += norm
            running, count = running + loss.item(), count + norm
            if group_end:
                for parameter in trainable:
                    if parameter.grad is not None:
                        parameter.grad.div_(window)
                accelerator.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                window, step = 0, step + 1
                if step % 10 == 0 and accelerator.is_main_process:
                    print(f'[step {step}/{total_steps}] loss={running / max(count, 1):.4f} '
                          f'lr={scheduler.get_last_lr()[0]:.2e}', flush=True)
                    running, count = 0.0, 0
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final = output / 'checkpoint-final'
        accelerator.unwrap_model(model).save_pretrained(str(final / 'adapter'))
        (final / 'training_summary.json').write_text(json.dumps(dict(
            steps=step, rows=len(rows), lr=args.lr, epochs=args.epochs, lora_r=args.lora_r,
            lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            effective_batch_size=args.effective_batch_size, seed=args.seed), indent=2))
        print(f'[train_understanding] saved {final / "adapter"}', flush=True)


if __name__ == '__main__':
    main()
