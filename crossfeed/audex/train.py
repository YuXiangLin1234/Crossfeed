import argparse
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

import soundfile as sf
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.checkpoint import checkpoint
from transformers import get_scheduler, set_seed

from crossfeed.audex.hf import load, rating_ids
from crossfeed.audex.runtime import add_official_paths, strip_tags
from crossfeed.prompts import QA_PROMPT
from crossfeed.utils import read_jsonl, rng_for


def checkpoint_layers(base):
    for layer in base.model.layers:
        original = layer.forward

        def forward(*args, _forward=original, **kwargs):
            if torch.is_grad_enabled():
                return checkpoint(_forward, *args, use_reentrant=False, **kwargs)
            return _forward(*args, **kwargs)
        layer.forward = forward


def prompt_text(row):
    return row['prompt'] if 'score_regression_target' in row else strip_tags(row['prompt'])


def shuffle_options(row, rng):
    if 'options' not in row:
        return row
    position = rng.randrange(4)
    correct = row['options'][row['answer_idx']]
    distractors = [o for i, o in enumerate(row['options']) if i != row['answer_idx']]
    rng.shuffle(distractors)
    options = distractors[:position] + [correct] + distractors[position:]
    rendered = '\n'.join(f'({letter}) {option}' for letter, option in zip('ABCD', options))
    return dict(row, options=options, answer_idx=position, prompt=QA_PROMPT.format(question=row['question'], options=rendered),
                answer=f'({"ABCD"[position]}) {correct}')


def token_batches(items, size, budget, cost):
    group, widest = [], 0
    for item in items:
        value = max(1, int(cost(item)))
        if group and (len(group) >= size or max(widest, value) * (len(group) + 1) > budget):
            yield group
            group, widest = [], 0
        group.append(item)
        widest = max(widest, value)
    if group:
        yield group


class Trainer:
    def __init__(self, args):
        add_official_paths()
        from inference_hf import build_inputs
        self.args = args
        self.generation = args.mode == 'generation'
        self.name = 'generation_training' if self.generation else 'understanding_training'
        self.seed = args.seed
        set_seed(self.seed)
        self.model_base, self.tokenizer, self.extractor, self.config = load(args.model)
        self.build_inputs = build_inputs
        targets = [name for name, module in self.model_base.named_modules()
                   if name.startswith('model.layers.') and isinstance(module, torch.nn.Linear)]
        self.targets = targets
        self.model = get_peft_model(self.model_base, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                                                                lora_dropout=0.0, target_modules=targets, bias='none',
                                                                task_type='CAUSAL_LM'))
        self.model.config.use_cache = False
        checkpoint_layers(self.model_base)
        self.model.enable_input_require_grads()
        self.model.train()
        self.anchors = torch.tensor(rating_ids(self.tokenizer), device='cuda')
        self.clip_seconds = float(getattr(self.config, 'sound_clip_duration', 30.0))
        self.embedding_size = int(getattr(self.config, 'sound_embedding_size', 750))

    def clips(self, row):
        info = sf.info(row['wav'])
        return max(1, math.ceil(info.frames / info.samplerate / self.clip_seconds))

    def cost(self, row):
        if self.generation:
            return max(len(self.tokenizer.encode(saved['prompt'], add_special_tokens=False)) + len(saved['token_ids'])
                       for saved in (read_jsonl(row['chosen'])[0], read_jsonl(row['rejected'])[0]))
        text = self.tokenizer.encode(prompt_text(row) + row.get('answer', ''), add_special_tokens=False)
        return self.clips(row) * self.embedding_size + len(text) + 64

    def example(self, row=None, token_file=None):
        if token_file:
            saved = read_jsonl(token_file)[0]
            inputs = {k: v.cuda() for k, v in self.tokenizer(saved['prompt'], return_tensors='pt',
                                                              add_special_tokens=False).items()}
            target, regression = saved['token_ids'], None
        else:
            ids, mask, features, _ = self.build_inputs(self.tokenizer, self.extractor, self.config, str(row['wav']),
                                                       prompt_text(row), 16000, False, 'none')
            inputs = dict(input_ids=ids.cuda(), attention_mask=mask.cuda(), input_features=features.unsqueeze(0).cuda())
            regression = row.get('score_regression_target')
            if regression is not None:
                newline = torch.tensor([self.tokenizer.encode('\n', add_special_tokens=False)], device='cuda')
                inputs['input_ids'] = torch.cat([inputs['input_ids'], newline], dim=1)
            target = (self.tokenizer.encode(row['answer'] + '<|im_end|>', add_special_tokens=False)
                      if 'answer' in row else [])
        prompt_length = inputs['input_ids'].shape[1]
        inputs['input_ids'] = torch.cat([inputs['input_ids'], inputs['input_ids'].new_tensor([target])], dim=1)
        inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
        if inputs['input_ids'].shape[1] > self.args.max_tokens:
            raise ValueError('training example exceeds --max_tokens')
        labels = inputs['input_ids'].clone()
        labels[:, :prompt_length] = -100
        return inputs, labels, regression

    def collate(self, examples):
        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else (self.tokenizer.eos_token_id or 0)
        width = max(inputs['input_ids'].shape[1] for inputs, _, _ in examples)
        n = len(examples)
        input_ids = torch.full((n, width), pad, dtype=torch.long, device='cuda')
        mask = torch.zeros((n, width), dtype=torch.long, device='cuda')
        labels = torch.full((n, width), -100, dtype=torch.long, device='cuda')
        for i, (inputs, target, _) in enumerate(examples):
            length = inputs['input_ids'].shape[1]
            input_ids[i, :length], mask[i, :length], labels[i, :length] = (inputs['input_ids'][0],
                                                                           inputs['attention_mask'][0], target[0])
        batch = dict(input_ids=input_ids, attention_mask=mask)
        features = [inputs.get('input_features') for inputs, _, _ in examples]
        if features[0] is not None:
            batch['input_features'] = torch.cat(features, dim=0)
        targets = torch.tensor([0.0 if r is None else float(r) for _, _, r in examples], device='cuda')
        regression_mask = torch.tensor([r is not None for _, _, r in examples], dtype=torch.bool, device='cuda')
        return batch, labels, targets, regression_mask

    def logp(self, batch, labels):
        logits = self.model(**batch, use_cache=False).logits[:, :-1]
        target = labels[:, 1:]
        losses = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), target.reshape(-1),
                                                   ignore_index=-100, reduction='none').reshape(target.shape)
        return -losses.sum(dim=1)

    def dpo_backward(self, rows, update_size):
        chosen = self.collate([self.example(token_file=row['chosen']) for row in rows])
        rejected = self.collate([self.example(token_file=row['rejected']) for row in rows])
        with self.model.disable_adapter(), torch.no_grad():
            reference = self.logp(*chosen[:2]) - self.logp(*rejected[:2])
        policy = self.logp(*chosen[:2]) - self.logp(*rejected[:2])
        loss = -torch.nn.functional.logsigmoid(self.args.beta * (policy - reference)).mean()
        (loss * (len(rows) / update_size)).backward()
        return loss.item()

    def understanding_backward(self, rows, update_size):
        batch, labels, targets, regression_mask = self.collate([self.example(row=row) for row in rows])
        logits = self.model(**batch, use_cache=False).logits
        shifted = labels[:, 1:]
        ce = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(), shifted.reshape(-1),
                                               ignore_index=-100, reduction='sum')
        regression = ce.new_zeros(())
        if regression_mask.any():
            positions = batch['attention_mask'].sum(dim=1) - 1
            first = logits[torch.arange(len(rows), device=logits.device), positions].float()[regression_mask]
            predictions = torch.softmax(first, dim=-1)[:, self.anchors] @ first.new_tensor([1, 2, 3, 4, 5])
            regression = torch.nn.functional.mse_loss(predictions, targets[regression_mask], reduction='sum')
        denominator = int((shifted != -100).sum()) + self.args.score_weight * int(regression_mask.sum())
        loss = (ce + self.args.score_weight * regression) / denominator
        (loss * (len(rows) / update_size)).backward()
        return loss.item()

    def run(self):
        args = self.args
        data = read_jsonl(args.data)
        effective = args.effective_batch_size
        total = args.epochs * math.ceil(len(data) / effective)
        optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=args.lr)
        scheduler = get_scheduler('constant_with_warmup', optimizer=optimizer, num_warmup_steps=math.floor(total * 0.05),
                                  num_training_steps=total)
        rng, option_rng = rng_for(args.seed, self.name, 'order'), rng_for(args.seed, self.name, 'options')
        bucket = (lambda row: 0) if self.generation else self.clips
        step = 0
        for epoch in range(args.epochs):
            rng.shuffle(data)
            if not self.generation:
                data = [shuffle_options(row, option_rng) for row in data]
            ordered = sorted(data, key=lambda row: (bucket(row), self.cost(row)))
            updates = [ordered[i:i + effective] for i in range(0, len(ordered), effective)]
            rng.shuffle(updates)
            optimizer.zero_grad(set_to_none=True)
            for update in updates:
                buckets = defaultdict(list)
                for row in update:
                    buckets[bucket(row)].append(row)
                loss = None
                for rows in buckets.values():
                    for micro in token_batches(rows, min(args.micro_batch_size, effective), args.micro_batch_tokens,
                                               self.cost):
                        loss = (self.dpo_backward if self.generation else self.understanding_backward)(micro, len(update))
                        if self.generation:
                            torch.cuda.empty_cache()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                print(f'[{self.name}] step={step}/{total} loss={loss:.5f} lr={scheduler.get_last_lr()[0]:.3g}', flush=True)
        self.save(step, len(data))

    def save(self, steps, examples):
        output = Path(self.args.output_dir) / 'checkpoint-final'
        output.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output / 'adapter')
        merged = self.model.merge_and_unload()
        merged.config.use_cache = True
        merged.config.audio_preprocessor_path = 'audio_preprocessor'
        merged.save_pretrained(output, safe_serialization=True)
        self.tokenizer.save_pretrained(output)
        self.extractor.save_pretrained(output / 'audio_preprocessor')
        for source in Path(self.args.model).glob('*.py'):
            shutil.copy2(source, output / source.name)
        (output / 'training_config.json').write_text(json.dumps(dict(
            mode=self.args.mode, examples=examples, steps=steps, lr=self.args.lr, epochs=self.args.epochs,
            effective_batch_size=self.args.effective_batch_size, lora_r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha, beta=self.args.beta, seed=self.seed), indent=2))
        (output / 'COMPLETE').write_text('ok\n')
        print(f'[{self.name}] saved {output}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['understanding', 'generation'], required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--effective_batch_size', type=int, default=None)
    parser.add_argument('--micro_batch_size', type=int, default=32)
    parser.add_argument('--micro_batch_tokens', type=int, default=32768)
    parser.add_argument('--max_tokens', type=int, default=8192)
    parser.add_argument('--lora_r', type=int, default=64)
    parser.add_argument('--lora_alpha', type=int, default=128)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--score_weight', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=531)
    args = parser.parse_args()
    generation = args.mode == 'generation'
    args.lr = args.lr or (1e-5 if generation else 5e-5)
    args.effective_batch_size = args.effective_batch_size or (64 if generation else 32)
    if (Path(args.output_dir) / 'checkpoint-final' / 'COMPLETE').is_file():
        print(f'[train] {args.output_dir} already complete', flush=True)
        return
    Trainer(args).run()


if __name__ == '__main__':
    main()
