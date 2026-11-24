# Flow-matching noise and latent preparation adapted from the UNISON training loop.
# https://github.com/lizhaoqing/UNISON/blob/7608c183f9f76e8da73261c95e923376b22f1e33/unison/pipelines/train.py#L546-L559

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.training_utils import EMAModel, compute_density_for_timestep_sampling
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from transformers import get_scheduler

from crossfeed.unison.generate import MODEL_CONFIG
from crossfeed.unison.merge import dit_weights
from crossfeed.utils import read_jsonl
from unison.models.text_encoders.omni_encoder import QwenOmniThinkerExtractor
from unison.models.transformers.backbone import UnisonBackbone


class PairDataset(Dataset):
    def __init__(self, path):
        self.rows = read_jsonl(path)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return dict(caption=row['caption'],
                    win=torch.from_numpy(np.load(row['win_latent_path']).astype(np.float32)),
                    lose=torch.from_numpy(np.load(row['lose_latent_path']).astype(np.float32)))


def collate(batch):
    return dict(caption=[b['caption'] for b in batch], win=torch.stack([b['win'] for b in batch]),
                lose=torch.stack([b['lose'] for b in batch]))


def load_backbone(path, config):
    model = UnisonBackbone(**config)
    path = dit_weights(path)
    state = load_file(str(path), device='cpu') if path.suffix == '.safetensors' else torch.load(path, map_location='cpu')
    if any(k.startswith('shadow_params') for k in state):
        ema = EMAModel(model.parameters(), decay=0.9999, model_cls=UnisonBackbone, model_config=config)
        ema.load_state_dict(state)
        ema.copy_to(model.parameters())
    else:
        model.load_state_dict({k: v for k, v in state.items() if k not in {'decay', 'num_updates', 'shadow_params'}},
                              strict=False)
    return model.float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--pretrained', required=True)
    parser.add_argument('--omni_model_path', required=True)
    parser.add_argument('--model_config', default=MODEL_CONFIG)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--beta', type=float, default=2000.0)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float, default=1000.0)
    parser.add_argument('--ema_decay', type=float, default=0.99)
    parser.add_argument('--ema_every', type=int, default=10)
    parser.add_argument('--seed', type=int, default=531)
    args = parser.parse_args()
    if (Path(args.output_dir) / 'checkpoint-final' / 'training_summary.json').is_file():
        print(f'[train_dit] {args.output_dir} already complete', flush=True)
        return

    accelerator = Accelerator(mixed_precision='bf16')
    set_seed(args.seed + accelerator.process_index)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path(args.model_config).read_text())
    depth = config.get('mm_double_blocks_depth', 0) + config.get('mm_single_blocks_depth', 0)
    text_encoder = QwenOmniThinkerExtractor(model_path=args.omni_model_path, dit_depth=depth, select_mode='interval',
                                            device=accelerator.device, dtype=torch.bfloat16,
                                            omni_last_layer_idx=config.get('omni_last_layer_idx', -1))
    text_encoder.eval().requires_grad_(False)
    omni_dim = config.get('omni_dim', 3584)
    backbone = {**config, 'concat_condition': True}
    backbone.pop('omni_last_layer_idx', None)
    model = load_backbone(args.pretrained, backbone)
    reference = load_backbone(args.pretrained, backbone).eval().requires_grad_(False).to(accelerator.device)
    ema = EMAModel(model.parameters(), decay=args.ema_decay, model_cls=UnisonBackbone, model_config=backbone)

    dataset = PairDataset(args.pairs)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate, pin_memory=True)
    total_steps = args.epochs * math.ceil(len(dataset) / args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_scheduler('constant_with_warmup', optimizer=optimizer,
                              num_warmup_steps=total_steps * int(args.warmup_ratio * 100) // 100,
                              num_training_steps=total_steps)
    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)
    ema.to(accelerator.device)
    print(f'[train_dit] {len(dataset)} pairs, {total_steps} steps, beta={args.beta}, lr={args.lr}', flush=True)

    step = 0
    while step < total_steps:
        model.train()
        for batch in loader:
            win, lose = batch['win'].to(accelerator.device), batch['lose'].to(accelerator.device)
            size = win.shape[0]
            latents = torch.cat([win, lose], dim=0).float()
            frames = latents.shape[2]
            source = torch.zeros(2 * size, latents.shape[1], frames, device=accelerator.device)
            mask = torch.zeros(2 * size, 1, frames, device=accelerator.device)
            with torch.no_grad():
                embeddings, embedding_mask, *extra = text_encoder(batch['caption'])
            last = extra[0] if extra else torch.zeros(size, 1, omni_dim, device=accelerator.device)
            embeddings = [torch.cat([e, e], dim=0).float() for e in embeddings]
            embedding_mask = torch.cat([embedding_mask, embedding_mask], dim=0)
            last = torch.cat([last, last], dim=0).float()
            noise = torch.randn_like(latents[:size]).repeat(2, 1, 1)
            t = compute_density_for_timestep_sampling(weighting_scheme='logit_normal', batch_size=size,
                                                      logit_mean=0.0, logit_std=1.0, mode_scale=1.29)
            t = t.to(device=accelerator.device, dtype=torch.float32).repeat(2)
            noisy = (1.0 - t[:, None, None]) * latents + t[:, None, None] * noise
            x = torch.cat([noisy, source, mask], dim=1)
            inputs = dict(x=x, t=t, duration=None, omni_emb_list=embeddings, omni_last_emb=last,
                          omni_mask=embedding_mask)
            with accelerator.autocast():
                prediction = model(**inputs)[0]
                with torch.no_grad():
                    prediction_ref = reference(**inputs)[0]
            velocity = noise - latents
            error = (prediction.float() - velocity).pow(2).mean(dim=(1, 2))
            error_ref = (prediction_ref.float() - velocity).pow(2).mean(dim=(1, 2))
            inside = -0.5 * args.beta * ((error[:size] - error_ref[:size]) - (error[size:] - error_ref[size:]))
            loss = -F.logsigmoid(inside).mean()
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            if step % args.ema_every == 0:
                ema.step(model.parameters())
            if step % 10 == 0 and accelerator.is_main_process:
                accuracy = ((error[:size] - error_ref[:size]) < (error[size:] - error_ref[size:])).float().mean()
                print(f'[step {step}/{total_steps}] loss={loss.item():.4f} implicit_acc={accuracy.item():.3f} '
                      f'lr={scheduler.get_last_lr()[0]:.2e}', flush=True)
            if step >= total_steps:
                break
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final = output / 'checkpoint-final'
        accelerator.save_model(model, str(final))
        torch.save(ema.state_dict(), os.path.join(final, 'ema_model.pt'))
        (final / 'training_summary.json').write_text(json.dumps(dict(
            steps=step, pairs=len(dataset), lr=args.lr, beta=args.beta, batch_size=args.batch_size,
            ema_decay=args.ema_decay, seed=args.seed), indent=2))
        print(f'[train_dit] saved {final}', flush=True)


if __name__ == '__main__':
    main()
