import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch
import yaml
from diffusers import FlowMatchEulerDiscreteScheduler
from tqdm import tqdm

from crossfeed.unison.omni import DEFAULT_MODEL
from crossfeed.utils import append_jsonl, nonempty_file, read_jsonl, seed_everything, seed_for, write_jsonl
from unison.pipelines.infer import (MAX_AUDIO_DURATION, _load_model, init_audio_vae, init_text_hidden_extractor,
                                    sample_latents, sync_omni_dim_with_text_encoder)

THIRD_PARTY = Path(__file__).resolve().parents[2] / 'third_party' / 'unison'
MODEL_CONFIG = str(THIRD_PARTY / 'config' / 'D24S0_O_20ch.yaml')
VAE_CONFIG = str(THIRD_PARTY / 'models' / 'mmaudio' / 'vae_config_16k.yaml')


def floats(value):
    return [float(x) for x in str(value).replace(',', ' ').split()]


def ints(value):
    return [int(x) for x in str(value).replace(',', ' ').split()]


def interval(value):
    values = floats(value)
    return tuple(values) if values else None


def trim_trailing_silence(wav, sample_rate, min_seconds, top_db=40.0, context_seconds=0.25):
    import librosa
    if not len(wav) or not np.isfinite(wav).all() or float(np.max(np.abs(wav))) <= 1e-8:
        return wav
    _, (_, end) = librosa.effects.trim(wav, top_db=top_db)
    if end <= 0 or (len(wav) - end) / sample_rate <= min_seconds:
        return wav
    cut = min(len(wav), int(end) + round(context_seconds * sample_rate))
    return wav[:cut] if 0 < cut < len(wav) else wav


class Generator:
    def __init__(self, dit_ckpt, omni_model_path=DEFAULT_MODEL, model_config=MODEL_CONFIG, vae_config=VAE_CONFIG,
                 device='cuda'):
        self.device = torch.device(device)
        with open(model_config) as handle:
            config = yaml.safe_load(handle)
        depth = config.get('mm_double_blocks_depth', 0) + config.get('mm_single_blocks_depth', 0)
        self.text_encoder = init_text_hidden_extractor('omni', omni_model_path, None, depth, self.device,
                                                       dtype=torch.bfloat16,
                                                       omni_last_layer_idx=config.get('omni_last_layer_idx', -1))
        sync_omni_dim_with_text_encoder(config, self.text_encoder)
        self.model = _load_model(SimpleNamespace(model_ckpt=dit_ckpt), self.device, config)
        self.vae, self.vae_scale, self.sample_rate = init_audio_vae(vae_config)
        self.vae.to(self.device).eval()
        self.scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
        with torch.no_grad():
            dummy = torch.zeros(1, int(MAX_AUDIO_DURATION * self.sample_rate), device=self.device)
            self.frames = int(self.vae.wrapped_encode(dummy).shape[-1])

    def latents(self, prompts, seeds, cfg, steps, cfg_interval):
        channels = self.model.config.in_channels
        noise = torch.stack([torch.randn(channels, self.frames, generator=torch.Generator().manual_seed(seed))
                             for seed in seeds])
        return sample_latents(self.model, self.scheduler, self.text_encoder, prompts, num_inference_steps=steps,
                              guidance_scale=cfg, device=self.device, target_frames=self.frames,
                              noise_init=noise, cfg_interval=cfg_interval)

    def decode(self, z):
        with torch.no_grad():
            audio = self.vae.wrapped_decode((z / self.vae_scale).to(getattr(self.vae, 'dtype', torch.bfloat16)))
        audio = audio.float().cpu()
        out = []
        for wav in audio:
            wav = wav.mean(0) if wav.dim() > 1 else wav
            out.append(wav[:int(MAX_AUDIO_DURATION * self.sample_rate)].clamp(-1, 1).numpy().astype(np.float32))
        return out


def generate(instructions, out_dir, dit_ckpt, omni_model_path=DEFAULT_MODEL, k=8, cfg=6.0, sample_seeds=None,
             cfg_interval=None, steps=50, batch_size=320, seed=531, save_latents=True, trim_silence=0.0,
             resume_missing=False):
    out_dir = Path(out_dir)
    sample_seeds = sample_seeds or list(range(1, k + 1))
    manifest = out_dir / 'gen_manifest.jsonl'
    completed = read_jsonl(manifest)
    if resume_missing:
        completed = [row for row in completed
                     if nonempty_file(out_dir / 'audio' / f'{row["inst_id"]}_k{row["k"]}.wav')]
    done = {(row['inst_id'], row['k']) for row in completed}
    todo = [(inst, j) for inst in instructions for j in range(k) if (inst['id'], j) not in done]
    print(f'[generate] {len(todo)} clips to generate ({len(done)} done)', flush=True)
    if not todo:
        return manifest
    seed_everything(seed)
    (out_dir / 'audio').mkdir(parents=True, exist_ok=True)
    if save_latents:
        (out_dir / 'latents').mkdir(parents=True, exist_ok=True)
    generator = Generator(dit_ckpt, omni_model_path)
    for start in tqdm(range(0, len(todo), batch_size), desc='[generate]'):
        batch = todo[start:start + batch_size]
        prompts = [inst.get('tts_prompt') or inst['prompt'] for inst, _ in batch]
        seeds = [seed_for(seed, 'noise', inst['id'], sample_seeds[j]) for inst, j in batch]
        z = generator.latents(prompts, seeds, cfg, steps, cfg_interval)
        rows = []
        for (inst, j), latent, wav, prompt in zip(batch, z, generator.decode(z), prompts):
            if trim_silence:
                wav = trim_trailing_silence(wav, generator.sample_rate, trim_silence)
            wav_path = out_dir / 'audio' / f'{inst["id"]}_k{j}.wav'
            output = wav_path.with_suffix('.tmp.wav') if resume_missing else wav_path
            sf.write(output, wav, generator.sample_rate, subtype='PCM_16')
            if resume_missing:
                output.replace(wav_path)
            row = dict(inst_id=inst['id'], k=j, wav=str(wav_path), prompt=prompt, task=inst.get('task', ''),
                       cfg=cfg, steps=steps, duration=round(len(wav) / generator.sample_rate, 2))
            if save_latents:
                latent_path = out_dir / 'latents' / f'{inst["id"]}_k{j}.npy'
                np.save(latent_path, latent.detach().to(torch.float16).cpu().numpy())
                row['latent'] = str(latent_path)
            rows.append(row)
        if resume_missing:
            completed.extend(rows)
            write_jsonl(manifest, completed)
        else:
            append_jsonl(manifest, rows)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instructions', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--dit_ckpt', required=True)
    parser.add_argument('--omni_model_path', default=DEFAULT_MODEL)
    parser.add_argument('--k', type=int, default=8)
    parser.add_argument('--cfg', type=float, default=6.0)
    parser.add_argument('--cfg_interval', type=interval, default=None)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=320)
    parser.add_argument('--seed', type=int, default=531)
    parser.add_argument('--save_latents', type=int, default=1)
    parser.add_argument('--trim_silence', type=float, default=0.0)
    args = parser.parse_args()
    generate(read_jsonl(args.instructions), args.out_dir, args.dit_ckpt, args.omni_model_path, args.k, args.cfg,
             cfg_interval=args.cfg_interval, steps=args.steps, batch_size=args.batch_size, seed=args.seed,
             save_latents=bool(args.save_latents), trim_silence=args.trim_silence)


if __name__ == '__main__':
    main()
