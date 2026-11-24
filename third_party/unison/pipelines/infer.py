# Vendored from UNISON and modified for CrossFeed.
# https://github.com/lizhaoqing/UNISON/blob/7608c183f9f76e8da73261c95e923376b22f1e33/unison/pipelines/infer.py

import argparse
import hashlib
import os
import re
from pathlib import Path
from typing import List, Optional
import json

import torch
import torch.nn.functional as F
import torch.distributed as dist
import soundfile as sf
import torchaudio.functional as AF
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import EMAModel
from safetensors.torch import load_file as safe_load_file
from tqdm.auto import tqdm
import yaml
import logging

from unison.models.transformers.backbone import UnisonBackbone
from unison.models.text_encoders.omni_encoder import (
    QwenOmniThinkerExtractor,
    resolve_omni_model_path,
)
from unison.models.mmaudio.features_utils import FeaturesUtils


_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VAE_CONFIG_PATH = str(_REPO_ROOT / "models/mmaudio/vae_config_16k.yaml")
DEFAULT_OMNI_MODEL_PATH = os.environ.get("QWEN_OMNI_MODEL_PATH", "Qwen/Qwen2.5-Omni-7B")
DEFAULT_MODEL_CONFIG_PATH = str(_REPO_ROOT / "config/D24S0_O_20ch.yaml")

MAX_AUDIO_DURATION = 10.0
AUDIO_FPS          = 31.25
TARGET_FRAMES      = int(MAX_AUDIO_DURATION * AUDIO_FPS)
DEFAULT_TARGET_SAMPLE_RATE = 16000
DEFAULT_VAE_SCALE_FACTOR   = 0.5
REF_DURATION       = 3.0
TRAINING_TARGET_RMS = 0.1


class _OmniFilter(logging.Filter):
    def filter(self, record):
        if record.levelno != logging.WARNING:
            return True
        return "System prompt modified" not in record.getMessage()

logging.getLogger().addFilter(_OmniFilter())


def init_text_hidden_extractor(
    text_encoder_type: str,
    omni_model_path: str,
    text_encoder_model_path: Optional[str],
    dit_depth: int,
    device,
    dtype=torch.bfloat16,
    omni_last_layer_idx=-1,
):
    omni_path = resolve_omni_model_path(
        text_encoder_type,
        text_encoder_model_path or omni_model_path,
    )
    print(f"  Omni preset={text_encoder_type} path={omni_path}")
    extractor = QwenOmniThinkerExtractor(
        model_path=omni_path,
        dit_depth=dit_depth,
        select_mode="interval",
        device=device,
        dtype=dtype,
        omni_last_layer_idx=omni_last_layer_idx,
    )
    extractor.eval()
    extractor.requires_grad_(False)
    return extractor


def sync_omni_dim_with_text_encoder(model_config, text_extractor):
    backbone = getattr(text_extractor, "text_backbone", None)
    if backbone is None:
        print("[WARN] Text encoder has no text_backbone; skip omni_dim sync.")
        return
    cfg = getattr(backbone, "config", None)
    hidden = getattr(cfg, "hidden_size", None) if cfg is not None else None
    if hidden is None:
        print("[WARN] Could not read text encoder hidden_size; skip omni_dim sync.")
        return
    prev = model_config.get("omni_dim")
    if prev != hidden:
        print(
            f"[WARN] model_config omni_dim={prev} != text encoder hidden_size={hidden}; "
            f"updating omni_dim to {hidden}."
        )
        model_config["omni_dim"] = hidden


def _resolve_vae_ckpt(path: Optional[str], config_path: str) -> Optional[str]:
    if not path or os.path.isabs(path):
        return path
    return str((Path(config_path).resolve().parent / path).resolve())


def init_audio_vae(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    vae_cfg = config.get("audio_vae", {})
    if vae_cfg.get("tod_vae_ckpt"):
        vae_cfg = {**vae_cfg, "tod_vae_ckpt": _resolve_vae_ckpt(vae_cfg["tod_vae_ckpt"], config_path)}
    if vae_cfg.get("bigvgan_vocoder_ckpt"):
        vae_cfg = {**vae_cfg, "bigvgan_vocoder_ckpt": _resolve_vae_ckpt(vae_cfg["bigvgan_vocoder_ckpt"], config_path)}

    vae = FeaturesUtils(
        tod_vae_ckpt=vae_cfg["tod_vae_ckpt"],
        bigvgan_vocoder_ckpt=vae_cfg.get("bigvgan_vocoder_ckpt"),
        mode=vae_cfg.get("mode", "16k"),
        need_vae_encoder=vae_cfg.get("need_vae_encoder", True),
    )
    scale_factor = vae_cfg.get("vae_scale_factor", DEFAULT_VAE_SCALE_FACTOR)
    sample_rate = vae_cfg.get("sample_rate", 16000)
    return vae, scale_factor, sample_rate


@torch.no_grad()
def sample_latents(
    model,
    scheduler,
    omni_extractor,
    prompts: List[str],
    source_latents=None,
    masks=None,
    num_inference_steps: int = 30,
    guidance_scale: float = 3.5,
    device=torch.device("cuda"),
    target_frames: int = None,
    latent_shape: tuple = None,
    inpaint_mask=None,
    inpaint_source=None,
    noise_init=None,
    cfg_interval=None,
):
    model.eval()
    batch_size = len(prompts)
    is_edit = source_latents is not None and masks is not None
    C = model.config.in_channels
    is_5d = latent_shape is not None and len(latent_shape) == 3


    omni_emb_cond, omni_mask_cond, omni_last_emb_cond = omni_extractor(prompts)
    omni_emb_cond  = [e.to(device) for e in omni_emb_cond]
    omni_mask_cond = omni_mask_cond.to(device)
    omni_last_emb_cond = omni_last_emb_cond.to(device)

    if is_edit:
        source_latents = source_latents.to(device=device, dtype=torch.bfloat16)
        masks = masks.to(device=device, dtype=torch.bfloat16)
        if noise_init is not None:
            noise = noise_init.to(device=device, dtype=torch.bfloat16)
        else:
            noise = torch.randn_like(source_latents)
        x = torch.cat([noise, source_latents, masks], dim=1)
    else:
        if is_5d:
            lt, lh, lw = latent_shape
            shape = (batch_size, C, lt, lh, lw)
        else:
            T = target_frames if target_frames is not None else TARGET_FRAMES
            shape = (batch_size, C, T)
        if noise_init is not None:
            noise = noise_init.to(device=device, dtype=torch.bfloat16)
        else:
            noise = torch.randn(*shape, device=device, dtype=torch.bfloat16)
        zero_src  = torch.zeros_like(noise)
        zero_mask = torch.zeros(shape[0], 1, *shape[2:], device=device, dtype=torch.bfloat16)
        x = torch.cat([noise, zero_src, zero_mask], dim=1)


    if inpaint_mask is not None:
        inpaint_mask = inpaint_mask.to(device=device)
    if inpaint_source is not None:
        inpaint_source = inpaint_source.to(device=device, dtype=torch.bfloat16)
    if noise_init is not None:
        noise_init = noise_init.to(device=device, dtype=torch.bfloat16)

    scheduler.set_timesteps(num_inference_steps, device=device)
    do_cfg = guidance_scale is not None and guidance_scale > 1.0

    noise_slice = slice(None, C)

    if do_cfg:
        omni_emb_uncond  = [torch.zeros_like(e) for e in omni_emb_cond]
        omni_mask_uncond = torch.zeros_like(omni_mask_cond)
        omni_last_emb_uncond = torch.zeros_like(omni_last_emb_cond)

        cfg_lo, cfg_hi = cfg_interval if cfg_interval is not None else (0.0, 1.0)

        timesteps = scheduler.timesteps
        for step_idx, t in enumerate(tqdm(timesteps, desc="Sampling (CFG)")):
            t_in = (t.float() / scheduler.config.num_train_timesteps).repeat(batch_size).to(device=device, dtype=torch.bfloat16)
            sigma = float(t) / scheduler.config.num_train_timesteps

            out_cond = model(x=x, t=t_in, duration=None,
                             omni_emb_list=omni_emb_cond, omni_last_emb=omni_last_emb_cond,
                             omni_mask=omni_mask_cond)
            pred_cond = out_cond[0]

            if cfg_lo <= sigma <= cfg_hi:
                out_unc = model(x=x, t=t_in, duration=None,
                                omni_emb_list=omni_emb_uncond, omni_last_emb=omni_last_emb_uncond,
                                omni_mask=omni_mask_uncond)
                pred_unc = out_unc[0]
                pred = pred_unc + guidance_scale * (pred_cond - pred_unc)
            else:
                pred = pred_cond
            x[:, noise_slice] = scheduler.step(model_output=pred, timestep=t, sample=x[:, noise_slice]).prev_sample


            if inpaint_mask is not None and noise_init is not None and inpaint_source is not None:

                if step_idx + 1 < len(timesteps):
                    t_next = timesteps[step_idx + 1]
                    sigma_next = t_next.float() / scheduler.config.num_train_timesteps
                else:
                    sigma_next = 0.0

                clamp_val = (1.0 - sigma_next) * inpaint_source + sigma_next * noise_init
                for dim_idx in range(C):
                    x[:, dim_idx][inpaint_mask[:, dim_idx]] = clamp_val[:, dim_idx][inpaint_mask[:, dim_idx]]
    else:
        timesteps = scheduler.timesteps
        for step_idx, t in enumerate(tqdm(timesteps, desc="Sampling")):
            t_in = (t.float() / scheduler.config.num_train_timesteps).repeat(batch_size).to(device=device, dtype=torch.bfloat16)
            out = model(x=x, t=t_in, duration=None,
                        omni_emb_list=omni_emb_cond, omni_last_emb=omni_last_emb_cond,
                        omni_mask=omni_mask_cond)
            pred = out[0]

            x[:, noise_slice] = scheduler.step(model_output=pred, timestep=t, sample=x[:, noise_slice]).prev_sample


            if inpaint_mask is not None and noise_init is not None and inpaint_source is not None:
                if step_idx + 1 < len(timesteps):
                    t_next = timesteps[step_idx + 1]
                    sigma_next = t_next.float() / scheduler.config.num_train_timesteps
                else:
                    sigma_next = 0.0
                clamp_val = (1.0 - sigma_next) * inpaint_source + sigma_next * noise_init
                for dim_idx in range(C):
                    x[:, dim_idx][inpaint_mask[:, dim_idx]] = clamp_val[:, dim_idx][inpaint_mask[:, dim_idx]]

    return x[:, noise_slice]


@torch.no_grad()
def decode_and_save(audio_vae, latents, durations, output_paths, sample_rate=None):
    if sample_rate is None:
        sample_rate = DEFAULT_TARGET_SAMPLE_RATE
    device  = next(audio_vae.parameters()).device
    dtype = getattr(audio_vae, 'dtype', torch.bfloat16)
    latents = latents.to(device=device, dtype=dtype)
    audio   = audio_vae.wrapped_decode(latents).cpu()

    for i, path in enumerate(output_paths):
        dur     = min(float(durations[i]), MAX_AUDIO_DURATION)
        max_smp = int(dur * sample_rate)
        wav     = audio[i]
        if wav.dim() > 1:
            wav = wav.mean(0)
        wav = wav[..., :max_smp]
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        sf.write(path, wav.float().numpy(), sample_rate)
        print(f"  Saved: {path}  ({dur:.1f}s)")


@torch.no_grad()
def decode_and_save_full(audio_vae, latents_full, ref_samples,
                         durations, output_paths, sample_rate=None,
                         fade_in_ms: float = 20.0,
                         max_duration: float = None,
                         ref_wav_raw=None,
                         ref_output_paths=None):
    if sample_rate is None:
        sample_rate = DEFAULT_TARGET_SAMPLE_RATE
    device = next(audio_vae.parameters()).device
    dtype = getattr(audio_vae, 'dtype', torch.bfloat16)
    latents_full = latents_full.to(device=device, dtype=dtype)
    audio_full = audio_vae.wrapped_decode(latents_full).cpu()

    batch_size = len(output_paths)
    if len(durations) != batch_size:
        raise ValueError(
            f"durations has {len(durations)} entries for {batch_size} output paths"
        )

    if isinstance(ref_samples, (list, tuple)):
        ref_samples_per_item = list(ref_samples)
    else:
        ref_samples_per_item = [ref_samples] * batch_size
    if len(ref_samples_per_item) != batch_size:
        raise ValueError(
            f"ref_samples has {len(ref_samples_per_item)} entries for "
            f"{batch_size} output paths"
        )

    if isinstance(ref_wav_raw, (list, tuple)):
        ref_wavs_per_item = list(ref_wav_raw)
    elif ref_wav_raw is None:
        ref_wavs_per_item = [None] * batch_size
    elif batch_size == 1:
        ref_wavs_per_item = [ref_wav_raw]
    else:
        raise ValueError(
            "batched decode requires ref_wav_raw to be a per-item list/tuple"
        )
    if len(ref_wavs_per_item) != batch_size:
        raise ValueError(
            f"ref_wav_raw has {len(ref_wavs_per_item)} entries for "
            f"{batch_size} output paths"
        )

    wav_total = audio_full.shape[-1]
    lat_total = latents_full.shape[-1]
    vae_hop = wav_total // lat_total if lat_total > 0 else 1

    save_cap = MAX_AUDIO_DURATION if max_duration is None else float(max_duration)

    for i, path in enumerate(output_paths):


        crop_start = round(int(ref_samples_per_item[i]) / vae_hop) * vae_hop
        crop_start_s = crop_start / sample_rate
        dur = min(float(durations[i]), save_cap)
        max_smp = int(dur * sample_rate)
        wav = audio_full[i]
        if wav.dim() > 1:
            wav = wav.mean(0)
        target_wav = wav[crop_start: crop_start + max_smp].clone()
        fade_len = min(int(fade_in_ms * sample_rate / 1000), target_wav.shape[-1])
        if fade_len > 1:
            target_wav[:fade_len] *= torch.linspace(0.0, 1.0, fade_len)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = Path(path).with_suffix(".tmp.wav")
        sf.write(tmp, target_wav.float().numpy(), sample_rate)
        tmp.replace(path)
        print(f"  Saved (target): {path}  ({dur:.1f}s, crop@{crop_start} = {crop_start_s:.3f}s, fade={fade_in_ms}ms)")


        ref_wav = ref_wavs_per_item[i]
        if ref_wav is not None and ref_output_paths is not None and i < len(ref_output_paths):
            ref_path = ref_output_paths[i]
            ref_out = ref_wav.squeeze().cpu().float()
            if ref_out.dim() == 0:
                ref_out = ref_out.unsqueeze(0)
            os.makedirs(os.path.dirname(ref_path) or ".", exist_ok=True)
            sf.write(ref_path, ref_out.numpy(), sample_rate)
            print(f"  Saved (ref):    {ref_path}  ({ref_out.shape[-1] / sample_rate:.3f}s)")


def _rms_normalize(wav: torch.Tensor, target_rms: float = TRAINING_TARGET_RMS) -> torch.Tensor:
    rms = torch.sqrt(torch.mean(wav ** 2))
    if rms > 0:
        wav = wav * (target_rms / (rms + 1e-8))
    return wav


def _load_audio(audio_path):
    audio, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(audio).transpose(0, 1).contiguous()
    return wav, int(sr)


def load_source_audio(audio_path, target_sr=DEFAULT_TARGET_SAMPLE_RATE,
                      target_length=None, max_duration=MAX_AUDIO_DURATION, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wav, sr = _load_audio(audio_path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != target_sr:
        wav = AF.resample(wav, sr, target_sr)
    wav = _rms_normalize(wav)
    L = target_length or int(max_duration * target_sr)
    if wav.shape[-1] > L:
        wav = wav[..., :L]
    elif wav.shape[-1] < L:
        wav = F.pad(wav, (0, L - wav.shape[-1]))
    return wav.to(device)


_whisper_model = None

def get_whisper_model(model_size="base"):
    global _whisper_model
    if _whisper_model is None:
        try:
            import whisper
        except ImportError:
            raise ImportError(
                "openai-whisper is required for auto-transcription. "
                "Install with: pip install openai-whisper"
            )
        print(f"  Loading Whisper ({model_size}) for auto-transcription...")
        _whisper_model = whisper.load_model(model_size)
    return _whisper_model


def transcribe_ref_audio(wav_tensor, sr, whisper_size="base"):
    model = get_whisper_model(whisper_size)
    wav_16k = wav_tensor.squeeze().cpu().float()
    if sr != 16000:
        wav_16k = AF.resample(wav_16k.unsqueeze(0), sr, 16000).squeeze(0)
    audio_np = wav_16k.numpy()
    result = model.transcribe(audio_np, fp16=torch.cuda.is_available())
    text = result["text"].strip()
    print(f"    [ASR] ref_text = \"{text}\"")
    return text


def _find_silence_cut(
    wav: torch.Tensor,
    sr: int,
    target_len: int,
    search_back_s: float = 0.5,
    win_ms: float = 20.0,
    threshold_db: float = -40.0,
) -> int:
    if wav.dim() == 2:
        mono = wav.mean(dim=0)
    else:
        mono = wav

    total = mono.shape[-1]
    if target_len <= 0 or total == 0:
        return max(0, target_len)
    if total <= target_len:
        return total

    win = max(int(win_ms * sr / 1000), 1)
    hop = max(win // 2, 1)
    search_back = max(int(search_back_s * sr), win)
    search_start = max(0, target_len - search_back)
    search_end = target_len
    if search_end - search_start < win:
        return target_len

    seg = mono[search_start:search_end].float()
    frames = seg.unfold(-1, win, hop)
    rms = frames.pow(2).mean(-1).sqrt()
    rms_db = 20.0 * torch.log10(rms.clamp(min=1e-10))
    is_silence = rms_db < threshold_db
    if not bool(is_silence.any()):
        return target_len

    silence_idx = torch.where(is_silence)[0]
    last = int(silence_idx[-1].item())
    cut = search_start + last * hop + win // 2
    return max(1, min(cut, target_len))


_TRAILING_PUNCT_CHARS = (
    " \t\r\n"
    ".,;:!?\"'`-—…"
    "。，；：！？、…—"
    "\u3002\uff0c\uff1b\uff1a\uff01\uff1f\u3001"
    "“”‘’《》「」『』"
)
_TRAILING_PUNCT_RE = re.compile(
    "[" + re.escape(_TRAILING_PUNCT_CHARS) + "]+$"
)
_CJK_RE = re.compile(r"[\u3400-\u9fff\uff00-\uffef]")


def _strip_trailing_punct(text: str) -> str:
    if not text:
        return ""
    return _TRAILING_PUNCT_RE.sub("", text).rstrip()


def _has_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))


def join_ref_target_text(ref_text: str, target_text: str) -> str:
    ref_text = _strip_trailing_punct(ref_text or "")
    target_text = (target_text or "").strip()
    if not ref_text:
        return target_text
    if not target_text:
        return ref_text
    sep = "，" if (_has_cjk(ref_text) or _has_cjk(target_text)) else ", "
    return f"{ref_text}{sep}{target_text}"


def load_ref_audio(
    audio_path,
    target_sr=DEFAULT_TARGET_SAMPLE_RATE,
    max_ref_duration=None,
    device=None,
    silence_search_s: float = 0.5,
    silence_threshold_db: float = -40.0,
    tail_pad_s: float = 0.1,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wav, sr = _load_audio(audio_path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != target_sr:
        wav = AF.resample(wav, sr, target_sr)

    if max_ref_duration is not None:
        ref_len = int(max_ref_duration * target_sr)
        if wav.shape[-1] > ref_len:
            if silence_search_s > 0:
                cut = _find_silence_cut(
                    wav, target_sr, ref_len,
                    search_back_s=silence_search_s,
                    threshold_db=silence_threshold_db,
                )
                cut_dur = cut / target_sr
                if cut < ref_len:
                    print(
                        f"    [ref-cut] snapped {max_ref_duration:.2f}s -> "
                        f"{cut_dur:.2f}s at silence (search_back={silence_search_s:.2f}s, "
                        f"thresh={silence_threshold_db:.0f}dB)"
                    )
                wav = wav[..., :cut]
            else:
                wav = wav[..., :ref_len]

    wav = _rms_normalize(wav)

    if tail_pad_s > 0:
        pad_len = int(tail_pad_s * target_sr)
        if pad_len > 0:
            wav = F.pad(wav, (0, pad_len))

    return wav.to(device)


def make_edit_mask(waveform_length, device):
    return torch.ones(1, 1, waveform_length, device=device, dtype=torch.float32)


def downsample_mask(mask_wav, latent_length):
    T = mask_wav.shape[-1]
    if T != latent_length:
        m = F.adaptive_avg_pool1d(mask_wav.float(), latent_length)
        return (m > 0.5).float()
    return mask_wav.float()


def stable_item_seed(base_seed: int, item_key: str) -> int:
    payload = f"{int(base_seed)}\\0{item_key}".encode("utf-8")


    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)


def make_item_noise(base_seed, item_keys, latent_shape, device,
                    dtype=torch.bfloat16):
    rows = []
    for item_key in item_keys:
        generator = torch.Generator(device=device)
        generator.manual_seed(stable_item_seed(base_seed, str(item_key)))
        rows.append(torch.randn(
            *latent_shape, generator=generator, device=device, dtype=dtype
        ))
    if not rows:
        raise ValueError("make_item_noise requires at least one item")
    return torch.stack(rows)


def run_zero_shot_tasks(args, zs_tasks, model, scheduler, omni_extractor,
                        audio_vae, vae_scale_factor, vae_sample_rate,
                        ckpt_tag, params_tag, device):
    if args.batch_size < 1:
        raise ValueError(f"--batch_size must be >= 1, got {args.batch_size}")

    max_total_duration = args.zs_duration


    ref_cache = {}

    for batch_start in range(0, len(zs_tasks), args.batch_size):
        task_chunk = zs_tasks[batch_start:batch_start + args.batch_size]
        prepared = []

        for offset, task in enumerate(task_chunk):
            i = batch_start + offset
            target_text = task.get("target_text") or task.get("prompt", "")
            ref_path = task["ref_audio"]
            supplied_ref_text = task.get("ref_text")
            duration = float(task.get("duration", max_total_duration))

            print(
                f"\n  [{i + 1}/{len(zs_tasks)}] target: "
                f"{target_text[:70]}{'...' if len(target_text) > 70 else ''}"
            )
            print(f"    ref: {ref_path}")

            cache_key = (
                str(Path(ref_path).resolve()), args.ref_duration,
                args.ref_silence_search_s, args.ref_silence_threshold_db,
                args.ref_tail_pad_s, vae_sample_rate,
            )
            cached = ref_cache.get(cache_key)
            if cached is None:
                max_ref_dur = None if args.ref_duration == 0 else args.ref_duration
                ref_wav = load_ref_audio(
                    ref_path, target_sr=vae_sample_rate,
                    max_ref_duration=max_ref_dur, device=device,
                    silence_search_s=args.ref_silence_search_s,
                    silence_threshold_db=args.ref_silence_threshold_db,
                    tail_pad_s=args.ref_tail_pad_s,
                )
                cached = {"wav": ref_wav.detach().cpu(), "auto_text": None}
                ref_cache[cache_key] = cached
            else:
                ref_wav = cached["wav"].to(device)

            ref_audio_duration = ref_wav.shape[-1] / vae_sample_rate
            speech_samples = ref_wav.shape[-1] - int(
                args.ref_tail_pad_s * vae_sample_rate
            )
            speech_samples = max(speech_samples, 1)
            if supplied_ref_text:
                ref_text = supplied_ref_text
                print(f"    ref_text = \"{ref_text}\"")
            else:
                if cached["auto_text"] is None:
                    cached["auto_text"] = transcribe_ref_audio(
                        ref_wav[..., :speech_samples], sr=vae_sample_rate,
                        whisper_size=args.whisper_model,
                    )
                ref_text = cached["auto_text"]
                print(f"    [ASR cache] ref_text = \"{ref_text}\"")

            if ref_audio_duration + 1.0 > duration:
                print(
                    f"    [WARN] ref ({ref_audio_duration:.1f}s) + min_target "
                    f"(1.0s) > duration ({duration:.1f}s), skipping."
                )
                continue
            print(
                f"    ref_duration={ref_audio_duration:.2f}s, "
                f"total_duration={duration:.1f}s, "
                f"target_duration={duration - ref_audio_duration:.2f}s"
            )

            if args.ref_text_join == "space":
                combined_text = (
                    f"{_strip_trailing_punct(ref_text)} {target_text.strip()}"
                ).strip()
            elif args.ref_text_join == "comma":
                ref_clean = _strip_trailing_punct(ref_text)
                tgt_clean = target_text.strip()
                if ref_clean and tgt_clean:
                    sep = "，" if (_has_cjk(ref_clean) or _has_cjk(tgt_clean)) else ", "
                    combined_text = f"{ref_clean}{sep}{tgt_clean}"
                else:
                    combined_text = ref_clean or tgt_clean
            else:
                combined_text = join_ref_target_text(ref_text, target_text)
            prompt = f"[Speech with voice] {combined_text}"
            print(f"    prompt = {prompt[:100]}{'...' if len(prompt) > 100 else ''}")

            safe = "".join(
                c if c.isalnum() or c in " -_" else "" for c in target_text[:70]
            ).strip().replace(" ", "_")
            ref_tag = Path(ref_path).stem
            item_key = task.get("id") or f"{ref_path}\\0{target_text}\\0{i}"
            output_path = Path(args.output_dir) / f"{ckpt_tag}zstts_{ref_tag}_{safe}_{params_tag}_{i:03d}.wav"
            if getattr(args, "zs_resume", False) and output_path.is_file() and output_path.stat().st_size > 0:
                continue
            prepared.append({
                "index": i,
                "prompt": prompt,
                "item_key": item_key,
                "duration": duration,
                "total_wav_len": int(duration * vae_sample_rate),
                "ref_wav": ref_wav,
                "ref_samples": int(ref_audio_duration * vae_sample_rate),
                "target_duration": duration - ref_audio_duration,
                "out_path": str(output_path),
                "ref_out_path": os.path.join(
                    args.output_dir,
                    f"{ckpt_tag}ref_{ref_tag}_{params_tag}_{i:03d}.wav",
                ),
            })


        duration_groups = {}
        for item in prepared:
            duration_groups.setdefault(item["total_wav_len"], []).append(item)

        for total_wav_len, items in duration_groups.items():
            batch_size = len(items)
            source_wavs = torch.zeros(
                batch_size, total_wav_len, device=device, dtype=torch.float32
            )
            mask_wavs = torch.zeros(
                batch_size, 1, total_wav_len, device=device, dtype=torch.float32
            )
            for row, item in enumerate(items):
                ref_1d = item["ref_wav"].reshape(-1)
                ref_len = min(ref_1d.shape[-1], total_wav_len)
                source_wavs[row, :ref_len] = ref_1d[:ref_len]
                mask_wavs[row, :, :ref_len] = 2.0

            with torch.no_grad():
                source_latents = audio_vae.wrapped_encode(source_wavs) * vae_scale_factor
            channels = source_latents.shape[1]
            total_lat_len = source_latents.shape[-1]
            masks = F.interpolate(
                mask_wavs, size=total_lat_len, mode="nearest"
            ).to(torch.bfloat16)

            if args.mask_zs_after_encode:
                ref_regions = (masks > 1.5).to(source_latents.dtype)
                source_latents = source_latents * ref_regions.expand_as(source_latents)

            noise = make_item_noise(
                args.seed, [item["item_key"] for item in items],
                (channels, total_lat_len), device, dtype=torch.bfloat16,
            )
            if args.use_inpainting:
                inpaint_sources = source_latents[:, :channels, :].clone()
                inpaint_masks = (masks > 1.5).expand(-1, channels, -1)
            else:
                inpaint_sources = None
                inpaint_masks = None

            latents = sample_latents(
                model, scheduler, omni_extractor,
                [item["prompt"] for item in items],
                source_latents=source_latents, masks=masks,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale, device=device,
                inpaint_mask=inpaint_masks,
                inpaint_source=inpaint_sources,
                noise_init=noise,
            )
            decode_and_save_full(
                audio_vae, latents * (1.0 / vae_scale_factor),
                [item["ref_samples"] for item in items],
                [item["target_duration"] for item in items],
                [item["out_path"] for item in items],
                sample_rate=vae_sample_rate,
                ref_wav_raw=[item["ref_wav"] for item in items],
                ref_output_paths=[item["ref_out_path"] for item in items],
            )
            print(
                f"  [batch] generated {batch_size} item(s), "
                f"indices {items[0]['index'] + 1}-{items[-1]['index'] + 1}"
            )


def parse_args():
    p = argparse.ArgumentParser(
        description="UNISON: inference pipeline (generation / editing / zero-shot TTS)"
    )

    p.add_argument("--model_ckpt",      type=str, required=True)
    p.add_argument("--vae_config",      type=str, default=DEFAULT_VAE_CONFIG_PATH)
    p.add_argument("--omni_model_path", type=str, default=DEFAULT_OMNI_MODEL_PATH,
                   help="Qwen2.5-Omni checkpoint. Used when --text_encoder_type=omni, "
                        "or as local override for omni-3b/omni-7b presets.")
    p.add_argument(
        "--text_encoder_type",
        type=str,
        default="omni",
        choices=["omni", "omni-3b", "omni-7b"],
    )
    p.add_argument(
        "--text_encoder_model_path",
        type=str,
        default=None,

    )
    p.add_argument("--dit_depth",       type=int, default=None,
                   help="(deprecated) Auto-computed from model_config. Only set to override.")
    p.add_argument("--model_config",    type=str, default=DEFAULT_MODEL_CONFIG_PATH)


    p.add_argument("--task_mode", type=str,
                   choices=["generation", "editing", "zeroshotts", "all"],
                   default="all")


    p.add_argument("--gen_prompt",      type=str, default=None,
                   help="Single generation prompt string.")
    p.add_argument("--gen_prompt_list", type=str, default=None,
                   help="Path to a text file with one prompt per line.")
    p.add_argument("--gen_duration",    type=float, default=10.0)


    p.add_argument("--edit_prompt",       type=str, default=None,
                   help="Single editing prompt (requires --edit_source_audio).")
    p.add_argument("--edit_source_audio", type=str, default=None,
                   help="Source audio path for single editing task.")
    p.add_argument("--edit_config",       type=str, default=None,
                   help="Path to JSON list of {prompt, source_audio} objects.")


    p.add_argument("--zs_prompt",      type=str, default=None,
                   help="Target text for single zero-shot TTS (what to say).")
    p.add_argument("--zs_ref_audio",   type=str, default=None,
                   help="Reference audio for single zero-shot TTS.")
    p.add_argument("--zs_ref_text",    type=str, default=None,
                   help="Transcription of ref audio. If omitted, auto-transcribed via Whisper.")
    p.add_argument("--zs_config",      type=str, default=None,
                   help="Path to JSON list of {target_text, ref_audio, ref_text(optional)} objects.")
    p.add_argument("--zs_duration",    type=float, default=10.0,
                   help="Max total duration (ref+target) in seconds.")
    p.add_argument("--zs_resume", action="store_true",
                   help="Reuse completed zero-shot clips in the output directory.")
    p.add_argument("--ref_duration",   type=float, default=REF_DURATION,
                   help="Max reference audio duration in seconds (default: 3.0)")
    p.add_argument("--whisper_model",  type=str, default="base",
                   help="Whisper model size for auto-transcription (tiny/base/small/medium/large)")
    p.add_argument("--batch_size",     type=int, default=1,
                   help="Zero-shot TTS inference batch size. Generation and "
                        "editing modes remain single-item.")


    p.add_argument("--ref_silence_search_s", type=float, default=0.5,
                   help="Search backward this many seconds from --ref_duration to "
                        "snap the cut at a silent frame. 0 disables silence-aware cut.")
    p.add_argument("--ref_silence_threshold_db", type=float, default=-40.0,
                   help="Frames with RMS below this dB are considered silence.")
    p.add_argument("--ref_tail_pad_s", type=float, default=0.1,
                   help="Seconds of true silence appended after ref to reset the "
                        "model's starting acoustic state. 0 disables.")
    p.add_argument("--ref_text_join", type=str, default="auto",
                   choices=["auto", "comma", "space"],
                   help="How to join ref_text and target_text after stripping ref's "
                        "trailing punctuation. 'auto' uses 「，」 for CJK and ', ' "
                        "otherwise; 'comma' forces a comma; 'space' forces a single "
                        "space (legacy behavior).")


    p.add_argument("--use_inpainting", action="store_true", default=False,
                   help="Use inpainting trick for zero-shot TTS (clamp ref xt each step)")
    p.add_argument("--mask_zs_after_encode", action="store_true", default=False,
                   help="Zero out target region of source latent after VAE encoding. "
                        "Must match training flag.")


    p.add_argument("--num_inference_steps", type=int,   default=50)
    p.add_argument("--guidance_scale",      type=float, default=4.5)
    p.add_argument("--output_dir",          type=str,   default="./outputs/timed_channelcat_infer")
    p.add_argument("--seed",                type=int,   default=42)
    p.add_argument("--device",              type=str,   default="cuda")

    return p.parse_args()


def _load_model(args, device, model_config):
    mcfg = dict(model_config)
    mcfg.pop("omni_last_layer_idx", None)
    mcfg["concat_condition"] = True

    model = UnisonBackbone(**mcfg)

    ckpt_path = args.model_ckpt
    if os.path.isdir(ckpt_path):
        for name in ("ema_model.pt", "model.safetensors", "pytorch_model.bin"):
            candidate = os.path.join(ckpt_path, name)
            if os.path.exists(candidate):
                ckpt_path = candidate
                break

    print(f"Loading checkpoint: {ckpt_path}")
    if ckpt_path.endswith(".safetensors"):
        sd = safe_load_file(ckpt_path, device="cpu")
    else:
        sd = torch.load(ckpt_path, map_location="cpu")

    is_ema = "shadow_params" in sd or any(k.startswith("shadow_params") for k in sd)
    if is_ema:
        print(">>> EMA wrapper detected - unwrapping...")
        ema = EMAModel(model.parameters(), decay=0.9999,
                       model_cls=UnisonBackbone, model_config=mcfg)
        ema.load_state_dict(sd)
        ema.copy_to(model.parameters())
        print(">>> EMA weights injected.")
    else:
        ignore = {'decay','num_updates','min_decay','update_after_step',
                  'use_ema_warmup','inv_gamma','power','shadow_params'}
        model.load_state_dict({k: v for k, v in sd.items() if k not in ignore}, strict=False)

    model.to(device=device, dtype=torch.bfloat16).eval()
    return model


def main():
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 60)
    print("UNISON Inference")
    print("=" * 60)


    with open(args.model_config) as _f:
        _mcfg = yaml.safe_load(_f)
    dit_depth = args.dit_depth
    if dit_depth is None:
        dit_depth = _mcfg.get("mm_double_blocks_depth", 0) + _mcfg.get("mm_single_blocks_depth", 0)
    print(f"  dit_depth = {dit_depth} (double={_mcfg.get('mm_double_blocks_depth')}, single={_mcfg.get('mm_single_blocks_depth')})")


    omni_last_layer_idx = _mcfg.get("omni_last_layer_idx", -1)
    print(
        f"  text_encoder_type={args.text_encoder_type} "
        f"(omni_path={args.omni_model_path if args.text_encoder_type.startswith('omni') else 'N/A'}, "
        f"text_encoder_model_path={args.text_encoder_model_path})"
    )
    omni_extractor = init_text_hidden_extractor(
        args.text_encoder_type,
        args.omni_model_path,
        args.text_encoder_model_path,
        dit_depth,
        device,
        dtype=torch.bfloat16,
        omni_last_layer_idx=omni_last_layer_idx,
    )
    sync_omni_dim_with_text_encoder(_mcfg, omni_extractor)
    model = _load_model(args, device, _mcfg)
    audio_vae, vae_scale_factor, vae_sample_rate = init_audio_vae(args.vae_config)
    audio_vae.to(device).eval()
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
    os.makedirs(args.output_dir, exist_ok=True)


    gen_target_frames = None
    dummy_len = int(MAX_AUDIO_DURATION * vae_sample_rate)
    with torch.no_grad():
        dummy_wav = torch.zeros(1, dummy_len, device=device)
        dummy_lat = audio_vae.wrapped_encode(dummy_wav)
        gen_target_frames = int(dummy_lat.shape[-1])
        del dummy_wav, dummy_lat
        torch.cuda.empty_cache()
    vae_hop = dummy_len // max(gen_target_frames, 1)
    audio_fps = vae_sample_rate / max(vae_hop, 1)
    if hasattr(audio_vae, 'hop_length'):
        reported_hop = int(audio_vae.hop_length)
        hop_note = f", reported_hop={reported_hop}"
        if reported_hop != vae_hop:
            hop_note += f" (MISMATCH with probed {vae_hop})"
    else:
        hop_note = " (no hop_length attr; probed from dummy encode)"
    print(
        f"  VAE: probed_hop={vae_hop}, latent FPS={audio_fps:.2f}, "
        f"gen frames ({MAX_AUDIO_DURATION}s)={gen_target_frames}{hop_note}"
    )

    print(f"  VAE scale factor: {vae_scale_factor}, sample rate: {vae_sample_rate}")
    print(f"  Inpainting trick: {'ON' if args.use_inpainting else 'OFF'}")
    print(f"  mask_zs_after_encode: {'ON' if args.mask_zs_after_encode else 'OFF'}")


    if os.path.isdir(args.model_ckpt):
        ckpt_tag = os.path.basename(args.model_ckpt) + "_"
    else:
        ckpt_tag = os.path.splitext(os.path.basename(args.model_ckpt))[0] + "_"
    ckpt_tag = ""
    params_tag = f"s{args.num_inference_steps}_cfg{args.guidance_scale}_seed{args.seed}"


    if args.task_mode in ("generation", "all"):
        print("\n[Generation Tasks]")
        if args.gen_prompt_list:
            with open(args.gen_prompt_list, encoding="utf-8") as f:
                gen_prompts = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        elif args.gen_prompt:
            gen_prompts = [args.gen_prompt]
        else:
            gen_prompts = []

        for i, prompt in enumerate(gen_prompts):
            print(f"\n  [{i+1}/{len(gen_prompts)}] {prompt[:70]}{'...' if len(prompt) > 70 else ''}")
            latents = sample_latents(model, scheduler, omni_extractor, [prompt],
                                     num_inference_steps=args.num_inference_steps,
                                     guidance_scale=args.guidance_scale, device=device,
                                     target_frames=gen_target_frames,
                                     latent_shape=None)
            latents = latents * (1.0 / vae_scale_factor)
            safe = "".join(c if c.isalnum() or c in " -_" else "" for c in prompt[:70]).strip().replace(" ", "_")
            out_path = os.path.join(args.output_dir, f"{ckpt_tag}gen_{safe}_{params_tag}_{i:03d}.wav")
            decode_and_save(audio_vae, latents, [args.gen_duration], [out_path], sample_rate=vae_sample_rate)


    if args.task_mode in ("editing", "all"):
        print("\n[Editing Tasks]")
        if args.edit_config:
            with open(args.edit_config, encoding="utf-8") as f:
                edit_tasks = json.load(f)
        elif args.edit_prompt and args.edit_source_audio:
            edit_tasks = [{"prompt": args.edit_prompt, "source_audio": args.edit_source_audio}]
        else:
            edit_tasks = []

        tgt_wav_len = int(MAX_AUDIO_DURATION * vae_sample_rate)
        for i, task in enumerate(edit_tasks):
            prompt   = task["prompt"]
            src_path = task["source_audio"]
            print(f"\n  [{i+1}/{len(edit_tasks)}] {prompt[:70]}{'...' if len(prompt) > 70 else ''}")
            print(f"    src: {src_path}")

            src_wav    = load_source_audio(src_path, target_sr=vae_sample_rate,
                                           target_length=tgt_wav_len, device=device)
            src_latent = audio_vae.wrapped_encode(src_wav) * vae_scale_factor

            if src_latent.dim() == 5:
                mask_lat = torch.ones(1, 1, *src_latent.shape[2:], device=device, dtype=torch.float32)
            else:
                mask_wav = make_edit_mask(src_wav.shape[-1], device)
                lat_len  = src_latent.shape[-1]
                mask_lat = downsample_mask(mask_wav, lat_len)

            latents = sample_latents(model, scheduler, omni_extractor, [prompt],
                                     source_latents=src_latent, masks=mask_lat,
                                     num_inference_steps=args.num_inference_steps,
                                     guidance_scale=args.guidance_scale, device=device)
            latents = latents * (1.0 / vae_scale_factor)
            dur     = src_wav.shape[-1] / vae_sample_rate
            safe    = "".join(c if c.isalnum() or c in " -_" else "" for c in prompt[:70]).strip().replace(" ", "_")
            src_tag = Path(src_path).stem
            out_path = os.path.join(args.output_dir, f"{ckpt_tag}edit_{safe}_{src_tag}_{params_tag}_{i:03d}.wav")
            decode_and_save(audio_vae, latents, [dur], [out_path], sample_rate=vae_sample_rate)


    if args.task_mode in ("zeroshotts", "all"):
        print("\n[Zero-shot TTS Tasks (Channel-Cat)]")
        if args.zs_config:
            with open(args.zs_config, encoding="utf-8") as f:
                zs_tasks = json.load(f)
        elif args.zs_prompt and args.zs_ref_audio:
            zs_tasks = [{"target_text": args.zs_prompt, "ref_audio": args.zs_ref_audio,
                         "ref_text": args.zs_ref_text}]
        else:
            zs_tasks = []

        run_zero_shot_tasks(
            args, zs_tasks, model, scheduler, omni_extractor, audio_vae,
            vae_scale_factor, vae_sample_rate, ckpt_tag, params_tag, device,
        )

    print(f"\nDone. Outputs -> {args.output_dir}")
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
