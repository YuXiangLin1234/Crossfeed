# The system, TTA and TTS prompt templates are adapted from NVIDIA Audex-2B.
# https://huggingface.co/nvidia/Nemotron-Labs-Audex-2B/blob/77b7e1a4de899769f22c1bc074db666601c28907/inference_scripts_vllm/audiogen_scripts/run_audio_gen_vllm_rvq_logit_mask.py
# Upstream Copyright (c) 2026, NVIDIA CORPORATION. Licensed under Apache-2.0.

import os
import re
import sys
from pathlib import Path

from crossfeed.utils import seed_for

SYSTEM_PROMPT = 'You are a helpful and harmless assistant.\n\nYou are not allowed to use any tools.'
DSP_PRESETS = {
    's00': None,
    's01': dict(formant=.86, semitones=-5., pitch_range=.75, duration=1.04),
    's11': dict(formant=1.10, semitones=5., pitch_range=1.05, duration=1.),
    's13': dict(formant=1.08, semitones=-4., pitch_range=.90, duration=1.),
    's14': dict(formant=.89, semitones=0., pitch_range=.60, duration=1.06),
    's15': dict(formant=1.07, semitones=0., pitch_range=1.45, duration=.94),
}


def model_root():
    return Path(os.environ['AUDEX_MODEL_ROOT']).resolve()


def base_model():
    return model_root() / 'checkpoint_folder_full'


def add_official_paths():
    root = model_root()
    paths = [str(root / 'nemotron_dense_vllm_plugin'), str(root / 'inference_scripts_vllm' / 'audioqa_scripts'),
             str(root / 'inference_scripts_vllm' / 'audiogen_scripts'), str(root / 'inference_scripts_hf')]
    for path in reversed(paths):
        if path not in sys.path:
            sys.path.insert(0, path)
    os.environ['PYTHONPATH'] = os.pathsep.join(paths + [os.environ.get('PYTHONPATH', '')])


def stable_seed(*identity, seed=None):
    return seed_for(int(os.environ.get('SEED', '531')) if seed is None else seed, *identity)


def strip_tags(text):
    return re.sub(r'\[(?:Audio|Speech|Mixed)\][ \t]*', '', text, flags=re.I).strip()


def generation_prompt(row):
    speech = row['task'] == 'speech'
    task, noun, marker = ('speech', 'transcription', 'speech') if speech else ('audio', 'caption', 'audio')
    text = strip_tags(row['text'] if speech else row['caption'])
    return (f'<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n'
            f'<|im_start|>user\n<|text to {task}|> Generate {task} for this {noun}. '
            f'{text}<|im_end|>\n<|im_start|>assistant\n<think></think><{marker}gen_start>')
