#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# Checkpoints, benchmarks and output
export OMNI="${OMNI:-}"              # Path to final_omni, defaults to the base Omni checkpoint
export DIT="${DIT:-}"                # Path to final_dit/model.safetensors, defaults to the base DiT weights
export OUT="${OUT:-}"                # Defaults to runs/unison/eval/base
export TASKS="${TASKS:-understanding judge tta tts}"
export BASELINES="${BASELINES:-0}"    # Set to 1 for understanding and judge baselines
export CF_EVAL_LIMIT="${CF_EVAL_LIMIT:-}" # Maximum items per benchmark, defaults to all items
export SEED="${SEED:-531}"

# Understanding and judge
export CF_EVAL_QA_BATCH="${CF_EVAL_QA_BATCH:-64}"
export CF_EVAL_JUDGE_CHUNK="${CF_EVAL_JUDGE_CHUNK:-256}"
export CF_EVAL_SC_SAMPLES="${CF_EVAL_SC_SAMPLES:-8}"       # Self-consistency (BASELINES=1)
export CF_EVAL_SC_TEMPERATURE="${CF_EVAL_SC_TEMPERATURE:-0.8}"

# Text-to-audio and text-to-speech (including zero-shot TTS)
export CF_EVAL_AUDIO_BATCH="${CF_EVAL_AUDIO_BATCH:-64}"
export CF_EVAL_TTA_CFG="${CF_EVAL_TTA_CFG:-4.5}"
export CF_EVAL_TTS_CFG="${CF_EVAL_TTS_CFG:-4.5}"
export CF_EVAL_DIT_STEPS="${CF_EVAL_DIT_STEPS:-100}"

PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" exec python -m crossfeed.evaluate unison "$@"
