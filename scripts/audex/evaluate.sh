#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# Checkpoint, benchmarks and output
export MODEL="${MODEL:-}"            # Trained checkpoint path, defaults to the Audex base checkpoint
export OUT="${OUT:-}"                # Defaults to runs/audex/eval/base
export TASKS="${TASKS:-understanding judge tta tts}"
export BASELINES="${BASELINES:-0}"    # Set to 1 for understanding and judge baselines
export CF_EVAL_LIMIT="${CF_EVAL_LIMIT:-}" # Maximum items per benchmark, defaults to all items
export SEED="${SEED:-531}"

# Understanding and judge
export CF_EVAL_QA_BATCH="${CF_EVAL_QA_BATCH:-64}"
export CF_EVAL_JUDGE_CHUNK="${CF_EVAL_JUDGE_CHUNK:-256}"
export CF_EVAL_SC_SAMPLES="${CF_EVAL_SC_SAMPLES:-8}"       # Self-consistency (BASELINES=1)
export CF_EVAL_SC_TEMPERATURE="${CF_EVAL_SC_TEMPERATURE:-0.8}"

# Text-to-audio and text-to-speech
export CF_EVAL_AUDIO_BATCH="${CF_EVAL_AUDIO_BATCH:-64}"    # Codec decoding batch size
export CF_EVAL_TTA_CFG="${CF_EVAL_TTA_CFG:-3.0}"
export CF_EVAL_TTS_CFG="${CF_EVAL_TTS_CFG:-1.0}"

PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" exec python -m crossfeed.evaluate audex "$@"
