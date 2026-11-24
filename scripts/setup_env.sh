#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV="${ENV:-$ROOT/.envs/crossfeed}"
source "$ROOT/config.sh"

conda create -y -p "$ENV" -c conda-forge python=3.13 ffmpeg=7 sox
export PYTHONUSERBASE="$ENV"
"$ENV/bin/pip" install -r "$ROOT/requirements.txt"

PLUGINS="$(mktemp -d)"
trap 'rm -rf "$PLUGINS"' EXIT
"$ENV/bin/hf" download nvidia/Nemotron-Labs-Audex-2B --revision 77b7e1a4de899769f22c1bc074db666601c28907 \
  --include "nemotron_dense_vllm_plugin/*" "inference_scripts_vllm/audioqa_scripts/*" --local-dir "$PLUGINS"
"$ENV/bin/pip" install --no-deps -r "$ROOT/requirements-nodeps.txt" \
  "$PLUGINS/nemotron_dense_vllm_plugin" "$PLUGINS/inference_scripts_vllm/audioqa_scripts"
echo "conda activate $ENV"
