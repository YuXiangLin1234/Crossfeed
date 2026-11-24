#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/config.sh"

hf download jac22/UNISON --revision 2ff46ae70a190b490606a1a52975262ca6346540 \
  --include "unison_D24S0_O_20ch/*" --local-dir "$CHECKPOINTS/UNISON"
hf download Qwen/Qwen2.5-Omni-7B --revision ae9e1690543ffd5c0221dc27f79834d0294cba00 \
  --local-dir "$CHECKPOINTS/Qwen2.5-Omni-7B"
hf download nvidia/Nemotron-Labs-Audex-2B --revision 77b7e1a4de899769f22c1bc074db666601c28907 \
  --include "checkpoint_folder_full/*" "inference_scripts_hf/*" "inference_scripts_vllm/*" \
  "nemotron_dense_vllm_plugin/*" "LICENSE" "license/*" --local-dir "$AUDEX_MODEL_ROOT"
hf download hf-audio/xcodec-hubert-general-balanced --revision ce75a648724c64dfd5ec05752caa2feafd8c03da \
  --local-dir "$XCODEC1_PATH"
hf download HKUSTAudio/xcodec2 --revision e412427ed30f0cf9d5e3c95562113deb10a32d03 \
  --include config.json model.safetensors --local-dir "$XCODEC2_PATH"
hf download AutoAudioReasoningMOS/CoTTAGE --revision f3d0fb04013e6d9e138f0ece6490d76d103328a2 \
  --local-dir "$CHECKPOINTS/CoTTAGE"

hf download facebook/w2v-bert-2.0 config.json
hf download openai/whisper-large-v3 --include "*.json" "*.txt" model.safetensors
hf download facebook/wav2vec2-large-960h-lv60-self --include "*.json" pytorch_model.bin

WEIGHTS="$ROOT/third_party/unison/models/mmaudio/data/ext_weights"
mkdir -p "$WEIGHTS"
for name in v1-16.pth best_netG.pt; do
  [ -s "$WEIGHTS/$name" ] || curl -fL --retry 5 -o "$WEIGHTS/$name" \
    "https://github.com/hkchengrex/MMAudio/releases/download/v0.1/$name"
done

python - <<'EOF'
import os

import torch
import whisper

whisper._download(whisper._MODELS['large-v3'], os.environ.get('WHISPER_CACHE') or
                  os.path.join(os.path.expanduser('~'), '.cache', 'whisper'), False)
torch.hub.load('tarepan/SpeechMOS:v1.2.0', 'utmos22_strong', trust_repo=True)

import laion_clap
laion_clap.CLAP_Module(enable_fusion=False).load_ckpt()

from audioldm_eval import EvaluationHelper
EvaluationHelper(16000, torch.device('cpu'), backbone='cnn14')
EOF
echo "[download_models] done"
