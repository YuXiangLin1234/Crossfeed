#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# Data and run paths
export N="${N:-5000}"                # Total scenes, split 1:1 between audio and speech
export UND_N="${UND_N:-}"            # Understanding scenes, defaults to N
export GEN_N="${GEN_N:-}"            # Generation scenes, defaults to N
export DATA_RUN="${DATA_RUN:-}"      # Defaults to runs/audex/data_n<N>
export OUT="${OUT:-}"                # Defaults to a path based on the scene budgets and MERGE
export BASE="${BASE:-}"              # Defaults to the Audex base checkpoint from config.sh
export SEED="${SEED:-531}"
export SKIP_DATA="${SKIP_DATA:-0}"    # Set to 1 to reuse generated training data
export MERGE="${MERGE:-1}"            # Set to 0 to disable self-merging

# Data generation and filtering
export CF_CANDIDATES="${CF_CANDIDATES:-8}"    # Candidates per original instruction
export CF_TEXT_ROUNDS="${CF_TEXT_ROUNDS:-500}" # Instruction pools, dialogue plans and scripts
export CF_QA_VOTE_MIN="${CF_QA_VOTE_MIN:-4}"   # Minimum agreement out of eight QA votes
export CF_PAIR_MARGIN="${CF_PAIR_MARGIN:-0.0625}" # 0-1

# Understanding and judge training
export CF_UND_LR="${CF_UND_LR:-5e-5}"
export CF_UND_EPOCHS="${CF_UND_EPOCHS:-1}"
export CF_UND_BATCH="${CF_UND_BATCH:-32}"
export CF_LORA_R="${CF_LORA_R:-64}"
export CF_LORA_ALPHA="${CF_LORA_ALPHA:-128}"

# Generation training
export CF_GEN_LR="${CF_GEN_LR:-1e-5}"
export CF_GEN_EPOCHS="${CF_GEN_EPOCHS:-1}"
export CF_GEN_BATCH="${CF_GEN_BATCH:-64}"
export CF_DPO_BETA="${CF_DPO_BETA:-0.1}"

PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" exec python -m crossfeed.train audex "$@"
