set -a
CROSSFEED_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$CROSSFEED_ROOT/config.local.sh" ] && source "$CROSSFEED_ROOT/config.local.sh"

export PYTHONPATH="$CROSSFEED_ROOT:$CROSSFEED_ROOT/third_party${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

: "${CHECKPOINTS:=$CROSSFEED_ROOT/checkpoints}"
: "${DATA:=$CROSSFEED_ROOT/data}"
: "${RUNS:=$CROSSFEED_ROOT/runs}"
: "${SEED:=531}"

: "${QWEN_OMNI:=$CHECKPOINTS/Qwen2.5-Omni-7B}"
: "${UNISON_DIT:=$CHECKPOINTS/UNISON/unison_D24S0_O_20ch/model.safetensors}"
: "${COTTAGE_MODEL:=$CHECKPOINTS/CoTTAGE}"
export AUDEX_MODEL_ROOT="${AUDEX_MODEL_ROOT:-$CHECKPOINTS/Nemotron-Labs-Audex-2B}"
export XCODEC1_PATH="${XCODEC1_PATH:-$CHECKPOINTS/xcodec-hubert-general-balanced}"
export XCODEC2_PATH="${XCODEC2_PATH:-$CHECKPOINTS/xcodec2}"
export QWEN_OMNI_MODEL_PATH="$QWEN_OMNI"

: "${BENCHMARKS:=$DATA/benchmarks}"
: "${MMAU_AUDIO:=$DATA/mmau_audio}"
: "${MMAR_AUDIO:=$DATA/mmar_audio}"
: "${MMSU_AUDIO:=$DATA/mmsu_audio}"
: "${TREA_ROOT:=$DATA/trea/TREA_dataset}"
: "${MMAU_TEST_ROOT:=$DATA/mmau_test}"

: "${PAM_ROOT:=$DATA/pam/human_eval/audio}"
: "${CLOTHO_CSV:=$DATA/clotho_mos/clotho_ovl_rel_test_set.csv}"
: "${CLOTHO_AUDIO:=$DATA/clotho_mos/wave_all_16k}"
: "${XACLE_AUDIO:=$DATA/xacle/test/wav}"
: "${XACLE_CSV:=$DATA/xacle/test/test_with_score.csv}"
: "${RELATE_AUDIO:=$DATA/relate/wav}"
: "${RELATE_CSV:=$DATA/relate/REL.csv}"
: "${BVCC_ROOT:=$DATA/BVCC/main/DATA}"
: "${QUALISPEECH_ROOT:=$DATA/qualispeech}"

: "${AUDIOCAPS_CSV:=$DATA/audiocaps/test.csv}"
: "${AUDIOCAPS_AUDIO:=$DATA/audiocaps/audio}"
: "${AUDIOTIME_ROOT:=$DATA/audiotime}"
: "${STEAM_ROOT:=$DATA/AudioTime/STEAMtool}"
: "${SEEDTTS_ROOT:=$DATA/seedtts_testset}"
: "${CV3_ROOT:=$DATA/CV3-Eval}"
: "${LIBRITTS_ROOT:=$DATA/LibriTTS}"
: "${REFERENCE_CACHE:=$DATA/tts_reference_cache}"
if [ -n "${CONDA_PREFIX:-}" ]; then
  PYTHONUSERBASE="$CONDA_PREFIX"
  LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
set +a
