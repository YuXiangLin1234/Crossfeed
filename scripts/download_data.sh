#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/config.sh"

STAGING="$DATA/_downloads"
mkdir -p "$STAGING"
log() { echo "[download_data] $*"; }

download() {
  local url=$1 output=$2
  [ -s "$output" ] && return
  log "downloading $url"
  mkdir -p "$(dirname "$output")"
  curl -fL --retry 5 --retry-delay 5 -C - -o "$output.part" "$url"
  mv "$output.part" "$output"
}

gdrive() {
  local id=$1 output=$2
  [ -s "$output" ] && return
  log "downloading Google Drive file $id"
  python -m gdown --continue "$id" -O "$output"
}

clone() {
  local url=$1 commit=$2 target=$3
  [ -d "$target/.git" ] || git clone "$url" "$target"
  git -C "$target" checkout -q "$commit"
}

unpack() {
  local archive=$1
  rm -rf "$STAGING/unpack"
  mkdir -p "$STAGING/unpack"
  case "$archive" in
    *.zip) unzip -q "$archive" -d "$STAGING/unpack" ;;
    *) tar -xf "$archive" -C "$STAGING/unpack" ;;
  esac
  rm -rf "$STAGING/unpack/__MACOSX"
}

install_dir() {
  local source=$1 target=$2
  [ -d "$source" ] || { echo "missing $source" >&2; exit 1; }
  mkdir -p "$(dirname "$target")"
  rm -rf "$target"
  mv "$source" "$target"
}

if [ ! -d "$MMAR_AUDIO/audio" ]; then
  hf download BoJack/MMAR mmar-audio.tar.gz --repo-type dataset --local-dir "$STAGING/mmar"
  mkdir -p "$MMAR_AUDIO"
  tar -xzf "$STAGING/mmar/mmar-audio.tar.gz" -C "$MMAR_AUDIO"
fi

if [ ! -d "$MMAU_TEST_ROOT/test-audios" ]; then
  hf download gamma-lab-umd/MMAU-test test.parquet test-audios.tar.gz --repo-type dataset --local-dir "$MMAU_TEST_ROOT"
  tar -xzf "$MMAU_TEST_ROOT/test-audios.tar.gz" -C "$MMAU_TEST_ROOT"
  rm -f "$MMAU_TEST_ROOT/test-audios.tar.gz"
fi

clone https://github.com/iiscleap/Audio-LLM-benchmarking-uncertainty.git 8092b70b753762ff0e65887d828c72db111568c5 \
  "$DATA/trea"

if [ ! -s "$PAM_ROOT/scores.csv" ]; then
  download "https://zenodo.org/records/10737388/files/human_eval.zip?download=1" "$STAGING/pam_human_eval.zip"
  unpack "$STAGING/pam_human_eval.zip"
  install_dir "$STAGING/unpack/human_eval" "$DATA/pam/human_eval"
fi

if [ ! -d "$CLOTHO_AUDIO" ] || [ ! -s "$CLOTHO_CSV" ]; then
  download https://github.com/lourson1091/audiobertscore/archive/d7d2a9a517a58967877110f11681a23068fa283a.zip \
    "$STAGING/audiobertscore.zip"
  unpack "$STAGING/audiobertscore.zip"
  repo="$(find "$STAGING/unpack" -mindepth 1 -maxdepth 1 -type d -print -quit)"
  install_dir "$repo/wave_all_16k" "$CLOTHO_AUDIO"
  mkdir -p "$(dirname "$CLOTHO_CSV")"
  cp "$repo/clotho_ovl_rel_test_set.csv" "$CLOTHO_CSV"
fi

if [ ! -d "$XACLE_AUDIO" ] || [ ! -s "$XACLE_CSV" ]; then
  download https://y-okamoto.sakura.ne.jp/XACLE_Challenge/2025/dataset/XACLE_test_data_with_score.zip \
    "$STAGING/XACLE_test_data_with_score.zip"
  unpack "$STAGING/XACLE_test_data_with_score.zip"
  install_dir "$STAGING/unpack/XACLE_test_data/wav" "$XACLE_AUDIO"
  mkdir -p "$(dirname "$XACLE_CSV")"
  cp "$STAGING/unpack/XACLE_test_data/meta_data/test_with_score.csv" "$XACLE_CSV"
fi

if [ ! -d "$RELATE_AUDIO" ]; then
  download https://sarulab.sakura.ne.jp/kanamori/RELATE_open_dataset/RELATE_wave.zip "$STAGING/RELATE_wave.zip"
  unpack "$STAGING/RELATE_wave.zip"
  source_dir="$(find "$STAGING/unpack" -type d -name audiocaps -printf '%h\n' -quit)"
  install_dir "$source_dir" "$RELATE_AUDIO"
fi
download https://raw.githubusercontent.com/sarulab-speech/RELATE/d76f65f3d880c41c13cf93b2eefb1dd6b273dd65/scores/REL.csv \
  "$RELATE_CSV"

if [ ! -d "$QUALISPEECH_ROOT/wav/test" ]; then
  hf download tsinghua-ee/QualiSpeech --repo-type dataset --local-dir "$QUALISPEECH_ROOT"
  unzip -q -o "$QUALISPEECH_ROOT/wav_part1.zip" -d "$QUALISPEECH_ROOT"
  rm -f "$QUALISPEECH_ROOT/wav_part1.zip"
fi
if [ -d "$BVCC_ROOT/wav" ]; then
  while read -r name; do
    [ -n "$name" ] && [ ! -e "$QUALISPEECH_ROOT/wav/test/$name" ] && cp "$BVCC_ROOT/wav/$name" "$QUALISPEECH_ROOT/wav/test/"
  done < <(tr -d '\r' < "$QUALISPEECH_ROOT/test_bvcc.txt")
else
  log "BVCC not found at $BVCC_ROOT: prepare it as described in the README, then rerun to complete QualiSpeech"
fi

download https://raw.githubusercontent.com/cdjkim/audiocaps/d004db3ea1b01cf4fd0347dd8d27db90cadc8809/dataset/test.csv \
  "$AUDIOCAPS_CSV"
[ -d "$AUDIOCAPS_AUDIO" ] || log "AudioCaps audio not found at $AUDIOCAPS_AUDIO: see the README"

if [ ! -d "$AUDIOTIME_ROOT/test500_duration" ]; then
  gdrive 1Xdpc7oY2oK4edUJCUW-vknQaNTBOpE1T "$STAGING/AudioTime-test.zip"
  mkdir -p "$AUDIOTIME_ROOT"
  unzip -q -o "$STAGING/AudioTime-test.zip" -d "$AUDIOTIME_ROOT"
  rm -rf "$AUDIOTIME_ROOT/__MACOSX"
fi

STEAM_REPO="$(dirname "$STEAM_ROOT")"
if [ ! -d "$STEAM_REPO/.git" ]; then
  clone https://github.com/zeyuxie29/AudioTime.git 9ac2023049c00e3e8375baab8145a6177cba4501 "$STEAM_REPO"
  git -C "$STEAM_REPO" apply "$ROOT/third_party/patches/steam.patch"
fi
GROUNDING="$STEAM_ROOT/steam/grounding_tool/grounding_ckpt"
if [ ! -s "$GROUNDING/best.pth" ]; then
  gdrive 1lnkX3AUhFiPvqUZVm-W558sIgyOmHgOZ "$STAGING/grounding_ckpt.zip"
  unpack "$STAGING/grounding_ckpt.zip"
  found="$(find "$STAGING/unpack" -name best.pth -printf '%h\n' -quit)"
  mkdir -p "$GROUNDING"
  cp "$found"/* "$GROUNDING/"
fi

if [ ! -s "$SEEDTTS_ROOT/en/meta.lst" ]; then
  gdrive 1GlSjVfSHkW3-leKKBlfrjuuTGqQ_xaLP "$STAGING/seedtts_testset.tar"
  unpack "$STAGING/seedtts_testset.tar"
  install_dir "$STAGING/unpack/seedtts_testset" "$SEEDTTS_ROOT"
fi

clone https://github.com/FunAudioLLM/CV3-Eval.git 68b7682a410d048308d1a12f2fbc266770cd23d5 "$CV3_ROOT"
download https://modelscope.cn/models/damo/speech_eres2net_sv_en_voxceleb_16k/resolve/master/pretrained_eres2net.ckpt \
  "$CV3_ROOT/utils/3D-Speaker/pretrained/speech_eres2net_sv_en_voxceleb_16k/pretrained_eres2net.ckpt"

if [ ! -d "$LIBRITTS_ROOT/test-clean" ]; then
  download https://www.openslr.org/resources/60/test-clean.tar.gz "$STAGING/libritts_test_clean.tar.gz"
  unpack "$STAGING/libritts_test_clean.tar.gz"
  install_dir "$STAGING/unpack/LibriTTS/test-clean" "$LIBRITTS_ROOT/test-clean"
fi

rm -rf "$STAGING/unpack"
log "done"
