<div align="center">

# CROSSFEED: Self-Improving Unified Audio Models across Understanding, Judging, and Generation with Zero External Data

[![Crossfeed-Unison](https://img.shields.io/badge/Crossfeed--Unison-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/Evan-Lin/Crossfeed-Unison)
[![Crossfeed-Audex-2B](https://img.shields.io/badge/Crossfeed--Audex--2B-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/Evan-Lin/Crossfeed-Audex-2B)

<!-- TODO: add the paper badge. -->

<p>
CROSSFEED is a post-training framework that turns a unified audio model's own capabilities into training supervision. The model synthesizes audio, speech and multi-speaker dialogues, improves its understanding with agreement-based answer verification and its judging with probability-based pseudo labels, and then uses the updated judge to build preference pairs for audio and speech generation. No external data or teacher model is used.
</p>

</div>

This repository implements CROSSFEED for two unified audio models with different architectures:

- **UNISON**: an LALM-conditioned diffusion model whose Qwen2.5-Omni-7B thinker understands and judges audio, and whose hidden states from multiple LALM layers condition the corresponding MM-DiT blocks to generate audio and speech
- **Audex-2B**: a single autoregressive model that understands audio and generates audio and speech codec tokens

## Contents

- [Installation](#installation)
- [Models](#models)
- [Quick start](#quick-start)
- [Inference](#inference)
- [Data](#data)
- [Evaluation](#evaluation)
- [Training](#training)
- [Acknowledgments](#acknowledgments)

## Installation

> **Disk:** the Conda environment takes about 14 GB in `.envs/crossfeed`, plus the pip download cache.

```bash
bash scripts/setup_env.sh
conda activate .envs/crossfeed
```

## Models

> **Disk:** the released checkpoints take 31 GB in total. `scripts/download_models.sh` downloads about 69 GB: 59 GB into `checkpoints/` and 10 GB of metric models into the Hugging Face, Whisper and Torch caches.

The released checkpoints are:

| Checkpoint | Size | Description |
|---|---|---|
| [`Evan-Lin/Crossfeed-Unison`](https://huggingface.co/Evan-Lin/Crossfeed-Unison) | 25.3 GB | Final UNISON checkpoint, with Qwen2.5-Omni at the root and the DiT in `dit/` |
| [`Evan-Lin/Crossfeed-Audex-2B`](https://huggingface.co/Evan-Lin/Crossfeed-Audex-2B) | 5.9 GB | Final Audex-2B checkpoint, in the layout of `checkpoint_folder_full` |

```bash
hf download Evan-Lin/Crossfeed-Unison --local-dir checkpoints/Crossfeed-Unison
hf download Evan-Lin/Crossfeed-Audex-2B --local-dir checkpoints/Crossfeed-Audex-2B
```

Inference and evaluation also need the Audex-2B plugins, codecs, VAE and metric models below, and training starts from the base models. Download them once:

```bash
bash scripts/download_models.sh
```

| Checkpoint | Size | Use |
|---|---|---|
| `Qwen/Qwen2.5-Omni-7B` | 22.4 GB | UNISON understanding, judging and text encoder |
| `jac22/UNISON` (`unison_D24S0_O_20ch`) | 2.9 GB | UNISON DiT |
| MMAudio `v1-16.pth`, `best_netG.pt` | 1.1 GB | UNISON 16 kHz VAE and vocoder |
| `nvidia/Nemotron-Labs-Audex-2B` | 5.9 GB | Audex-2B and its vLLM plugins |
| `hf-audio/xcodec-hubert-general-balanced`, `HKUSTAudio/xcodec2` | 4.0 GB | Audex-2B audio and speech decoders |
| `AutoAudioReasoningMOS/CoTTAGE` | 22.4 GB | CoTTAGE judge for text-to-audio evaluation |
| Whisper large-v3, Wav2Vec2, LAION-CLAP, UTMOS, CNN14 | 10.1 GB | evaluation metrics |

All paths are defined in `config.sh` and default to locations inside the repository. Put machine-specific paths in `config.local.sh`, which `config.sh` reads first.

## Quick start

From the repository root, prepare the models and evaluation data in the activated Conda environment.

```bash
bash scripts/download_models.sh
bash scripts/download_data.sh
```

Complete the manual AudioCaps and BVCC steps in [Data](#data) before running the full evaluation.

Train UNISON with the paper recipe (5,000 scenes, K=8 candidates, self-merging), then evaluate the resulting checkpoints.

```bash
python -m crossfeed.train unison
python -m crossfeed.evaluate unison \
    --omni runs/unison/u5000_g5000_merge1/final_omni \
    --dit runs/unison/u5000_g5000_merge1/final_dit/model.safetensors \
    --out-dir runs/unison/eval/crossfeed
```

For Audex-2B, use the final unified checkpoint.

```bash
python -m crossfeed.train audex
python -m crossfeed.evaluate audex \
    --model runs/audex/u5000_g5000_merge1/final \
    --out-dir runs/audex/eval/crossfeed
```

Each evaluation writes `report.md` in its output directory. To evaluate the released checkpoints instead, pass `--omni checkpoints/Crossfeed-Unison --dit checkpoints/Crossfeed-Unison/dit/model.safetensors` or `--model checkpoints/Crossfeed-Audex-2B`.

## Inference

Run each example as a Python file from the repository root after `source ./config.sh`, which sets `PYTHONPATH` and the model paths. The models are served with vLLM, which starts worker processes, so keep the code under `if __name__ == '__main__':`. Each example needs one GPU.

### UNISON

Text and audio question answering:

```python
from crossfeed.unison.omni import OmniVLLM, load_audio

if __name__ == '__main__':
    checkpoint = 'checkpoints/Crossfeed-Unison'
    # checkpoint = 'Evan-Lin/Crossfeed-Unison'  # Load from the Hub and cache the weights locally.
    model = OmniVLLM(checkpoint)
    question = 'What is the main sound in the recording?\n(A) A dog barking\n(B) Rain\n(C) Speech\n(D) Music'
    print(model.generate([
        dict(prompt='Describe a rainy street in one sentence.', audio=None),
        dict(prompt=question, audio=load_audio('clip.wav')),
    ], max_new_tokens=128))
```

Text-to-audio and text-to-speech. UNISON selects the task with the `[Audio]` or `[Speech]` tag:

```python
from crossfeed.evaluation.synthesis import synthesize

if __name__ == '__main__':
    omni = 'checkpoints/Crossfeed-Unison'
    dit = f'{omni}/dit/model.safetensors'
    # from huggingface_hub import hf_hub_download
    # omni = 'Evan-Lin/Crossfeed-Unison'
    # dit = hf_hub_download(omni, filename='dit/model.safetensors')
    wavs = synthesize('unison', [
        dict(id='dog', prompt='[Audio] A dog barks while rain falls on a roof'),
        dict(id='hello', prompt='[Speech] A voice saying "Hello, and welcome to the demo."'),
    ], 'outputs/unison', model=dit, omni=omni)
    print(wavs)
```

### Audex-2B

Text and audio question answering:

```python
from crossfeed.evaluation.engines import AudexGenerator

if __name__ == '__main__':
    checkpoint = 'checkpoints/Crossfeed-Audex-2B'
    # from huggingface_hub import snapshot_download
    # checkpoint = snapshot_download('Evan-Lin/Crossfeed-Audex-2B')
    model = AudexGenerator(checkpoint)
    question = 'What is the main sound in the recording?\n(A) A dog barking\n(B) Rain\n(C) Speech\n(D) Music'
    print(model.generate([
        dict(prompt='Describe a rainy street in one sentence.'),
        dict(prompt=question, wav='clip.wav'),
    ], max_new_tokens=128))
```

Text-to-audio and text-to-speech:

```python
from crossfeed.evaluation.synthesis import synthesize

if __name__ == '__main__':
    checkpoint = 'checkpoints/Crossfeed-Audex-2B'
    # from huggingface_hub import snapshot_download
    # checkpoint = snapshot_download('Evan-Lin/Crossfeed-Audex-2B')
    wavs = synthesize('audex', [
        dict(id='dog', task='audio', caption='A dog barks while rain falls on a roof'),
        dict(id='hello', task='speech', text='Hello, and welcome to the demo.'),
    ], 'outputs/audex', model=checkpoint)
    print(wavs)
```

`synthesize` writes 16 kHz clips to `<output>/audio/<id>_k0.wav` and returns their paths. UNISON always generates 10 s clips, so short speech ends with silence.

## Data

Training uses no external data. Everything below is for evaluation only.

> **Disk:** `scripts/download_data.sh` uses about 58 GB: 52 GB under `data/` and 4 GB of MMAU-test-mini and MMSU in the Hugging Face cache. About 12 GB of it are the original archives in `data/_downloads`, which can be deleted once the script has finished. BVCC needs another 45 GB of manual downloads.

### Benchmarks

| Benchmark | Task | Disk | Source |
|---|---|---|---|
| MMAU-test-mini, MMAU-test | Understanding | 1.6 GB, 14 GB | [`gamma-lab-umd/MMAU-test-mini`](https://huggingface.co/datasets/gamma-lab-umd/MMAU-test-mini), [`gamma-lab-umd/MMAU-test`](https://huggingface.co/datasets/gamma-lab-umd/MMAU-test) |
| MMAR | Understanding | 8 GB | [`BoJack/MMAR`](https://huggingface.co/datasets/BoJack/MMAR) |
| MMSU | Understanding | 5 GB | [`ddwang2000/MMSU`](https://huggingface.co/datasets/ddwang2000/MMSU) |
| TREA | Understanding | 2.5 GB | [Audio-LLM-benchmarking-uncertainty](https://github.com/iiscleap/Audio-LLM-benchmarking-uncertainty) (`count`, `duration`, `order`) |
| PAM | Audio-REL, Audio-OVL | 2.8 GB | [PAM](https://github.com/soham97/PAM) human evaluation set on [Zenodo](https://zenodo.org/records/10737388) |
| Clotho | Audio-REL, Audio-OVL | 0.6 GB | [AudioBERTScore](https://github.com/lourson1091/audiobertscore) |
| XACLE | Audio-REL | 1.7 GB | [XACLE Challenge](https://xacle.org/) test set with scores |
| RELATE | Audio-REL | 9.6 GB | [RELATE](https://github.com/sarulab-speech/RELATE) test split |
| BVCC | Speech-OVL | 45 GB download | [VoiceMOS 2022 main track](https://zenodo.org/records/6572573) (manual, see below) |
| QualiSpeech | Speech-OVL | 2.1 GB | [`tsinghua-ee/QualiSpeech`](https://huggingface.co/datasets/tsinghua-ee/QualiSpeech) test split |
| AudioCaps | Text-to-audio | manual | [AudioCaps](https://github.com/cdjkim/audiocaps) test split (audio is manual, see below) |
| AudioTime | Text-to-audio | 1.9 GB | [AudioTime](https://github.com/zeyuxie29/AudioTime) test500 and STEAMtool |
| Seed-TTS-Eval | Text-to-speech | 2.5 GB | [seed-tts-eval](https://github.com/BytedanceSpeech/seed-tts-eval) `test-en` |
| CV3-Eval | Text-to-speech | 2.5 GB | [CV3-Eval](https://github.com/FunAudioLLM/CV3-Eval) `en` and `hard_en`, with 3D-Speaker ERes2Net for SIM |
| LibriTTS | Text-to-speech | 2.7 GB | [LibriTTS](https://www.openslr.org/60/) `test-clean` |

### Download

```bash
bash scripts/download_data.sh
```

The script fetches every benchmark from its original publisher: Hugging Face, Zenodo, GitHub, Google Drive (through `gdown`), ModelScope and OpenSLR. Repositories are pinned to the commits used in the paper, STEAMtool is patched with `third_party/patches/steam.patch` so that it runs on current PyTorch, and existing directories are skipped, so the script can be rerun at any time. MMAU-test-mini and MMSU are read directly from Hugging Face when the benchmarks are exported.

Two datasets cannot be redistributed and have to be prepared by hand:

- **AudioCaps audio.** The script downloads the official `test.csv`. Obtain the audio of each test clip yourself (the 10 s YouTube segment starting at `start_time`) and save it as `data/audiocaps/audio/<youtube_id>_<start_time>.wav`. Clips without audio are skipped.
- **BVCC.** Download `main.tar.gz` from [Zenodo](https://zenodo.org/records/6572573), extract it to `data/BVCC/`, and follow its `README.txt` (`01_get.py`, `02_gather.py`, `03_preprocess.py`, with SoX and sv56) to gather the Blizzard Challenge samples, which needs about 45 GB of downloads. Then rerun `scripts/download_data.sh`, which copies the 718 BVCC clips of the QualiSpeech test set into place.

Everything lands under `data/`:

```
data/
├── mmar_audio/                       audio/ (original), 16k/ (16 kHz copies made at export)
├── mmau_audio/, mmsu_audio/          MMAU-test-mini and MMSU clips written at export
├── mmau_test/                        test.parquet, test-audios/
├── trea/TREA_dataset/                count/, duration/, order/
├── pam/human_eval/audio/             PAM clips and scores.csv
├── clotho_mos/                       clotho_ovl_rel_test_set.csv, wave_all_16k/
├── xacle/test/                       test_with_score.csv, wav/
├── relate/                           REL.csv, wav/
├── BVCC/main/DATA/                   sets/, wav/ (manual)
├── qualispeech/                      test.csv, wav/test/
├── audiocaps/                        test.csv, audio/ (manual)
├── audiotime/                        test500_{duration,frequency,ordering,timestamp}/
├── AudioTime/STEAMtool/              STEAM metrics and grounding checkpoint
├── seedtts_testset/en/               meta.lst, prompt-wavs/
├── CV3-Eval/                         data/zero_shot/{en,hard_en}/, 3D-Speaker
├── LibriTTS/test-clean/
└── benchmarks/*.json                 exported understanding benchmarks
```

Set custom paths in `config.local.sh` using the variables in `config.sh`. The Python evaluation entry point exports understanding benchmarks to `data/benchmarks/*.json` on first use. It converts MMAU-test-mini and MMAR clips to 16 kHz, as in the paper. Reference prompts for zero-shot TTS are prepared on first use and cached in `data/tts_reference_cache`.

## Evaluation

```bash
python -m crossfeed.evaluate unison --baselines
python -m crossfeed.evaluate audex --baselines
```

The editable launchers [Audex-2B](scripts/audex/evaluate.sh) and [UNISON](scripts/unison/evaluate.sh) expose checkpoint paths, tasks, sample limits, self-consistency settings, batch sizes and audio generation settings. Edit their defaults or set values for a single run.

```bash
TASKS="judge tts" CF_EVAL_LIMIT=20 OUT=runs/unison/eval/smoke bash scripts/unison/evaluate.sh
```

| Task | Benchmarks | Metrics |
|---|---|---|
| `understanding` | MMAU-test-mini, MMAR, MMSU, TREA, MMAU-test | accuracy, greedy decoding with the training prompt |
| `judge` | PAM, Clotho, XACLE, RELATE, BVCC, QualiSpeech | Pearson, Spearman and MSE against human MOS (1 to 5) |
| `tta` | AudioCaps, AudioTime | FAD, FD, KL, IS, CLAP; order error rate, duration and frequency L1, timestamp F1; CoTTAGE REL and OVL |
| `tts` | Seed-TTS-Eval, CV3-Eval, CV3-Hard, LibriTTS | corpus WER (Whisper large-v3), UTMOS; zero-shot WER, UTMOS and SIM for UNISON |

`--baselines` also runs the paper baselines: self-consistency (maj@8) for understanding, and Prompting (greedy and avg@8), AQAScore, G-Eval and AQAScore + G-Eval for judging. The trained judge is scored with G-Eval.

`--tasks` selects whole tasks, and `--limit N` keeps only the first N items of each benchmark:

```bash
python -m crossfeed.evaluate unison --tasks judge --baselines --limit 100 \
    --out-dir runs/unison/eval/judge-smoke
```

Evaluation directories record the checkpoint and run settings in `run_config.json`. Rerun a command with the same settings to resume generation, or add more tasks for that model in the same directory. Choose a separate `--out-dir` when comparing checkpoints or changing a task's sample limit or generation settings. The single benchmark TTA and TTS commands apply the same check to `--out_dir`. For outputs written before `run_config.json` was introduced, use a fresh directory so the provenance of each result is recorded.

### A single benchmark

Load the configuration first, then call the task module directly. The examples use Audex-2B; for UNISON, use `--family unison`, pass the Omni checkpoint as `--model` for understanding, and pass `--model <dit>/model.safetensors --omni <omni>` for generation. Every command accepts `--limit N`.

```bash
source ./config.sh
MODEL=checkpoints/Crossfeed-Audex-2B
# MODEL="$(hf download Evan-Lin/Crossfeed-Audex-2B --quiet)"

# Understanding: mmau (MMAU-test-mini), mmar, mmsu, trea or mmau_test
bash scripts/prepare_benchmarks.sh
python -m crossfeed.evaluation.understanding evaluate --family audex --model "$MODEL" \
    --benchmark mmar --data "$BENCHMARKS/mmar.json" --out runs/audex/eval/single/mmar.json

# Text-to-audio: audiocaps, or audiotime with --tasks duration,frequency,ordering,timestamp
python -m crossfeed.evaluation.tta audiotime --family audex --model "$MODEL" --tasks ordering \
    --audiotime_root "$AUDIOTIME_ROOT" --steam_root "$STEAM_ROOT" --out_dir runs/audex/eval/single/audiotime
python -m crossfeed.evaluation.cottage --model "$COTTAGE_MODEL" --dirs runs/audex/eval/single/audiotime

# Text-to-speech: seedtts, cv3, cv3_hard or libritts
python -m crossfeed.evaluation.tts cv3_hard --family audex --model "$MODEL" --cv3_root "$CV3_ROOT" \
    --out_dir runs/audex/eval/single/cv3-hard
```

The six judge benchmarks are evaluated together with `--tasks judge`. For AudioCaps, pass `--audiocaps_csv "$AUDIOCAPS_CSV" --audiocaps_audio "$AUDIOCAPS_AUDIO"`; for Seed-TTS-Eval and LibriTTS, pass `--seedtts_root "$SEEDTTS_ROOT"` or `--libritts_root "$LIBRITTS_ROOT"`. For UNISON zero-shot TTS, add `--zero_shot --reference_cache "$REFERENCE_CACHE"`.

## Training

> **Disk:** with the default 5,000 scenes, a run writes the generated clips and several full checkpoints under `runs/`. Reserve about 70 GB for UNISON and 35 GB for Audex-2B.

### Pipeline

`python -m crossfeed.train unison` and `python -m crossfeed.train audex` run the pipeline on one GPU. Both recipes split the scene budget equally between audio and speech, including standalone utterances and dialogues in the speech budget. Both use LoRA dropout of 0. Each instruction pool, dialogue plan and dialogue script generation loop runs for up to 500 rounds by default.

1. **Instructions**: the model proposes sound events, speech texts and dialogue plans.
2. **Generation**: K candidates per instruction, plus perturbed instructions and voice-cloned dialogues.
3. **Pseudo labels**: AQAScore and G-Eval relevance and quality scores; speech relevance is 1 - WER from the model's own ASR.
4. **QA**: questions about the best clip are answered with eight balanced option orders and kept when at least four votes agree. Both models count replies containing a `Final answer: (X)` marker with an option from A to D.
5. **Understanding and judging**: LoRA training on the QA and pseudo-label data.
6. **Preference pairs**: the trained judge rescores every candidate, and weak-Pareto pairs with a margin of 0.0625 are combined with perturbation pairs.
7. **Generation training**: Diffusion-DPO for the UNISON DiT, DPO on codec tokens for Audex-2B.
8. **Self-merging**: the understanding and generation updates are interpolated with weight 0.5.

### Options

The editable launchers [Audex-2B](scripts/audex/train.sh) and [UNISON](scripts/unison/train.sh) expose scene budgets, learning rates, epochs, batch sizes, LoRA rank and alpha, and DPO settings. Both launch the same Python entry point.

```bash
N=1000 CF_UND_LR=3e-5 OUT=runs/audex/lr-3e-5 bash scripts/audex/train.sh
```

Choose a fresh `OUT` when changing training settings and a fresh `DATA_RUN` when changing data generation or QA filtering. Existing paths can reuse completed stages.

| Option | Environment variable | Default | Meaning |
|---|---|---|---|
| `--n` | `N` | 5000 | scenes generated by the model |
| `--und-n` | `UND_N` | `N` | scenes used for understanding and judge training |
| `--gen-n` | `GEN_N` | `N` | scenes used for generation preference pairs |
| `--no-merge` | `MERGE=0` | self-merging enabled | train the variant without self-merging |
| `--data-run` | `DATA_RUN` | `runs/<family>/data_n$N` | generated data |
| `--out-dir` | `OUT` | `runs/<family>/u${UND_N}_g${GEN_N}_merge$MERGE` | checkpoints |
| `--skip-data` | `SKIP_DATA=1` | off | reuse the data already in the data directory |
| `--seed` | `SEED` | 531 | base seed |

Scenes are interleaved by task and category, so `UND_N` and `GEN_N` select balanced prefixes of one generated run. Older UNISON data runs are rescored when their QA stage is run again. `--skip-data` reuses the labels already on disk. To train several variants from the same data, generate once and reuse it:

```bash
python -m crossfeed.train unison --n 5000
python -m crossfeed.train unison --skip-data --und-n 3000 --gen-n 3000
python -m crossfeed.train unison --skip-data --no-merge
```

Every stage resumes from its outputs, so an interrupted run continues where it stopped.

## Acknowledgments

Crossfeed builds on [UNISON](https://huggingface.co/jac22/UNISON), [Audex-2B](https://huggingface.co/nvidia/Nemotron-Labs-Audex-2B), and [Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B). Evaluation uses [Whisper](https://github.com/openai/whisper), [LAION-CLAP](https://github.com/LAION-AI/CLAP), [AudioLDM Eval](https://github.com/haoheliu/audioldm_eval), [UTMOS](https://github.com/tarepan/SpeechMOS), [3D-Speaker](https://github.com/modelscope/3D-Speaker), [AudioTime](https://github.com/zeyuxie29/AudioTime), and [CoTTAGE](https://huggingface.co/AutoAudioReasoningMOS/CoTTAGE). Please follow the licenses of the underlying models, datasets, and dependencies.
