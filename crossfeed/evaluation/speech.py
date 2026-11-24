import math
import multiprocessing
import os
import random
import string
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from crossfeed.utils import seed_for, write_json, write_jsonl

_WHISPER = None


def normalize(text):
    from zhon.hanzi import punctuation
    for mark in punctuation + string.punctuation:
        if mark != "'":
            text = text.replace(mark, '')
    return text.replace('  ', ' ').lower()


def score_text(reference, hypothesis):
    import jiwer
    truth, hypo = normalize(reference), normalize(hypothesis)
    measures = jiwer.process_words(truth, hypo)
    n = measures.hits + measures.substitutions + measures.deletions
    errors = measures.substitutions + measures.deletions + measures.insertions
    return dict(errors=errors, n_reference=n, wer=errors / n, normalized_ref=truth, normalized_hyp=hypo)


def trim_for_asr(source, target):
    import librosa
    import soundfile as sf
    import whisper
    wav = whisper.load_audio(str(source))
    start, end = 0, len(wav)
    if float(np.max(np.abs(wav))) > 1e-8:
        _, (a, b) = librosa.effects.trim(wav, top_db=40)
        start, end = max(0, int(a) - 4000), min(len(wav), int(b) + 4000)
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    sf.write(target, wav[start:end], 16000, subtype='FLOAT', format='WAV')
    return str(target)


def _init_whisper(device):
    global _WHISPER
    import torch
    import whisper
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _WHISPER = whisper.load_model('large-v3', device=device, download_root=os.environ.get('WHISPER_CACHE'))


def _transcribe(job):
    import torch
    path, seed = job
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return _WHISPER.transcribe(path, language='en')['text'].strip()


def transcribe(paths, seeds, workers=6, device='cuda'):
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '1'
    context = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(min(workers, len(paths)), mp_context=context, initializer=_init_whisper,
                             initargs=(device,)) as pool:
        return list(pool.map(_transcribe, zip(map(str, paths), seeds), chunksize=1))


def corpus_wer(items, out_dir, dataset, trim, seed=531, workers=6):
    out_dir = Path(out_dir)
    scored = [trim_for_asr(item['wav'], out_dir / 'trimmed' / Path(item['wav']).name) if trim else str(item['wav'])
              for item in items]
    hypotheses = transcribe(scored, [seed_for(seed, 'wer', dataset, item['id']) for item in items], workers)
    rows = [dict(id=item['id'], ref=item['text'], hyp=hyp, wav=path, **score_text(item['text'], hyp))
            for item, hyp, path in zip(items, hypotheses, scored)]
    errors, total = sum(r['errors'] for r in rows), sum(r['n_reference'] for r in rows)
    summary = dict(n=len(rows), corpus_wer=100 * errors / total, total_errors=errors, total_reference_words=total)
    write_jsonl(out_dir / 'wer.jsonl', rows)
    write_json(out_dir / 'wer_summary.json', summary)
    return summary, {r['id']: r['wav'] for r in rows}


def utmos(groups, out_dir, device='cuda'):
    import librosa
    import soundfile as sf
    import torch
    out_dir = Path(out_dir)
    model = torch.hub.load('tarepan/SpeechMOS:v1.2.0', 'utmos22_strong', trust_repo=True).to(device).eval()
    trimmed = out_dir / 'utmos_trimmed'
    trimmed.mkdir(parents=True, exist_ok=True)
    rows = []
    for parent, wavs in groups.items():
        scores = []
        for wav in wavs:
            y, sr = librosa.load(str(wav), sr=None, mono=True)
            _, (a, b) = librosa.effects.trim(y, top_db=40)
            pad = int(0.25 * sr)
            target = trimmed / Path(wav).name
            sf.write(str(target), y[max(0, a - pad):min(len(y), b + pad)], sr)
            y, sr = librosa.load(str(target), sr=None, mono=True)
            with torch.no_grad():
                scores.append(float(model(torch.from_numpy(y).unsqueeze(0).to(device), sr)))
        rows.append(dict(id=parent, utmos=float(np.mean(scores)), segments=len(scores)))
    summary = dict(n=len(rows), utmos=float(np.mean([r['utmos'] for r in rows])))
    write_jsonl(out_dir / 'utmos.jsonl', rows)
    write_json(out_dir / 'utmos_summary.json', summary)
    return summary


def resample_prompt(path, out_dir):
    import soundfile as sf
    if sf.info(str(path)).samplerate == 16000:
        return str(path)
    target = Path(out_dir) / 'prompts_16k' / Path(path).name
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['sox', '-D', str(path), '-e', 'floating-point', '-b', '32', str(target), 'rate', '16000'],
                       check=True)
    return str(target)


def similarity(targets, prompts, out_dir, cv3_root):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = {uid: resample_prompt(prompts[uid], out_dir) for uid in targets}
    (out_dir / 'hyp_wavs.scp').write_text(''.join(f'{uid} {targets[uid]}\n' for uid in sorted(targets)))
    (out_dir / 'prompt_wavs.scp').write_text(''.join(f'{uid} {prompts[uid]}\n' for uid in sorted(targets)))
    speaker = Path(cv3_root) / 'utils' / '3D-Speaker'
    log = out_dir / 'spk_simi_scores.txt'
    subprocess.run([sys.executable, str(Path(cv3_root) / 'scripts' / 'eval_speaker_similarity.py'),
                    '--model_id', 'damo/speech_eres2net_sv_en_voxceleb_16k',
                    '--local_model_dir', str(speaker / 'pretrained'), '--prompt_wavs', str(out_dir / 'prompt_wavs.scp'),
                    '--hyp_wavs', str(out_dir / 'hyp_wavs.scp'), '--log_file', str(log), '--devices', '0'],
                   check=True, env={**os.environ, 'PYTHONPATH': os.pathsep.join([str(speaker),
                                                                                os.environ.get('PYTHONPATH', '')])})
    scores, average = {}, None
    for line in log.read_text().splitlines():
        uid, _, value = line.split()
        if uid == 'avg':
            average = float(value)
        else:
            scores[uid] = float(value)
    if set(scores) != set(targets) or average is None or not all(map(math.isfinite, scores.values())):
        raise RuntimeError(f'incomplete speaker similarity scores in {log}')
    summary = dict(n=len(scores), sim=average / 100)
    write_json(out_dir / 'sim_summary.json', summary)
    return summary
