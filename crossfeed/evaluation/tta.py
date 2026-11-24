import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from crossfeed.evaluation.run_config import RunConfigError, file_identity, samples_identity
from crossfeed.evaluation.synthesis import add_arguments, benchmark_output, from_args
from crossfeed.utils import write_json, write_jsonl

AUDIOTIME_TASKS = ('duration', 'frequency', 'ordering', 'timestamp')
STEAM_METRIC = re.compile(r'^####\s*(.*?):\s*([0-9.]+)')
SR = 16000


def audiocaps_rows(csv_path):
    rows, seen = [], set()
    with open(csv_path, newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            caption = (row.get('caption') or '').strip()
            key = (row.get('youtube_id', ''), row.get('start_time', ''))
            if not caption or not row.get('audiocap_id') or key in seen:
                continue
            seen.add(key)
            rows.append(dict(id=f'{key[0]}_{key[1]}', prompt=f'[Audio] {caption}', task='audio', caption=caption))
    return rows


def audiotime_rows(root, task):
    data = json.loads((Path(root) / f'test500_{task}' / f'{task}_captions.json').read_text(encoding='utf-8'))
    return [dict(id=uid, prompt=f'[Audio] {item["caption"].strip()}', task='audio', caption=item['caption'].strip())
            for uid, item in data.items() if (item.get('caption') or '').strip()]


def paired_dirs(generated, ref_dir, work):
    import librosa
    import soundfile as sf
    gen_out, ref_out = work / 'gen', work / 'ref'
    gen_out.mkdir(parents=True, exist_ok=True)
    ref_out.mkdir(parents=True, exist_ok=True)
    for cache in work.glob('*cache*'):
        cache.unlink()
    kept = []
    for uid, path in sorted(generated.items()):
        ref = Path(ref_dir) / f'{uid}.wav'
        if not ref.exists():
            continue
        shutil.copy2(path, gen_out / f'{uid}.wav')
        if not (ref_out / f'{uid}.wav').exists():
            sf.write(ref_out / f'{uid}.wav', librosa.load(str(ref), sr=SR, mono=True)[0], SR, subtype='PCM_16')
        kept.append(uid)
    return gen_out, ref_out, kept


def distribution_metrics(gen_dir, ref_dir):
    import torch
    from audioldm_eval import EvaluationHelper
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
    helper = EvaluationHelper(SR, torch.device('cuda' if torch.cuda.is_available() else 'cpu'), backbone='cnn14')
    metrics = helper.main(str(gen_dir), str(ref_dir))
    return dict(FAD=float(metrics['frechet_audio_distance']), FD=float(metrics['frechet_distance']),
                KL=float(metrics['kullback_leibler_divergence_softmax']), IS=float(metrics['inception_score_mean']))


def clap_score(paths, captions, batch_size=32):
    import laion_clap
    import torch
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()
    ids, scores = sorted(paths), []
    for start in range(0, len(ids), batch_size):
        chunk = ids[start:start + batch_size]
        with torch.no_grad():
            audio = model.get_audio_embedding_from_filelist(x=[str(paths[u]) for u in chunk], use_tensor=True)
            text = model.get_text_embedding([captions[u] for u in chunk], use_tensor=True)
        audio = audio / audio.norm(dim=-1, keepdim=True)
        text = text / text.norm(dim=-1, keepdim=True)
        scores.extend((audio * text).sum(-1).cpu().tolist())
    return float(np.mean(scores))


def steam(gen_dir, captions_json, task, steam_root, work):
    ref_dir = work / 'steam_ref'
    ref_dir.mkdir(parents=True, exist_ok=True)
    have = {p.stem for p in Path(gen_dir).glob('*.wav')}
    refs = {k: v for k, v in json.loads(Path(captions_json).read_text()).items() if k in have}
    (ref_dir / f'{task}_caption.json').write_text(json.dumps(refs, indent=4))
    root = Path(steam_root)
    result = subprocess.run([sys.executable, str(root / 'steam' / 'runner' / 'steam_eval.py'),
                             '-g', str(root / 'steam' / 'grounding_tool' / 'grounding_ckpt'), '-p', str(gen_dir),
                             '-r', str(ref_dir), '-t', task], capture_output=True, text=True,
                            env={**os.environ, 'PYTHONPATH': str(root)}, cwd=work)
    (work / f'steam_{task}.log').write_text(result.stdout + result.stderr)
    metrics = [STEAM_METRIC.match(line) for line in result.stdout.splitlines()]
    metrics = [m for m in metrics if m]
    if result.returncode or not metrics:
        raise RuntimeError(f'STEAM failed for {task}; see {work / f"steam_{task}.log"}')
    return float(metrics[-1].group(2))


def run_audiocaps(args):
    out = Path(args.out_dir).absolute()
    rows = audiocaps_rows(args.audiocaps_csv)[:args.limit or None]
    references = samples_identity([file_identity(Path(args.audiocaps_audio) / f'{r["id"]}.wav') for r in rows])
    with benchmark_output(out, args, task='tta', benchmark='audiocaps',
                          samples=samples_identity(rows), references=references):
        write_jsonl(out / 'instructions.jsonl', rows)
        generated = from_args(args, rows, out)
        gen_dir, ref_dir, kept = paired_dirs(generated, args.audiocaps_audio, out / 'metrics')
        captions = {r['id']: r['caption'] for r in rows}
        summary = dict(n=len(kept), **distribution_metrics(gen_dir, ref_dir),
                       CLAP=clap_score({u: gen_dir / f'{u}.wav' for u in kept}, captions))
        write_json(out / 'metrics' / 'summary.json', summary)
        print(f'[audiocaps] {json.dumps(summary)}', flush=True)


def run_audiotime(args):
    root = Path(args.out_dir).absolute()
    names = dict(duration='duration_l1', frequency='frequency_l1', ordering='order_error_rate', timestamp='timestamp_f1')
    datasets = {task: audiotime_rows(args.audiotime_root, task)[:args.limit or None]
                for task in args.tasks.split(',')}
    captions = {task: Path(args.audiotime_root) / f'test500_{task}' / f'{task}_captions.json' for task in datasets}
    samples = {task: samples_identity(rows) for task, rows in datasets.items()}
    references = {task: file_identity(path) for task, path in captions.items()}
    steam_root = Path(args.steam_root).resolve()
    with benchmark_output(root, args, task='tta', benchmark='audiotime',
                          samples=samples, references=references, steam_root=str(steam_root)):
        summary = {}
        for task, rows in datasets.items():
            out = root / task
            write_jsonl(out / 'instructions.jsonl', rows)
            generated = from_args(args, rows, out)
            steam_in = out / 'metrics' / 'steam_in'
            steam_in.mkdir(parents=True, exist_ok=True)
            for uid, path in generated.items():
                shutil.copy2(path, steam_in / f'{uid}.wav')
            summary[names[task]] = steam(steam_in, captions[task], task, steam_root, out / 'metrics')
            print(f'[audiotime] {task}: {names[task]} = {summary[names[task]]:.4f}', flush=True)
        write_json(root / 'summary.json', summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('benchmark', choices=['audiocaps', 'audiotime'])
    add_arguments(parser)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--audiocaps_csv', default=None)
    parser.add_argument('--audiocaps_audio', default=None)
    parser.add_argument('--audiotime_root', default=None)
    parser.add_argument('--steam_root', default=None)
    parser.add_argument('--tasks', default=','.join(AUDIOTIME_TASKS))
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    try:
        (run_audiocaps if args.benchmark == 'audiocaps' else run_audiotime)(args)
    except RunConfigError as error:
        parser.exit(2, f'{error}\n')


if __name__ == '__main__':
    main()
