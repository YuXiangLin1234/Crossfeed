import argparse
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from crossfeed.evaluation.speech import corpus_wer, similarity, utmos
from crossfeed.evaluation.run_config import RunConfigError, file_identity, samples_identity
from crossfeed.evaluation.synthesis import UNISON_CFG, add_arguments, benchmark_output, from_args
from crossfeed.utils import nonempty_file, write_json, write_jsonl

THIRD_PARTY = Path(__file__).resolve().parents[2] / 'third_party'
ABBREVIATIONS = {'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr', 'st', 'vs', 'e.g', 'i.e', 'etc', 'no', 'fig'}
CLOSERS = '"\'”’»)]}」』'
SPLIT = {'cv3_hard', 'libritts'}


def split_sentences(text):
    text = text.strip()
    parts, start, i = [], 0, 0
    while i < len(text):
        if text[i] not in '.!?。！？':
            i += 1
            continue
        j = i + 1
        while j < len(text) and text[j] in '.!?。！？':
            j += 1
        if text[i] == '.':
            if j < len(text) and not (text[j].isspace() or text[j] in CLOSERS):
                i = j
                continue
            token = re.search(r'([A-Za-z.]+)$', text[:i])
            token = token.group(1) if token else ''
            if j < len(text) and (token.lower() in ABBREVIATIONS or re.fullmatch(r'[A-Z]', token)
                                  or re.fullmatch(r'(?:[A-Za-z]\.)+[A-Za-z]', token)):
                i = j
                continue
        while j < len(text) and text[j] in CLOSERS:
            j += 1
        if text[start:j].strip():
            parts.append(text[start:j].strip())
        start, i = j, j
    if text[start:].strip():
        parts.append(text[start:].strip())
    return parts


def kaldi(path):
    return dict(line.split(maxsplit=1) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip())


def load_dataset(name, args):
    if name == 'seedtts':
        meta = Path(args.seedtts_root) / 'en' / 'meta.lst'
        rows = []
        for line in meta.read_text(encoding='utf-8').splitlines():
            parts = line.strip().split('|')
            if len(parts) == 4:
                rows.append(dict(id=parts[0].strip(), text=parts[3].strip(), ref_text=parts[1].strip(),
                                 ref_audio=str((meta.parent / parts[2].strip()).resolve())))
        return rows
    if name in ('cv3', 'cv3_hard'):
        root = Path(args.cv3_root)
        folder = root / 'data' / 'zero_shot' / ('en' if name == 'cv3' else 'hard_en')
        prompts, audio = kaldi(folder / 'prompt_text'), kaldi(folder / 'prompt_wav.scp')
        return [dict(id=uid, text=text.strip(), ref_text=prompts[uid].strip(),
                     ref_audio=str((root / audio[uid].strip()).resolve())) for uid, text in kaldi(folder / 'text').items()]
    files = sorted((Path(args.libritts_root) / 'test-clean').glob('*/*/*.normalized.txt'))
    return [dict(id=f.name.removesuffix('.normalized.txt'), text=f.read_text(encoding='utf-8').strip()) for f in files]


def chunk_rows(parents, split):
    chunks = []
    for row in parents:
        pieces = split_sentences(row['text']) if split else [row['text']]
        for index, piece in enumerate(pieces):
            chunks.append(dict(id=row['id'] if len(pieces) == 1 else f'{row["id"]}__sentence{index:03d}', parent=row['id'],
                               chunk_idx=index, text=piece, ref_audio=row.get('ref_audio'), task='speech',
                               prompt=f'[Speech] A voice saying "{piece}"'))
    return chunks


def stitch(chunks, segment_dir, out_dir, gap=0.2, top_db=40.0):
    import librosa
    import soundfile as sf
    groups = defaultdict(list)
    for chunk in chunks:
        groups[chunk['parent']].append(chunk)
    out_dir.mkdir(parents=True, exist_ok=True)
    for parent, rows in groups.items():
        sources = [segment_dir / f'{r["id"]}_k0.wav' for r in sorted(rows, key=lambda r: r['chunk_idx'])]
        target = out_dir / f'{parent}_k0.wav'
        if len(sources) == 1:
            shutil.copyfile(sources[0], target)
            continue
        pieces = []
        for source in sources:
            y, sr = sf.read(source, dtype='float32')
            if float(np.max(np.abs(y))) > 1e-8:
                _, (a, b) = librosa.effects.trim(y, top_db=top_db)
                pad = round(0.25 * sr)
                y = y[max(0, a - pad):min(len(y), b + pad)]
            if pieces:
                pieces.append(np.zeros(round(gap * sr), dtype='float32'))
            pieces.append(y)
        sf.write(target, np.concatenate(pieces), sr, subtype='PCM_16')
    return groups


def publish_zero_shot(raw, audio, missing):
    pending = []
    for index, chunk in missing:
        matches = list(raw.glob(f'zstts_*_{index:03d}.wav'))
        if len(matches) > 1:
            raise RuntimeError(f'expected one zero-shot output for {chunk["id"]}: {matches}')
        if not matches or not nonempty_file(matches[0]):
            pending.append((index, chunk))
            continue
        target = audio / f'{chunk["id"]}_k0.wav'
        tmp = target.with_suffix('.tmp.wav')
        shutil.copyfile(matches[0], tmp)
        tmp.replace(target)
    return pending


def zero_shot(chunks, parents, out, args):
    audio = out / 'segments' / 'audio'
    raw = out / 'raw'
    audio.mkdir(parents=True, exist_ok=True)
    missing = [(index, c) for index, c in enumerate(chunks) if not nonempty_file(audio / f'{c["id"]}_k0.wav')]
    missing = publish_zero_shot(raw, audio, missing)
    if not missing:
        return
    from crossfeed.evaluation.references import prepare_references
    refs = prepare_references(parents, args.reference_cache)
    tasks = [dict(id=f'{args.dataset}__{c["id"]}', target_text=c['text'], ref_audio=refs[c['ref_audio']]['wav'],
                  ref_text=refs[c['ref_audio']]['text'], duration=10) for c in chunks]
    write_json(raw / 'tasks.json', tasks)
    root = THIRD_PARTY / 'unison'
    subprocess.run([sys.executable, '-m', 'unison.pipelines.infer', '--task_mode', 'zeroshotts',
                    '--zs_config', str(raw / 'tasks.json'), '--output_dir', str(raw), '--model_ckpt', args.model,
                    '--omni_model_path', args.omni, '--model_config', str(root / 'config' / 'D24S0_O_20ch.yaml'),
                    '--vae_config', str(root / 'models' / 'mmaudio' / 'vae_config_16k.yaml'), '--zs_duration', '10',
                    '--ref_duration', '0', '--ref_silence_search_s', '0', '--ref_tail_pad_s', '0',
                    '--mask_zs_after_encode', '--guidance_scale', str(UNISON_CFG if args.cfg is None else args.cfg),
                    '--num_inference_steps', str(args.steps), '--zs_resume',
                    '--batch_size', str(args.batch_size), '--seed', str(args.seed)], check=True)
    missing = publish_zero_shot(raw, audio, missing)
    if missing:
        raise RuntimeError(f'missing zero-shot outputs for {[chunk["id"] for _, chunk in missing[:5]]}')


def run(args):
    out = Path(args.out_dir).absolute()
    parents = load_dataset(args.dataset, args)[:args.limit or None]
    split = args.family == 'unison' and args.dataset in SPLIT
    if args.family == 'unison':
        chunks = chunk_rows(parents, split)
    else:
        chunks = [dict(id=r['id'], parent=r['id'], chunk_idx=0, text=r['text'], task='speech') for r in parents]
    samples = dict(utterances=samples_identity(parents), segments=samples_identity(chunks))
    settings = dict(zero_shot=args.zero_shot)
    references = None
    if args.zero_shot:
        references = samples_identity([file_identity(p) for p in sorted({r['ref_audio'] for r in parents})])
        settings['reference_cache'] = str(Path(args.reference_cache).resolve())
    with benchmark_output(out, args, task='tts', benchmark=args.dataset, samples=samples, references=references,
                          **settings):
        evaluate_utterances(args, out, parents, chunks)


def evaluate_utterances(args, out, parents, chunks):
    write_jsonl(out / 'instructions.jsonl', parents)
    write_jsonl(out / 'segments' / 'instructions.jsonl', chunks)
    if args.zero_shot:
        zero_shot(chunks, parents, out, args)
    else:
        from_args(args, chunks, out / 'segments', task='tts')
    groups = stitch(chunks, out / 'segments' / 'audio', out / 'audio')
    metrics = out / 'metrics'
    items = [dict(id=r['id'], text=r['text'], wav=out / 'audio' / f'{r["id"]}_k0.wav') for r in parents]
    summary, scored = corpus_wer(items, metrics, args.dataset, trim=args.family == 'unison', seed=args.seed,
                                 workers=args.wer_workers)
    segments = {parent: [out / 'segments' / 'audio' / f'{r["id"]}_k0.wav' for r in rows]
                for parent, rows in groups.items()}
    summary.update(utmos=utmos(segments, metrics)['utmos'])
    if args.zero_shot:
        summary.update(sim=similarity(scored, {r['id']: r['ref_audio'] for r in parents}, metrics / 'sim',
                                      args.cv3_root)['sim'])
    write_json(out / 'summary.json', summary)
    print(f'[tts] {args.dataset}: {summary}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['seedtts', 'cv3', 'cv3_hard', 'libritts'])
    add_arguments(parser)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--zero_shot', action='store_true')
    parser.add_argument('--seedtts_root', default=None)
    parser.add_argument('--cv3_root', default=None)
    parser.add_argument('--libritts_root', default=None)
    parser.add_argument('--reference_cache', default=None)
    parser.add_argument('--wer_workers', type=int, default=6)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    if args.zero_shot and (args.family != 'unison' or args.dataset == 'libritts'):
        parser.error('zero-shot TTS is evaluated for UNISON on seedtts, cv3 and cv3_hard')
    try:
        run(args)
    except RunConfigError as error:
        parser.exit(2, f'{error}\n')


if __name__ == '__main__':
    main()
