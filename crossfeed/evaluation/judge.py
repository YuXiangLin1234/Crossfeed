import argparse
import csv
import math
import re
import statistics
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from crossfeed.evaluation.engines import generator, scorer
from crossfeed.prompts import (AQA_QUALITY_PROMPTS, AQA_RELEVANCE_PROMPT, AQA_RELEVANCE_SYSTEM, DIRECT_PROMPTS,
                               GEVAL_PROMPTS, SYSTEM_PROMPT)
from crossfeed.utils import seed_for, write_json

GROUPS = {
    'Audio-REL': ['pam/audio_relevance', 'clotho/audio_relevance', 'xacle/audio_relevance', 'relate/audio_relevance'],
    'Audio-OVL': ['pam/audio_quality', 'clotho/audio_quality'],
    'Speech-OVL': ['bvcc/speech_quality', 'qualispeech/overall'],
}
METHODS = ['geval', 'aqascore', 'aqascore_geval', 'prompting_greedy', 'prompting_avg8']
_DECIMAL = re.compile(r'\s*([1-4]\.[0-9]{2}|5\.00)\s*')


def finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def item(name, index, wav, gt, caption=None, source_id=None):
    axis = 'audio_relevance' if name.endswith('relevance') else 'audio_quality' if name.endswith('audio_quality') \
        else 'speech_quality'
    return dict(key=f'{name}:{index}', id=source_id or Path(wav).name, axis=axis, wav=str(wav), gt=finite(gt),
                caption=caption)


def mos_csv(prefix, csv_path, audio_root, encoding):
    relevance, quality = [], []
    with open(csv_path, encoding=encoding) as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            filename = row['File Name'] if row['File Name'].endswith('.wav') else row['File Name'] + '.wav'
            wav = Path(audio_root) / row['Model'] / filename
            if not wav.is_file():
                continue
            source = f"{row['Model']}/{filename}"
            relevance.append(item(f'{prefix}/audio_relevance', index, wav, row.get('REL'), row.get('Text', ''), source))
            quality.append(item(f'{prefix}/audio_quality', index, wav, row.get('OVL'), None, source))
    return {f'{prefix}/audio_relevance': relevance, f'{prefix}/audio_quality': quality}


def relevance_rows(name, scores, audio_root):
    rows = []
    for index, filename in enumerate(sorted(scores)):
        wav = Path(audio_root) / filename
        if wav.is_file():
            text, values = scores[filename]
            rows.append(item(name, index, wav, 1 + 0.4 * statistics.mean(values), text))
    return {name: rows}


def xacle(csv_path, audio_root):
    with open(csv_path, encoding='utf-8-sig', newline='') as handle:
        scores = {row['wav_file_name']: (row['text'], [float(row['average_score'])]) for row in csv.DictReader(handle)}
    return relevance_rows('xacle/audio_relevance', scores, audio_root)


def relate(csv_path, audio_root):
    scores = {}
    with open(csv_path, encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            if row['in RELATE dataset'] == 'test':
                scores.setdefault(row['wavname'].lstrip('/'), (row['text'], []))[1].append(float(row['score']))
    return relevance_rows('relate/audio_relevance', scores, audio_root)


def qualispeech(root):
    import pandas as pd
    rows = []
    for index, row in enumerate(pd.read_csv(Path(root) / 'test.csv').to_dict('records')):
        filename = str(row.get('id') or '').strip()
        wav = Path(root) / 'wav' / 'test' / filename
        if filename and wav.is_file():
            rows.append(item('qualispeech/overall', index, wav, row.get('Overall quality')))
    return {'qualispeech/overall': rows}


def bvcc(root):
    rows = []
    with (Path(root) / 'sets' / 'test_mos_list.txt').open(encoding='utf-8-sig', newline='') as handle:
        for index, row in enumerate(csv.reader(handle)):
            wav = Path(root) / 'wav' / row[0].strip()
            if wav.is_file():
                rows.append(item('bvcc/speech_quality', index, wav, row[1].strip()))
    return {'bvcc/speech_quality': rows}


def load_benchmarks(args):
    benchmarks = {}
    benchmarks.update(mos_csv('pam', Path(args.pam_root) / 'scores.csv', args.pam_root, 'utf-8'))
    benchmarks.update(mos_csv('clotho', args.clotho_csv, args.clotho_audio, 'utf-8-sig'))
    benchmarks.update(xacle(args.xacle_csv, args.xacle_audio))
    benchmarks.update(relate(args.relate_csv, args.relate_audio))
    benchmarks.update(bvcc(args.bvcc_root))
    benchmarks.update(qualispeech(args.qualispeech_root))
    for name, rows in benchmarks.items():
        print(f'[judge] {name}: {len(rows)} items', flush=True)
    return {name: rows[:args.limit or None] for name, rows in benchmarks.items()}


def geval_prompt(row, prompts=GEVAL_PROMPTS):
    template = prompts[row['axis']]
    return template.format(instruction=row['caption'] or '') if row['axis'] == 'audio_relevance' else template


def aqa_request(row, family):
    if row['axis'] == 'audio_relevance':
        return dict(wav=row['wav'], prompt=AQA_RELEVANCE_PROMPT.format(text=row['caption']), system=AQA_RELEVANCE_SYSTEM)
    prompt = AQA_QUALITY_PROMPTS['speech' if row['axis'] == 'speech_quality' else 'audio']
    return dict(wav=row['wav'], prompt=prompt, system=SYSTEM_PROMPT if family == 'unison' else None)


def score_logits(args, rows, methods):
    from crossfeed.unison.omni import load_audio
    model = scorer(args.family, args.model)
    batch = {'geval': 16 if args.family == 'unison' else 1, 'aqascore': 1}
    try:
        for method in methods:
            for start in range(0, len(rows), args.chunk):
                chunk = rows[start:start + args.chunk]
                if method == 'geval':
                    requests = [dict(wav=r['wav'], prompt=geval_prompt(r)) for r in chunk]
                else:
                    requests = [aqa_request(r, args.family) for r in chunk]
                if args.family == 'unison':
                    requests = [dict(q, audio=load_audio(q['wav'])) for q in requests]
                    function = model.geval_scores if method == 'geval' else model.aqa_scores
                    values = function(requests, batch_size=batch[method])
                else:
                    values = (model.geval_scores if method == 'geval' else model.aqa_scores)(requests)
                for row, value in zip(chunk, values):
                    row[method] = float(value)
                print(f'[judge] {method} {min(start + args.chunk, len(rows))}/{len(rows)}', flush=True)
    finally:
        model.close()


def prompting(args, rows):
    model = generator(args.family, args.model, args.seed)
    try:
        for mode, n, temperature in (('greedy', 1, 0.0), ('avg8', 8, 0.8)):
            for start in range(0, len(rows), args.chunk):
                chunk = rows[start:start + args.chunk]
                requests = [dict(wav=r['wav'], prompt=geval_prompt(r, DIRECT_PROMPTS),
                                 seed=seed_for(args.seed, 'judge_prompting', mode, r['key'])) for r in chunk]
                texts = model.generate(requests, max_new_tokens=32, temperature=temperature, n=n, top_p=0.95)
                for i, row in enumerate(chunk):
                    replies = texts[i * n:(i + 1) * n]
                    parsed = [float(m.group(1)) if (m := _DECIMAL.fullmatch(t or '')) else None for t in replies]
                    valid = [p for p in parsed if p is not None]
                    row[f'prompting_{mode}_replies'] = replies
                    row[f'prompting_{mode}'] = statistics.mean(valid) if valid else None
                print(f'[judge] prompting {mode} {min(start + args.chunk, len(rows))}/{len(rows)}', flush=True)
    finally:
        model.close()


def prediction(row, method):
    if method == 'aqascore':
        return 1 + 4 * row['aqascore']
    if method == 'aqascore_geval':
        return min(5.0, max(1.0, ((1 + 4 * row['aqascore']) + row['geval']) / 2))
    return row.get(method)


def correlation(rows, method):
    pairs = [(r['gt'], prediction(r, method)) for r in rows]
    pairs = np.asarray([(g, p) for g, p in pairs if g is not None and p is not None], dtype=float)
    gt, pred = pairs.T
    return dict(n=len(pairs), pearson=float(pearsonr(gt, pred)[0]), spearman=float(spearmanr(gt, pred)[0]),
                mse=float(np.mean((gt - pred) ** 2)))


def summarize(benchmarks, methods):
    summary = {}
    for method in methods:
        per = {name: correlation(rows, method) for name, rows in benchmarks.items()}
        groups = {group: {metric: float(np.mean([per[name][metric] for name in names]))
                          for metric in ('pearson', 'spearman', 'mse')}
                  for group, names in GROUPS.items()}
        summary[method] = dict(benchmarks=per, groups=groups)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', required=True, choices=['unison', 'audex'])
    parser.add_argument('--model', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--methods', default='geval')
    parser.add_argument('--pam_root', required=True)
    parser.add_argument('--clotho_csv', required=True)
    parser.add_argument('--clotho_audio', required=True)
    parser.add_argument('--xacle_csv', required=True)
    parser.add_argument('--xacle_audio', required=True)
    parser.add_argument('--relate_csv', required=True)
    parser.add_argument('--relate_audio', required=True)
    parser.add_argument('--bvcc_root', required=True)
    parser.add_argument('--qualispeech_root', required=True)
    parser.add_argument('--chunk', type=int, default=256)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--seed', type=int, default=531)
    args = parser.parse_args()
    methods = [m for m in args.methods.split(',') if m]
    if 'all' in methods:
        methods = METHODS
    unknown = set(methods) - set(METHODS)
    if unknown:
        parser.error(f'unknown methods: {sorted(unknown)}')
    benchmarks = load_benchmarks(args)
    rows = [row for name in benchmarks for row in benchmarks[name]]
    logit_methods = [m for m in ('geval', 'aqascore') if m in methods or 'aqascore_geval' in methods]
    if logit_methods:
        score_logits(args, rows, logit_methods)
    if any(m.startswith('prompting') for m in methods):
        prompting(args, rows)
    out = Path(args.out_dir)
    for name, items in benchmarks.items():
        write_json(out / f'{name}.json', items)
    summary = summarize(benchmarks, methods)
    write_json(out / 'summary.json', summary)
    for method, result in summary.items():
        print(f'[judge] {method}: ' + '  '.join(
            f'{group} r={v["pearson"]:.3f} rho={v["spearman"]:.3f} mse={v["mse"]:.3f}'
            for group, v in result['groups'].items()), flush=True)


if __name__ == '__main__':
    main()
