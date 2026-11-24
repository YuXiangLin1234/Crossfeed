import argparse
import csv
import difflib
import glob
import io
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

from crossfeed.evaluation.engines import generator
from crossfeed.mcq import LETTERS, extract_letter, qa_prompt
from crossfeed.utils import read_json, seed_for, write_json

COT_SUFFIX = ('Think it through step by step, then end your reply with "Final answer:" followed by the option you chose, '
              'written exactly as it appears above.')
SCORERS = dict(mmau='mmau', mmau_test='mmau', mmar='mmar', mmsu='mmau', trea='mmau')
SC_SAMPLES = 8
SC_TEMPERATURE = 0.8
_OPTION_PREFIX = re.compile(r'^\s*\([A-H]\)\s*')
_TURN_BOUNDARY = re.compile(r'\n(?:Human|User)(?=\s|:|assistant\b)', re.IGNORECASE)
_STANDALONE_LETTER = re.compile(r'\s*(?:([A-D])|[（(]\s*([A-D])\s*[）)])\s*')
_FINAL_ANSWER = re.compile(r'final\s+answer\s*(?:[:：]|is\b)\s*([^\n]*)', re.IGNORECASE)
_BARE_LETTER = re.compile(r'\(?([A-H])\)?[.:]?')
_MAGIC = ((b'RIFF', '.wav'), (b'ID3', '.mp3'), (b'\xff\xfb', '.mp3'), (b'\xff\xf3', '.mp3'), (b'\xff\xf2', '.mp3'),
          (b'fLaC', '.flac'), (b'OggS', '.ogg'))


def strip_prefix(text):
    return _OPTION_PREFIX.sub('', text or '')


def tokens(text):
    return set(re.findall(r'\b\w+\b', (text or '').lower()))


def string_match(answer, prediction, choices, strip_prefixes=False):
    if strip_prefixes:
        answer, choices = strip_prefix(answer), [strip_prefix(c) for c in choices]
    predicted, gold_tokens = tokens(prediction), tokens(answer)
    if not predicted:
        return False
    wrong = set()
    for choice in choices:
        if tokens(choice) != gold_tokens:
            wrong |= tokens(choice) - gold_tokens
    return gold_tokens <= predicted and predicted.isdisjoint(wrong)


def normalize_text(text):
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', ' ', (text or '').lower())).strip()


def match_choice_letter(text, choices):
    text = normalize_text(text)
    if not text:
        return None
    hits = sorted(((len(c), i) for i, c in enumerate(normalize_text(strip_prefix(c)) for c in choices)
                   if c and c in text), reverse=True)
    if not hits or (len(hits) > 1 and hits[0][0] == hits[1][0]):
        return None
    return LETTERS[hits[0][1]]


def canonical(prediction, choices, scorer):
    if scorer == 'mmar':
        match = _STANDALONE_LETTER.fullmatch(prediction or '')
        if match:
            index = ord(match.group(1) or match.group(2)) - ord('A')
            if index < len(choices):
                return strip_prefix(str(choices[index]))
        return prediction or ''
    letter = extract_letter(prediction)
    if letter is None:
        letter = match_choice_letter(prediction, choices)
    if letter is not None and ord(letter) - ord('A') < len(choices):
        return str(choices[ord(letter) - ord('A')])
    return (prediction or '').strip()


def correct(prediction, answer, choices, scorer):
    return string_match(answer, canonical(prediction, choices, scorer), choices, strip_prefixes=scorer == 'mmar')


def trim_turn(text):
    text = text or ''
    positions = [m.start() for m in _TURN_BOUNDARY.finditer(text)]
    if '<|im_end|>' in text:
        positions.append(text.find('<|im_end|>'))
    return text[:min(positions)].strip() if positions else text.strip()


def _norm(text):
    return re.sub(r'[^a-z0-9 ]+', ' ', (text or '').lower()).strip()


def resolve_answer(answer, choices, bare_letters=False):
    a = _norm(answer)
    for i, c in enumerate(choices):
        if _norm(c) == a and a:
            return i
    bare = re.fullmatch(r'\(?([a-h])\)?', a) if bare_letters else None
    if bare and LETTERS.index(bare.group(1).upper()) < len(choices):
        return LETTERS.index(bare.group(1).upper())
    contained = [i for i, c in enumerate(choices) if _norm(c) and (_norm(c) in a or a in _norm(c))]
    if len(contained) == 1:
        return contained[0]
    scores = [difflib.SequenceMatcher(None, a, _norm(c)).ratio() for c in choices]
    best = max(range(len(choices)), key=lambda i: scores[i])
    return best if scores[best] >= 0.5 else None


def prefixed(choices):
    return [f'({LETTERS[i]}) {str(c).strip()}' for i, c in enumerate(choices)]


def gold(choices, index, raw):
    return prefixed(choices)[index] if index is not None else raw


def materialize(audio, path, sniff=False):
    if isinstance(audio, dict) and audio.get('path') and os.path.exists(audio['path']):
        return audio['path']
    if sniff:
        path = f"{path}{next((ext for magic, ext in _MAGIC if audio['bytes'].startswith(magic)), '.wav')}"
    path = Path(path)
    if not path.exists() or not path.stat().st_size:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(audio['bytes'])
    return str(path)


def resample_16k(path):
    import librosa
    import soundfile as sf
    if sf.info(str(path)).samplerate != 16000:
        data, rate = sf.read(str(path))
        sf.write(str(path), librosa.resample(data.T, orig_sr=rate, target_sr=16000).T, 16000)
    return str(path)


def mono_16k(audio, path):
    import soundfile as sf
    path = Path(path)
    if not path.exists() or not path.stat().st_size:
        path.parent.mkdir(parents=True, exist_ok=True)
        data, rate = sf.read(io.BytesIO(audio['bytes']))
        sf.write(str(path), data.mean(axis=1) if data.ndim > 1 else data, rate)
        resample_16k(path)
    return str(path)


def copy_16k(source, path):
    import soundfile as sf
    path = Path(path)
    if not path.exists() or not path.stat().st_size:
        path.parent.mkdir(parents=True, exist_ok=True)
        data, rate = sf.read(str(source))
        sf.write(str(path), data, rate)
        resample_16k(path)
    return str(path)


def plain(value):
    return value.tolist() if hasattr(value, 'tolist') else value


def export(benchmark, out, audio_dir=None, source=None):
    rows = []
    if benchmark == 'mmau':
        from datasets import Audio, load_dataset
        dataset = load_dataset('gamma-lab-umd/MMAU-test-mini', split='test').cast_column('context', Audio(decode=False))
        for index, row in enumerate(dataset):
            other = row.get('other_attributes') or '{}'
            rows.append(dict(id=index, question=row['instruction'], choices=list(row['choices']), answer=row['answer'],
                             task=json.loads(other).get('task', 'unknown'),
                             audio=mono_16k(row['context'], Path(audio_dir) / f'audio_{index}.wav')))
    elif benchmark == 'mmar':
        from huggingface_hub import hf_hub_download
        meta = json.loads(Path(hf_hub_download('BoJack/MMAR', 'MMAR-meta.json', repo_type='dataset')).read_text())
        for index, row in enumerate(meta):
            choices = [str(c) for c in row['choices']]
            relative = row['audio_path'].lstrip('./')
            rows.append(dict(id=index, question=row['question'], choices=prefixed(choices),
                             answer=gold(choices, resolve_answer(row['answer'], choices), row['answer']),
                             task=row.get('modality', 'unknown'),
                             audio=copy_16k(Path(audio_dir) / relative, Path(audio_dir) / '16k' / relative)))
    elif benchmark == 'mmsu':
        import pyarrow.parquet as pq
        from huggingface_hub import snapshot_download
        root = source or snapshot_download('ddwang2000/MMSU', repo_type='dataset', allow_patterns=['data/*.parquet'])
        for shard in sorted(glob.glob(os.path.join(root, '**', '*.parquet'), recursive=True)):
            for row in pq.read_table(shard).to_pylist():
                choices = [str(row[c]).strip() for c in ('choice_a', 'choice_b', 'choice_c', 'choice_d')
                           if row.get(c) not in (None, '')]
                rows.append(dict(id=len(rows), question=row['question'], choices=prefixed(choices),
                                 answer=gold(choices, resolve_answer(row['answer_gt'], choices, True), row['answer_gt']),
                                 task=row.get('category') or 'unknown',
                                 audio=materialize(row['audio'], Path(audio_dir) / str(row['id']), sniff=True)))
    elif benchmark == 'trea':
        root = Path(source)
        for task in ('count', 'duration', 'order'):
            with (root / task / f'{task}.csv').open(newline='', encoding='utf-8-sig') as handle:
                for row in csv.DictReader(handle):
                    uid = row['id'].strip()
                    choices = [f'({letter}) {row[f"option{letter}"].strip()}' for letter in 'ABCD']
                    rows.append(dict(id=f'{task}:{uid}', question=row['question'].strip(), choices=choices,
                                     answer=choices['ABCD'.index(row['correct'].strip().upper().strip('(). '))],
                                     task=task, audio=str(root / task / 'audios' / f'{uid}.wav')))
    elif benchmark == 'mmau_test':
        import pandas as pd
        root = Path(source)
        for meta in pd.read_parquet(root / 'test.parquet').to_dict(orient='records'):
            meta = {k: plain(v) for k, v in meta.items()}
            audio_id = Path(str(meta['audio_id']))
            relative = Path(str(audio_id).removeprefix('./'))
            candidates = [root / relative, Path(audio_dir) / audio_id.name, Path(audio_dir) / relative]
            choices = [str(c) for c in meta['choices']]
            rows.append(dict(id=str(meta['id']), question=meta['question'], choices=choices,
                             prompt_choices=prefixed(choices), answer=None, task=meta.get('task', 'unknown'),
                             audio=str(next((p for p in candidates if p.is_file()), candidates[0])), metadata=meta))
    missing = [r['audio'] for r in rows if not os.path.exists(r['audio'])]
    if missing:
        raise FileNotFoundError(f'{len(missing)} audio files missing, e.g. {missing[:3]}')
    write_json(out, rows)
    print(f'[export] {len(rows)} {benchmark} rows -> {out}', flush=True)


def final_answer(raw, options, keep_raw, native_think):
    text = (raw or '') if keep_raw else trim_turn(raw)
    closed = '</think>' in text
    if closed:
        text = text.rsplit('</think>', 1)[1]
    if native_think and not closed:
        return ''
    lines = [line.replace('*', '').strip() for line in text.splitlines()]
    answer = None
    for i in range(len(lines) - 1, -1, -1):
        matches = _FINAL_ANSWER.findall(lines[i])
        if matches:
            answer = matches[-1].strip() or next((line for line in lines[i + 1:] if line), '')
            break
    if answer is None:
        answer = next((line for line in reversed(lines) if line), '')
    letter = _BARE_LETTER.fullmatch(answer)
    if letter and ord(letter.group(1)) - ord('A') < len(options):
        answer = options[ord(letter.group(1)) - ord('A')]
    return answer


def vote(labels):
    counts, first = Counter(), {}
    for i, label in enumerate(labels):
        if label is not None:
            counts[label] += 1
            first.setdefault(label, i)
    if not counts:
        return None
    top = max(counts.values())
    return first[min((label for label, count in counts.items() if count == top), key=first.__getitem__)]


def self_consistent(row, texts, scorer, keep_raw, native_think):
    options = row.get('prompt_choices', row['choices'])
    answers = [final_answer(text, options, keep_raw, native_think) for text in texts]
    labels = [tuple(j for j, c in enumerate(row['choices']) if correct(a, c, row['choices'], scorer)) or None
              if a.strip() else None for a in answers]
    winner = vote(labels)
    if winner is not None:
        return answers[winner], True
    if row['answer'] is None:
        return next((a for a in answers if a.strip()), next((trim_turn(t) for t in texts if (t or '').strip()), '')), False
    return answers[0], False


def evaluate(args):
    rows = read_json(args.data)[:args.limit or None]
    scorer = SCORERS[args.benchmark]
    sc = args.mode == 'sc'
    keep_raw = args.benchmark == 'mmar'
    native_think = sc and args.family == 'audex'
    max_tokens = args.max_new_tokens or (1024 if sc or keep_raw else 64)
    model = generator(args.family, args.model, args.seed, reasoning=native_think)
    n = args.samples if sc else 1
    outputs = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        items = []
        for row in batch:
            prompt = qa_prompt(row['question'], row.get('prompt_choices', row['choices']), pre_prefixed=True)
            if sc:
                prompt = prompt[:-len('Answer:')] + COT_SUFFIX
            items.append(dict(prompt=prompt, wav=row['audio'],
                              seed=seed_for(args.seed, 'understanding', args.benchmark, args.mode, str(row['id']))))
        texts = model.generate(items, max_new_tokens=max_tokens, temperature=args.temperature if sc else 0.0, n=n)
        outputs.extend(texts[i * n:(i + 1) * n] for i in range(len(batch)))
        print(f'[understanding] {min(start + args.batch_size, len(rows))}/{len(rows)}', flush=True)
    model.close()
    records, per_task, no_vote = [], defaultdict(lambda: [0, 0]), 0
    for row, texts in zip(rows, outputs):
        if sc:
            prediction, voted = self_consistent(row, texts, scorer, keep_raw, native_think)
            no_vote += int(not voted)
        else:
            prediction = texts[0] if keep_raw else trim_turn(texts[0])
        record = dict(id=row['id'], task=row['task'], question=row['question'], choices=row['choices'],
                      answer=row['answer'], model_prediction=prediction, raw=texts)
        if row['answer'] is not None:
            record['correct'] = correct(prediction, row['answer'], row['choices'], scorer)
            per_task[row['task']][0] += int(record['correct'])
            per_task[row['task']][1] += 1
        records.append(record)
    out = Path(args.out)
    write_json(out, records)
    if args.benchmark == 'mmau_test':
        submission = [dict(row['metadata'], model_prediction=canonical(record['model_prediction'], row['choices'], 'mmau'))
                      for row, record in zip(rows, records)]
        empty = [s['id'] for s in submission if not str(s['model_prediction']).strip()]
        if empty:
            raise RuntimeError(f'{len(empty)} rows have empty predictions, e.g. {empty[:10]}')
        path = out.with_name(out.stem + '_submission.json')
        write_json(path, submission)
        print(f'[understanding] submission -> {path}', flush=True)
        return
    hits, total = sum(v[0] for v in per_task.values()), sum(v[1] for v in per_task.values())
    summary = dict(benchmark=args.benchmark, mode=args.mode, n=total, correct=hits, overall_acc=round(hits / total, 4),
                   per_task={k: dict(acc=round(h / m, 4), n=m) for k, (h, m) in sorted(per_task.items())},
                   max_new_tokens=max_tokens, seed=args.seed, model=args.model)
    if sc:
        summary.update(samples=n, temperature=args.temperature, no_vote=no_vote)
    write_json(out.with_suffix('.summary.json'), summary)
    print(f'[understanding] {args.benchmark} {args.mode} accuracy = {summary["overall_acc"]:.4f} ({hits}/{total})',
          flush=True)


def submit(path):
    from gradio_client import Client, handle_file
    print(Client('sonalkum/MMAU-Eval').predict(handle_file(str(path)), api_name='/predict'))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    exp = sub.add_parser('export')
    exp.add_argument('--benchmark', required=True, choices=list(SCORERS))
    exp.add_argument('--out', required=True)
    exp.add_argument('--audio_dir', default=None)
    exp.add_argument('--source', default=None)
    run = sub.add_parser('evaluate')
    run.add_argument('--family', required=True, choices=['unison', 'audex'])
    run.add_argument('--model', required=True)
    run.add_argument('--benchmark', required=True, choices=list(SCORERS))
    run.add_argument('--data', required=True)
    run.add_argument('--out', required=True)
    run.add_argument('--mode', default='greedy', choices=['greedy', 'sc'])
    run.add_argument('--samples', type=int, default=SC_SAMPLES)
    run.add_argument('--temperature', type=float, default=SC_TEMPERATURE)
    run.add_argument('--max_new_tokens', type=int, default=None)
    run.add_argument('--batch_size', type=int, default=64)
    run.add_argument('--limit', type=int, default=None)
    run.add_argument('--seed', type=int, default=531)
    sm = sub.add_parser('submit')
    sm.add_argument('--predictions', required=True)
    args = parser.parse_args()
    if args.command == 'export':
        export(args.benchmark, args.out, args.audio_dir, args.source)
    elif args.command == 'evaluate':
        evaluate(args)
    else:
        submit(args.predictions)


if __name__ == '__main__':
    main()
