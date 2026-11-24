import argparse
import io
import re
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from crossfeed.prompts import COTTAGE_PROMPTS
from crossfeed.utils import read_jsonl, write_json

SCORE = re.compile(r'<score>\s*([0-9]+(?:\.[0-9]+)?)\s*</score>')
SYSTEM = 'You are a helpful assistant.'


def load_audio(path):
    wav = librosa.load(str(path), sr=16000)[0]
    buffer = io.BytesIO()
    sf.write(buffer, wav, 16000, format='WAV', subtype='PCM_16')
    buffer.seek(0)
    return librosa.load(buffer, sr=16000)[0]


def load_rows(eval_dir):
    instructions = {row['id']: row for row in read_jsonl(eval_dir / 'instructions.jsonl')}
    rows = []
    for wav in sorted((eval_dir / 'audio').glob('*.wav')):
        row = instructions[re.sub(r'_k\d+$', '', wav.stem)]
        rows.append(dict(id=row['id'], wav=str(wav),
                         text=row.get('caption') or re.sub(r'^\[Audio\]\s*', '', row['prompt'])))
    if len(rows) != len(instructions):
        raise RuntimeError(f'{eval_dir}: {len(rows)} wavs for {len(instructions)} instructions')
    return rows


def score_dir(model, eval_dir):
    rows = load_rows(eval_dir)
    audio = [load_audio(row['wav']) for row in rows]
    summary = dict(eval_dir=str(eval_dir), n=len(rows))
    for dim, template in COTTAGE_PROMPTS.items():
        items = [dict(prompt=template.format(instruction=row['text']) if dim == 'REL' else template, audio=a,
                      system=SYSTEM) for row, a in zip(rows, audio)]
        texts = model.generate(items, max_new_tokens=16000, temperature=0.0)
        results = [dict(row, response=text, score=float(m.group(1)) if (m := SCORE.search(text or '')) else None)
                   for row, text in zip(rows, texts)]
        write_json(eval_dir / 'metrics' / f'cottage_{dim}.json', results)
        values = [r['score'] for r in results if r['score'] is not None]
        summary[dim] = dict(mean=float(np.mean(values)), n=len(values), unparsed=len(results) - len(values))
    write_json(eval_dir / 'metrics' / 'cottage_summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--dirs', nargs='+', required=True)
    args = parser.parse_args()
    from crossfeed.unison.omni import OmniVLLM
    model = OmniVLLM(args.model, max_model_len=25232, gpu_mem=0.8)
    try:
        for path in map(Path, args.dirs):
            subsets = [path] if (path / 'instructions.jsonl').is_file() else sorted(
                p.parent for p in path.glob('*/instructions.jsonl'))
            results = [score_dir(model, subset) for subset in subsets]
            summary = {dim: float(np.mean([r[dim]['mean'] for r in results])) for dim in COTTAGE_PROMPTS}
            write_json(path / 'cottage.json', dict(summary, subsets=[str(s) for s in subsets]))
            print(f'[cottage] {path}: REL={summary["REL"]:.3f} OVL={summary["OVL"]:.3f}', flush=True)
    finally:
        model.close()


if __name__ == '__main__':
    main()
