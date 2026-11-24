import argparse
from collections import Counter
from pathlib import Path

from crossfeed.prompts import GEVAL_PROMPTS
from crossfeed.unison.label import axes
from crossfeed.utils import interleave_scenes, read_jsonl, scene_prefix, write_json, write_jsonl


def write_scene_order(run, seed):
    run = Path(run)
    scenes = [dict(scene=row['scene'], task=row['task'], category=row.get('instruction_type', 'general'))
              for row in read_jsonl(run / 'instructions.jsonl')]
    scenes += [dict(scene=row['scene'], task='speech', category='dialogue/' + row['capability'])
               for row in read_jsonl(run / 'dialogue' / 'scripts.jsonl')]
    write_json(run / 'scenes.json', interleave_scenes(scenes, seed))


def score_rows(scores):
    rows = []
    for clip in scores:
        for axis in axes(clip['task']):
            relevance = axis == 'audio_relevance'
            prompt = GEVAL_PROMPTS[axis].format(instruction=clip['caption']) if relevance else GEVAL_PROMPTS[axis]
            rows.append(dict(kind='score', wav=clip['wav'], scene=clip['scene'], axis=axis, prompt=prompt,
                             target=float(clip['relevance' if relevance else 'quality'])))
    return rows


def mcq_rows(path, source):
    return [dict(kind='mcq', wav=row['wav'], scene=row['scene'], question=row['question'], options=row['options'],
                 answer_idx=row['answer_idx'], source=source) for row in read_jsonl(path)]


def build_understanding(run, und_n=0, out=None):
    run = Path(run)
    out = Path(out) if out else run / 'understanding_train.jsonl'
    selected = scene_prefix(run, und_n)
    rows = (mcq_rows(run / 'qa' / 'qa_train.jsonl', 'main') + mcq_rows(run / 'perturb' / 'qa' / 'qa_train.jsonl',
                                                                       'perturb')
            + mcq_rows(run / 'dialogue' / 'qa' / 'qa_train.jsonl', 'dialogue')
            + score_rows(read_jsonl(run / 'scores.jsonl')))
    rows = [row for row in rows if row['scene'] in selected]
    write_jsonl(out, rows)
    stats = Counter(row.get('source', row.get('axis')) for row in rows)
    print(f'[data] {len(rows)} understanding/judge rows from {len(selected)} scenes: {dict(stats)}', flush=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--und_n', type=int, default=0)
    parser.add_argument('--out', default=None)
    parser.add_argument('--scene_order', action='store_true')
    parser.add_argument('--seed', type=int, default=531)
    args = parser.parse_args()
    if args.scene_order:
        write_scene_order(args.run, args.seed)
    else:
        build_understanding(args.run, args.und_n, args.out)


if __name__ == '__main__':
    main()
