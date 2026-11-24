import argparse
from collections import defaultdict
from pathlib import Path

from crossfeed.unison.label import finalize, release, score_clips
from crossfeed.unison.omni import OmniHF
from crossfeed.utils import read_jsonl, rng_for, scene_prefix, write_jsonl


def rescore(rows, judge_path, batch_size=16, chunk=1024):
    rows = [dict(row) for row in rows]
    model = OmniHF(judge_path)
    try:
        for start in range(0, len(rows), chunk):
            score_clips(model, rows[start:start + chunk], methods=('geval',), batch_size=batch_size)
            print(f'[preference] rescored {min(start + chunk, len(rows))}/{len(rows)}', flush=True)
    finally:
        release(model)
    return [finalize(row) for row in rows]


def qualifies(winner, loser, margin):
    quality = winner['quality_p'] - loser['quality_p']
    relevance = winner['relevance_p'] - loser['relevance_p']
    return min(quality, relevance) >= 0 and max(quality, relevance) >= margin


def pair(winner, loser, caption, source):
    return dict(inst_id=winner['inst_id'], scene=winner['scene'], caption=caption, task=winner['task'],
                win_audio_path=winner['wav'], lose_audio_path=loser['wav'],
                win_latent_path=winner['latent'], lose_latent_path=loser['latent'], duration=10.0,
                win_quality_p=winner['quality_p'], win_relevance_p=winner['relevance_p'],
                lose_quality_p=loser['quality_p'], lose_relevance_p=loser['relevance_p'],
                margin=round(winner['quality_p'] + winner['relevance_p'] - loser['quality_p'] - loser['relevance_p'], 6),
                pair_source=source)


def build_pairs(scores, perturb_scores, perturb_instructions, margin=0.0625, seed=531, reverse=True,
                max_pairs=100000):
    rng = rng_for(seed, 'dpo_pair_selection')
    by_inst = defaultdict(list)
    for row in scores:
        by_inst[row['inst_id']].append(row)
    pairs = []
    for inst_id, clips in sorted(by_inst.items()):
        candidates = [(w, l) for w in clips for l in clips if l['k'] != w['k'] and qualifies(w, l, margin)]
        for winner, loser in rng.sample(candidates, min(max_pairs, len(candidates))):
            pairs.append(pair(winner, loser, winner['prompt'], 'score'))
    parent = {row['id']: row['perturb_of'] for row in perturb_instructions}
    variants = defaultdict(list)
    for row in perturb_scores:
        if row['inst_id'] in parent:
            variants[parent[row['inst_id']]].append(row)
    for origin, rows in sorted(variants.items()):
        if not by_inst.get(origin):
            continue
        best = max(by_inst[origin], key=lambda r: (r['quality_p'] + r['relevance_p'], -r['k']))
        combos = []
        for variant in sorted(rows, key=lambda r: (r['inst_id'], r['k'])):
            combos.append((best['prompt'], best, variant, 'perturb'))
            if reverse:
                combos.append((variant['prompt'], variant, best, 'perturb_rev'))
        if len(combos) > max_pairs:
            combos = rng.sample(combos, max_pairs)
        for caption, winner, loser, source in combos:
            pairs.append(pair(winner, loser, caption, source))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--judge', required=True)
    parser.add_argument('--gen_n', type=int, default=0)
    parser.add_argument('--out_dir', default=None)
    parser.add_argument('--margin', type=float, default=0.0625)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=531)
    args = parser.parse_args()
    run = Path(args.run)
    out = Path(args.out_dir) if args.out_dir else run / 'generation'
    for name, source in (('scores.jsonl', run / 'scores.jsonl'), ('perturb_scores.jsonl', run / 'perturb' / 'scores.jsonl')):
        if not (out / name).is_file():
            write_jsonl(out / name, rescore(read_jsonl(source), args.judge, args.batch_size))
    selected = scene_prefix(run, args.gen_n)
    pairs = build_pairs(read_jsonl(out / 'scores.jsonl'), read_jsonl(out / 'perturb_scores.jsonl'),
                        read_jsonl(run / 'perturb' / 'instructions.jsonl'), args.margin, args.seed)
    pairs = [row for row in pairs if row['scene'] in selected]
    if not pairs:
        raise SystemExit('no preference pairs passed the margin')
    write_jsonl(out / 'dpo_pairs.jsonl', pairs)
    print(f'[preference] {len(pairs)} DPO pairs -> {out / "dpo_pairs.jsonl"}', flush=True)


if __name__ == '__main__':
    main()
