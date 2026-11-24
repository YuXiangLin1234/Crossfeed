import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from crossfeed.mcq import BALANCED_PERMUTATIONS, LETTERS, balanced_shuffle, letter_index, vote_option_index
from crossfeed.prompts import DIALOGUE_CAPABILITIES, DIALOGUE_QUESTION_PROMPT, QUESTION_PROMPT, VOTE_PROMPT
from crossfeed.unison.omni import DEFAULT_MODEL, build_omni, load_audio
from crossfeed.utils import append_jsonl, read_jsonl, rng_for, seed_for, write_json, write_jsonl


def parse_questions(text, n_expected):
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for item in items[:n_expected * 2] if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        question = str(item.get('question', '')).strip()
        options = item.get('options')
        if not question or not isinstance(options, list):
            continue
        seen, unique = set(), []
        for option in (str(o).strip() for o in options if str(o).strip()):
            key = re.sub(r'\W', '', option.lower())
            if key and key not in seen:
                seen.add(key)
                unique.append(option)
        answer = item.get('answer_index', item.get('answer'))
        if isinstance(answer, str):
            answer = letter_index(answer.strip().strip('()').upper()[:1])
        if len(unique) == 4 and isinstance(answer, int) and 0 <= answer < 4:
            out.append(dict(question=question, options=unique, claimed_idx=answer))
    return out[:n_expected]


def question_prompt(source, n, dialogue):
    reference = source.get('caption') or source.get('prompt', '')
    if not dialogue:
        return QUESTION_PROMPT.format(n=n, ref=reference)
    capability = source['capability']
    return DIALOGUE_QUESTION_PROMPT.format(n=n, ref=reference, capability=capability,
                                           capability_description=DIALOGUE_CAPABILITIES[capability])


def make_qa(clips, instructions, out_dir, model_path=DEFAULT_MODEL, backend='vllm', n_questions=8, max_keep=8,
            vote_min=4, vote_temperature=0.8, gen_temperature=0.8, gen_batch=128, chunk=128, seed=531,
            dialogue=False, require_claim=False, extra_rows=()):
    out_dir = Path(out_dir)
    sources = {row['id']: row for row in instructions}
    clips = [c for c in clips if c['inst_id'] in sources]
    scored_path = out_dir / 'qa_scored_final_answer.jsonl'
    done = {(row['inst_id'], row['k']) for row in read_jsonl(scored_path)}
    todo = [c for c in clips if (c['inst_id'], c['k']) not in done]
    print(f'[qa] {len(todo)}/{len(clips)} clips to process', flush=True)
    engine = build_omni(model_path, backend=backend) if todo else None
    for start in range(0, len(todo), chunk):
        batch = todo[start:start + chunk]
        audios = {(c['inst_id'], c['k']): load_audio(c['wav']) for c in batch}
        requests = [dict(prompt=question_prompt(sources[c['inst_id']], n_questions, dialogue),
                         audio=audios[(c['inst_id'], c['k'])],
                         seed=seed_for(seed, 'qa_generate', c['inst_id'], c['k'])) for c in batch]
        generated = engine.generate(requests, max_new_tokens=800, temperature=gen_temperature,
                                    batch_size=gen_batch, seed=seed_for(seed, 'qa_generate_batch', start))
        questions = []
        for clip, text in zip(batch, generated):
            for index, qa in enumerate(parse_questions(text, n_questions)):
                qa['qid'] = f'{clip["inst_id"]}_k{clip["k"]}_mq{index}'
                questions.append((clip, qa))
        votes_requests, meta = [], []
        for index, (clip, qa) in enumerate(questions):
            for permutation in BALANCED_PERMUTATIONS:
                options = '\n'.join(f'({LETTERS[d]}) {qa["options"][o]}' for d, o in enumerate(permutation))
                votes_requests.append(dict(prompt=VOTE_PROMPT.format(question=qa['question'], options=options),
                                           audio=audios[(clip['inst_id'], clip['k'])],
                                           seed=seed_for(seed, 'qa_vote', clip['inst_id'], clip['k'], qa['qid'],
                                                         list(permutation))))
                meta.append((index, permutation))
        answers = engine.generate(votes_requests, max_new_tokens=160, temperature=vote_temperature,
                                  batch_size=gen_batch, seed=seed_for(seed, 'qa_vote_batch', start)) if votes_requests else []
        votes = defaultdict(list)
        for text, (index, permutation) in zip(answers, meta):
            option = vote_option_index(text, permutation)
            if option is not None:
                votes[index].append(option)
        by_clip = defaultdict(list)
        for index, (clip, qa) in enumerate(questions):
            top, count = Counter(votes[index]).most_common(1)[0] if votes[index] else (None, 0)
            qa['votes'] = votes[index]
            if top is None or count < vote_min:
                qa['status'] = 'rejected_consensus'
            elif require_claim and top != qa['claimed_idx']:
                qa['status'] = 'rejected_claim'
            else:
                qa.update(status='kept', label_idx=top)
            by_clip[(clip['inst_id'], clip['k'])].append(qa)
        append_jsonl(scored_path, [dict(inst_id=c['inst_id'], k=c['k'], wav=c['wav'], scene=c['scene'],
                                        family=c.get('task', ''), qas=by_clip[(c['inst_id'], c['k'])]) for c in batch])
        print(f'[qa] {min(start + chunk, len(todo))}/{len(todo)} clips', flush=True)
    if engine is not None:
        engine.close()
    kept, status = [], Counter()
    for row in read_jsonl(scored_path):
        count = 0
        for qa in row['qas']:
            status[qa.get('status')] += 1
            if qa.get('status') == 'kept' and count < max_keep:
                count += 1
                kept.append(dict(inst_id=row['inst_id'], wav=row['wav'], scene=row['scene'], family=row['family'],
                                 qid=qa['qid'], question=qa['question'], options=qa['options'],
                                 answer_idx=qa['label_idx']))
    kept.extend(extra_rows)
    balanced_shuffle(kept, rng_for(seed, 'qa_balance'))
    write_jsonl(out_dir / 'qa_train.jsonl', kept)
    write_json(out_dir / 'qa_stats.json', dict(status=dict(status), kept=len(kept), extra=len(extra_rows)))
    print(f'[qa] kept {len(kept)} questions ({dict(status)})', flush=True)


def best_clip(scores):
    best = {}
    for row in scores:
        key = (-row['relevance_p'], -row['quality_p'], row['k'])
        if row['inst_id'] not in best or key < best[row['inst_id']][0]:
            best[row['inst_id']] = (key, row)
    return [row for _, row in best.values()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--source', default='main', choices=['main', 'perturb', 'dialogue'])
    parser.add_argument('--model_path', default=DEFAULT_MODEL)
    parser.add_argument('--backend', default='vllm', choices=['vllm', 'hf'])
    parser.add_argument('--n_questions', type=int, default=8)
    parser.add_argument('--max_keep', type=int, default=8)
    parser.add_argument('--vote_min', type=int, default=4)
    parser.add_argument('--seed', type=int, default=531)
    args = parser.parse_args()
    run = Path(args.run)
    options = dict(model_path=args.model_path, backend=args.backend, n_questions=args.n_questions,
                   max_keep=args.max_keep, vote_min=args.vote_min, seed=args.seed)
    if args.source == 'dialogue':
        base = run / 'dialogue'
        make_qa(read_jsonl(base / 'clips.jsonl'), read_jsonl(base / 'instructions.jsonl'), base / 'qa',
                dialogue=True, require_claim=True, extra_rows=read_jsonl(base / 'ground_truth_qa.jsonl'), **options)
        return
    base = run if args.source == 'main' else run / 'perturb'
    make_qa(best_clip(read_jsonl(base / 'scores.jsonl')), read_jsonl(base / 'instructions.jsonl'), base / 'qa',
            **options)


if __name__ == '__main__':
    main()
