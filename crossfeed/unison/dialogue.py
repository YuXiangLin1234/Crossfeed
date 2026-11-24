import argparse
import json
import re
from collections import Counter
from pathlib import Path

from crossfeed.prompts import DIALOGUE_CAPABILITIES, DIALOGUE_PLAN_PROMPT, DIALOGUE_SCRIPT_PROMPT
from crossfeed.unison.omni import DEFAULT_MODEL, build_omni
from crossfeed.utils import DEFAULT_TEXT_GENERATION_ROUNDS, json_list, read_json, read_jsonl, rng_for, seed_for, write_jsonl

PLAN_FIELDS = ('topic', 'setting', 'participants', 'summary', 'key_details')
_STOPWORDS = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'in', 'into', 'is', 'it', 'of', 'on',
              'or', 'that', 'the', 'their', 'then', 'they', 'to', 'with', 'who', 'will'}


def clean_text(value):
    return re.sub(r'\s+', ' ', str(value).strip())


def validate_plan(raw, capability):
    if not isinstance(raw, dict) or set(raw) - set(PLAN_FIELDS):
        return None
    plan = {field: clean_text(raw.get(field, '')) for field in ('topic', 'setting', 'summary')}
    participants, details = raw.get('participants'), raw.get('key_details')
    if not all(plan.values()) or not isinstance(participants, list) or not 2 <= len(participants) <= 6:
        return None
    if not isinstance(details, list) or len(details) < 2:
        return None
    participants, details = [clean_text(x) for x in participants], [clean_text(x) for x in details]
    if not all(participants) or not all(details):
        return None
    return dict(capability=capability, **plan, participants=participants, key_details=details)


def plan_tokens(plan):
    text = ' '.join(str(plan.get(k, '')) for k in ('topic', 'setting', 'summary'))
    return {t for t in re.findall(r'[a-z0-9]+', text.lower()) if len(t) > 2 and t not in _STOPWORDS}


def plan_similarity(left, right):
    a, b = plan_tokens(left), plan_tokens(right)
    return len(a & b) / len(a | b) if a | b else 1.0


def validate_script(raw, capability):
    if not isinstance(raw, dict):
        return None
    speakers, turns = raw.get('speakers'), raw.get('turns')
    if not isinstance(speakers, list) or not 2 <= len(speakers) <= 6:
        return None
    speakers = [str(x).strip() for x in speakers]
    if len(set(speakers)) != len(speakers) or not all(speakers):
        return None
    if isinstance(turns, list):
        turns = turns[:10]
    if not isinstance(turns, list) or len(turns) < 5:
        return None
    cleaned = []
    for turn in turns:
        if not isinstance(turn, dict):
            return None
        speaker, text = str(turn.get('speaker', '')).strip(), clean_text(turn.get('text', ''))
        if speaker not in speakers or not text or any(c in text for c in '[]{}') or re.search(r'\blaugh\b', text, re.I):
            return None
        if cleaned and cleaned[-1]['speaker'] == speaker:
            return None
        cleaned.append(dict(turn_id=len(cleaned) + 1, speaker=speaker, text=text))
    if set(t['speaker'] for t in cleaned) != set(speakers):
        return None
    return dict(capability=capability, scenario=str(raw.get('scenario', '')).strip(), speakers=speakers, turns=cleaned)


def least_used(pool, excluded, usage, rng):
    candidates = [row for row in pool if row['speaker_id'] not in excluded]
    minimum = min(usage[row['speaker_id']] for row in candidates)
    return rng.choice([row for row in candidates if usage[row['speaker_id']] == minimum])


def assign_speakers(scripts, bank, seed):
    rng, usage = rng_for(seed, 'dialogue_speakers'), Counter()
    by_gender = {gender: [row for row in bank if row['gender'] == gender] for gender in ('female', 'male')}
    turns = []
    for index, script in enumerate(scripts, 1):
        selected, excluded = {}, set()
        for local_index, local in enumerate(script['speakers']):
            pick = least_used(by_gender['female' if (index + local_index) % 2 == 0 else 'male'], excluded, usage, rng)
            selected[local] = pick
            excluded.add(pick['speaker_id'])
            usage[pick['speaker_id']] += 1
        script['speaker_map'] = {local: {k: row[k] for k in ('speaker_id', 'gender', 'ref_wav', 'reference_text')}
                                 for local, row in selected.items()}
        for turn in script['turns']:
            speaker = selected[turn['speaker']]
            turns.append(dict(turn_key=f'{script["dialogue_id"]}_turn_{turn["turn_id"]:02d}',
                              dialogue_id=script['dialogue_id'], turn_id=turn['turn_id'], text=turn['text'],
                              speaker_id=speaker['speaker_id'], ref_audio=speaker['ref_wav'],
                              ref_text=speaker['reference_text']))
    return turns


def propose(engine, capability, target, args, offset, previous):
    plans = []
    prompt = DIALOGUE_PLAN_PROMPT.format(k=args.batch_k, capability=capability,
                                         capability_description=DIALOGUE_CAPABILITIES[capability])
    for iteration in range(args.max_iters):
        if len(plans) >= target:
            break
        outputs = engine.generate([dict(prompt=prompt, audio=None)], max_new_tokens=900, temperature=args.temperature,
                                  batch_size=args.gen_batch, seed=seed_for(args.seed, 'dialogue_plan', capability, iteration),
                                  n=args.parallel)
        for output in outputs:
            for raw in json_list(output):
                plan = validate_plan(raw, capability)
                if plan is not None and max((plan_similarity(plan, old) for old in previous + plans), default=0.0) \
                        <= args.plan_max_similarity:
                    plans.append(plan)
                if len(plans) >= target:
                    break
        print(f'[dialogue] plans {capability}: {len(plans)}/{target}', flush=True)
    if len(plans) < target:
        raise SystemExit(f'only {len(plans)}/{target} plans for {capability}; raise --max_iters')
    return plans[:target]


def write_scripts(engine, capability, plans, args, offset, seen):
    scripts, pending = [None] * len(plans), list(range(len(plans)))
    for iteration in range(args.max_iters):
        if not pending:
            break
        prompts = [dict(prompt=DIALOGUE_SCRIPT_PROMPT.format(
            capability=capability, capability_description=DIALOGUE_CAPABILITIES[capability],
            planning_context=json.dumps({k: plans[i][k] for k in PLAN_FIELDS}, ensure_ascii=False)), audio=None,
            seed=seed_for(args.seed, 'dialogue_script', capability, i, iteration)) for i in pending]
        outputs = engine.generate(prompts, max_new_tokens=1600, temperature=args.temperature, batch_size=args.gen_batch,
                                  seed=seed_for(args.seed, 'dialogue_script_batch', capability, iteration), n=1)
        remaining = []
        for index, output in zip(pending, outputs):
            candidates = json_list(output)
            script = validate_script(candidates[0], capability) if len(candidates) == 1 else None
            if script is not None:
                key = re.sub(r'\W', '', ' '.join(t['text'].lower() for t in script['turns']))
                if key in seen:
                    script = None
                else:
                    seen.add(key)
                    script['planning_context'] = plans[index]
                    scripts[index] = script
            if script is None:
                remaining.append(index)
        pending = remaining
        print(f'[dialogue] scripts {capability}: {len(plans) - len(pending)}/{len(plans)}', flush=True)
    if pending:
        raise SystemExit(f'only {len(plans) - len(pending)}/{len(plans)} scripts for {capability}; raise --max_iters')
    return scripts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--speaker_bank', required=True)
    parser.add_argument('--model_path', default=DEFAULT_MODEL)
    parser.add_argument('--backend', default='vllm', choices=['vllm', 'hf'])
    parser.add_argument('--batch_k', type=int, default=1)
    parser.add_argument('--parallel', type=int, default=128)
    parser.add_argument('--gen_batch', type=int, default=32)
    parser.add_argument('--max_iters', type=int, default=DEFAULT_TEXT_GENERATION_ROUNDS)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--plan_max_similarity', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=531)
    args = parser.parse_args()

    out = Path(args.run) / 'dialogue'
    if (out / 'scripts.jsonl').is_file():
        print(f'[dialogue] {out / "scripts.jsonl"} exists, skipping', flush=True)
        return
    quotas = read_json(Path(args.run) / 'plan.json')['dialogue_quotas']
    engine = build_omni(args.model_path, backend=args.backend)
    scripts, seen, previous = [], set(), []
    for offset, capability in enumerate(DIALOGUE_CAPABILITIES):
        if quotas.get(capability, 0):
            plans = propose(engine, capability, quotas[capability], args, offset, previous)
            previous.extend(plans)
            scripts.extend(write_scripts(engine, capability, plans, args, offset, seen))
    engine.close()
    rng_for(args.seed, 'dialogue_order').shuffle(scripts)
    for index, script in enumerate(scripts, 1):
        script['dialogue_id'] = script['scene'] = f'dialogue_{index:06d}'
    turns = assign_speakers(scripts, read_jsonl(args.speaker_bank), args.seed)
    write_jsonl(out / 'scripts.jsonl', scripts)
    write_jsonl(out / 'turns.jsonl', turns)
    print(f'[dialogue] {len(scripts)} dialogues / {len(turns)} turns -> {out}', flush=True)


if __name__ == '__main__':
    main()
