import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

from crossfeed import timeline
from crossfeed.prompts import (AUDIO_CATEGORIES, DIALOGUE_CAPABILITIES, EVENT_EXAMPLES, EVENT_PROMPT,
                               SPEECH_CATEGORIES)
from crossfeed.unison.omni import DEFAULT_MODEL, build_omni
from crossfeed.utils import (DEFAULT_TEXT_GENERATION_ROUNDS, apportion, json_string_list, rng_for, seed_for,
                             split_even, write_json, write_jsonl)

GENDERS = ['female', 'male']
_NON_LATIN = {
    'han': re.compile(r'[㐀-䶿一-鿿豈-﫿]'),
    'kana': re.compile(r'[぀-ヿㇰ-ㇿ]'),
    'hangul': re.compile(r'[ᄀ-ᇿ㄰-㆏가-힯]'),
    'thai': re.compile(r'[฀-๿]'),
    'fullwidth': re.compile(r'[　-〿＀-￯]'),
    'other': re.compile(r'[Ͱ-ۿऀ-෿]'),
}
_TIMED_SEGMENT = re.compile(r'From\s+(\d+(?:\.\d+)?)s\s+to\s+(\d+(?:\.\d+)?)s\s*,\s*([^.]+?)\s*\.', re.I)
_NATIVE_SPAN = re.compile(r'\bfrom\s+(\d+(?:\.\d+)?)s\s+to\s+(\d+(?:\.\d+)?)s\b', re.I)
_FREE_SPAN = re.compile(r'\bfrom\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)\s+seconds?\b', re.I)
COUNT_WORDS = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten')


def latin_only(text):
    return '�' not in text and not any(pattern.search(text) for pattern in _NON_LATIN.values())


def spans_feasible(text):
    spans = [(float(a), float(b)) for a, b in _FREE_SPAN.findall(text)]
    spans += [(float(a), float(b)) for a, b in _NATIVE_SPAN.findall(text)]
    return all(a < b <= 10.0 for a, b in spans)


def clean_event(text):
    text = re.sub(r'^\s*\d+[\.\)]\s*', '', text).strip().strip('"').strip().lower()
    text = re.sub(r'\s+', ' ', text)
    return text if text and latin_only(text) else None


def category_cleaner(task, entry):
    lo = int(entry.get('min_words', 1 if task == 'audio' else 4))
    hi = int(entry.get('max_words', 35 if task == 'audio' else 22))

    def clean(text):
        if task == 'audio':
            text = clean_event(text)
            if text is None or not spans_feasible(text):
                return None
            if (entry.get('timed') or entry.get('counted')) and not timeline.split_events(text):
                return None
        else:
            text = re.sub(r'^\s*\d+[\.\)]\s*', '', text).strip().strip('"').strip().replace('"', "'")
            if not text:
                return None
        if not lo <= len(text.split()) <= hi:
            return None
        if entry.get('max_chars') and len(text) > int(entry['max_chars']):
            return None
        return text if latin_only(text) else None
    return clean


def shot_item(template, examples, batch_k, turns):
    base = template.format(k='{k}', examples=' ')
    shots = [(base.format(k=len(group)), json.dumps(group, ensure_ascii=False)) for group in split_even(examples, turns)]
    return {'prompt': base.format(k=batch_k), 'audio': None, 'shots': shots}


def propose_pool(engine, template, cleaner, target, key, defaults, rng, args, bootstrap=False):
    fixed = shot_item(template, defaults, args.batch_k, args.icl_turns)
    pool, seen, stall = [], set(), 0
    for iteration in range(args.max_iters):
        if len(pool) >= target:
            break
        if bootstrap and len(pool) >= args.n_examples:
            items = [shot_item(template, rng.sample(pool, args.n_examples), args.batch_k, args.icl_turns)
                     for _ in range(args.parallel)]
            outputs = engine.generate(items, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                                      batch_size=args.gen_batch, seed=seed_for(args.seed, 'pool', key, iteration), n=1)
        else:
            outputs = engine.generate([dict(fixed)], max_new_tokens=args.max_new_tokens,
                                      temperature=args.temperature, batch_size=args.gen_batch,
                                      seed=seed_for(args.seed, 'pool', key, iteration), n=args.parallel)
        added = 0
        for output in outputs:
            for raw in json_string_list(output) or []:
                item = cleaner(raw)
                key = re.sub(r'\W', '', item.lower()) if item else ''
                if key and key not in seen:
                    seen.add(key)
                    pool.append(item)
                    added += 1
        print(f'    iter {iteration}: +{added} ({len(pool)}/{target})', flush=True)
        stall = stall + 1 if added == 0 else 0
        if stall >= 5 and len(pool) >= args.pool_min:
            break
    return pool[:target]


def timed_caption(segments):
    return ', and '.join(f'{event} from {start:g} to {end:g} seconds' for start, end, event in segments)


def parse_timed(body):
    segments = list(_TIMED_SEGMENT.finditer(body))
    if not segments or re.sub(r'\s+', '', ''.join(m.group(0) for m in segments)) != re.sub(r'\s+', '', body):
        return None
    return [(float(m.group(1)), float(m.group(2)), m.group(3).strip()) for m in segments]


def counted_body(body, uid, args):
    parts = timeline.split_events(body)
    if not parts:
        return None, {}
    rng = rng_for(args.seed, 'frequency', uid)
    index = rng.randrange(len(parts))
    count = rng.randint(*args.timed_repeat)
    sentences = []
    for i, clause in enumerate(parts):
        clause = clause.strip().rstrip('.')
        if i == index:
            clause += ' exactly once' if count == 1 else f' exactly {COUNT_WORDS[count]} times'
        sentences.append(clause[:1].upper() + clause[1:] + '.')
    return ' '.join(sentences), dict(frequency_target=parts[index], frequency_count=count,
                                     frequency_target_index=index)


def pair(value, cast=float):
    lo, hi = (cast(x) for x in value.split(','))
    return lo, hi


def compose(args, events, audio_pools, speech_pools, rng):
    weights = [float(x) for x in args.task_weights.split(',')]
    audio_budget, speech_budget = apportion(args.n, weights)
    single_names = sorted(speech_pools)
    joint = [f'single:{name}' for name in single_names] + [f'dialogue:{name}' for name in DIALOGUE_CAPABILITIES]
    quotas = dict(zip(joint, apportion(speech_budget, [1] * len(joint))))
    single_quotas = {name: quotas[f'single:{name}'] for name in single_names}
    dialogue_quotas = {name: quotas[f'dialogue:{name}'] for name in DIALOGUE_CAPABILITIES}
    targets = {'audio': audio_budget, 'speech': sum(single_quotas.values())}
    audio_names = ['general', *sorted(audio_pools)]
    rows, seen, seen_timed, mix = [], set(), set(), Counter()
    attempts, limit = 0, max(args.n, sum(targets.values())) * 50
    while len(rows) < sum(targets.values()) and attempts < limit:
        attempts += 1
        remaining = [max(0, targets['audio'] - sum(mix[f'audio:{c}'] for c in audio_names)),
                     max(0, targets['speech'] - sum(mix[f'speech:{c}'] for c in single_names))]
        if not any(remaining):
            break
        task = rng.choices(['audio', 'speech'], weights=remaining, k=1)[0]
        uid = f'inst{len(rows):05d}'
        if task == 'speech':
            eligible = [name for name in single_names if mix[f'speech:{name}'] < single_quotas[name]]
            if not eligible:
                continue
            deficit = max(single_quotas[name] - mix[f'speech:{name}'] for name in eligible)
            category = rng.choice([name for name in eligible if single_quotas[name] - mix[f'speech:{name}'] == deficit])
            item = rng.choice(speech_pools[category])
            speech = f'A {item["gender"]} voice saying "{item["text"]}"'
            prompt = f'[Speech] {speech}'
            if prompt in seen:
                continue
            seen.add(prompt)
            row = dict(id=uid, task='speech', prompt=prompt, caption=speech, text=item['text'])
        else:
            counts = {name: mix[f'audio:{name}'] for name in audio_names}
            category = rng.choice([name for name in audio_names if counts[name] == min(counts.values())])
            if category == 'general':
                event = rng.choice(events)
                prompt = f'[Audio] {event}'
                if prompt in seen:
                    continue
                seen.add(prompt)
                row = dict(id=uid, task='audio', prompt=prompt, caption=event, text=None)
            else:
                entry = AUDIO_CATEGORIES[category]
                body, extra = rng.choice(audio_pools[category]), {}
                if entry.get('counted'):
                    body, extra = counted_body(body, uid, args)
                    if body is None:
                        continue
                if entry.get('timed'):
                    body = timeline.native_from_pool_item(
                        body, entry['timed'], rng_for(args.seed, 'timeline', uid), decimals=args.timed_decimals,
                        fill=args.timed_fill, even=args.timed_even, dur_range=args.timed_dur) or ''
                    if not body:
                        continue
                segments = parse_timed(body)
                caption = timed_caption(segments) if segments else body
                prompt = f'[Audio] {caption}'
                timed_key = (category, tuple(re.sub(r'\W+', ' ', e.casefold()).strip() for _, _, e in segments)) \
                    if segments else None
                if prompt in seen or (timed_key is not None and timed_key in seen_timed):
                    continue
                seen.add(prompt)
                if timed_key is not None:
                    seen_timed.add(timed_key)
                row = dict(id=uid, task='audio', prompt=prompt, caption=caption, text=None, **extra)
                if segments:
                    row['tts_prompt'] = f'[Audio] {body}'
        row.update(instruction_type=category, scene=uid)
        rows.append(row)
        mix[f'{task}:{category}'] += 1
    plan = dict(scene_budget=args.n, task_weights=weights, audio_budget=audio_budget, speech_budget=speech_budget,
                single_utterance_quotas=single_quotas, dialogue_quotas=dialogue_quotas,
                dialogue_total=sum(dialogue_quotas.values()), instruction_total=len(rows),
                category_mix=dict(sorted(mix.items())))
    return rows, plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--n', type=int, default=5000)
    parser.add_argument('--task_weights', default='1,1')
    parser.add_argument('--model_path', default=DEFAULT_MODEL)
    parser.add_argument('--backend', default='vllm', choices=['vllm', 'hf'])
    parser.add_argument('--batch_k', type=int, default=20)
    parser.add_argument('--parallel', type=int, default=128)
    parser.add_argument('--gen_batch', type=int, default=32)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--max_iters', type=int, default=DEFAULT_TEXT_GENERATION_ROUNDS)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--n_examples', type=int, default=3)
    parser.add_argument('--icl_turns', type=int, default=3)
    parser.add_argument('--pool_min', type=int, default=20)
    parser.add_argument('--timed_fill', type=pair, default=(0.3, 0.5))
    parser.add_argument('--timed_even', type=float, default=0.0)
    parser.add_argument('--timed_dur', type=pair, default=(0.2, 5.0))
    parser.add_argument('--timed_repeat', type=lambda v: pair(v, int), default=(1, 3))
    parser.add_argument('--timed_decimals', type=int, default=1)
    parser.add_argument('--seed', type=int, default=531)
    args = parser.parse_args()

    out = Path(args.out_dir)
    if (out / 'instructions.jsonl').is_file():
        print(f'[instructions] {out / "instructions.jsonl"} exists, skipping', flush=True)
        return
    weights = [float(x) for x in args.task_weights.split(',')]
    rng = rng_for(args.seed, 'compose')
    engine = build_omni(args.model_path, backend=args.backend)
    print('[instructions] general sound events', flush=True)
    events = propose_pool(engine, EVENT_PROMPT, clean_event, args.n, 'audio_general', EVENT_EXAMPLES, rng, args)
    audio_need = args.n * weights[0] / sum(weights) / (1 + len(AUDIO_CATEGORIES))
    audio_pools = {}
    for name, entry in AUDIO_CATEGORIES.items():
        target = max(args.pool_min, math.ceil(audio_need * 1.25))
        print(f'[instructions] audio category {name} (target {target})', flush=True)
        pool = propose_pool(engine, entry['prompt'], category_cleaner('audio', entry), target,
                            f'audio_{name}', entry['examples'], rng, args, bootstrap=True)
        if len(pool) >= args.pool_min:
            audio_pools[name] = pool
    speech_need = args.n * weights[1] / sum(weights) / (len(SPEECH_CATEGORIES) + len(DIALOGUE_CAPABILITIES))
    speech_pools = {}
    for name, entry in SPEECH_CATEGORIES.items():
        target = max(args.pool_min, math.ceil(speech_need * 1.25))
        print(f'[instructions] speech category {name} (target {target})', flush=True)
        pool = propose_pool(engine, entry['prompt'], category_cleaner('speech', entry), target,
                            f'speech_{name}', entry['examples'], rng, args, bootstrap=True)
        if len(pool) >= args.pool_min:
            speech_pools[name] = [dict(text=t, gender=rng.choice(GENDERS)) for t in pool]
    engine.close()
    rows, plan = compose(args, events, audio_pools, speech_pools, rng)
    write_jsonl(out / 'instructions.jsonl', rows)
    write_json(out / 'plan.json', plan)
    write_json(out / 'pools.json', {'audio_general': events, **{f'audio_{k}': v for k, v in audio_pools.items()},
                                    **{f'speech_{k}': [x['text'] for x in v] for k, v in speech_pools.items()}})
    print(f'[instructions] wrote {len(rows)} instructions; plan {plan}', flush=True)


if __name__ == '__main__':
    main()
