import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from crossfeed import timeline
from crossfeed.asr import extract_audex_transcript, word_error_rate
from crossfeed.audex.runtime import DSP_PRESETS, base_model, generation_prompt, stable_seed, strip_tags
from crossfeed.mcq import BALANCED_PERMUTATIONS, integer_distractors, ordinal, vote_option_index
from crossfeed.prompts import (AQA_QUALITY_PROMPTS, AQA_RELEVANCE_PROMPT, AQA_RELEVANCE_SYSTEM, ASR_PROMPT,
                               AUDIO_CATEGORIES, DIALOGUE_CAPABILITIES, DIALOGUE_PLAN_PROMPT,
                               DIALOGUE_QUESTION_PROMPT, DIALOGUE_SCRIPT_PROMPT, EVENT_EXAMPLES, EVENT_PROMPT,
                               GEVAL_PROMPTS, PERTURB_AUDIO_PROMPT, PERTURB_AUDIO_SHOTS, PERTURB_SPEECH_PROMPT,
                               PERTURB_SPEECH_SHOTS, QA_PROMPT, QUESTION_PROMPT, SPEECH_CATEGORIES,
                               TIMED_EVENTS_EXAMPLES, TIMED_EVENTS_PROMPT, VOTE_PROMPT)
from crossfeed.unison.dialogue import plan_similarity, validate_plan
from crossfeed.utils import (DEFAULT_TEXT_GENERATION_ROUNDS, apportion, first_json, interleave_scenes, read_json,
                             read_jsonl, rng_for, scene_prefix, split_even, write_json, write_jsonl)

_FORBIDDEN_SCRIPTS = re.compile(r'[Ͱ-ۿऀ-෿฀-๿ᄀ-ᇿ぀-ヿ'
                                r'㄰-㆏㐀-鿿豈-﫿＀-￯]')
COUNT_WORDS = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten')


def engine(path, profile='text'):
    from crossfeed.audex.engine import AudexEngine
    return AudexEngine(path, profile=profile)


def clean_pool_text(value, task='audio', entry=None):
    if not isinstance(value, str):
        return None
    value = re.sub(r'^\s*\d+[\.\)]\s*', '', strip_tags(value)).strip().strip('"').strip()
    value = re.sub(r'\s+', ' ', value).replace('"', "'")
    if task == 'audio':
        value = value.lower()
    if not value or '�' in value or _FORBIDDEN_SCRIPTS.search(value):
        return None
    entry = entry or {}
    lo, hi = int(entry.get('min_words', 1 if task == 'audio' else 4)), int(entry.get('max_words', 35 if task == 'audio' else 22))
    if not lo <= len(value.split()) <= hi:
        return None
    if (entry.get('timed') or entry.get('counted')) and not timeline.split_events(value):
        return None
    return value


def pool_item(template, examples, k, turns=3):
    base = template.format(k='{k}', examples=' ')
    shots = [(base.format(k=len(group)), json.dumps(group, ensure_ascii=False)) for group in split_even(examples, turns)]
    return base.format(k=k), shots


def propose_pool(model, template, defaults, target, key, task, args, entry=None, bootstrap=False):
    rng = rng_for(args.seed, 'pool', key)
    pool, seen, stall = [], set(), 0
    for iteration in range(args.max_iters):
        if len(pool) >= target:
            break
        outputs = []
        if bootstrap and len(pool) >= args.n_examples:
            requests = []
            for sample in range(args.parallel):
                prompt, shots = pool_item(template, rng.sample(pool, args.n_examples), args.batch_k)
                requests.append(dict(prompt=prompt, n=1, temperature=args.temperature, max_tokens=args.max_tokens,
                                     shots=shots, seed=stable_seed('pool', key, iteration, sample)))
            for offset in range(0, len(requests), args.gen_batch):
                outputs.extend(t for group in model.answers(requests[offset:offset + args.gen_batch]) for t in group)
        else:
            prompt, shots = pool_item(template, list(defaults), args.batch_k)
            for offset in range(0, args.parallel, args.gen_batch):
                outputs.extend(model.answers([dict(prompt=prompt, n=min(args.gen_batch, args.parallel - offset),
                                                   temperature=args.temperature, max_tokens=args.max_tokens,
                                                   shots=shots, seed=stable_seed('pool', key, iteration, offset))])[0])
        added = 0
        for output in outputs:
            try:
                values = first_json(output)
            except ValueError:
                continue
            for value in values if isinstance(values, list) else []:
                cleaned = clean_pool_text(value, task, entry)
                canonical = re.sub(r'\W', '', cleaned.lower()) if cleaned else ''
                if canonical and canonical not in seen:
                    seen.add(canonical)
                    pool.append(cleaned)
                    added += 1
        print(f'[instructions:{key}] iter={iteration} +{added} ({len(pool)}/{target})', flush=True)
        stall = stall + 1 if added == 0 else 0
        if stall >= 5 and len(pool) >= args.pool_min:
            break
    return pool[:target]


def temporal_caption(body, entry, rng, args):
    parts = timeline.split_events(body)
    if not parts:
        return None, {}
    if entry.get('counted'):
        counts = [rng.randint(*args.timed_repeat) for _ in parts]
        phrases = ['once' if n == 1 else 'twice' if n == 2 else f'{COUNT_WORDS[n]} times' for n in counts]
        return (', followed by '.join(f'{e} {p}' for e, p in zip(parts, phrases)) + '.',
                dict(frequency_targets=parts, frequency_counts=counts))
    if entry['timed'] == 'sequence':
        return ', followed by '.join(parts) + '.', {}
    try:
        segments = timeline.snap(timeline.sample_segments(parts, entry['timed'], rng, fill=args.timed_fill,
                                                          even=args.timed_even, dur_range=args.timed_dur),
                                 args.timed_decimals)
    except ValueError:
        return None, {}
    d = args.timed_decimals
    if entry['timed'] == 'durations':
        return ', and '.join(f'{e} for {b - a:.{d}f} seconds' for a, b, e in segments) + '.', {}
    return ', and '.join(f'{e} from {a:.{d}f} to {b:.{d}f} seconds' for a, b, e in segments) + '.', {}


def stage_instructions(args):
    run = Path(args.run)
    if (run / 'instructions.jsonl').is_file():
        return
    audio_budget, speech_budget = apportion(args.n, [1, 1])
    audio_entries = {name: dict(entry, prompt=TIMED_EVENTS_PROMPT, examples=TIMED_EVENTS_EXAMPLES)
                     if entry.get('counted') else entry for name, entry in AUDIO_CATEGORIES.items()}
    joint = [f'single:{name}' for name in sorted(SPEECH_CATEGORIES)] + [f'dialogue:{n}' for n in DIALOGUE_CAPABILITIES]
    quotas = dict(zip(joint, apportion(speech_budget, [1] * len(joint))))
    single_quotas = {name: quotas[f'single:{name}'] for name in sorted(SPEECH_CATEGORIES)}
    dialogue_quotas = {name: quotas[f'dialogue:{name}'] for name in DIALOGUE_CAPABILITIES}
    model = engine(args.model)
    events = propose_pool(model, EVENT_PROMPT, EVENT_EXAMPLES, args.n, 'audio_general', 'audio', args)
    audio_pools, speech_pools = {}, {}
    for name, entry in audio_entries.items():
        target = max(args.pool_min, math.ceil(audio_budget / (1 + len(audio_entries)) * 1.25))
        pool = propose_pool(model, entry['prompt'], entry['examples'], target, f'audio_{name}', 'audio', args,
                            entry, bootstrap=True)
        if len(pool) >= args.pool_min:
            audio_pools[name] = pool
    for name, entry in SPEECH_CATEGORIES.items():
        target = max(args.pool_min, math.ceil(single_quotas[name] * 1.25))
        pool = propose_pool(model, entry['prompt'], entry['examples'], target, f'speech_{name}', 'speech', args,
                            entry, bootstrap=True)
        if len(pool) >= args.pool_min:
            speech_pools[name] = pool
    model.close()
    audio_names = ['general', *sorted(audio_pools)]
    audio_quotas = dict(zip(audio_names, apportion(audio_budget, [1] * len(audio_names))))
    rng = rng_for(args.seed, 'compose')
    rows, seen, seen_timed = [], set(), set()

    def add(task, category, body, entry=None):
        uid = f'i{len(rows):07d}'
        caption, extra = body, {}
        if task == 'audio' and entry and (entry.get('counted') or entry.get('timed')):
            caption, extra = temporal_caption(body, entry, rng_for(args.seed, 'timeline', uid), args)
            if not caption:
                return
        row = dict(id=uid, scene=uid, task=task, caption=caption, text=body if task == 'speech' else '',
                   instruction_type=category, **extra)
        key = generation_prompt(row)
        timed_key = None
        if entry and entry.get('timed'):
            timed_key = (category, tuple(re.sub(r'\W+', ' ', e.casefold()).strip()
                                         for e in timeline.split_events(body)))
        if key in seen or (timed_key is not None and timed_key in seen_timed):
            return
        seen.add(key)
        if timed_key is not None:
            seen_timed.add(timed_key)
        rows.append(row)

    sources = {'general': events, **audio_pools}
    for task, category_quotas, pools, entries in (('audio', audio_quotas, sources, audio_entries),
                                                  ('speech', single_quotas, speech_pools, SPEECH_CATEGORIES)):
        for category, quota in category_quotas.items():
            if category not in pools:
                continue
            attempts = 0
            while sum(r['task'] == task and r['instruction_type'] == category for r in rows) < quota:
                if attempts > max(100, quota * 50):
                    raise RuntimeError(f'cannot compose distinct {task}/{category} rows')
                add(task, category, rng.choice(pools[category]), entries.get(category))
                attempts += 1
    write_jsonl(run / 'instructions.jsonl', rows)
    write_json(run / 'plan.json', dict(scene_budget=args.n, audio_quotas=audio_quotas,
                                       single_utterance_quotas=single_quotas, dialogue_quotas=dialogue_quotas,
                                       dialogue_total=sum(dialogue_quotas.values()), instruction_total=len(rows)))
    write_json(run / 'pools.json', {'audio_general': events, **{f'audio_{k}': v for k, v in audio_pools.items()},
                                    **{f'speech_{k}': v for k, v in speech_pools.items()}})
    print(f'[instructions] {len(rows)} instructions', flush=True)


def generate(run, rows, manifest, model_path, k, args, retries=2, resume_missing=False):
    run = Path(run)
    model, records, pending = None, [], []

    def flush():
        nonlocal model
        work = list(pending)
        pending.clear()
        while work:
            model = model or engine(model_path, 'generation')
            outputs = model.audio_tokens([(dict(row, generation_attempt=int(row.get('generation_attempt', 0)) + retry),
                                           key) for row, key, _, retry in work], tta_cfg=args.tta_cfg,
                                         tts_cfg=args.tts_cfg)
            again = []
            for (row, key, path, retry), tokens in zip(work, outputs):
                write_jsonl(path, [dict(instruction=row, retry=retry, **tokens)])
                if tokens['truncated'] and retry < retries:
                    again.append((row, key, path, retry + 1))
            work = again

    for row in rows:
        for j in range(int(row.get('generation_samples', k))):
            key = f'{row["id"]}_k{j}'
            path = run / 'tokens' / f'{key}.jsonl'
            retry = 0 if not path.is_file() else None
            if resume_missing and retry is None:
                saved = next(iter(read_jsonl(path)), None)
                if saved is None or saved['truncated']:
                    last_attempt = saved.get('retry', -1) if saved else -1
                    retry = last_attempt + 1 if last_attempt < retries else 0
            if retry is not None:
                pending.append((row, key, path, retry))
                if len(pending) >= args.gen_batch_size:
                    flush()
            records.append((row, key, j, path))
    flush()
    if model is not None:
        model.close()
    result = []
    for row, key, j, path in records:
        saved = read_jsonl(path)[0]
        if not saved['truncated']:
            result.append(dict(row, inst_id=row['id'], clip_id=key, k=j, wav=str(run / 'audio' / f'{key}.wav'),
                               tokens=str(path)))
    write_jsonl(run / manifest, result)
    from crossfeed.audex.codec import decode
    decode(result, args.decode_batch_size)
    print(f'[generate] {len(result)}/{len(records)} clips -> {run / manifest}', flush=True)
    return result


def axes(task):
    return ['speech_quality'] if task == 'speech' else ['audio_quality', 'audio_relevance']


def judge_items(row, axis, method):
    relevance = axis == 'audio_relevance'
    task = 'speech' if axis == 'speech_quality' else 'audio'
    if method == 'geval':
        prompt = GEVAL_PROMPTS[axis].format(instruction=row['caption']) if relevance else GEVAL_PROMPTS[axis]
        return dict(wav=row['wav'], prompt=prompt)
    return dict(wav=row['wav'], prompt=AQA_RELEVANCE_PROMPT.format(text=row['caption']) if relevance
                else AQA_QUALITY_PROMPTS[task], system=AQA_RELEVANCE_SYSTEM if relevance else None)


def judge_rows(rows, judge_path, methods=('aqa', 'geval'), batch_size=8, chunk=256):
    from crossfeed.audex.hf import AudexJudge
    judge = AudexJudge(judge_path, batch_size)
    try:
        for start in range(0, len(rows), chunk):
            batch = rows[start:start + chunk]
            requests = [(row, axis) for row in batch for axis in axes(row['task'])]
            values = {method: (judge.geval_scores if method == 'geval' else judge.aqa_scores)(
                [judge_items(row, axis, method) for row, axis in requests]) for method in methods}
            for index, (row, axis) in enumerate(requests):
                parts = {method: values[method][index] for method in methods}
                score = (min(5.0, max(1.0, ((1 + 4 * parts['aqa']) + parts['geval']) / 2)) if 'aqa' in methods
                         else min(5.0, max(1.0, parts['geval'])))
                name = 'quality' if axis.endswith('quality') else 'relevance'
                row.update({name: score, f'{name}_p': (score - 1) / 4, f'{name}_components': parts})
            print(f'[label] judged {min(start + chunk, len(rows))}/{len(rows)}', flush=True)
    finally:
        judge.close()
    return rows


def transcribe(rows, model_path, batch_size=128):
    model = engine(model_path)
    try:
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            answers = model.answers([dict(prompt=ASR_PROMPT, wav=row['wav'], n=1, temperature=0.0, max_tokens=128,
                                          seed=stable_seed('asr', row['clip_id'])) for row in batch])
            for row, answer in zip(batch, answers):
                wer = word_error_rate(row['text'], extract_audex_transcript(answer[0]))
                row.update(asr_hyp=answer[0].strip(), asr_wer=round(wer, 6), asr_p=round(max(0.0, 1 - wer), 6))
    finally:
        model.close()
    return rows


def finalize(row):
    if row['task'] == 'speech':
        row['relevance_p'] = row['asr_p']
        row['relevance'] = 1 + 4 * row['asr_p']
    row['score'] = (row['quality_p'] + row['relevance_p']) / 2
    return row


def label(run, manifest, out, judge_path, args):
    run = Path(run)
    if (run / out).is_file():
        return read_jsonl(run / out)
    rows = judge_rows(read_jsonl(run / manifest), judge_path, batch_size=args.judge_batch_size)
    speech = [row for row in rows if row['task'] == 'speech']
    if speech:
        transcribe(speech, args.model)
    rows = [finalize(row) for row in rows]
    write_jsonl(run / out, rows)
    return rows


def stage_generate(args):
    run = Path(args.run)
    generate(run, read_jsonl(run / 'instructions.jsonl'), 'gen_manifest.jsonl', args.model, args.k, args)
    label(run, 'gen_manifest.jsonl', 'scores.jsonl', args.model, args)


def stage_perturb(args):
    run = Path(args.run)
    path = run / 'perturb_instructions.jsonl'
    if not path.is_file():
        scored = {row['inst_id'] for row in read_jsonl(run / 'scores.jsonl')}
        sources = [row for row in read_jsonl(run / 'instructions.jsonl') if row['id'] in scored]
        rng_for(args.seed, 'perturb').shuffle(sources)
        templates = {'audio': PERTURB_AUDIO_PROMPT, 'speech': PERTURB_SPEECH_PROMPT}
        demos = {'audio': PERTURB_AUDIO_SHOTS, 'speech': PERTURB_SPEECH_SHOTS}
        render = lambda template, body, k: template.replace('{k}', str(k)).replace('{instruction}', body)
        specs = []
        for index, row in enumerate(sources):
            template = templates[row['task']]
            body = row['text'] if row['task'] == 'speech' else row['caption']
            shots = [(render(template, original, len(v)), json.dumps(v, ensure_ascii=False))
                     for original, v in demos[row['task']]]
            specs.append((index, row, body, dict(prompt=render(template, body, args.perturb_k), n=1,
                                                 temperature=args.temperature, max_tokens=1024, shots=shots,
                                                 seed=stable_seed('perturb', row['id']))))
        model, rows = engine(args.model), []
        for start in range(0, len(specs), args.gen_batch):
            batch = specs[start:start + args.gen_batch]
            for (index, row, body, _), outputs in zip(batch, model.answers([s[3] for s in batch])):
                try:
                    values = first_json(outputs[0])
                except ValueError:
                    continue
                seen = set()
                for variant_index, value in enumerate(values[:args.perturb_k] if isinstance(values, list) else []):
                    changed = clean_pool_text(value, row['task'])
                    key = re.sub(r'\W', '', changed.lower()) if changed else ''
                    if not changed or changed == body or key in seen:
                        continue
                    seen.add(key)
                    rows.append(dict(id=f'perturb{index:07d}_v{variant_index}', scene=row['scene'], task=row['task'],
                                     caption=changed, text=changed if row['task'] == 'speech' else '',
                                     instruction_type=row['instruction_type'], source_inst_id=row['id'],
                                     generation_samples=1))
        model.close()
        write_jsonl(path, rows)
    generate(run, read_jsonl(path), 'perturb_manifest.jsonl', args.model, 1, args)
    label(run, 'perturb_manifest.jsonl', 'perturb_scores.jsonl', args.model, args)


def validate_script(raw, capability):
    if not isinstance(raw, dict):
        return None
    speakers, turns = raw.get('speakers'), raw.get('turns')
    if not isinstance(speakers, list) or not 2 <= len(speakers) <= 6:
        return None
    speakers = [str(v).strip() for v in speakers]
    if not all(speakers) or len(set(speakers)) != len(speakers) or not isinstance(turns, list) or not 5 <= len(turns) <= 10:
        return None
    cleaned = []
    for turn in turns:
        if not isinstance(turn, dict):
            return None
        speaker, text = str(turn.get('speaker', '')).strip(), re.sub(r'\s+', ' ', str(turn.get('text', '')).strip())
        if (speaker not in speakers or not text or any(c in text for c in '[]{}') or re.search(r'\blaugh\b', text, re.I)
                or (cleaned and cleaned[-1]['speaker'] == speaker)):
            return None
        cleaned.append(dict(speaker=speaker, text=text))
    if {t['speaker'] for t in cleaned} != set(speakers):
        return None
    return dict(capability=capability, scenario=str(raw.get('scenario', '')).strip(), speakers=speakers, turns=cleaned)


def plan_dialogues(model, quotas, args):
    entries, previous, seen = [], [], set()
    for capability in DIALOGUE_CAPABILITIES:
        target, plans = quotas.get(capability, 0), []
        prompt = DIALOGUE_PLAN_PROMPT.format(k=args.dialogue_batch_k, capability=capability,
                                             capability_description=DIALOGUE_CAPABILITIES[capability])
        for iteration in range(args.dialogue_max_iters):
            if len(plans) >= target:
                break
            outputs = []
            for offset in range(0, args.parallel, args.gen_batch):
                outputs.extend(model.answers([dict(prompt=prompt, n=min(args.gen_batch, args.parallel - offset),
                                                   temperature=args.temperature, max_tokens=900,
                                                   seed=stable_seed('dialogue_plan', capability, iteration, offset))])[0])
            for output in outputs:
                try:
                    candidates = first_json(output)
                except ValueError:
                    continue
                for raw in candidates if isinstance(candidates, list) else []:
                    plan = validate_plan(raw, capability) if isinstance(raw, dict) and set(raw) == {
                        'topic', 'setting', 'participants', 'summary', 'key_details'} else None
                    if plan and max((plan_similarity(plan, old) for old in previous + plans), default=0.0) \
                            <= args.dialogue_max_similarity:
                        plans.append(plan)
                    if len(plans) >= target:
                        break
                if len(plans) >= target:
                    break
        if len(plans) < target:
            raise RuntimeError(f'dialogue plan {capability}: only {len(plans)}/{target}')
        resolved, pending = {}, list(range(len(plans)))
        for attempt in range(args.dialogue_max_iters):
            if not pending:
                break
            retry = []
            for start in range(0, len(pending), args.gen_batch):
                batch = pending[start:start + args.gen_batch]
                requests = [dict(prompt=DIALOGUE_SCRIPT_PROMPT.format(
                    capability=capability, capability_description=DIALOGUE_CAPABILITIES[capability],
                    planning_context=json.dumps({k: plans[i][k] for k in ('topic', 'setting', 'participants',
                                                                           'summary', 'key_details')},
                                                ensure_ascii=False)),
                    n=1, temperature=args.temperature, max_tokens=1600,
                    seed=stable_seed('dialogue_script', capability, i, attempt)) for i in batch]
                for index, outputs in zip(batch, model.answers(requests)):
                    try:
                        candidates = first_json(outputs[0])
                    except ValueError:
                        candidates = None
                    script = (validate_script(candidates[0], capability)
                              if isinstance(candidates, list) and len(candidates) == 1 else None)
                    key = re.sub(r'\W', '', ' '.join(t['text'].lower() for t in script['turns'])) if script else ''
                    if script and key not in seen:
                        seen.add(key)
                        resolved[index] = (plans[index], script)
                    else:
                        retry.append(index)
            pending = retry
        if pending:
            raise RuntimeError(f'dialogue script {capability}: only {len(resolved)}/{len(plans)}')
        entries.extend(resolved[i] for i in range(len(plans)))
        previous.extend(plans)
    return entries


def assign_voices(script, usage, rng):
    pool = list(DSP_PRESETS)
    mapping, available = {}, set(pool)
    for local in script['speakers']:
        minimum = min(usage[s] for s in available)
        voice = rng.choice([s for s in pool if s in available and usage[s] == minimum])
        mapping[local] = voice
        available.remove(voice)
        usage[voice] += 1
    return mapping


def dsp_voice(path, voice):
    import parselmouth
    from parselmouth.praat import call
    preset = DSP_PRESETS[voice]
    sound = parselmouth.Sound(str(path))
    voiced = sound.to_pitch(time_step=None, pitch_floor=60., pitch_ceiling=600.).selected_array['frequency']
    voiced = voiced[voiced > 0]
    if not len(voiced):
        raise RuntimeError(f'no voiced F0 in {path}')
    target = float(np.median(voiced)) * 2 ** (preset['semitones'] / 12)
    changed = call(sound, 'Change gender', 60., 600., preset['formant'], target, preset['pitch_range'], preset['duration'])
    audio = np.asarray(changed.values[0], dtype=np.float64)
    source = np.asarray(sound.values[0], dtype=np.float64)
    out_rms = math.sqrt(float(np.mean(audio ** 2)))
    gain = math.sqrt(float(np.mean(source ** 2))) / out_rms if out_rms else 1.0
    peak = float(np.max(np.abs(audio)))
    if peak:
        gain = min(gain, .999 / peak)
    return audio * gain, round(changed.sampling_frequency)


def turn_problem(row, min_duration=0.2, min_rms=1e-4):
    audio, sample_rate = sf.read(row['wav'], always_2d=False)
    audio = np.asarray(audio)
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        return 'invalid_waveform'
    if len(audio) / sample_rate < min_duration:
        return 'too_short'
    if math.sqrt(float(np.mean(audio.astype(np.float64) ** 2))) < min_rms:
        return 'low_rms'
    if row['voice'] != 's00':
        import parselmouth
        pitch = parselmouth.Sound(audio, sampling_frequency=sample_rate).to_pitch(time_step=None, pitch_floor=60.,
                                                                                  pitch_ceiling=600.)
        if not (pitch.selected_array['frequency'] > 0).sum():
            return 'no_voiced_f0'
    return None


def voiced_turns(run, turns, args, attempts=3):
    for attempt in range(attempts + 1):
        clips = generate(run, turns, 'dialogue_manifest.jsonl', args.model, 1, args)
        bad = set()
        for row in clips:
            if turn_problem(row):
                bad.add(row['inst_id'])
                continue
            if row['voice'] != 's00':
                target = Path(row['wav']).with_name(Path(row['wav']).stem + f'__{row["voice"]}.wav')
                if not target.is_file():
                    audio, sample_rate = dsp_voice(row['wav'], row['voice'])
                    sf.write(target, audio, sample_rate, subtype='PCM_16')
                row['wav'] = str(target)
        if not bad:
            return clips
        if attempt == attempts:
            raise RuntimeError(f'{len(bad)} dialogue turns stayed invalid')
        for turn in turns:
            if turn['id'] in bad:
                turn['generation_attempt'] = int(turn.get('generation_attempt', 0)) + 1
                token = Path(run) / 'tokens' / f'{turn["id"]}_k0.jsonl'
                token.unlink(missing_ok=True)
                wav = Path(run) / 'audio' / f'{turn["id"]}_k0.wav'
                wav.unlink(missing_ok=True)
                for transformed in wav.parent.glob(wav.stem + '__s*.wav'):
                    transformed.unlink()
        write_jsonl(Path(run) / 'dialogue_turns.jsonl', turns)


def numeric_question(scene, family, question, correct, distractors, formatter, rng):
    values = [correct, *distractors[:3]]
    rng.shuffle(values)
    return dict(inst_id=scene['inst_id'], clip_id=scene['inst_id'], scene=scene['scene'], wav=scene['wav'],
                family=f'dialogue_ground_truth/{family}', qid=f'{scene["inst_id"]}_gt_{family}', question=question,
                options=[formatter(v) for v in values], answer_idx=values.index(correct))


def ground_truth(scene):
    rng = rng_for(int(os.environ.get('SEED', '531')), 'ground_truth', scene['inst_id'])
    speakers, turns, duration = len(scene['speakers']), len(scene['turn_timeline']), scene['duration']
    rounded = max(1, int(round(duration)))
    rows = [numeric_question(scene, 'speaker_count', 'How many distinct speakers are heard in the dialogue?', speakers,
                             integer_distractors(speakers, 1, 12), str, rng),
            numeric_question(scene, 'turn_count', 'How many speaking turns are heard in the dialogue?', turns,
                             integer_distractors(turns, 2, 12), str, rng),
            numeric_question(scene, 'total_duration', 'Approximately how long is the complete dialogue?', rounded,
                             integer_distractors(rounded, 1, max(rounded + 8, 12)), lambda v: f'{v} seconds', rng)]
    chosen = rng.randrange(1, turns)
    tick = int(round(scene['turn_timeline'][chosen]['start_s'] * 2))
    rows.append(numeric_question(scene, 'utterance_start_time',
                                 f'Approximately when does the {ordinal(chosen + 1)} utterance begin?', tick,
                                 integer_distractors(tick, 0, max(3, int(round(duration * 2)))),
                                 lambda v: f'{v / 2:.1f} seconds', rng))
    return rows


def stage_dialogue(args):
    run = Path(args.run)
    scripts_path = run / 'dialogue_scripts.jsonl'
    if not scripts_path.is_file():
        model = engine(args.model)
        entries = plan_dialogues(model, read_json(run / 'plan.json')['dialogue_quotas'], args)
        model.close()
        rng = rng_for(args.seed, 'dialogue_speakers')
        rng.shuffle(entries)
        usage, scripts, turns = Counter(), [], []
        for index, (plan, script) in enumerate(entries):
            dialogue_id = f'dialogue{index:07d}'
            mapping = assign_voices(script, usage, rng)
            voices = [mapping[s] for s in script['speakers']]
            scripts.append(dict(dialogue_id=dialogue_id, scene=dialogue_id, index=index, plan=plan, **script,
                                voice_map=mapping, voices=voices))
            for turn_index, turn in enumerate(script['turns']):
                turns.append(dict(id=f'{dialogue_id}_t{turn_index}', scene=dialogue_id, task='speech', caption=turn['text'],
                                  text=turn['text'], dialogue_id=dialogue_id, turn=turn_index, voice=mapping[turn['speaker']],
                                  generation_samples=1))
        write_jsonl(scripts_path, scripts)
        write_jsonl(run / 'dialogue_turns.jsonl', turns)
    scripts, turns = read_jsonl(scripts_path), read_jsonl(run / 'dialogue_turns.jsonl')
    clips = {row['inst_id']: row for row in voiced_turns(run, turns, args)}
    scenes, questions = [], []
    for script in scripts:
        keys = [f'{script["dialogue_id"]}_t{i}' for i in range(len(script['turns']))]
        if not all(key in clips for key in keys):
            continue
        audio = [sf.read(clips[key]['wav']) for key in keys]
        sample_rate = audio[0][1]
        gap_rng = rng_for(args.seed, 'dialogue_gap', script['dialogue_id'])
        pieces, timeline_rows, cursor = [], [], 0
        for turn_index, ((array, _), turn) in enumerate(zip(audio, script['turns'])):
            if turn_index:
                silence = np.zeros(round(gap_rng.uniform(*args.dialogue_gap) * sample_rate), dtype=array.dtype)
                pieces.append(silence)
                cursor += len(silence)
            start = cursor / sample_rate
            pieces.append(array)
            cursor += len(array)
            timeline_rows.append(dict(turn_id=turn_index, speaker=turn['speaker'], text=turn['text'],
                                      start_s=round(start, 4), end_s=round(cursor / sample_rate, 4)))
        combined = np.concatenate(pieces)
        if len(combined) / sample_rate > args.dialogue_max_duration:
            continue
        key = f'dialogue_scene{script["index"]:07d}'
        wav = run / 'audio' / f'{key}.wav'
        sf.write(wav, combined, sample_rate)
        transcript = '\n'.join(f'Speaker {script["voices"].index(script["voice_map"][t["speaker"]]) + 1}: {t["text"]}'
                               for t in script['turns'])
        scene = dict(inst_id=key, clip_id=key, scene=script['scene'], task='speech', dialogue=True, wav=str(wav),
                     caption=transcript, capability=script['capability'], speakers=script['voices'],
                     duration=round(len(combined) / sample_rate, 4), turn_timeline=timeline_rows)
        scenes.append(scene)
        questions.extend(ground_truth(scene))
    write_jsonl(run / 'dialogue_scenes.jsonl', scenes)
    write_jsonl(run / 'dialogue_ground_truth_qa.jsonl', questions)
    print(f'[dialogue] {len(scenes)}/{len(scripts)} dialogues assembled', flush=True)


def clean_questions(value, maximum):
    result = []
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict):
            continue
        question, options, answer = item.get('question'), item.get('options'), item.get('answer_index')
        if (not isinstance(question, str) or not question.strip() or not isinstance(options, list) or len(options) != 4
                or any(not isinstance(o, str) or not o.strip() for o in options)
                or len({o.strip() for o in options}) != 4 or type(answer) is not int or not 0 <= answer < 4):
            continue
        result.append(dict(question=question.strip(), options=[o.strip() for o in options], answer_index=answer))
        if len(result) == maximum:
            break
    return result


def qa_row(row, question, options, answer_idx, **extra):
    rendered = '\n'.join(f'({letter}) {option}' for letter, option in zip('ABCD', options))
    return dict(inst_id=row.get('inst_id'), clip_id=row.get('clip_id'), scene=row['scene'], wav=row['wav'],
                prompt=QA_PROMPT.format(question=question, options=rendered),
                answer=f'({"ABCD"[answer_idx]}) {options[answer_idx]}', question=question, options=options,
                answer_idx=answer_idx, **extra)


def qa_candidates(run):
    run = Path(run)
    best = {}
    for row in read_jsonl(run / 'scores.jsonl') + read_jsonl(run / 'perturb_scores.jsonl'):
        key = (row['relevance_p'], row['quality_p'])
        if row['inst_id'] not in best or key > best[row['inst_id']][0]:
            best[row['inst_id']] = (key, row)
    return [row for _, row in best.values()] + read_jsonl(run / 'dialogue_scenes.jsonl')


def stage_qa(args):
    run = Path(args.run)
    cache = run / 'qa_attempts.jsonl'
    attempts = read_jsonl(cache)
    done = {row['clip_id'] for row in attempts}
    pending = [row for row in qa_candidates(run) if row['clip_id'] not in done]
    model = engine(args.model) if pending else None
    for start in range(0, len(pending), args.qa_batch_size):
        batch = pending[start:start + args.qa_batch_size]
        requests = []
        for row in batch:
            if row.get('dialogue'):
                prompt = DIALOGUE_QUESTION_PROMPT.format(ref=row['caption'], n=args.n_questions,
                                                         capability=row['capability'],
                                                         capability_description=DIALOGUE_CAPABILITIES[row['capability']])
            else:
                prompt = QUESTION_PROMPT.format(ref=row.get('caption') or row.get('text', ''), n=args.n_questions)
            requests.append(dict(prompt=prompt, wav=row['wav'], n=1, temperature=args.temperature, max_tokens=1024,
                                 seed=stable_seed('qa_generate', row['clip_id'])))
        generated = {}
        for row, outputs in zip(batch, model.answers(requests)):
            try:
                generated[row['clip_id']] = clean_questions(first_json(outputs[0]), args.n_questions)
            except (IndexError, ValueError):
                generated[row['clip_id']] = []
        votes_requests, specs = [], []
        for row in batch:
            for q_index, item in enumerate(generated[row['clip_id']]):
                for v_index, permutation in enumerate(BALANCED_PERMUTATIONS):
                    options = '\n'.join(f'({letter}) {item["options"][i]}' for letter, i in zip('ABCD', permutation))
                    votes_requests.append(dict(prompt=VOTE_PROMPT.format(question=item['question'], options=options),
                                               wav=row['wav'], n=1, temperature=args.temperature, max_tokens=160,
                                               seed=stable_seed('qa_vote', row['clip_id'], q_index, v_index)))
                    specs.append((row['clip_id'], q_index, permutation))
        votes = defaultdict(list)
        for (clip_id, q_index, permutation), outputs in zip(specs, model.answers(votes_requests) if votes_requests else []):
            option = vote_option_index(outputs[0] if outputs else '', permutation)
            if option is not None:
                votes[(clip_id, q_index)].append(option)
        for row in batch:
            kept = []
            for q_index, item in enumerate(generated[row['clip_id']]):
                counts = Counter(votes[(row['clip_id'], q_index)])
                if not counts:
                    continue
                consensus, count = counts.most_common(1)[0]
                if count >= args.vote_min and (not row.get('dialogue') or consensus == item['answer_index']):
                    kept.append(qa_row(row, item['question'], item['options'], consensus,
                                       qid=f'{row["clip_id"]}_q{q_index}', votes=count,
                                       source='dialogue' if row.get('dialogue') else 'audio'))
                if len(kept) >= args.max_keep:
                    break
            attempts.append(dict(clip_id=row['clip_id'], proposed=len(generated[row['clip_id']]), qas=kept))
        write_jsonl(cache, attempts)
        print(f'[qa] {min(start + args.qa_batch_size, len(pending))}/{len(pending)} clips', flush=True)
    if model is not None:
        model.close()
    rows = [qa for record in read_jsonl(cache) for qa in record['qas']]
    rows += [qa_row(row, row['question'], row['options'], row['answer_idx'], qid=row['qid'], source='ground_truth')
             for row in read_jsonl(run / 'dialogue_ground_truth_qa.jsonl')]
    write_jsonl(run / 'qa.jsonl', rows)
    print(f'[qa] {len(rows)} questions', flush=True)


def scene_order(run, seed):
    run = Path(run)
    scenes = [dict(scene=row['scene'], task=row['task'], category=row['instruction_type'])
              for row in read_jsonl(run / 'instructions.jsonl')]
    scenes += [dict(scene=row['scene'], task='speech', category='dialogue/' + row['capability'])
               for row in read_jsonl(run / 'dialogue_scripts.jsonl')]
    write_json(run / 'scenes.json', interleave_scenes(scenes, seed))


def stage_data(args):
    run = Path(args.run)
    if not (run / 'scenes.json').is_file():
        scene_order(run, args.seed)
    selected = scene_prefix(run, args.und_n)
    out = Path(args.out) if args.out else run / 'understanding_train.jsonl'
    rows = [row for row in read_jsonl(run / 'qa.jsonl') if row['scene'] in selected]
    for clip in read_jsonl(run / 'scores.jsonl'):
        if clip['scene'] not in selected:
            continue
        for axis in axes(clip['task']):
            relevance = axis == 'audio_relevance'
            rows.append(dict(scene=clip['scene'], wav=clip['wav'], axis=axis,
                             prompt=GEVAL_PROMPTS[axis].format(instruction=clip['caption']) if relevance
                             else GEVAL_PROMPTS[axis], score_regression_target=float(
                                 clip['relevance' if relevance else 'quality'])))
    write_jsonl(out, rows)
    print(f'[data] {len(rows)} understanding/judge rows from {len(selected)} scenes', flush=True)


def qualifies(winner, loser, margin):
    quality = (winner['quality'] - loser['quality']) / 4
    relevance = (winner['relevance'] - loser['relevance']) / 4
    return min(quality, relevance) >= 0 and max(quality, relevance) >= margin


def proxy(run, target_tokens, condition_tokens, name):
    target, condition = read_jsonl(target_tokens)[0], read_jsonl(condition_tokens)[0]
    path = Path(run) / 'tokens' / 'dpo_proxy' / f'{name}.jsonl'
    if not path.is_file():
        write_jsonl(path, [dict(target, prompt=condition['prompt'], retargeted_from=str(target_tokens))])
    return str(path)


def build_pairs(run, scores, perturb_scores, margin, reverse=True):
    rng = rng_for(int(os.environ.get('SEED', '531')), 'dpo_pair_selection')
    rows = scores + perturb_scores
    by_inst = defaultdict(list)
    for row in rows:
        by_inst[row['inst_id']].append(row)
    pairs = []
    for clips in by_inst.values():
        candidates = [(w, l) for w in clips for l in clips if w['k'] != l['k'] and qualifies(w, l, margin)]
        for winner, loser in rng.sample(candidates, len(candidates)):
            pairs.append(dict(scene=winner['scene'], chosen=winner['tokens'], rejected=loser['tokens'],
                              pair_source='score'))
    best = {}
    for row in scores:
        if row['inst_id'] not in best or row['score'] > best[row['inst_id']]['score']:
            best[row['inst_id']] = row
    for variant in perturb_scores:
        source = best.get(variant['source_inst_id'])
        if source is None:
            continue
        pairs.append(dict(scene=source['scene'], chosen=source['tokens'], pair_source='perturb',
                          rejected=proxy(run, variant['tokens'], source['tokens'],
                                         f'{source["clip_id"]}__reject__{variant["clip_id"]}')))
        if reverse:
            pairs.append(dict(scene=source['scene'], chosen=variant['tokens'], pair_source='perturb_rev',
                              rejected=proxy(run, source['tokens'], variant['tokens'],
                                             f'{variant["clip_id"]}__reject__{source["clip_id"]}')))
    return pairs


def stage_preference(args):
    run = Path(args.run)
    out = Path(args.out_dir) if args.out_dir else run / 'generation_judge'
    for name in ('scores.jsonl', 'perturb_scores.jsonl'):
        if not (out / name).is_file():
            rows = judge_rows(read_jsonl(run / name), args.judge, methods=('geval',), batch_size=args.judge_batch_size)
            write_jsonl(out / name, [finalize(row) for row in rows])
    if not (run / 'scenes.json').is_file():
        scene_order(run, args.seed)
    selected = scene_prefix(run, args.gen_n)
    pairs = [row for row in build_pairs(run, read_jsonl(out / 'scores.jsonl'), read_jsonl(out / 'perturb_scores.jsonl'),
                                        args.margin) if row['scene'] in selected]
    if not pairs:
        raise SystemExit('no preference pairs passed the margin')
    write_jsonl(out / 'dpo_pairs.jsonl', pairs)
    print(f'[preference] {len(pairs)} DPO pairs', flush=True)


STAGES = dict(instructions=stage_instructions, generate=stage_generate, perturb=stage_perturb, dialogue=stage_dialogue,
              qa=stage_qa, data=stage_data, preference=stage_preference)


def pair(value, cast=float):
    return tuple(cast(x) for x in value.split(','))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=list(STAGES))
    parser.add_argument('--run', required=True)
    parser.add_argument('--model', default=None)
    parser.add_argument('--judge', default=None)
    parser.add_argument('--n', type=int, default=5000)
    parser.add_argument('--k', type=int, default=8)
    parser.add_argument('--und_n', type=int, default=0)
    parser.add_argument('--gen_n', type=int, default=0)
    parser.add_argument('--batch_k', type=int, default=20)
    parser.add_argument('--parallel', type=int, default=128)
    parser.add_argument('--gen_batch', type=int, default=128)
    parser.add_argument('--max_iters', type=int, default=DEFAULT_TEXT_GENERATION_ROUNDS)
    parser.add_argument('--max_tokens', type=int, default=1024)
    parser.add_argument('--n_examples', type=int, default=3)
    parser.add_argument('--pool_min', type=int, default=20)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--timed_fill', type=pair, default=(0.3, 0.5))
    parser.add_argument('--timed_even', type=float, default=0.0)
    parser.add_argument('--timed_dur', type=pair, default=(0.2, 5.0))
    parser.add_argument('--timed_repeat', type=lambda v: pair(v, int), default=(1, 3))
    parser.add_argument('--timed_decimals', type=int, default=1)
    parser.add_argument('--tta_cfg', type=float, default=3.0)
    parser.add_argument('--tts_cfg', type=float, default=1.0)
    parser.add_argument('--gen_batch_size', type=int, default=128)
    parser.add_argument('--decode_batch_size', type=int, default=64)
    parser.add_argument('--judge_batch_size', type=int, default=8)
    parser.add_argument('--perturb_k', type=int, default=3)
    parser.add_argument('--dialogue_batch_k', type=int, default=1)
    parser.add_argument('--dialogue_max_iters', type=int, default=DEFAULT_TEXT_GENERATION_ROUNDS)
    parser.add_argument('--dialogue_max_similarity', type=float, default=0.5)
    parser.add_argument('--dialogue_gap', type=pair, default=(0.10, 0.50))
    parser.add_argument('--dialogue_max_duration', type=float, default=60.0)
    parser.add_argument('--qa_batch_size', type=int, default=128)
    parser.add_argument('--n_questions', type=int, default=8)
    parser.add_argument('--max_keep', type=int, default=8)
    parser.add_argument('--vote_min', type=int, default=4)
    parser.add_argument('--margin', type=float, default=0.0625)
    parser.add_argument('--out', default=None)
    parser.add_argument('--out_dir', default=None)
    parser.add_argument('--seed', type=int, default=531)
    args = parser.parse_args()
    args.model = args.model or str(base_model())
    os.environ['SEED'] = str(args.seed)
    Path(args.run).mkdir(parents=True, exist_ok=True)
    STAGES[args.stage](args)


if __name__ == '__main__':
    main()
