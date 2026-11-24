import argparse
import json
import re
from pathlib import Path

from crossfeed.prompts import PERTURB_AUDIO_PROMPT, PERTURB_AUDIO_SHOTS, PERTURB_SPEECH_PROMPT, PERTURB_SPEECH_SHOTS
from crossfeed.unison.instructions import latin_only, spans_feasible
from crossfeed.unison.omni import DEFAULT_MODEL, build_omni
from crossfeed.utils import json_string_list, read_jsonl, seed_for, write_jsonl

_FORBIDDEN_AUDIO = re.compile(r'\b(music|musical|song|melody|speech|voice|speaking|talking)\b', re.I)
_DURATION = re.compile(r'\bfor\s+(\d+(?:\.\d+)?)\s*seconds?\b', re.I)


def normalize(text):
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9 ]', ' ', text.lower())).strip()


def clean_reply(text):
    text = str(text).strip()
    text = re.sub(r'^(rewritten (description|sentence|instruction)|description|sentence|instruction|variant|rewrite)'
                  r'\s*[:：]\s*', '', text, flags=re.I)
    text = re.sub(r'^\s*\d+[\.\)]\s*', '', text).strip().strip('"').strip("'").strip()
    return re.sub(r'\s+', ' ', text)


def feasible(text):
    if not spans_feasible(text):
        return False
    durations = [float(d) for d in _DURATION.findall(text)]
    return not durations or (all(0.5 <= d <= 9.5 for d in durations)
                             and sum(durations) + 0.25 * (len(durations) - 1) <= 10.0)


def validate(original, candidate, task):
    candidate = clean_reply(candidate)
    if task == 'speech':
        candidate = candidate.replace('"', "'")
    if not candidate or normalize(candidate) == normalize(original) or not latin_only(candidate):
        return None
    n_original, n_candidate = len(original.split()), len(candidate.split())
    if not max(1, n_original // 2 - 2) <= n_candidate <= max(int(n_original * 1.7), n_original + 6):
        return None
    if n_candidate > (35 if task == 'audio' else 26):
        return None
    if task == 'audio' and (_FORBIDDEN_AUDIO.search(candidate) or not feasible(candidate)):
        return None
    return candidate


def variant_row(row, body, index):
    variant = dict(row, id=f'{row["id"]}_pb{index}', perturb_of=row['id'])
    variant.pop('tts_prompt', None)
    if row['task'] == 'audio':
        variant.update(prompt=f'[Audio] {body}', caption=body)
    else:
        variant.update(prompt=row['prompt'].replace(row['text'], body),
                       caption=row['caption'].replace(row['text'], body), text=body)
    return variant


def render(template, body, k):
    return template.replace('{instruction}', body).replace('{k}', str(k))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--model_path', default=DEFAULT_MODEL)
    parser.add_argument('--backend', default='vllm', choices=['vllm', 'hf'])
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--gen_batch', type=int, default=32)
    parser.add_argument('--seed', type=int, default=531)
    args = parser.parse_args()

    run = Path(args.run)
    out = run / 'perturb' / 'instructions.jsonl'
    if out.is_file():
        print(f'[perturb] {out} exists, skipping', flush=True)
        return
    scored = {row['inst_id'] for row in read_jsonl(run / 'scores.jsonl')}
    sources = [row for row in read_jsonl(run / 'instructions.jsonl') if row['id'] in scored]
    templates = {'audio': PERTURB_AUDIO_PROMPT, 'speech': PERTURB_SPEECH_PROMPT}
    shots = {task: [(render(templates[task], source, len(variants)), json.dumps(variants, ensure_ascii=False))
                    for source, variants in demos]
             for task, demos in (('audio', PERTURB_AUDIO_SHOTS), ('speech', PERTURB_SPEECH_SHOTS))}
    bodies = [row['caption'] if row['task'] == 'audio' else row['text'] for row in sources]
    items = [dict(prompt=render(templates[row['task']], body, args.k), audio=None, shots=shots[row['task']],
                  seed=seed_for(args.seed, 'perturb', row['id'])) for row, body in zip(sources, bodies)]
    engine = build_omni(args.model_path, backend=args.backend)
    outputs = engine.generate(items, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                              batch_size=args.gen_batch, seed=seed_for(args.seed, 'perturb_batch'), n=1)
    engine.close()
    rows = []
    for row, body, output in zip(sources, bodies, outputs):
        seen, kept = {normalize(body)}, 0
        for candidate in json_string_list(output) or [output]:
            variant = validate(body, candidate, row['task'])
            if variant is None or normalize(variant) in seen:
                continue
            seen.add(normalize(variant))
            rows.append(variant_row(row, variant, kept))
            kept += 1
            if kept >= args.k:
                break
    write_jsonl(out, rows)
    print(f'[perturb] wrote {len(rows)} variants for {len(sources)} instructions -> {out}', flush=True)


if __name__ == '__main__':
    main()
