import argparse
import gc
import glob
from pathlib import Path

import torch

from crossfeed.asr import extract_transcript, word_error_rate
from crossfeed.prompts import (AQA_QUALITY_PROMPTS, AQA_RELEVANCE_PROMPT, AQA_RELEVANCE_SYSTEM, ASR_PROMPT,
                               GEVAL_PROMPTS, SYSTEM_PROMPT)
from crossfeed.unison.omni import DEFAULT_MODEL, OmniHF, build_omni, load_audio
from crossfeed.utils import append_jsonl, read_jsonl, seed_for, write_jsonl


def axes(task):
    return ['speech_quality'] if task == 'speech' else ['audio_quality', 'audio_relevance']


def score_inputs(axis, caption):
    relevance = axis == 'audio_relevance'
    task = 'speech' if axis == 'speech_quality' else 'audio'
    return dict(geval=GEVAL_PROMPTS[axis].format(instruction=caption) if relevance else GEVAL_PROMPTS[axis],
                aqa=AQA_RELEVANCE_PROMPT.format(text=caption) if relevance else AQA_QUALITY_PROMPTS[task],
                aqa_system=AQA_RELEVANCE_SYSTEM if relevance else SYSTEM_PROMPT)


def fuse(aqa, geval):
    return min(5.0, max(1.0, ((1 + 4 * aqa) + geval) / 2))


def release(model):
    model.close()
    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def score_clips(model, clips, methods=('aqa', 'geval'), batch_size=8):
    requests = [dict(wav=c['wav'], **score_inputs(axis, c['caption'])) for c in clips for axis in axes(c['task'])]
    values = {method: [] for method in methods}
    for start in range(0, len(requests), batch_size):
        chunk = requests[start:start + batch_size]
        audios = [load_audio(r['wav']) for r in chunk]
        if 'aqa' in methods:
            values['aqa'].extend(model.aqa_scores(
                [dict(prompt=r['aqa'], audio=a, system=r['aqa_system']) for r, a in zip(chunk, audios)],
                batch_size=batch_size))
        if 'geval' in methods:
            values['geval'].extend(model.geval_scores(
                [dict(prompt=r['geval'], audio=a) for r, a in zip(chunk, audios)], batch_size=batch_size))
    position = 0
    for clip in clips:
        for axis in axes(clip['task']):
            parts = {method: values[method][position] for method in methods}
            score = fuse(parts['aqa'], parts['geval']) if 'aqa' in methods else min(5.0, max(1.0, parts['geval']))
            name = 'quality' if axis.endswith('quality') else 'relevance'
            clip.update({name: score, f'{name}_p': (score - 1) / 4, f'{name}_components': parts})
            position += 1
    return clips


def transcribe(clips, model_path, backend, seed, batch_size=64):
    model = build_omni(model_path, backend=backend)
    try:
        for start in range(0, len(clips), batch_size):
            chunk = clips[start:start + batch_size]
            requests = [dict(prompt=ASR_PROMPT, audio=load_audio(c['wav']),
                             seed=seed_for(seed, 'asr', c['inst_id'], c['k'])) for c in chunk]
            hypotheses = model.generate(requests, max_new_tokens=128, temperature=0.0, batch_size=batch_size,
                                        seed=seed_for(seed, 'asr_batch', start))
            for clip, hypothesis in zip(chunk, hypotheses):
                wer = word_error_rate(clip['text'], extract_transcript(hypothesis))
                clip.update(asr_hyp=hypothesis.strip(), asr_wer=round(wer, 6), asr_p=round(max(0.0, 1 - wer), 6))
    finally:
        release(model)
    return clips


def finalize(clip):
    if clip['task'] == 'speech':
        clip['relevance_p'] = clip['asr_p']
        clip['relevance'] = 1 + 4 * clip['asr_p']
    clip['score'] = (clip['quality_p'] + clip['relevance_p']) / 2
    return clip


def load_clips(instructions, manifest):
    sources = {row['id']: row for row in read_jsonl(instructions)}
    clips = []
    for path in sorted(glob.glob(manifest)):
        for row in read_jsonl(path):
            source = sources.get(row['inst_id'])
            if source is not None:
                clips.append(dict(row, task=source['task'], scene=source['scene'], text=source.get('text'),
                                  caption=source.get('caption') or source.get('text') or row['prompt']))
    return clips


def label(instructions, manifest, out, model_path, batch_size=8, chunk=1024, asr_backend='vllm', seed=531):
    out = Path(out)
    if out.is_file():
        print(f'[label] {out} exists, skipping', flush=True)
        return
    partial = out.with_name(out.stem + '.partial.jsonl')
    clips = load_clips(instructions, manifest)
    done = {(row['inst_id'], row['k']) for row in read_jsonl(partial)}
    pending = [c for c in clips if (c['inst_id'], c['k']) not in done]
    print(f'[label] {len(pending)} clips to score ({len(done)} cached)', flush=True)
    if pending:
        model = OmniHF(model_path)
        try:
            for start in range(0, len(pending), chunk):
                append_jsonl(partial, score_clips(model, pending[start:start + chunk], batch_size=batch_size))
                print(f'[label] scored {min(start + chunk, len(pending))}/{len(pending)}', flush=True)
        finally:
            release(model)
    scored = read_jsonl(partial)
    speech = [c for c in scored if c['task'] == 'speech' and 'asr_p' not in c]
    if speech:
        transcribe(speech, model_path, asr_backend, seed)
    write_jsonl(out, [finalize(c) for c in scored])
    print(f'[label] wrote {len(scored)} clip scores -> {out}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instructions', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--model_path', default=DEFAULT_MODEL)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--asr_backend', default='vllm', choices=['vllm', 'hf'])
    parser.add_argument('--seed', type=int, default=531)
    args = parser.parse_args()
    label(args.instructions, args.manifest, args.out, args.model_path, args.batch_size,
          asr_backend=args.asr_backend, seed=args.seed)


if __name__ == '__main__':
    main()
