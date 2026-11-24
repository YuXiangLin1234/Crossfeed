import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from crossfeed.mcq import integer_distractors, ordinal
from crossfeed.unison.generate import Generator, interval
from crossfeed.unison.omni import DEFAULT_MODEL
from crossfeed.utils import append_jsonl, read_jsonl, rng_for, seed_everything, seed_for, write_json, write_jsonl
from unison.pipelines.infer import join_ref_target_text, sample_latents


def fixed_reference(path, sample_rate, duration):
    wav, sr = sf.read(path, dtype='float32', always_2d=True)
    wav = wav.mean(axis=1)
    if sr != sample_rate:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=sample_rate)
    out = np.zeros(int(round(duration * sample_rate)), dtype=np.float32)
    out[:min(len(out), len(wav))] = wav[:len(out)]
    return out


def trim_target(wav, sample_rate, top_db=40.0):
    trimmed, _ = librosa.effects.trim(np.asarray(wav, dtype=np.float32), top_db=top_db)
    if len(trimmed) < int(0.15 * sample_rate):
        return np.asarray(wav, dtype=np.float32)
    fade = min(int(0.01 * sample_rate), len(trimmed) // 4)
    if fade:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        trimmed[:fade] *= ramp
        trimmed[-fade:] *= ramp[::-1]
    return trimmed


def synthesize(turns, out_dir, dit_ckpt, omni_model_path, duration=10.0, ref_duration=3.0, steps=50, cfg=4.5,
               cfg_interval=(0.2, 0.9), batch_size=64, seed=531):
    manifest = out_dir / 'turn_audio.jsonl'
    done = {row['turn_key'] for row in read_jsonl(manifest)}
    todo = [turn for turn in turns if turn['turn_key'] not in done]
    print(f'[dialogue_audio] {len(todo)}/{len(turns)} turns to synthesize', flush=True)
    if not todo:
        return manifest
    seed_everything(seed)
    (out_dir / 'turns').mkdir(parents=True, exist_ok=True)
    generator = Generator(dit_ckpt, omni_model_path)
    sample_rate = generator.sample_rate
    total, ref_samples = int(round(duration * sample_rate)), int(round(ref_duration * sample_rate))
    for start in range(0, len(todo), batch_size):
        batch = todo[start:start + batch_size]
        sources = []
        for turn in batch:
            source = np.zeros(total, dtype=np.float32)
            source[:ref_samples] = fixed_reference(turn['ref_audio'], sample_rate, ref_duration)[:ref_samples]
            sources.append(torch.from_numpy(source))
        with torch.no_grad():
            latents = generator.vae.wrapped_encode(torch.stack(sources).to(generator.device)) * generator.vae_scale
        frames = latents.shape[-1]
        mask = torch.zeros(len(batch), 1, total, device=generator.device)
        mask[:, :, :ref_samples] = 2.0
        masks = F.interpolate(mask, size=frames, mode='nearest').to(torch.bfloat16)
        latents = latents * (masks > 1.5).to(latents.dtype).expand_as(latents)
        noise = torch.stack([torch.randn(generator.model.config.in_channels, frames,
                                         generator=torch.Generator().manual_seed(seed_for(seed, 'dialogue_turn', t['turn_key'])))
                             for t in batch])
        prompts = ['[Speech with voice] ' + join_ref_target_text(t['ref_text'], t['text']) for t in batch]
        z = sample_latents(generator.model, generator.scheduler, generator.text_encoder, prompts,
                           source_latents=latents, masks=masks, num_inference_steps=steps, guidance_scale=cfg,
                           device=generator.device, noise_init=noise, cfg_interval=cfg_interval)
        with torch.no_grad():
            decoded = generator.vae.wrapped_decode(
                (z / generator.vae_scale).to(getattr(generator.vae, 'dtype', torch.bfloat16))).float().cpu()
        rows = []
        for turn, wav in zip(batch, decoded):
            wav = wav.mean(0) if wav.ndim > 1 else wav
            path = out_dir / 'turns' / f'{turn["turn_key"]}.wav'
            sf.write(path, trim_target(wav[ref_samples:total].numpy(), sample_rate), sample_rate, subtype='PCM_16')
            rows.append(dict(turn_key=turn['turn_key'], wav=str(path.resolve())))
        append_jsonl(manifest, rows)
        print(f'[dialogue_audio] {min(start + batch_size, len(todo))}/{len(todo)} turns', flush=True)
    return manifest


def numeric_question(dialogue_id, scene, wav, family, question, correct, distractors, formatter, rng):
    values = [correct, *distractors[:3]]
    rng.shuffle(values)
    return dict(inst_id=dialogue_id, scene=scene, wav=wav, family=f'dialogue_ground_truth/{family}',
                qid=f'{dialogue_id}_gt_{family}', question=question, options=[formatter(v) for v in values],
                answer_idx=values.index(correct))


def ground_truth_questions(script, wav, duration, timeline, seed):
    rng = rng_for(seed, 'ground_truth', script['dialogue_id'])
    did, scene = script['dialogue_id'], script['scene']
    speakers, turns = len(script['speakers']), len(script['turns'])
    rounded = max(1, int(round(duration)))
    questions = [
        numeric_question(did, scene, wav, 'speaker_count', 'How many distinct speakers are heard in the dialogue?',
                         speakers, integer_distractors(speakers, 1, 12), str, rng),
        numeric_question(did, scene, wav, 'turn_count', 'How many speaking turns are heard in the dialogue?',
                         turns, integer_distractors(turns, 2, 12), str, rng),
        numeric_question(did, scene, wav, 'total_duration', 'Approximately how long is the complete dialogue?',
                         rounded, integer_distractors(rounded, 1, max(rounded + 8, 12)), lambda v: f'{v} seconds', rng),
    ]
    chosen = rng.randrange(1, turns)
    tick = int(round(timeline[chosen]['start_s'] * 2))
    questions.append(numeric_question(did, scene, wav, 'utterance_start_time',
                                      f'Approximately when does the {ordinal(chosen + 1)} utterance begin?', tick,
                                      integer_distractors(tick, 0, max(3, int(round(duration * 2)))),
                                      lambda v: f'{v / 2:.1f} seconds', rng))
    return questions


def assemble(scripts, turn_audio, out_dir, gap=(0.10, 0.50), max_duration=60.0, sample_rate=16000, seed=531):
    audio = {row['turn_key']: row['wav'] for row in read_jsonl(turn_audio)}
    (out_dir / 'audio').mkdir(parents=True, exist_ok=True)
    instructions, clips, questions = [], [], []
    for script in scripts:
        did = script['dialogue_id']
        keys = [f'{did}_turn_{turn["turn_id"]:02d}' for turn in script['turns']]
        if not all(key in audio for key in keys):
            continue
        pieces, timeline, cursor = [], [], 0
        rng = rng_for(seed, 'dialogue_gap', did)
        for turn, key in zip(script['turns'], keys):
            wav, _ = librosa.load(audio[key], sr=sample_rate, mono=True)
            start = cursor / sample_rate
            pieces.append(np.asarray(wav, dtype=np.float32))
            cursor += len(wav)
            timeline.append(dict(turn_id=turn['turn_id'], speaker=turn['speaker'], text=turn['text'],
                                 start_s=round(start, 4), end_s=round(cursor / sample_rate, 4)))
            if turn['turn_id'] < len(script['turns']):
                silence = int(round(rng.uniform(*gap) * sample_rate))
                pieces.append(np.zeros(silence, dtype=np.float32))
                cursor += silence
        dialogue = np.concatenate(pieces)
        duration = len(dialogue) / sample_rate
        if duration > max_duration:
            continue
        peak = float(np.abs(dialogue).max())
        if peak > 0.999:
            dialogue = dialogue * (0.999 / peak)
        path = out_dir / 'audio' / f'{did}.wav'
        sf.write(path, dialogue, sample_rate, subtype='PCM_16')
        wav = str(path.resolve())
        transcript = '\n'.join(f'Speaker {script["speakers"].index(t["speaker"]) + 1}: {t["text"]}'
                               for t in script['turns'])
        instructions.append(dict(id=did, scene=script['scene'], task='dialogue', capability=script['capability'],
                                 instruction_type=script['capability'], caption=transcript, prompt=transcript,
                                 duration=round(duration, 4), turn_timeline=timeline))
        clips.append(dict(inst_id=did, k=0, wav=wav, scene=script['scene'], task=f'dialogue/{script["capability"]}'))
        questions.extend(ground_truth_questions(script, wav, duration, timeline, seed))
    write_jsonl(out_dir / 'instructions.jsonl', instructions)
    write_jsonl(out_dir / 'clips.jsonl', clips)
    write_jsonl(out_dir / 'ground_truth_qa.jsonl', questions)
    write_json(out_dir / 'assembly_stats.json', dict(dialogues=len(clips), scripts=len(scripts),
                                                     ground_truth_questions=len(questions)))
    print(f'[dialogue_audio] assembled {len(clips)}/{len(scripts)} dialogues', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--dit_ckpt', required=True)
    parser.add_argument('--omni_model_path', default=DEFAULT_MODEL)
    parser.add_argument('--cfg', type=float, default=4.5)
    parser.add_argument('--cfg_interval', type=interval, default=(0.2, 0.9))
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=531)
    args = parser.parse_args()
    out = Path(args.run) / 'dialogue'
    manifest = synthesize(read_jsonl(out / 'turns.jsonl'), out, args.dit_ckpt, args.omni_model_path, steps=args.steps,
                          cfg=args.cfg, cfg_interval=args.cfg_interval, batch_size=args.batch_size, seed=args.seed)
    assemble(read_jsonl(out / 'scripts.jsonl'), manifest, out, seed=args.seed)


if __name__ == '__main__':
    main()
