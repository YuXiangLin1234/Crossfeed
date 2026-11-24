import argparse
import shutil
from collections import Counter, deque
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from crossfeed.unison.generate import generate
from crossfeed.unison.omni import DEFAULT_MODEL
from crossfeed.utils import read_jsonl, rng_for, write_json, write_jsonl


def load_features(path, sample_rate=16000):
    wav, sr = sf.read(path, dtype='float32', always_2d=True)
    wav = wav.mean(axis=1)
    if sr != sample_rate:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=sample_rate)
    wav, _ = librosa.effects.trim(wav, top_db=40)
    if len(wav) < sample_rate // 4:
        raise ValueError(f'too little non-silent audio: {path}')
    rms = np.sqrt(np.mean(wav ** 2))
    if rms > 0:
        wav = wav * (0.1 / rms)
    mfcc = librosa.feature.mfcc(y=wav, sr=sample_rate, n_mfcc=20, n_fft=512, hop_length=160)[1:]
    return np.concatenate([mfcc, librosa.feature.delta(mfcc, order=1),
                           librosa.feature.delta(mfcc, order=2)], axis=0).T


def dtw_distance(first, second):
    costs, path = librosa.sequence.dtw(X=first.T, Y=second.T, metric='euclidean')
    return float(costs[-1, -1] / len(path))


def candidates(n, text):
    rows = []
    for index in range(n):
        gender = 'female' if index % 2 == 0 else 'male'
        rows.append(dict(id=f'speaker_candidate_{index + 1:06d}', task='speech', text=text, gender=gender,
                         prompt=f'[Speech] A {gender} voice saying "{text}"'))
    return rows


def crop(rows, audio_dir, out_dir, duration=3.0, sample_rate=16000):
    out_dir.mkdir(parents=True, exist_ok=True)
    n_samples = int(round(duration * sample_rate))
    cropped = []
    for row in rows:
        wav, sr = sf.read(Path(audio_dir) / f'{row["id"]}_k0.wav', dtype='float32', always_2d=True)
        wav = wav.mean(axis=1)
        if sr != sample_rate:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=sample_rate)
        fixed = np.zeros(n_samples, dtype=np.float32)
        fixed[:min(n_samples, len(wav))] = wav[:n_samples]
        path = out_dir / f'{row["id"]}.wav'
        sf.write(path, fixed, sample_rate, subtype='PCM_16')
        if float(np.sqrt(np.mean(fixed ** 2))) > 1e-5:
            cropped.append(dict(candidate_id=row['id'], gender=row['gender'], ref_wav=str(path.resolve()),
                                reference_text=row['text']))
    return cropped


def interleaved(rows, seed):
    rng = rng_for(seed, 'speaker_bank')
    queues = {}
    for gender in ('female', 'male'):
        members = [row for row in rows if row['gender'] == gender]
        rng.shuffle(members)
        queues[gender] = deque(members)
    while any(queues.values()):
        for gender in ('female', 'male'):
            if queues[gender]:
                yield queues[gender].popleft()


def select(rows, target, threshold, neighbors, seed):
    features, embeddings, usable = {}, {}, []
    for row in rows:
        try:
            feature = load_features(row['ref_wav'])
        except ValueError:
            continue
        features[row['candidate_id']] = feature
        embeddings[row['candidate_id']] = np.concatenate([feature.mean(axis=0), feature.std(axis=0)])
        usable.append(row)
    accepted, accepted_embeddings = [], []
    for row in interleaved(usable, seed):
        nearest = None
        if accepted:
            distances = np.linalg.norm(np.stack(accepted_embeddings) - embeddings[row['candidate_id']][None], axis=1)
            count = min(neighbors, len(accepted))
            for index in np.argpartition(distances, count - 1)[:count]:
                distance = dtw_distance(features[row['candidate_id']], features[accepted[int(index)]['candidate_id']])
                nearest = distance if nearest is None else min(nearest, distance)
                if distance <= threshold:
                    break
        if nearest is not None and nearest <= threshold:
            continue
        accepted.append(row)
        accepted_embeddings.append(embeddings[row['candidate_id']])
        if len(accepted) >= target:
            break
    return accepted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--dit_ckpt', required=True)
    parser.add_argument('--omni_model_path', default=DEFAULT_MODEL)
    parser.add_argument('--target', type=int, default=200)
    parser.add_argument('--candidates', type=int, default=2000)
    parser.add_argument('--reference_text', default='What I cannot create, I do not understand.')
    parser.add_argument('--duration', type=float, default=3.0)
    parser.add_argument('--threshold', type=float, default=70.0)
    parser.add_argument('--neighbors', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=480)
    parser.add_argument('--seed', type=int, default=531)
    args = parser.parse_args()

    out = Path(args.out_dir)
    if (out / 'speaker_bank.jsonl').is_file():
        print(f'[speaker_bank] {out / "speaker_bank.jsonl"} exists, skipping', flush=True)
        return
    rows = candidates(args.candidates, args.reference_text)
    write_jsonl(out / 'candidates.jsonl', rows)
    generate(rows, out / 'generation', args.dit_ckpt, args.omni_model_path, k=1, cfg=4.5, sample_seeds=[1],
             cfg_interval=(0.2, 0.9), steps=50, batch_size=args.batch_size, seed=args.seed, save_latents=False)
    cropped = crop(rows, out / 'generation' / 'audio', out / 'candidates', args.duration)
    accepted = select(cropped, args.target, args.threshold, args.neighbors, args.seed)
    (out / 'accepted').mkdir(parents=True, exist_ok=True)
    bank = []
    for index, row in enumerate(accepted, 1):
        path = out / 'accepted' / f'speaker_{index:06d}.wav'
        shutil.copy2(row['ref_wav'], path)
        bank.append(dict(speaker_id=f'speaker_{index:06d}', gender=row['gender'], ref_wav=str(path.resolve()),
                         reference_text=row['reference_text']))
    write_jsonl(out / 'speaker_bank.jsonl', bank)
    write_json(out / 'summary.json', dict(target=args.target, accepted=len(bank), candidates=len(rows),
                                          by_gender=dict(Counter(row['gender'] for row in bank)),
                                          threshold=args.threshold, neighbors=args.neighbors))
    if len(bank) < args.target:
        raise SystemExit(f'only {len(bank)}/{args.target} speakers survived; increase --candidates')
    print(f'[speaker_bank] {len(bank)} speakers -> {out / "speaker_bank.jsonl"}', flush=True)


if __name__ == '__main__':
    main()
