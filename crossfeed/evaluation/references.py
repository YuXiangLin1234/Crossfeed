import gc
import re
from pathlib import Path

import numpy as np

from crossfeed.asr import edit_distance
from crossfeed.evaluation.run_config import file_identity, output_lock, samples_identity
from crossfeed.utils import nonempty_file, read_json, write_json

ASR_KWARGS = dict(language='english', task='transcribe', num_beams=1, max_new_tokens=96,
                  condition_on_prev_tokens=False, temperature=0.0)


def cache_key(source, transcript):
    return samples_identity([file_identity(source), transcript])['sha256'][:24]


def words(text):
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    return EnglishTextNormalizer({})(text).split()


def audit(text, truth, seconds):
    hypothesis, reference = words(text), words(truth)
    mismatch, prefix = min((edit_distance(reference[:n], hypothesis) / max(n, len(hypothesis), 1), n)
                           for n in range(1, len(reference) + 1))
    flags = []
    if not hypothesis:
        flags.append('empty')
    if mismatch > .5:
        flags.append('reference_prefix_mismatch')
    if len(hypothesis) > 6 * seconds + 5:
        flags.append('implausible_transcript_length')
    if any('؀' <= c <= 'ۿ' or '一' <= c <= '鿿' for c in text):
        flags.append('unexpected_script')
    if len(hypothesis) > len(reference) + 3:
        flags.append('longer_than_full_reference')
    return dict(prefix_normalized_edit_rate=mismatch, best_prefix_words=prefix, flags=flags)


def transcribe_batch(asr, waves, kwargs):
    window = getattr(asr.feature_extractor, 'chunk_length', 30)
    short = [i for i, w in enumerate(waves) if len(w['raw']) <= window * w['sampling_rate']]
    outputs = [None] * len(waves)
    if short:
        for i, output in zip(short, asr([waves[i] for i in short], generate_kwargs=kwargs)):
            outputs[i] = output
    for i in set(range(len(waves))) - set(short):
        outputs[i] = asr(waves[i], generate_kwargs=kwargs, return_timestamps=True)
    return outputs


def ctc_align(log_probs, labels, blank):
    labels = np.asarray(labels, dtype=np.int64)
    states = np.full(2 * len(labels) + 1, blank, dtype=np.int64)
    states[1::2] = labels
    frames, size = len(log_probs), len(states)
    dp = np.full(size, -np.inf)
    dp[0], dp[1] = log_probs[0, blank], log_probs[0, states[1]]
    back = np.zeros((frames, size), dtype=np.uint8)
    skip = np.zeros(size, dtype=bool)
    skip[2:] = (states[2:] != blank) & (states[2:] != states[:-2])
    for t in range(1, frames):
        previous = np.concatenate(([-np.inf], dp[:-1]))
        previous2 = np.concatenate(([-np.inf, -np.inf], dp[:-2]))
        previous2[~skip] = -np.inf
        choices = np.stack([dp, previous, previous2])
        which = choices.argmax(0)
        back[t] = which
        dp = choices[which, np.arange(size)] + log_probs[t, states]
    s = size - 1 if dp[-1] > dp[-2] else size - 2
    if not np.isfinite(dp[s]):
        raise ValueError('no valid alignment')
    path = np.zeros(frames, dtype=np.int64)
    path[-1] = s
    for t in range(frames - 1, 0, -1):
        s -= int(back[t, s])
        path[t - 1] = s
    spans = []
    for i, label in enumerate(labels):
        hit = np.flatnonzero(path == 2 * i + 1)
        if not len(hit):
            raise ValueError('missing aligned token')
        spans.append((int(hit[0]), int(hit[-1] + 1)))
    return spans


def recover(item, asr, ctc, processor):
    import soundfile as sf
    import torch
    from unison.pipelines.infer import _rms_normalize
    wave, sr = sf.read(item['source'], dtype='float32')
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    full = asr(dict(raw=wave.copy(), sampling_rate=sr), return_timestamps='word',
               generate_kwargs=dict(ASR_KWARGS, num_beams=5, max_new_tokens=160))
    truth, hypothesis = words(item['transcript']), words(full['text'])
    if not hypothesis or edit_distance(truth, hypothesis) / max(len(truth), len(hypothesis), 1) > .35:
        raise ValueError(f'unresolved reference: {item["source"]}')
    if sr != 16000:
        from torchaudio.functional import resample
        wave = resample(torch.from_numpy(wave), sr, 16000).numpy()
    labels = processor.tokenizer(re.sub("[^A-Z' ]", '', ' '.join(truth).upper()), add_special_tokens=False).input_ids
    inputs = processor(wave, sampling_rate=16000, return_tensors='pt')
    with torch.inference_mode():
        log_probs = ctc(**{k: v.to('cuda') for k, v in inputs.items()}).logits[0].log_softmax(-1).cpu().numpy()
    spans = ctc_align(log_probs, labels, ctc.config.pad_token_id)
    scale = len(wave) / 16000 / len(log_probs)
    aligned, begin = [], 0
    for i in range(len(labels) + 1):
        if i == len(labels) or labels[i] == processor.tokenizer.word_delimiter_token_id:
            if i > begin:
                aligned.append(dict(text=processor.tokenizer.decode(labels[begin:i], group_tokens=False),
                                    start=spans[begin][0] * scale, end=spans[i - 1][1] * scale))
            begin = i + 1
    start = max(0, aligned[0]['start'] - .10)
    eligible = [w for w in aligned if w['end'] + .08 - start <= 3.0]
    if not eligible:
        raise ValueError(f'no complete reference word: {item["source"]}')
    end = min(len(wave) / 16000, eligible[-1]['end'] + .08)
    speech = _rms_normalize(torch.from_numpy(wave[round(start * 16000):round(end * 16000)]).reshape(1, -1))
    speech = speech.reshape(-1).numpy()
    expected = ' '.join(w['text'] for w in eligible)
    cropped = asr(dict(raw=speech.copy(), sampling_rate=16000), generate_kwargs=dict(ASR_KWARGS, num_beams=5))
    a, b = words(expected), words(cropped['text'])
    check = audit(cropped['text'], item['transcript'], len(speech) / 16000)
    if check['flags'] or edit_distance(a, b) / max(len(a), len(b), 1) > .35:
        raise ValueError(f'unresolved CTC crop: {item["source"]}')
    return np.pad(speech, (0, 1600)), expected


def prepare_references(parents, cache):
    cache = Path(cache).resolve()
    with output_lock(cache, '.references.lock'):
        return prepare_cached_references(parents, cache)


def prepare_cached_references(parents, cache):
    import soundfile as sf
    import torch
    from transformers import AutoModelForCTC, AutoProcessor, pipeline
    from unison.pipelines.infer import load_ref_audio
    requested = {r['ref_audio']: r['ref_text'] for r in parents}
    receipts, pending = {}, []
    for source, transcript in sorted(requested.items()):
        key = cache_key(source, transcript)
        wav = cache / f'{key}.wav'
        receipt = wav.with_suffix('.json')
        if receipt.is_file() and nonempty_file(wav):
            receipts[source] = read_json(receipt)
            continue
        wave = load_ref_audio(source, target_sr=16000, max_ref_duration=3, device='cpu', silence_search_s=.5,
                              silence_threshold_db=-40, tail_pad_s=.1)
        pending.append(dict(source=source, transcript=transcript, wave=wave.reshape(-1).numpy(), wav=wav))
    if not pending:
        return receipts
    asr = pipeline('automatic-speech-recognition', model='openai/whisper-large-v3', dtype=torch.bfloat16,
                   device='cuda', batch_size=32)
    ctc = processor = None
    for start in range(0, len(pending), 32):
        batch = pending[start:start + 32]
        outputs = transcribe_batch(asr, [dict(raw=r['wave'][:-1600].copy(), sampling_rate=16000) for r in batch],
                                   ASR_KWARGS)
        for item, output in zip(batch, outputs):
            speech = item['wave'][:-1600]
            text = output['text'].strip()
            check = audit(text, item['transcript'], len(speech) / 16000)
            if check['flags']:
                retry = asr(dict(raw=speech.copy(), sampling_rate=16000),
                            generate_kwargs=dict(ASR_KWARGS, num_beams=5))['text'].strip()
                other = audit(retry, item['transcript'], len(speech) / 16000)
                if (len(other['flags']), other['prefix_normalized_edit_rate']) < (
                        len(check['flags']), check['prefix_normalized_edit_rate']):
                    text, check = retry, other
            wave = item['wave']
            if check['flags']:
                if ctc is None:
                    name = 'facebook/wav2vec2-large-960h-lv60-self'
                    processor = AutoProcessor.from_pretrained(name)
                    ctc = AutoModelForCTC.from_pretrained(name).to('cuda').eval()
                wave, text = recover(item, asr, ctc, processor)
            wav = item['wav']
            tmp = wav.with_suffix('.tmp.wav')
            sf.write(tmp, wave, 16000, subtype='FLOAT', format='WAV')
            tmp.replace(wav)
            receipts[item['source']] = dict(source=item['source'], wav=str(wav), text=text, recovered=bool(check['flags']))
            write_json(wav.with_suffix('.json'), receipts[item['source']])
    del asr, ctc
    gc.collect()
    torch.cuda.empty_cache()
    return receipts
