# CFG request pairing is adapted from the NVIDIA Audex-2B inference example.
# https://huggingface.co/nvidia/Nemotron-Labs-Audex-2B/blob/77b7e1a4de899769f22c1bc074db666601c28907/inference_scripts_vllm/audiogen_scripts/run_audio_gen_vllm_rvq_logit_mask.py
# Upstream Copyright (c) 2026, NVIDIA CORPORATION. Licensed under Apache-2.0.

import hashlib
import os
import re
from pathlib import Path

import librosa
import numpy as np

from crossfeed.audex.runtime import SYSTEM_PROMPT, add_official_paths, generation_prompt, stable_seed, strip_tags

TEXT_IDS = [i for i in range(131072) if i not in (29, 30, 31)]


def codec_vocabulary(vocab, family):
    count = 4096 if family == 'audio' else 65536
    codec = {tid: int(m.group(1)) for token, tid in vocab.items()
             if (m := re.fullmatch(fr'<{family}codec_(\d+)>', token)) and int(m.group(1)) < count}
    if set(codec.values()) != set(range(count)):
        raise ValueError(f'incomplete {family} codec vocabulary')
    return codec


class AudexEngine:
    def __init__(self, path, profile='text', gpu_mem=0.92, max_model_len=16384, max_num_seqs=512,
                 max_batched_tokens=131072, enforce_eager=False):
        add_official_paths()
        os.environ['VLLM_PLUGINS'] = 'register_nemotron_dense,audex_2b'
        from audex_2b_vllm.register import register_audex
        from run_audioqa_vllm import build_prompt, check_checkpoint_shards
        from vllm import LLM, ModelRegistry, SamplingParams
        from crossfeed.audex.processors import (PairedCFGLogitsProcessor, TextVocabularyLogitsProcessor,
                                                TTARVQPhaseMaskLogitsProcessor)
        check_checkpoint_shards(str(path))
        register_audex()
        ModelRegistry.register_model('NemotronDenseForCausalLM', 'crossfeed.audex.textmodel:TextOnlyAudex')
        self.text = profile == 'text'
        self.build_prompt = build_prompt
        self.SamplingParams = SamplingParams
        processors = [PairedCFGLogitsProcessor, TTARVQPhaseMaskLogitsProcessor]
        if self.text:
            processors.append(TextVocabularyLogitsProcessor)
        self.llm = LLM(model=str(path), trust_remote_code=True, dtype='bfloat16', seed=stable_seed('vllm_engine'),
                       tensor_parallel_size=1, gpu_memory_utilization=gpu_mem, max_model_len=max_model_len,
                       max_num_seqs=max_num_seqs, max_num_batched_tokens=max_batched_tokens,
                       enforce_eager=enforce_eager, async_scheduling=False, limit_mm_per_prompt={'audio': 1},
                       scheduler_cls='crossfeed.audex.scheduler.AudexCFGScheduler',
                       enable_prefix_caching=self.text, logits_processors=processors)
        self.tokenizer = self.llm.get_tokenizer()
        self.vocab = self.tokenizer.get_vocab()
        self.codecs = {family: codec_vocabulary(self.vocab, family) for family in ('audio', 'speech')}

    def request(self, prompt, audio=None, audio_uuid=None, shots=None, reasoning=False):
        prompt = strip_tags(prompt)
        if audio is not None:
            request = dict(prompt=self.build_prompt(prompt, reasoning=reasoning),
                           multi_modal_data={'audio': [(np.asarray(audio, dtype=np.float32), 16000)]})
            if audio_uuid:
                request['multi_modal_uuids'] = {'audio': [audio_uuid]}
            return request
        text = f'<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n'
        for user, assistant in shots or []:
            text += f'<|im_start|>user\n{strip_tags(user)}<|im_end|>\n<|im_start|>assistant\n{strip_tags(assistant)}<|im_end|>\n'
        return text + f'<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n' + (
            '<think>\n' if reasoning else '<think></think>')

    def text_params(self, n=1, temperature=0.8, max_tokens=1024, seed=None, top_p=0.9):
        kwargs = dict(n=n, temperature=temperature, top_p=top_p if temperature > 0 else 1.0, top_k=0,
                      max_tokens=max_tokens, seed=seed, stop=['<|im_end|>', '<|end_of_text|>'],
                      skip_special_tokens=False)
        if not self.text:
            kwargs['allowed_token_ids'] = TEXT_IDS
        return self.SamplingParams(**kwargs)

    def answers(self, items):
        wavs = list(dict.fromkeys(item['wav'] for item in items if item.get('wav')))
        audio = {wav: librosa.load(str(wav), sr=16000, mono=True)[0] for wav in wavs}
        uuids = {wav: 'audex-wav-' + hashlib.sha256(str(Path(wav).resolve()).encode()).hexdigest() for wav in wavs}
        requests, params = [], []
        for item in items:
            wav = item.get('wav')
            requests.append(self.request(item['prompt'], audio.get(wav), uuids.get(wav), item.get('shots'),
                                         item.get('reasoning', False)))
            params.append(self.text_params(item.get('n', 1), item.get('temperature', 0.8),
                                           item.get('max_tokens', 1024), item.get('seed'), item.get('top_p', 0.9)))
        outputs = self.llm.generate(requests, params, use_tqdm=False, tokenization_kwargs={'add_special_tokens': False})
        result = []
        for output in outputs:
            texts = []
            for completion in output.outputs:
                value = completion.text
                for marker in ('<|im_end|>', '<|end_of_text|>'):
                    value = value.split(marker, 1)[0]
                texts.append(value.strip())
            result.append(texts)
        return result

    def audio_tokens(self, items, tta_cfg=3.0, tts_cfg=1.0, temperature=1.0, top_p=1.0, top_k=80, max_tokens=4096,
                     tta_codec_cap=4000):
        from run_audio_gen_vllm_rvq_logit_mask import build_tta_null_prompt, build_tts_null_prompt, tokenize_cfg_pair
        requests, params, records = [], [], []
        for row, key in items:
            speech = row['task'] == 'speech'
            family = 'speech' if speech else 'audio'
            codec, eos = self.codecs[family], self.vocab[f'<{family}gen_end>']
            prompt = generation_prompt(row)
            attempt = int(row.get('generation_attempt', 0))
            seed = stable_seed('generate', key, attempt)
            common = dict(temperature=temperature, top_p=top_p, top_k=top_k or -1, max_tokens=max_tokens, seed=seed,
                          stop_token_ids=[eos], detokenize=False, allowed_token_ids=list(codec) + [eos],
                          min_tokens=1 if speech else 4)
            extra = {}
            if not speech:
                extra['tta_rvq'] = dict(start_tid=self.vocab['<audiogen_start>'], end_tid=eos, start_in_prompt=True,
                                        codec_cap=tta_codec_cap, phase_token_ids=[
                                            [t for t, c in codec.items() if phase * 1024 <= c < (phase + 1) * 1024]
                                            for phase in range(4)])
            cfg = tts_cfg if speech else tta_cfg
            records.append((len(requests), row, key, prompt, codec, eos, seed))
            if cfg > 1:
                null = (build_tts_null_prompt if speech else build_tta_null_prompt)(prompt, self.tokenizer)
                cond, uncond = tokenize_cfg_pair(prompt, null, self.tokenizer)
                bos = self.tokenizer.bos_token_id
                if cond[:1] == [bos] and uncond[:1] == [bos]:
                    cond, uncond = cond[1:], uncond[1:]
                for role, ids in (('cond', cond), ('uncond', uncond)):
                    requests.append({'prompt_token_ids': ids})
                    params.append(self.SamplingParams(**common, extra_args=dict(extra, cfg_scale=cfg, cfg_role=role,
                                                                                 cfg_pair_id=key)))
            else:
                requests.append({'prompt_token_ids': list(self.tokenizer.encode(prompt, add_special_tokens=False))})
                params.append(self.SamplingParams(**common, extra_args=extra))
        outputs = self.llm.generate(requests, params, use_tqdm=False)
        result = []
        for index, row, key, prompt, codec, eos, seed in records:
            output = outputs[index].outputs[0]
            ids = list(output.token_ids)
            stopped = output.finish_reason == 'stop' and bool(ids) and ids[-1] == eos
            if not stopped:
                result.append(dict(prompt=prompt, seed=seed, truncated=True, generated_tokens=len(ids)))
                continue
            codes = [codec[i] for i in ids[:-1] if i in codec]
            if len(codes) != len(ids) - 1 or not codes:
                raise RuntimeError(f'{key}: empty or invalid codec sequence')
            if row['task'] == 'audio' and (len(codes) % 4 or any(c // 1024 != i % 4 for i, c in enumerate(codes))):
                raise RuntimeError(f'{key}: invalid RVQ frames')
            result.append(dict(prompt=prompt, token_ids=ids, codec_ids=codes, seed=seed, truncated=False,
                               decoder='xcodec2' if row['task'] == 'speech' else 'xcodec1'))
        return result

    def close(self):
        llm, self.llm = self.llm, None
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()
