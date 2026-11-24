import gc
import os

import numpy as np
import torch

from crossfeed.prompts import SYSTEM_PROMPT

SR = 16000
DEFAULT_MODEL = os.environ.get('QWEN_OMNI_MODEL_PATH', 'Qwen/Qwen2.5-Omni-7B')


def load_audio(path):
    import librosa
    wav, _ = librosa.load(path, sr=SR, mono=True)
    return wav.astype(np.float32)


def load_processor(model_path=DEFAULT_MODEL):
    from transformers import Qwen2_5OmniProcessor
    return Qwen2_5OmniProcessor.from_pretrained(model_path)


def load_thinker(model_path=DEFAULT_MODEL, device='cuda', dtype=torch.bfloat16, drop_visual=True,
                 attn='flash_attention_2'):
    from transformers import Qwen2_5OmniThinkerForConditionalGeneration
    error = None
    for implementation in dict.fromkeys([attn, 'sdpa']):
        try:
            model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=dtype, attn_implementation=implementation)
            break
        except Exception as exc:
            error = exc
    else:
        raise RuntimeError(f'failed to load thinker: {error}')
    if drop_visual and hasattr(model, 'visual'):
        del model.visual
    return model.to(device).eval()


def chat_messages(user_text, with_audio, shots=None, system=None):
    content = [{'type': 'audio', 'audio': 'placeholder.wav'}] if with_audio else []
    content.append({'type': 'text', 'text': user_text})
    messages = []
    if system != '':
        messages.append({'role': 'system', 'content': [
            {'type': 'text', 'text': SYSTEM_PROMPT if system is None else system}]})
    for shot_user, shot_assistant in shots or []:
        messages.append({'role': 'user', 'content': [{'type': 'text', 'text': shot_user}]})
        messages.append({'role': 'assistant', 'content': [{'type': 'text', 'text': shot_assistant}]})
    messages.append({'role': 'user', 'content': content})
    return messages


def rating_token_ids(tokenizer):
    ids = [tokenizer.encode(str(score), add_special_tokens=False) for score in range(1, 6)]
    if any(len(x) != 1 for x in ids) or len({x[0] for x in ids}) != 5:
        raise ValueError(f'rating anchors must be five distinct single tokens: {ids}')
    return [x[0] for x in ids]


def yes_no_token_ids(tokenizer):
    ids = {answer: tokenizer.encode(answer, add_special_tokens=False) for answer in ('Yes', 'No')}
    if any(len(x) != 1 for x in ids.values()) or ids['Yes'] == ids['No']:
        raise ValueError(f'Yes/No must be distinct single tokens: {ids}')
    return {answer: x[0] for answer, x in ids.items()}


class OmniHF:
    def __init__(self, model_path=DEFAULT_MODEL, device='cuda', attn='flash_attention_2'):
        self.device = device
        self.processor = load_processor(model_path)
        self.tokenizer = self.processor.tokenizer
        self.model = load_thinker(model_path, device=device, attn=attn)
        self.rating_ids = rating_token_ids(self.tokenizer)
        self.yes_no_ids = yes_no_token_ids(self.tokenizer)

    def chat_text(self, user_text, with_audio, shots=None, system=None):
        return self.processor.apply_chat_template(
            chat_messages(user_text, with_audio, shots, system), tokenize=False,
            add_generation_prompt=True)

    def process(self, texts, audios, padding_side='left'):
        old = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side
        try:
            kwargs = dict(text=texts, return_tensors='pt', padding=True, padding_side=padding_side)
            if audios is not None:
                kwargs['audio'] = list(audios)
            inputs = self.processor(**kwargs, sampling_rate=SR)
        finally:
            self.tokenizer.padding_side = old
        return {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

    def _batch(self, items):
        texts = [self.chat_text(it['prompt'], it['audio'] is not None, it.get('shots'), it.get('system'))
                 for it in items]
        audios = [it['audio'] for it in items]
        has_audio = audios[0] is not None
        if any((a is not None) != has_audio for a in audios):
            raise ValueError('a batch must be homogeneous in audio presence')
        return self.process(texts, audios if has_audio else None)

    @torch.no_grad()
    def last_logits(self, items):
        inputs = self._batch(items)
        logits = self.model(**inputs, logits_to_keep=1, use_cache=False).logits
        return logits[:, -1, :].float()

    @torch.no_grad()
    def geval_scores(self, items, batch_size=8):
        weights = torch.arange(1, 6, dtype=torch.float32, device=self.device)
        out = []
        for start in range(0, len(items), batch_size):
            logits = self.last_logits(items[start:start + batch_size])
            probs = torch.softmax(logits, dim=-1)[:, self.rating_ids]
            out.extend(float(x) for x in (probs * weights).sum(-1).tolist())
        return out

    @torch.no_grad()
    def aqa_scores(self, items, batch_size=8):
        out = []
        for start in range(0, len(items), batch_size):
            logits = self.last_logits(items[start:start + batch_size]).double()
            margin = logits[:, self.yes_no_ids['Yes']] - logits[:, self.yes_no_ids['No']]
            out.extend(float(x) for x in torch.sigmoid(margin).tolist())
        return out

    @torch.no_grad()
    def generate(self, items, max_new_tokens=128, temperature=0.0, batch_size=8, seed=None, n=1):
        if seed is not None:
            torch.manual_seed(seed)
        sampling = bool(temperature and temperature > 0)
        step = max(1, batch_size // max(n, 1))
        outputs = []
        for start in range(0, len(items), step):
            inputs = self._batch(items[start:start + step])
            kwargs = dict(max_new_tokens=max_new_tokens, eos_token_id=self.tokenizer.eos_token_id,
                          pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
            if sampling:
                kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
                if n > 1:
                    kwargs['num_return_sequences'] = n
            else:
                kwargs['do_sample'] = False
            generated = self.model.generate(**inputs, **kwargs)[:, inputs['input_ids'].shape[1]:]
            decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            if not sampling and n > 1:
                decoded = [d for d in decoded for _ in range(n)]
            outputs.extend(d.strip() for d in decoded)
        return outputs

    def close(self):
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()


class OmniVLLM:
    def __init__(self, model_path=DEFAULT_MODEL, max_model_len=None, gpu_mem=0.9, seed=None,
                 generation_config=None):
        from vllm import LLM
        self.processor = load_processor(model_path)
        kwargs = dict(model=model_path, trust_remote_code=True, dtype='bfloat16',
                      gpu_memory_utilization=gpu_mem, enable_prefix_caching=True,
                      limit_mm_per_prompt={'audio': 1, 'image': 0, 'video': 0})
        if max_model_len is not None:
            kwargs['max_model_len'] = max_model_len
        if generation_config is not None:
            kwargs['generation_config'] = generation_config
        if seed is not None:
            kwargs['seed'] = seed
        self.llm = LLM(**kwargs)

    def _request(self, item):
        prompt = self.processor.apply_chat_template(
            chat_messages(item['prompt'], item['audio'] is not None, item.get('shots'), item.get('system')),
            tokenize=False, add_generation_prompt=True)
        request = {'prompt': prompt}
        if item['audio'] is not None:
            request['multi_modal_data'] = {'audio': [(item['audio'], SR)]}
        return request

    def generate(self, items, max_new_tokens=128, temperature=0.0, n=1, seed=None, batch_size=None, top_p=0.95):
        from vllm import SamplingParams
        sampling = temperature is not None and temperature > 0
        kwargs = dict(n=n, temperature=temperature if sampling else 0.0, top_p=top_p if sampling else 1.0,
                      max_tokens=max_new_tokens, stop_token_ids=[self.processor.tokenizer.eos_token_id])
        item_seeds = [it.get('seed') for it in items]
        if any(value is not None for value in item_seeds):
            params = [SamplingParams(**kwargs, seed=seed if value is None else int(value))
                      for value in item_seeds]
        else:
            params = SamplingParams(**kwargs, seed=seed)
        results = self.llm.generate([self._request(it) for it in items], params, use_tqdm=False)
        return [completion.text.strip() for result in results for completion in result.outputs]

    def close(self):
        engine = getattr(self.llm, 'llm_engine', None)
        core = getattr(engine, 'engine_core', engine)
        if callable(getattr(core, 'shutdown', None)):
            core.shutdown()
        self.llm = None


def build_omni(model_path=DEFAULT_MODEL, backend='vllm', **kwargs):
    if backend == 'vllm':
        import vllm
        return OmniVLLM(model_path, **kwargs)
    return OmniHF(model_path, **kwargs)
