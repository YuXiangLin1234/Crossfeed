# AudexJudge.prepare follows the audio feature and input formatting in NVIDIA Audex-2B's build_inputs.
# https://huggingface.co/nvidia/Nemotron-Labs-Audex-2B/blob/77b7e1a4de899769f22c1bc074db666601c28907/inference_scripts_hf/inference_hf.py#L99

import gc
from pathlib import Path

import librosa
import numpy as np
import torch

from crossfeed.audex.runtime import add_official_paths, strip_tags


def load(path):
    add_official_paths()
    from inference_hf import load_model
    return load_model(str(path), 'single', 'cuda', 'bfloat16', 'none', False)


def yes_no_ids(tokenizer):
    ids = {answer: tokenizer.encode(answer, add_special_tokens=False) for answer in ('Yes', 'No')}
    if any(len(x) != 1 for x in ids.values()):
        raise ValueError(f'Yes/No must be single tokens: {ids}')
    return ids['Yes'][0], ids['No'][0]


def rating_ids(tokenizer):
    ids = [tokenizer.encode(str(score), add_special_tokens=False) for score in range(1, 6)]
    if any(len(x) != 1 for x in ids) or len({x[0] for x in ids}) != 5:
        raise ValueError(f'rating anchors must be five distinct single tokens: {ids}')
    return [x[0] for x in ids]


class AudexJudge:
    def __init__(self, path, batch_size=8):
        add_official_paths()
        from audio_utils import build_prompt_template, expand_sound_placeholder, extract_whisper_features
        from transformers import AutoConfig, AutoFeatureExtractor, AutoModelForCausalLM, AutoTokenizer
        path = Path(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        self.extractor = AutoFeatureExtractor.from_pretrained(path / 'audio_preprocessor')
        self.model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, config=self.config,
                                                          torch_dtype=torch.bfloat16, device_map={'': 'cuda'}).eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id or 0
        self.tokenizer.padding_side = 'right'
        self.template, self.expand, self.features = build_prompt_template, expand_sound_placeholder, extract_whisper_features
        self.batch_size = batch_size
        self.rating = rating_ids(self.tokenizer)
        self.yes, self.no = yes_no_ids(self.tokenizer)

    def prepare(self, item, strip):
        audio = item.get('audio')
        if audio is None:
            audio, _ = librosa.load(item['wav'], sr=16000, mono=True)
        features = self.features(self.extractor, np.asarray(audio, dtype=np.float32), sample_rate=16000,
                                 clip_duration=float(getattr(self.config, 'sound_clip_duration', 30.0)))
        count = features.shape[0] * int(getattr(self.config, 'sound_embedding_size', 750))
        text = strip_tags(item['prompt']) if strip else item['prompt']
        prompt = self.expand(self.template(text, reasoning=False) + '\n', count)
        if item.get('system'):
            prompt = f'<|im_start|>system\n{item["system"]}<|im_end|>\n' + prompt
        return prompt, features

    @torch.inference_mode()
    def last_logits(self, items, strip):
        prepared = [self.prepare(item, strip) for item in items]
        groups = {}
        for index, (prompt, features) in enumerate(prepared):
            groups.setdefault(int(features.shape[0]), []).append((index, prompt, features))
        logits = [None] * len(items)
        for group in groups.values():
            for start in range(0, len(group), self.batch_size):
                chunk = group[start:start + self.batch_size]
                tokens = self.tokenizer([entry[1] for entry in chunk], return_tensors='pt', padding=True,
                                        add_special_tokens=False)
                input_ids, mask = tokens.input_ids.cuda(), tokens.attention_mask.cuda()
                positions = (mask.long().cumsum(-1) - 1).masked_fill_(mask == 0, 0)
                embeds = self.model.prepare_inputs_embeds(input_ids=input_ids,
                                                          input_features=torch.stack([e[2] for e in chunk]).cuda())
                hidden = self.model._dense_forward_from_embeds(inputs_embeds=embeds, attention_mask=mask,
                                                               position_ids=positions, use_cache=False,
                                                               return_dict=True).last_hidden_state
                last = hidden[torch.arange(len(chunk), device=hidden.device), mask.sum(dim=-1) - 1]
                for entry, row in zip(chunk, self.model.lm_head(last).float()):
                    logits[entry[0]] = row
        return torch.stack(logits)

    def geval_scores(self, items):
        probs = torch.softmax(self.last_logits(items, strip=False), dim=-1)[:, self.rating]
        return (probs * torch.arange(1, 6, device=probs.device, dtype=torch.float32)).sum(-1).tolist()

    def aqa_scores(self, items):
        logits = self.last_logits(items, strip=True).double()
        return torch.sigmoid(logits[:, self.yes] - logits[:, self.no]).tolist()

    def close(self):
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()
