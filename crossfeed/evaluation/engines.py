import os


class UnisonGenerator:
    def __init__(self, model_path, seed=531):
        from crossfeed.unison.omni import OmniVLLM
        self.engine = OmniVLLM(model_path, seed=seed, generation_config='vllm')

    def generate(self, items, max_new_tokens=64, temperature=0.0, n=1, top_p=0.95):
        from crossfeed.unison.omni import load_audio
        items = [dict(item, audio=item['audio'] if 'audio' in item else load_audio(item['wav'])) for item in items]
        return self.engine.generate(items, max_new_tokens=max_new_tokens, temperature=temperature, n=n, top_p=top_p)

    def close(self):
        self.engine.close()


class AudexGenerator:
    def __init__(self, model_path, seed=531, reasoning=False):
        os.environ['SEED'] = str(seed)
        from crossfeed.audex.engine import AudexEngine
        self.engine = AudexEngine(model_path, profile='generation', gpu_mem=0.95, max_num_seqs=256)
        self.reasoning = reasoning

    def generate(self, items, max_new_tokens=64, temperature=0.0, n=1, top_p=0.9):
        requests = [dict(prompt=item['prompt'], wav=item.get('wav'), n=n, temperature=temperature, top_p=top_p,
                         max_tokens=max_new_tokens, seed=item.get('seed'), reasoning=self.reasoning)
                    for item in items]
        return [text for outputs in self.engine.answers(requests) for text in outputs]

    def close(self):
        self.engine.close()


def generator(family, model_path, seed=531, reasoning=False):
    if family == 'unison':
        return UnisonGenerator(model_path, seed)
    return AudexGenerator(model_path, seed, reasoning)


def scorer(family, model_path):
    if family == 'unison':
        from crossfeed.unison.omni import OmniHF
        return OmniHF(model_path)
    from crossfeed.audex.hf import AudexJudge
    return AudexJudge(model_path, batch_size=1)
