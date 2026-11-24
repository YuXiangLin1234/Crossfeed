import torch

from crossfeed.audex.runtime import add_official_paths

add_official_paths()

from cfg_logits_processor import CFGLogitsProcessor
from tta_rvq_logits_processor import TTARVQPhaseMaskLogitsProcessor
from vllm.v1.sample.logits_processor import LogitsProcessor

TEXT_VOCAB_SIZE = 131072
RESERVED_TEXT_TOKEN_IDS = (29, 30, 31)


class PairedCFGLogitsProcessor(CFGLogitsProcessor):
    def apply(self, logits):
        if not self._info:
            self._pairs.clear()
            return logits
        return super().apply(logits)


class TextVocabularyLogitsProcessor(LogitsProcessor):
    def __init__(self, vllm_config, device, is_pin_memory):
        self.reserved = torch.tensor(RESERVED_TEXT_TOKEN_IDS, dtype=torch.long, device=device)

    def apply(self, logits):
        logits[..., TEXT_VOCAB_SIZE:] = float('-inf')
        logits[..., self.reserved] = float('-inf')
        return logits

    def is_argmax_invariant(self):
        return False

    def update_state(self, batch_update):
        pass


__all__ = ['PairedCFGLogitsProcessor', 'TTARVQPhaseMaskLogitsProcessor', 'TextVocabularyLogitsProcessor']
