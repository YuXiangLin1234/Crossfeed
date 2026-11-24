from crossfeed.audex.runtime import add_official_paths

add_official_paths()

from nemotron_dense_vllm.nemotron_dense import NemotronDenseForCausalLM


class TextOnlyAudex(NemotronDenseForCausalLM):
    def load_weights(self, weights):
        return super().load_weights((name, value) for name, value in weights
                                    if not name.startswith(('audio_encoder.', 'audio_projector.')))
