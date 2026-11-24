import importlib
import importlib.metadata
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from crossfeed.utils import nonempty_file, read_jsonl


def xcodec2_class():
    codec = importlib.metadata.distribution('xcodec2')
    tune = importlib.metadata.distribution('torchtune')
    directory = tempfile.TemporaryDirectory(prefix='audex_codec_')
    root = Path(directory.name)
    package = root / 'audex_original_xcodec2'
    shutil.copytree(codec.locate_file('xcodec2'), package, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(tune.locate_file('torchtune/modules/position_embeddings.py'), package / '_rope.py')
    for source in (package / 'vq').glob('*.py'):
        text = source.read_text()
        source.write_text(text.replace('from torchtune.modules import RotaryPositionalEmbeddings',
                                       'from .._rope import RotaryPositionalEmbeddings'))
    model_source = package / 'modeling_xcodec2.py'
    model_source.write_text(model_source.read_text().replace(
        'from transformers import AutoFeatureExtractor, Wav2Vec2BertModel',
        'from transformers import AutoConfig, AutoFeatureExtractor, Wav2Vec2BertModel').replace(
        '''self.semantic_model = Wav2Vec2BertModel.from_pretrained(
            "facebook/w2v-bert-2.0",
            output_hidden_states=True
        )''', '''semantic_config = AutoConfig.from_pretrained("facebook/w2v-bert-2.0")
        semantic_config.output_hidden_states = True
        self.semantic_model = Wav2Vec2BertModel(semantic_config)'''))
    activation = package / 'vq' / 'activations.py'
    activation.write_text(activation.read_text().replace('self.bias', 'self.beta'))
    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module('audex_original_xcodec2.modeling_xcodec2')
        module.XCodec2Model.get_init_context = classmethod(lambda cls, *args, **kwargs: [])
        module.XCodec2Model.all_tied_weights_keys = {}
        module._source_directory = directory
        return module.XCodec2Model
    finally:
        sys.path.remove(str(root))


def load_codec(task):
    if task == 'speech':
        return xcodec2_class().from_pretrained(os.environ.get('XCODEC2_PATH', 'HKUSTAudio/xcodec2')).eval().cuda()
    from transformers import AutoModel
    return AutoModel.from_pretrained(os.environ['XCODEC1_PATH'], trust_remote_code=True).eval().cuda()


@torch.inference_mode()
def decode_codes(codec, task, code_rows):
    values = torch.tensor(code_rows, dtype=torch.long, device='cuda')
    if task == 'speech':
        return codec.decode_code(values[:, None, :])[:, 0].float().cpu().numpy()
    values = values.reshape(len(code_rows), -1, 4).transpose(1, 2)
    values = values - torch.arange(4, device='cuda')[None, :, None] * 1024
    return codec.decode(values).audio_values[:, 0].float().cpu().numpy()


def decode(rows, batch_size=64):
    for task in ('speech', 'audio'):
        pending = [row for row in rows if row['task'] == task and not nonempty_file(row['wav'])]
        if not pending:
            continue
        codec = load_codec(task)
        groups = defaultdict(list)
        for row in pending:
            groups[len(read_jsonl(row['tokens'])[0]['codec_ids'])].append(row)
        for _, group in sorted(groups.items()):
            for start in range(0, len(group), batch_size):
                batch = group[start:start + batch_size]
                waves = decode_codes(codec, task, [read_jsonl(row['tokens'])[0]['codec_ids'] for row in batch])
                for row, wav in zip(batch, waves):
                    if not len(wav) or not np.isfinite(wav).all():
                        raise RuntimeError(f'invalid decoded waveform: {row["clip_id"]}')
                    target = Path(row['wav'])
                    target.parent.mkdir(parents=True, exist_ok=True)
                    tmp = target.with_suffix('.tmp.wav')
                    sf.write(tmp, wav, 16000)
                    tmp.replace(target)
        del codec
        torch.cuda.empty_cache()
