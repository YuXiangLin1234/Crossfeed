import argparse
import json
import os
import shutil
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

MERGE_PREFIXES = ('thinker.model.', 'thinker.lm_head.')
DIT_WEIGHTS = ('ema_model.pt', 'model.safetensors', 'pytorch_model.bin')


def resolve(path):
    path = Path(path).expanduser()
    if path.is_dir():
        return path.resolve()
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(str(path))).resolve()


def complete(path):
    try:
        index = json.loads((Path(path) / 'model.safetensors.index.json').read_text())
    except (OSError, ValueError):
        return False
    return (Path(path) / 'config.json').is_file() and all((Path(path) / s).is_file()
                                                          for s in set(index['weight_map'].values()))


def copy_metadata(source, target):
    for item in source.iterdir():
        if item.is_file() and not item.name.endswith('.safetensors'):
            shutil.copy2(item, target / item.name)


def fold_lora(adapter, base, output):
    from peft import PeftModel
    from crossfeed.unison.omni import load_thinker
    base, output = resolve(base), Path(output)
    if complete(output):
        return output
    thinker = load_thinker(str(base), device='cpu', drop_visual=False, attn='sdpa')
    thinker = PeftModel.from_pretrained(thinker, str(adapter)).merge_and_unload()
    folded = {f'thinker.{k}': v for k, v in thinker.state_dict().items()}
    output.mkdir(parents=True, exist_ok=True)
    index = json.loads((base / 'model.safetensors.index.json').read_text())
    for shard in sorted(set(index['weight_map'].values())):
        with safe_open(base / shard, 'pt') as handle:
            tensors = {k: folded[k].contiguous() if k in folded else handle.get_tensor(k) for k in handle.keys()}
            save_file(tensors, output / shard, metadata=handle.metadata() or {'format': 'pt'})
    copy_metadata(base, output)
    return output


def interpolate_omni(base, tuned, output, weight=0.5):
    base, tuned, output = resolve(base), Path(tuned), Path(output)
    if complete(output):
        return output
    output.mkdir(parents=True, exist_ok=True)
    base_map = json.loads((base / 'model.safetensors.index.json').read_text())['weight_map']
    tuned_map = json.loads((tuned / 'model.safetensors.index.json').read_text())['weight_map']
    for shard in sorted(set(base_map.values())):
        names = [n for n, s in base_map.items() if s == shard and n.startswith(MERGE_PREFIXES)]
        target = output / shard
        target.unlink(missing_ok=True)
        if not names:
            try:
                os.link(base / shard, target)
            except OSError:
                shutil.copy2(base / shard, target)
            continue
        with ExitStack() as stack:
            tuned_files = {s: stack.enter_context(safe_open(tuned / s, framework='pt', device='cpu'))
                           for s in {tuned_map[n] for n in names}}
            with safe_open(base / shard, framework='pt', device='cpu') as handle:
                tensors = {}
                for name in handle.keys():
                    value = handle.get_tensor(name)
                    if name in names and value.is_floating_point():
                        trained = tuned_files[tuned_map[name]].get_tensor(name).float()
                        value = value.float().add_(trained - value.float(), alpha=weight).to(value.dtype)
                    tensors[name] = value.contiguous()
                save_file(tensors, target, metadata=handle.metadata() or {'format': 'pt'})
    copy_metadata(base, output)
    return output


def dit_weights(path):
    path = Path(path)
    if path.is_file():
        return path
    for name in DIT_WEIGHTS:
        if (path / name).is_file():
            return path / name
    raise FileNotFoundError(f'no DiT weights in {path}')


def dit_state(path, model, base_state=None):
    path = dit_weights(path)
    state = load_file(str(path), device='cpu') if path.suffix == '.safetensors' else torch.load(
        path, map_location='cpu', weights_only=True)
    if 'shadow_params' not in state:
        return state
    names = [name for name, _ in model.named_parameters()]
    result = dict(base_state)
    result.update(zip(names, state['shadow_params']))
    return result


def interpolate_dit(base, tuned, output, weight=0.5, model_config=None):
    import yaml
    from unison.models.transformers.backbone import UnisonBackbone
    from crossfeed.unison.generate import MODEL_CONFIG
    output = Path(output)
    if (output / 'model.safetensors').is_file():
        return output
    config = yaml.safe_load(Path(model_config or MODEL_CONFIG).read_text())
    config.pop('omni_last_layer_idx', None)
    config['concat_condition'] = True
    with torch.device('meta'):
        model = UnisonBackbone(**config)
    base_state = dit_state(base, model)
    trained = dit_state(tuned, model, base_state)
    merged = {name: torch.lerp(value.float(), trained[name].float(), weight).contiguous()
              if value.is_floating_point() else value.clone() for name, value in base_state.items()}
    output.mkdir(parents=True, exist_ok=True)
    save_file(merged, str(output / 'model.safetensors'))
    return output


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    omni = sub.add_parser('omni')
    omni.add_argument('--adapter', required=True)
    omni.add_argument('--base', required=True)
    omni.add_argument('--finetuned', required=True)
    omni.add_argument('--merged', default='')
    omni.add_argument('--weight', type=float, default=0.5)
    dit = sub.add_parser('dit')
    dit.add_argument('--base', required=True)
    dit.add_argument('--tuned', required=True)
    dit.add_argument('--output', required=True)
    dit.add_argument('--weight', type=float, default=0.5)
    args = parser.parse_args()
    if args.command == 'omni':
        folded = fold_lora(args.adapter, args.base, args.finetuned)
        if args.merged:
            interpolate_omni(args.base, folded, args.merged, args.weight)
    else:
        interpolate_dit(args.base, args.tuned, args.output, args.weight)


if __name__ == '__main__':
    main()
