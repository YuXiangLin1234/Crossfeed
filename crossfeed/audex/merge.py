import argparse
import json
import os
import shutil
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

PREFIXES = ('model.', 'lm_head.', 'backbone.')


def weight_map(root):
    return json.loads((Path(root) / 'model.safetensors.index.json').read_text())['weight_map']


def merge(base, understanding, generation, output, understanding_weight=0.5, generation_weight=0.5):
    base, understanding, generation, output = (Path(p).resolve() for p in (base, understanding, generation, output))
    if (output / 'COMPLETE').is_file():
        return output
    output.mkdir(parents=True, exist_ok=True)
    maps = {'base': weight_map(base), 'und': weight_map(understanding), 'gen': weight_map(generation)}
    roots = {'base': base, 'und': understanding, 'gen': generation}
    names = {name for name in maps['base'] if name.startswith(PREFIXES)}
    with ExitStack() as stack:
        files = {label: {shard: stack.enter_context(safe_open(roots[label] / shard, framework='pt', device='cpu'))
                         for shard in set(maps[label].values())} for label in roots}
        for shard in sorted(set(maps['base'].values())):
            target = output / shard
            target.unlink(missing_ok=True)
            members = [n for n, s in maps['base'].items() if s == shard]
            if not any(n in names for n in members):
                try:
                    os.link(base / shard, target)
                except OSError:
                    shutil.copy2(base / shard, target)
                continue
            tensors = {}
            for name in files['base'][shard].keys():
                value = files['base'][shard].get_tensor(name)
                if name in names and value.is_floating_point():
                    b = value.float()
                    u = files['und'][maps['und'][name]].get_tensor(name).float()
                    g = files['gen'][maps['gen'][name]].get_tensor(name).float()
                    value = (b + understanding_weight * (u - b) + generation_weight * (g - b)).to(value.dtype)
                tensors[name] = value.contiguous()
            save_file(tensors, target, metadata=files['base'][shard].metadata() or {'format': 'pt'})
    for item in base.iterdir():
        if item.is_file() and not item.name.endswith('.safetensors'):
            shutil.copy2(item, output / item.name)
        elif item.is_dir():
            shutil.copytree(item, output / item.name, dirs_exist_ok=True, symlinks=True)
    (output / 'COMPLETE').write_text('ok\n')
    print(f'[merge] {output}', flush=True)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--understanding', required=True)
    parser.add_argument('--generation', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--understanding_weight', type=float, default=0.5)
    parser.add_argument('--generation_weight', type=float, default=0.5)
    args = parser.parse_args()
    merge(args.base, args.understanding, args.generation, args.output, args.understanding_weight,
          args.generation_weight)


if __name__ == '__main__':
    main()
