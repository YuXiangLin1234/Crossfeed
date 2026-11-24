import hashlib
import heapq
import json
import math
import random
import re
from pathlib import Path


DEFAULT_TEXT_GENERATION_ROUNDS = 500


def nonempty_file(path):
    path = Path(path)
    return path.is_file() and path.stat().st_size > 0


def read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    tmp.replace(path)


def append_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def seed_for(base_seed, *identity):
    payload = json.dumps([int(base_seed), *identity], ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), 'little') % (2 ** 31)


def rng_for(base_seed, *identity):
    return random.Random(seed_for(base_seed, *identity))


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


def apportion(total, weights):
    if sum(weights) <= 0:
        raise ValueError('at least one weight must be positive')
    raw = [total * weight / sum(weights) for weight in weights]
    result = [math.floor(value) for value in raw]
    order = sorted(range(len(weights)), key=lambda i: (-(raw[i] - result[i]), i))
    for i in order[:total - sum(result)]:
        result[i] += 1
    return result


def split_even(items, n):
    items = list(items)
    n = max(1, min(n, len(items)))
    q, r = divmod(len(items), n)
    groups, offset = [], 0
    for index in range(n):
        size = q + (index < r)
        groups.append(items[offset:offset + size])
        offset += size
    return [group for group in groups if group]


def json_string_list(text):
    match = re.search(r'\[.*?\]', text, re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if isinstance(value, list) and all(isinstance(x, str) for x in value):
        return value
    return None


def json_list(text):
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if not match:
        return []
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return value if isinstance(value, list) else []


def first_json(text):
    decoder = json.JSONDecoder()
    for i, char in enumerate(text):
        if char in '[{':
            try:
                return decoder.raw_decode(text[i:])[0]
            except ValueError:
                pass
    raise ValueError(f'no JSON in response: {text[:160]!r}')


def interleave_scenes(scenes, seed):
    groups = {}
    for scene in scenes:
        groups.setdefault((scene['task'], scene['category']), []).append(scene)
    heap = []
    for key, rows in groups.items():
        rows.sort(key=lambda row: hashlib.sha256(json.dumps([seed, row['scene']]).encode()).hexdigest())
        heapq.heappush(heap, (0.5 / len(rows), key, 0))
    ordered = []
    while heap:
        _, key, index = heapq.heappop(heap)
        rows = groups[key]
        ordered.append(rows[index])
        if index + 1 < len(rows):
            heapq.heappush(heap, ((index + 1.5) / len(rows), key, index + 1))
    return ordered


def scene_prefix(run, count):
    order = read_json(Path(run) / 'scenes.json')
    if count and count > len(order):
        raise ValueError(f'requested {count} scenes but only {len(order)} were generated')
    return {scene['scene'] for scene in order[:count or len(order)]}
