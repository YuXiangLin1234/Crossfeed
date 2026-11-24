import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

from crossfeed.utils import read_json, write_json


CONFIG_NAME = 'run_config.json'
LOCK_NAME = '.run.lock'
SCHEMA_VERSION = 1
MODEL_SUFFIXES = {
    '.bin', '.gguf', '.jinja', '.json', '.model', '.pt', '.pth', '.py', '.safetensors',
    '.tiktoken', '.txt', '.yaml', '.yml',
}


class RunConfigError(ValueError):
    pass


def file_identity(path):
    path = Path(path).resolve()
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {'path': str(path), 'missing': True}
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                ctime_ns=stat.st_ctime_ns, inode=stat.st_ino)


def checkpoint(value, *, local=False, env=None):
    if not value:
        raise RunConfigError('A checkpoint path is required.')
    path = Path(value).expanduser()
    if not path.exists():
        if local or path.is_absolute() or path.is_symlink() or str(value).startswith('.') or len(path.parts) > 2:
            raise RunConfigError(f'Checkpoint does not exist: {path}')
        from huggingface_hub import snapshot_download
        env = os.environ if env is None else env
        cache = env.get('HF_HUB_CACHE') or env.get('HUGGINGFACE_HUB_CACHE')
        if not cache and env.get('HF_HOME'):
            cache = Path(env['HF_HOME']).expanduser() / 'hub'
        offline = any(env.get(key, '').upper() in ('1', 'TRUE', 'YES', 'ON')
                      for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'))
        path = Path(snapshot_download(str(value), cache_dir=cache, local_files_only=offline,
                                      token=env.get('HF_TOKEN')))
    if path.is_file():
        return str(path.parent.resolve() / path.name), file_identity(path)

    path = path.resolve()
    files = {p for p in path.rglob('*') if p.is_file() and p.suffix in MODEL_SUFFIXES
             and not any(part.startswith('.') or part == '__pycache__' for part in p.relative_to(path).parts)}
    audio_preprocessor = path / 'audio_preprocessor'
    if audio_preprocessor.is_symlink():
        files.update(p for p in audio_preprocessor.iterdir() if p.is_file() and p.suffix in MODEL_SUFFIXES)
    if not files:
        raise RunConfigError(f'No checkpoint files found in {path}')
    return str(path), dict(path=str(path), files={str(p.relative_to(path)): file_identity(p) for p in sorted(files)})


def samples_identity(rows):
    payload = json.dumps(rows, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    return dict(count=len(rows), sha256=hashlib.sha256(payload.encode('utf-8')).hexdigest())


@contextmanager
def output_lock(directory, filename=LOCK_NAME):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / filename).open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RunConfigError(f'{directory} is in use by another evaluation.') from error
        yield


def load_config(path):
    try:
        config = read_json(path)
    except (OSError, ValueError) as error:
        raise RunConfigError(f'Cannot read {path}. Use a new evaluation output directory.') from error
    if not isinstance(config, dict) or config.get('schema_version') != SCHEMA_VERSION:
        raise RunConfigError(f'Unrecognized run configuration at {path}. Use a new evaluation output directory.')
    if config.get('kind') == 'evaluation' and not all(isinstance(config.get(key), dict)
                                                       for key in ('models', 'tasks')):
        raise RunConfigError(f'Incomplete run configuration at {path}. Use a new evaluation output directory.')
    return config


def changed_fields(previous, requested, depth=2):
    changed = []
    for key in sorted(previous.keys() | requested.keys()):
        old, new = previous.get(key), requested.get(key)
        if old == new and key in previous and key in requested:
            continue
        if (depth and key not in ('auxiliary', 'files', 'samples', 'references')
                and isinstance(old, dict) and isinstance(new, dict)):
            changed.extend(f'{key}.{name}' for name in changed_fields(old, new, depth - 1))
        else:
            changed.append(key)
    return changed


def check_config(directory, previous, requested):
    changed = changed_fields(previous, requested)
    if changed:
        raise RunConfigError(f'{directory} has results for a different evaluation configuration '
                             f'(changed: {", ".join(changed)}). Use a new --out-dir or --out_dir.')


@contextmanager
def evaluation_output(directory, config):
    directory = Path(directory)
    path = directory / CONFIG_NAME
    config = dict(schema_version=SCHEMA_VERSION, **config)
    with output_lock(directory):
        if path.exists() or path.is_symlink():
            previous = load_config(path)
            if previous.get('kind') == config['kind'] == 'evaluation':
                requested = dict(config)
                comparison = dict(previous)
                for key in ('tasks', 'models'):
                    requested[key] = {**previous[key], **config[key]}
                    comparison[key] = {**config[key], **previous[key]}
                check_config(directory, comparison, requested)
                if requested != previous:
                    write_json(path, requested)
            else:
                check_config(directory, previous, config)
        elif any(p.name not in (LOCK_NAME, f'{CONFIG_NAME}.tmp') for p in directory.iterdir()):
            raise RunConfigError(f'{directory} has existing results without {CONFIG_NAME}. '
                                 'Use a new --out-dir or --out_dir to preserve those results.')
        else:
            write_json(path, config)
        yield


def check_parent_run(directory, family, models, task, settings):
    for parent in Path(directory).resolve().parents:
        path = parent / CONFIG_NAME
        if not (parent / LOCK_NAME).is_file() or not path.is_file():
            continue
        config = load_config(path)
        if config.get('kind') == 'evaluation':
            requested = dict(family=family, models=models, tasks={task: settings})
            previous = dict(family=config['family'], models={k: config['models'].get(k) for k in models},
                            tasks={task: config['tasks'].get(task)})
            check_config(parent, previous, requested)
            return
