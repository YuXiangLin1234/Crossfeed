import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ('unison', 'audex')


def configured_environment():
    result = subprocess.run(
        ['bash', '--noprofile', '--norc', '-c', 'source "$1" >&2 && env -0',
         'crossfeed-config', str(ROOT / 'config.sh')],
        stdout=subprocess.PIPE, check=True,
    )
    return dict(os.fsdecode(entry).split('=', 1) for entry in result.stdout.split(b'\0') if entry)


def environment_options(env, **variables):
    options = []
    for flag, variable in variables.items():
        if value := env.get(variable):
            options.extend((f'--{flag}', value))
    return options


def run_module(env, module, *args):
    command = [sys.executable, '-m', module, *(str(arg) for arg in args)]
    print(f'[crossfeed] {shlex.join(command)}', flush=True)
    result = subprocess.run(command, env=env)
    if result.returncode:
        raise SystemExit(result.returncode if result.returncode > 0 else 128 - result.returncode)
