from functools import partial
from pathlib import Path

from crossfeed._cli import configured_environment, run_module


def prepare_benchmarks(run, env):
    root = Path(env['BENCHMARKS'])
    mmau_test = Path(env['MMAU_TEST_ROOT'])
    inputs = {
        'mmau': ('--audio_dir', env['MMAU_AUDIO']),
        'mmar': ('--audio_dir', env['MMAR_AUDIO']),
        'mmsu': ('--audio_dir', env['MMSU_AUDIO']),
        'trea': ('--source', env['TREA_ROOT']),
        'mmau_test': ('--source', mmau_test, '--audio_dir', mmau_test / 'test-audios'),
    }
    for name, options in inputs.items():
        output = root / f'{name}.json'
        if not output.is_file() or not output.stat().st_size:
            run('crossfeed.evaluation.understanding', 'export', '--benchmark', name, '--out', output, *options)
    return tuple(inputs)


def main():
    env = configured_environment()
    prepare_benchmarks(partial(run_module, env), env)


if __name__ == '__main__':
    main()
