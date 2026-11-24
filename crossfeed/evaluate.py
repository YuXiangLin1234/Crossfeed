import argparse
from functools import partial
from pathlib import Path

from crossfeed._cli import FAMILIES, configured_environment, environment_options, run_module
from crossfeed.evaluation.prepare import prepare_benchmarks
from crossfeed.evaluation.run_config import RunConfigError, checkpoint, evaluation_output
from crossfeed.evaluation.understanding import SC_SAMPLES, SC_TEMPERATURE


TASKS = ('understanding', 'judge', 'tta', 'tts')
TTS_DATASETS = ('seedtts', 'cv3', 'cv3_hard', 'libritts')


def understanding(run, env, out, model, sc, benchmarks):
    batch = environment_options(env, batch_size='CF_EVAL_QA_BATCH')
    for benchmark in benchmarks:
        data = Path(env['BENCHMARKS']) / f'{benchmark}.json'
        command = ('evaluate', *model, '--benchmark', benchmark, '--data', data, *batch)
        run('crossfeed.evaluation.understanding', *command, '--out', out / 'understanding' / f'{benchmark}.json',
            '--mode', 'greedy')
        if sc is not None:
            run('crossfeed.evaluation.understanding', *command, '--out', out / 'understanding_sc' / f'{benchmark}.json',
                '--mode', 'sc', '--samples', sc['samples'], '--temperature', sc['temperature'])


def judge(run, env, out, model, baselines):
    run('crossfeed.evaluation.judge', *model, '--out_dir', out / 'judge',
        '--methods', 'all' if baselines else 'geval', '--pam_root', env['PAM_ROOT'],
        '--clotho_csv', env['CLOTHO_CSV'], '--clotho_audio', env['CLOTHO_AUDIO'],
        '--xacle_csv', env['XACLE_CSV'], '--xacle_audio', env['XACLE_AUDIO'],
        '--relate_csv', env['RELATE_CSV'], '--relate_audio', env['RELATE_AUDIO'],
        '--bvcc_root', env['BVCC_ROOT'], '--qualispeech_root', env['QUALISPEECH_ROOT'],
        *environment_options(env, chunk='CF_EVAL_JUDGE_CHUNK'))


def synthesis_options(env, cfg):
    return environment_options(env, cfg=cfg, batch_size='CF_EVAL_AUDIO_BATCH', steps='CF_EVAL_DIT_STEPS')


def tta(run, raw_run, env, out, model):
    generation = synthesis_options(env, 'CF_EVAL_TTA_CFG')
    run('crossfeed.evaluation.tta', 'audiocaps', *model, '--out_dir', out / 'audiocaps',
        '--audiocaps_csv', env['AUDIOCAPS_CSV'], '--audiocaps_audio', env['AUDIOCAPS_AUDIO'], *generation)
    run('crossfeed.evaluation.tta', 'audiotime', *model, '--out_dir', out / 'audiotime',
        '--audiotime_root', env['AUDIOTIME_ROOT'], '--steam_root', Path(env['STEAM_ROOT']).absolute(), *generation)
    raw_run('crossfeed.evaluation.cottage', '--model', env['COTTAGE_MODEL'], '--dirs',
            out / 'audiocaps', out / 'audiotime')


def tts(run, env, out, model, zero_shot):
    data = ('--seedtts_root', env['SEEDTTS_ROOT'], '--cv3_root', env['CV3_ROOT'],
            '--libritts_root', env['LIBRITTS_ROOT'], *synthesis_options(env, 'CF_EVAL_TTS_CFG'))
    for dataset in TTS_DATASETS:
        run('crossfeed.evaluation.tts', dataset, *model, '--out_dir', out / 'tts' / dataset, *data)
    if zero_shot:
        for dataset in TTS_DATASETS[:-1]:
            run('crossfeed.evaluation.tts', dataset, *model, '--zero_shot',
                '--out_dir', out / 'zero_shot_tts' / dataset, '--reference_cache', env['REFERENCE_CACHE'], *data)


def main():
    parser = argparse.ArgumentParser(description='Evaluate a CrossFeed model on the paper benchmarks.')
    parser.add_argument('family', choices=FAMILIES)
    parser.add_argument('--model', help='Audex-2B checkpoint')
    parser.add_argument('--omni', help='UNISON Omni checkpoint')
    parser.add_argument('--dit', help='UNISON DiT checkpoint')
    parser.add_argument('--out-dir', type=Path, help='evaluation output directory')
    parser.add_argument('--tasks', nargs='+', choices=TASKS, help='evaluation tasks (default: all)')
    parser.add_argument('--baselines', action='store_true', help='include understanding and judge baselines')
    parser.add_argument('--limit', type=int, help='limit the number of items per benchmark')
    args = parser.parse_args()
    is_unison = args.family == 'unison'
    if is_unison and args.model:
        parser.error('use --omni and --dit for UNISON')
    if not is_unison and (args.omni or args.dit):
        parser.error('use --model for Audex-2B')
    env = configured_environment()
    if args.limit is None and env.get('CF_EVAL_LIMIT'):
        try:
            args.limit = int(env['CF_EVAL_LIMIT'])
        except ValueError:
            parser.error('CF_EVAL_LIMIT must be an integer')
    tasks = args.tasks or (env.get('TASKS') or ' '.join(TASKS)).split()
    invalid = set(tasks) - set(TASKS)
    if invalid:
        parser.error(f'unknown tasks: {", ".join(sorted(invalid))}')
    out = (args.out_dir or Path(env.get('OUT') or Path(env['RUNS']) / args.family / 'eval' / 'base')).absolute()
    baselines = args.baselines or env.get('BASELINES') == '1'

    try:
        evaluate(args, env, tasks, out, baselines)
    except RunConfigError as error:
        parser.exit(2, f'{error}\n')


def evaluate(args, env, tasks, out, baselines):
    is_unison = args.family == 'unison'
    models = {}
    if is_unison:
        omni = args.omni or env.get('OMNI') or env['QWEN_OMNI']
        model, models['omni'] = checkpoint(omni, env=env)
        if 'tta' in tasks or 'tts' in tasks:
            dit = args.dit or env.get('DIT') or env['UNISON_DIT']
            dit, models['dit'] = checkpoint(dit, local=True)
            generator = ('--family', args.family, '--model', dit, '--omni', model)
    else:
        audex = args.model or env.get('MODEL') or Path(env['AUDEX_MODEL_ROOT']) / 'checkpoint_folder_full'
        model, models['audex'] = checkpoint(audex, local=True)
        generator = ('--family', args.family, '--model', model)
    if 'tta' in tasks:
        cottage, models['cottage'] = checkpoint(env['COTTAGE_MODEL'], env=env)
        env = {**env, 'COTTAGE_MODEL': cottage}
    scorer = ('--family', args.family, '--model', model)
    raw_run = partial(run_module, env)
    options = ('--seed', env['SEED'])
    if args.limit is not None:
        options += ('--limit', args.limit)

    def run(module, *command):
        raw_run(module, *command, *options)

    task_settings = {task: dict(seed=int(env['SEED']), limit=args.limit or None) for task in tasks}
    sc = None
    if baselines and 'understanding' in tasks:
        sc = dict(samples=int(env.get('CF_EVAL_SC_SAMPLES') or SC_SAMPLES),
                  temperature=float(env.get('CF_EVAL_SC_TEMPERATURE') or SC_TEMPERATURE))
        task_settings['understanding_sc'] = {**task_settings['understanding'], **sc}
    config = dict(kind='evaluation', family=args.family, models=models, tasks=task_settings)
    with evaluation_output(out, config):
        for task in tasks:
            if task == 'understanding':
                benchmarks = prepare_benchmarks(raw_run, env)
                understanding(run, env, out, scorer, sc, benchmarks)
            elif task == 'judge':
                judge(run, env, out, scorer, baselines)
            elif task == 'tta':
                tta(run, raw_run, env, out, generator)
            elif task == 'tts':
                tts(run, env, out, generator, is_unison)
        raw_run('crossfeed.evaluation.report', out)


if __name__ == '__main__':
    main()
