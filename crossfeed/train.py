import argparse
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

from crossfeed._cli import FAMILIES, configured_environment, environment_options, run_module


def link_checkpoint(link, target):
    if not target.is_dir():
        raise FileNotFoundError(f'Checkpoint directory is missing: {target}')
    if link.exists() and not link.is_symlink():
        raise FileExistsError(f'Checkpoint link path is already in use: {link}')
    with TemporaryDirectory(dir=link.parent, prefix=f'.{link.name}-') as directory:
        staged = Path(directory) / link.name
        staged.symlink_to(target.relative_to(link.parent), target_is_directory=True)
        staged.replace(link)


def prepare_unison(run, env, data, omni, dit, n, seed):
    rounds = environment_options(env, max_iters='CF_TEXT_ROUNDS')
    generation = ('--dit_ckpt', dit, '--omni_model_path', omni, '--cfg', 6, '--cfg_interval', '0.2,0.9',
                  '--steps', 50, '--trim_silence', 2, '--seed', seed)
    run('crossfeed.unison.instructions', '--out_dir', data, '--n', n, '--model_path', omni, '--seed', seed, *rounds)
    run('crossfeed.unison.generate', '--instructions', data / 'instructions.jsonl', '--out_dir', data,
        *generation, *environment_options(env, k='CF_CANDIDATES'))
    run('crossfeed.unison.label', '--instructions', data / 'instructions.jsonl', '--manifest',
        data / 'gen_manifest.jsonl', '--out', data / 'scores.jsonl', '--model_path', omni, '--seed', seed)
    run('crossfeed.unison.perturb', '--run', data, '--model_path', omni, '--seed', seed)
    perturb = data / 'perturb'
    run('crossfeed.unison.generate', '--instructions', perturb / 'instructions.jsonl', '--out_dir', perturb,
        '--k', 1, *generation)
    run('crossfeed.unison.label', '--instructions', perturb / 'instructions.jsonl', '--manifest',
        perturb / 'gen_manifest.jsonl', '--out', perturb / 'scores.jsonl', '--model_path', omni, '--seed', seed)
    run('crossfeed.unison.speaker_bank', '--out_dir', data / 'speaker_bank', '--dit_ckpt', dit,
        '--omni_model_path', omni, '--seed', seed)
    run('crossfeed.unison.dialogue', '--run', data, '--speaker_bank', data / 'speaker_bank' / 'speaker_bank.jsonl',
        '--model_path', omni, '--seed', seed, *rounds)
    run('crossfeed.unison.dialogue_audio', '--run', data, '--dit_ckpt', dit, '--omni_model_path', omni, '--seed', seed)
    for source in ('main', 'perturb', 'dialogue'):
        run('crossfeed.unison.qa', '--run', data, '--source', source, '--model_path', omni, '--seed', seed,
            *environment_options(env, vote_min='CF_QA_VOTE_MIN'))
    run('crossfeed.unison.data', '--run', data, '--scene_order', '--seed', seed)


def understanding_options(env):
    return environment_options(env, lr='CF_UND_LR', epochs='CF_UND_EPOCHS', effective_batch_size='CF_UND_BATCH',
                               lora_r='CF_LORA_R', lora_alpha='CF_LORA_ALPHA')


def generation_options(env, batch_flag):
    return environment_options(env, lr='CF_GEN_LR', epochs='CF_GEN_EPOCHS', beta='CF_DPO_BETA',
                               **{batch_flag: 'CF_GEN_BATCH'})


def train_unison(run, env, data, out, n, und_n, gen_n, merge, skip_data, seed):
    omni, dit = env['QWEN_OMNI'], env['UNISON_DIT']
    if not skip_data:
        prepare_unison(run, env, data, omni, dit, n, seed)

    training_data = out / 'understanding_train.jsonl'
    understanding = out / 'understanding'
    finetuned = out / 'finetuned_omni_full'
    merged = out / 'merged_omni_full'
    generation = out / 'generation'
    dit_dpo = out / 'dit_dpo'
    final_dit = out / 'final_dit'
    run('crossfeed.unison.data', '--run', data, '--und_n', und_n, '--out', training_data)
    run('crossfeed.unison.train_understanding', '--data', training_data, '--output_dir', understanding,
        '--model_path', omni, '--seed', seed, *understanding_options(env))
    merge_args = ('--merged', merged) if merge else ()
    run('crossfeed.unison.merge', 'omni', '--adapter', understanding / 'checkpoint-final' / 'adapter',
        '--base', omni, '--finetuned', finetuned, *merge_args)
    run('crossfeed.unison.preference', '--run', data, '--judge', finetuned, '--gen_n', gen_n,
        '--out_dir', generation, '--seed', seed, *environment_options(env, margin='CF_PAIR_MARGIN'))

    conditioner = merged if merge else finetuned
    run('crossfeed.unison.train_dit', '--pairs', generation / 'dpo_pairs.jsonl', '--output_dir', dit_dpo,
        '--pretrained', dit, '--omni_model_path', conditioner, '--seed', seed, *generation_options(env, 'batch_size'))
    run('crossfeed.unison.merge', 'dit', '--base', dit, '--tuned', dit_dpo / 'checkpoint-final',
        '--output', final_dit, '--weight', 0.5 if merge else 1.0)
    link_checkpoint(out / 'final_omni', conditioner)
    print(f'[unison] done: DiT={final_dit} Omni={out / "final_omni"}', flush=True)


def train_audex(run, env, data, out, n, und_n, gen_n, merge, skip_data, seed):
    base = env.get('BASE') or Path(env['AUDEX_MODEL_ROOT']) / 'checkpoint_folder_full'

    def stage(name, *args):
        run('crossfeed.audex.pipeline', name, '--run', data, '--model', base, '--seed', seed, *args)

    if not skip_data:
        stage('instructions', '--n', n, *environment_options(env, max_iters='CF_TEXT_ROUNDS'))
        stage('generate', *environment_options(env, k='CF_CANDIDATES'))
        stage('perturb')
        stage('dialogue', *environment_options(env, dialogue_max_iters='CF_TEXT_ROUNDS'))
        stage('qa', *environment_options(env, vote_min='CF_QA_VOTE_MIN'))

    training_data = out / 'understanding_train.jsonl'
    understanding = out / 'understanding'
    understanding_ckpt = understanding / 'checkpoint-final'
    generation = out / 'generation'
    dpo = out / 'generation_dpo'
    stage('data', '--und_n', und_n, '--out', training_data)
    run('crossfeed.audex.train', '--mode', 'understanding', '--model', base, '--data', training_data,
        '--output_dir', understanding, '--seed', seed, *understanding_options(env))
    stage('preference', '--judge', understanding_ckpt, '--gen_n', gen_n, '--out_dir', generation,
          *environment_options(env, margin='CF_PAIR_MARGIN'))
    run('crossfeed.audex.train', '--mode', 'generation', '--model', base if merge else understanding_ckpt,
        '--data', generation / 'dpo_pairs.jsonl', '--output_dir', dpo, '--seed', seed,
        *generation_options(env, 'effective_batch_size'))

    checkpoint = dpo / 'checkpoint-final'
    if merge:
        merged = out / 'merged'
        run('crossfeed.audex.merge', '--base', base, '--understanding', understanding_ckpt,
            '--generation', checkpoint, '--output', merged)
        checkpoint = merged
    link_checkpoint(out / 'final', checkpoint)
    print(f'[audex] done: {out / "final"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description='Run the CrossFeed training pipeline.')
    parser.add_argument('family', choices=FAMILIES)
    parser.add_argument('--n', type=int, help='number of scenes to generate (default: 5000)')
    parser.add_argument('--und-n', type=int, help='scene budget for understanding and judge training')
    parser.add_argument('--gen-n', type=int, help='scene budget for generation training')
    parser.add_argument('--out-dir', type=Path, help='training output directory')
    parser.add_argument('--data-run', type=Path, help='generated training data directory')
    parser.add_argument('--skip-data', action='store_true', help='reuse generated training data')
    parser.add_argument('--no-merge', action='store_true', help='train without self-merging')
    parser.add_argument('--seed', type=int, help='base random seed')
    args = parser.parse_args()

    env = configured_environment()
    n = args.n if args.n is not None else int(env.get('N') or 5000)
    und_n = args.und_n if args.und_n is not None else int(env.get('UND_N') or n)
    gen_n = args.gen_n if args.gen_n is not None else int(env.get('GEN_N') or n)
    merge = not args.no_merge and (env.get('MERGE') or '1') == '1'
    skip_data = args.skip_data or env.get('SKIP_DATA') == '1'
    seed = args.seed if args.seed is not None else int(env['SEED'])
    runs = Path(env['RUNS']) / args.family
    data = args.data_run or Path(env.get('DATA_RUN') or runs / f'data_n{n}')
    out = args.out_dir or Path(env.get('OUT') or runs / f'u{und_n}_g{gen_n}_merge{int(merge)}')
    train = train_unison if args.family == 'unison' else train_audex
    train(partial(run_module, env), env, data, out, n, und_n, gen_n, merge, skip_data, seed)


if __name__ == '__main__':
    main()
