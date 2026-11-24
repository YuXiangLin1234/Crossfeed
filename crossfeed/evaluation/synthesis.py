import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from crossfeed.evaluation.run_config import checkpoint, check_parent_run, evaluation_output, file_identity
from crossfeed.utils import nonempty_file


UNISON_CFG = 4.5
AUDEX_TTA_CFG = 3.0
AUDEX_TTS_CFG = 1.0


@contextmanager
def benchmark_output(out, args, *, task, benchmark, samples, references=None, **settings):
    shared = dict(seed=args.seed, limit=args.limit or None)
    generation = dict(batch_size=args.batch_size, **shared)
    if args.family == 'unison':
        args.model, dit = checkpoint(args.model, local=True)
        args.omni, omni = checkpoint(args.omni)
        models = dict(dit=dit, omni=omni)
        generation.update(cfg=UNISON_CFG if args.cfg is None else args.cfg, steps=args.steps)
        unison = Path(__file__).resolve().parents[2] / 'third_party' / 'unison'
        mmaudio = unison / 'models' / 'mmaudio'
        auxiliary = {p.name: file_identity(p) for p in (
            unison / 'config' / 'D24S0_O_20ch.yaml', mmaudio / 'vae_config_16k.yaml',
            mmaudio / 'data' / 'ext_weights' / 'v1-16.pth',
            mmaudio / 'data' / 'ext_weights' / 'best_netG.pt',
        )}
    else:
        args.model, audex = checkpoint(args.model, local=True)
        models = dict(audex=audex)
        if task == 'tta':
            variable, codec = 'XCODEC1_PATH', os.environ.get('XCODEC1_PATH')
            default_cfg = AUDEX_TTA_CFG
        else:
            variable, codec = 'XCODEC2_PATH', os.environ.get('XCODEC2_PATH', 'HKUSTAudio/xcodec2')
            default_cfg = AUDEX_TTS_CFG
        generation['cfg'] = default_cfg if args.cfg is None else args.cfg
        codec_path, codec_identity = checkpoint(codec)
        auxiliary = {variable: codec_identity}
    generation.update(settings)
    check_parent_run(out, args.family, models, task, shared)
    config = dict(kind='benchmark', family=args.family, task=task, benchmark=benchmark, models=models,
                  auxiliary=auxiliary, samples=samples, references=references, settings=generation)
    with evaluation_output(out, config):
        if args.family == 'audex':
            os.environ[variable] = codec_path
        yield


def synthesize(family, rows, out_dir, model, omni=None, cfg=None, steps=100, batch_size=64, seed=531, *, task='tta'):
    out_dir = Path(out_dir)
    if family == 'unison':
        from crossfeed.unison.generate import generate
        generate(rows, out_dir, model, omni, k=1, cfg=UNISON_CFG if cfg is None else cfg, steps=steps,
                 batch_size=batch_size, seed=seed, save_latents=False, resume_missing=True)
    else:
        os.environ['SEED'] = str(seed)
        from crossfeed.audex.pipeline import generate
        args = SimpleNamespace(tta_cfg=cfg if task == 'tta' and cfg is not None else AUDEX_TTA_CFG,
                               tts_cfg=cfg if task == 'tts' and cfg is not None else AUDEX_TTS_CFG,
                               gen_batch_size=128, decode_batch_size=batch_size)
        generate(out_dir, rows, 'gen_manifest.jsonl', model, 1, args, resume_missing=True)
    missing = [r['id'] for r in rows if not nonempty_file(out_dir / 'audio' / f'{r["id"]}_k0.wav')]
    if missing:
        raise RuntimeError(f'{len(missing)} clips were not generated, e.g. {missing[:5]}')
    return {r['id']: out_dir / 'audio' / f'{r["id"]}_k0.wav' for r in rows}


def add_arguments(parser):
    parser.add_argument('--family', required=True, choices=['unison', 'audex'])
    parser.add_argument('--model', required=True)
    parser.add_argument('--omni', default=None)
    parser.add_argument('--cfg', type=float, default=None)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=531)


def from_args(args, rows, out_dir, *, task='tta'):
    return synthesize(args.family, rows, out_dir, args.model, args.omni, args.cfg, args.steps, args.batch_size,
                      args.seed, task=task)

