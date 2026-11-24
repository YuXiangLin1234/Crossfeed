import json
import sys
from pathlib import Path


def load(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else None


def fmt(value, digits=3):
    return '--' if value is None else f'{value:.{digits}f}'


def table(title, header, rows):
    lines = [f'### {title}', '', '| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    lines += ['| ' + ' | '.join(row) + ' |' for row in rows]
    return '\n'.join(lines) + '\n'


def understanding(out):
    sections = []
    for name, folder in (('greedy', 'understanding'), ('SC maj@8', 'understanding_sc')):
        values = [load(out / folder / f'{b}.summary.json') for b in ('mmau', 'mmar', 'mmsu', 'trea')]
        if any(values):
            row = [fmt(v and 100 * v['overall_acc'], 2) for v in values]
            sections.append([name] + row)
    return table('Understanding (accuracy %)', ['mode', 'MMAU-mini', 'MMAR', 'MMSU', 'TREA'], sections) if sections else ''


def judge(out):
    summary = load(out / 'judge' / 'summary.json')
    if not summary:
        return ''
    rows = []
    for method, result in summary.items():
        row = [method]
        for group in ('Audio-REL', 'Audio-OVL', 'Speech-OVL'):
            g = result['groups'][group]
            row += [fmt(g['pearson']), fmt(g['spearman']), fmt(g['mse'])]
        rows.append(row)
    header = ['method'] + [f'{g} {m}' for g in ('REL', 'A-OVL', 'S-OVL') for m in ('r', 'rho', 'MSE')]
    return table('Judge', header, rows)


def tta(out):
    caps, caps_judge = load(out / 'audiocaps' / 'metrics' / 'summary.json'), load(out / 'audiocaps' / 'cottage.json')
    time, time_judge = load(out / 'audiotime' / 'summary.json'), load(out / 'audiotime' / 'cottage.json')
    if not any((caps, time)):
        return ''
    caps, caps_judge, time, time_judge = caps or {}, caps_judge or {}, time or {}, time_judge or {}
    row = [fmt(caps.get('FAD'), 2), fmt(caps.get('FD'), 2), fmt(caps.get('KL')), fmt(caps.get('IS'), 2),
           fmt(caps.get('CLAP')), fmt(caps_judge.get('REL')), fmt(caps_judge.get('OVL')),
           fmt(time.get('order_error_rate')), fmt(time.get('duration_l1')), fmt(time.get('frequency_l1')),
           fmt(time.get('timestamp_f1')), fmt(time_judge.get('REL')), fmt(time_judge.get('OVL'))]
    header = ['FAD', 'FD', 'KL', 'IS', 'CLAP', 'CoTTAGE REL', 'CoTTAGE OVL', 'Order ER', 'Duration L1',
              'Frequency L1', 'Timestamp F1', 'CoTTAGE REL', 'CoTTAGE OVL']
    return table('Text-to-audio (AudioCaps | AudioTime)', header, [row])


def tts(out, folder, datasets, sim):
    rows = []
    for dataset in datasets:
        summary = load(out / folder / dataset / 'summary.json')
        if summary:
            row = [dataset, fmt(summary['corpus_wer'], 2), fmt(summary['utmos'], 2)]
            rows.append(row + [fmt(summary.get('sim'))] if sim else row)
    if not rows:
        return ''
    header = ['dataset', 'WER %', 'UTMOS'] + (['SIM'] if sim else [])
    return table('Zero-shot TTS' if sim else 'Text-to-speech', header, rows)


def main():
    out = Path(sys.argv[1])
    report = '\n'.join(part for part in (
        understanding(out), judge(out), tta(out),
        tts(out, 'tts', ('seedtts', 'cv3', 'cv3_hard', 'libritts'), False),
        tts(out, 'zero_shot_tts', ('seedtts', 'cv3', 'cv3_hard'), True)) if part)
    (out / 'report.md').write_text(report)
    print(report)


if __name__ == '__main__':
    main()
