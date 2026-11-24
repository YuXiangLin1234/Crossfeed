import re
from collections import defaultdict

from crossfeed.prompts import QA_PROMPT

LETTERS = 'ABCDEFGH'

BALANCED_PERMUTATIONS = (
    (0, 1, 2, 3),
    (0, 1, 3, 2),
    (1, 2, 0, 3),
    (1, 3, 0, 2),
    (2, 0, 3, 1),
    (2, 3, 1, 0),
    (3, 0, 2, 1),
    (3, 2, 1, 0),
)

_LETTER_PATTERNS = [
    re.compile(r'^\s*\(([A-H])\)'),
    re.compile(r'[Ff]inal answer[:\s]*\(?([A-H])\)?'),
    re.compile(r'[Aa]nswer[:\s]+\(?([A-H])\)?'),
    re.compile(r'\(([A-H])\)'),
    re.compile(r'^\s*([A-H])[\.\):\s]'),
    re.compile(r'\b([A-H])\s*$'),
]
_FINAL_ANSWER_VOTE = re.compile(r'Final answer:\s*\(([A-D])\)', re.I)


def render_options(options, pre_prefixed=False):
    if pre_prefixed:
        return '\n'.join(options)
    return '\n'.join(f'({LETTERS[i]}) {o}' for i, o in enumerate(options))


def qa_prompt(question, options, pre_prefixed=False):
    return QA_PROMPT.format(question=question, options=render_options(options, pre_prefixed))


def answer_text(options, answer_idx, pre_prefixed=False):
    if pre_prefixed:
        return options[answer_idx]
    return f'({LETTERS[answer_idx]}) {options[answer_idx]}'


def extract_letter(text):
    if not text:
        return None
    for pattern in _LETTER_PATTERNS:
        match = pattern.search(text.strip())
        if match:
            return match.group(1)
    return None


def letter_index(letter):
    return LETTERS.index(letter) if letter and letter in LETTERS else None


def vote_option_index(text, permutation):
    match = _FINAL_ANSWER_VOTE.search(text or '')
    if match:
        return permutation[LETTERS.index(match.group(1).upper())]
    return None


def balanced_shuffle(rows, rng):
    groups = defaultdict(list)
    for row in rows:
        groups[len(row['options'])].append(row)
    for n, items in groups.items():
        targets = [i % n for i in range(len(items))]
        rng.shuffle(targets)
        for row, target in zip(items, targets):
            correct = row['options'][row['answer_idx']]
            rest = [o for j, o in enumerate(row['options']) if j != row['answer_idx']]
            rng.shuffle(rest)
            row['options'] = rest[:target] + [correct] + rest[target:]
            row['answer_idx'] = target
    return rows


def ordinal(n):
    suffix = 'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f'{n}{suffix}'


def integer_distractors(correct, low, high):
    values = []
    for delta in (-1, 1, 2, -2, 3, -3, 4, -4):
        value = correct + delta
        if low <= value <= high and value != correct and value not in values:
            values.append(value)
        if len(values) == 3:
            return values
    value = low
    while len(values) < 3:
        if value != correct and value not in values:
            values.append(value)
        value += 1
    return values
