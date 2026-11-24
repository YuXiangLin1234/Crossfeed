import re

from crossfeed.normalizer import EnglishTextNormalizer

_ENGLISH = EnglishTextNormalizer()

_WRAPPER = re.compile(
    r'^\s*(?:the\s+)?(?:original\s+|spoken\s+)?(?:english\s+)?'
    r'(?:audio|transcription|transcript|transcribed\s+text|content|text)'
    r'\b[^:\n]{0,80}?\s*:\s*'
    r'(?P<open>[\x27\x22‘“])(?P<body>.+?)\s*$',
    flags=re.IGNORECASE | re.DOTALL,
)

_AUDEX_WRAPPER = re.compile(
    r'^\s*'
    r'(?:Language\s*:\s*[A-Za-z][A-Za-z -]{0,39}\.\s*)?'
    r'(?:The\s+)?(?:spoken\s+)?'
    r'(?:content|transcription|transcript)\s+of\s+'
    r'(?:(?:the|this)\s+)?'
    r'(?:input\s+(?:audio|file)|audio|file|input)\s+'
    r'(?:is|reads?)\b[\s:]*'
    r'(?P<open>[\'"‘“])?\s*(?P<body>.*?)\s*$',
    flags=re.IGNORECASE | re.DOTALL,
)

_AUDEX_LABEL_WRAPPER = re.compile(
    r'^\s*(?:(?:The\s+)?language\s+is|Language\s*:)\s*'
    r'[A-Za-z][A-Za-z -]{0,39}\.\s*'
    r'(?:(?:transcription|transcript)\s*:|'
    r'(?:The\s+)?spoken\s+content(?:\s+of\s+(?:the\s+)?audio)?\s+is)\s*'
    r'(?P<open>[\'"‘“])?\s*(?P<body>.*?)\s*$',
    flags=re.IGNORECASE | re.DOTALL,
)

_AUDEX_TRANSLATION_WRAPPER = re.compile(
    r'^\s*Source\s+language\s*:\s*[A-Za-z][A-Za-z -]{0,39}\.\s*'
    r'The\s+spoken\s+content(?:\s+of\s+(?:the\s+)?audio)?\s+is\s+'
    r'[\'"‘“].*?[\'"’”]\s*\.\s*'
    r'Translat(?:ing|ed)\s+(?:it\s+)?to\s+English\s*,?\s*'
    r'(?:the\s+)?content\s+is\s*'
    r'(?P<open>[\'"‘“])?\s*(?P<body>.*?)\s*$',
    flags=re.IGNORECASE | re.DOTALL,
)

_CLOSING = {"'": "'", '"': '"', '‘': '’', '“': '”'}


def extract_transcript(text):
    if not isinstance(text, str):
        return text
    match = _WRAPPER.match(text)
    if match is None:
        return text.strip()
    body = match.group('body').strip()
    closing = _CLOSING[match.group('open')]
    candidate = body[:-1].rstrip() if body.endswith('.') else body
    if candidate.endswith(closing):
        body = candidate[:-1]
    return body.strip()


def extract_audex_transcript(text):
    if not isinstance(text, str):
        return extract_transcript(text)
    match = (_AUDEX_TRANSLATION_WRAPPER.match(text) or _AUDEX_LABEL_WRAPPER.match(text)
             or _AUDEX_WRAPPER.match(text))
    if match is None:
        return extract_transcript(text)
    body = match.group('body').strip()
    opening = match.group('open')
    if opening is not None:
        closing = _CLOSING[opening]
        if body.endswith(closing):
            body = body[:-len(closing)].rstrip()
        else:
            for punctuation in '.!?':
                if body.endswith(closing + punctuation):
                    body = body[:-len(closing + punctuation)].rstrip()
                    break
    return body


def words(text):
    return _ENGLISH(text or '').split()


def edit_distance(reference, hypothesis):
    distances = list(range(len(hypothesis) + 1))
    for i, ref_token in enumerate(reference, 1):
        previous, distances[0] = distances[0], i
        for j, hyp_token in enumerate(hypothesis, 1):
            current = min(distances[j] + 1, distances[j - 1] + 1, previous + (ref_token != hyp_token))
            previous, distances[j] = distances[j], current
    return distances[-1]


def word_error_rate(reference, hypothesis):
    ref = words(reference)
    return edit_distance(ref, words(hypothesis)) / max(len(ref), 1)
