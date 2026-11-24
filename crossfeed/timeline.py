import re

CLIP_SECONDS = 10.0
FILL = (0.4, 0.7)
OCC = (0.3, 1.2)
EVEN = 0.5
EVENT_SEP = re.compile(r'\s*[;|]\s*')
NUMBERY = re.compile(
    r'\b(?:once|twice|thrice|one|two|three|four|five|six|seven|eight|nine|'
    r'ten|times|seconds?|minutes?)\b', re.I)


def slot_onsets(n, rng, even=EVEN, clip=CLIP_SECONDS):
    weights = [even + (1.0 - even) * rng.random() * n for _ in range(n)]
    total = sum(weights)
    t, out = 0.0, []
    for weight in weights:
        out.append(t)
        t += clip * weight / total
    return out


def fill_spans(onsets, rng, fill=FILL, clip=CLIP_SECONDS):
    out = []
    for i, start in enumerate(onsets):
        room = (onsets[i + 1] if i + 1 < len(onsets) else clip) - start
        if room <= 0:
            raise ValueError(f'onset {start} has no room before the next event')
        out.append((start, start + room * rng.uniform(*fill)))
    return out


def occ_spans(onsets, rng, occ=OCC, fill=FILL, clip=CLIP_SECONDS):
    out = []
    for i, start in enumerate(onsets):
        room = (onsets[i + 1] if i + 1 < len(onsets) else clip) - start
        if room <= 0:
            raise ValueError(f'onset {start} has no room before the next event')
        out.append((start, start + min(rng.uniform(*occ), fill[1] * room)))
    return out


def packed_spans(durations, rng, clip=CLIP_SECONDS):
    slack = clip - sum(durations)
    if slack < 0:
        raise ValueError(f'durations sum to {sum(durations):.2f}s > {clip}s')
    weights = [rng.random() + 1e-6 for _ in range(len(durations) + 1)]
    gaps = [slack * w / sum(weights) for w in weights]
    out, t = [], gaps[0]
    for i, duration in enumerate(durations):
        out.append((t, t + duration))
        t += duration + gaps[i + 1]
    return out


def snap(segments, decimals, clip=CLIP_SECONDS):
    step = 10.0 ** -decimals
    out = []
    for start, end, event in segments:
        a = round(max(0.0, start), decimals)
        b = min(round(min(end, clip), decimals), clip)
        if b <= a:
            a = round(max(0.0, b - step), decimals)
            if b <= a:
                b = round(min(clip, a + step), decimals)
        if not 0.0 <= a < b <= clip:
            raise ValueError(f'bad span ({a}, {b}) for {event!r}')
        out.append((a, b, event))
    return sorted(out)


def render(segments, decimals):
    return ' '.join(f'From {start:.{decimals}f}s to {end:.{decimals}f}s, {event}.'
                    for start, end, event in segments)


def _uncapitalise(phrase):
    head = phrase.split(' ', 1)[0]
    if len(head) > 1 and head.isupper():
        return phrase
    return phrase[0].lower() + phrase[1:]


def split_events(body, lo=2, hi=4):
    parts = [re.sub(r'\s+', ' ', p).strip().strip('.,;').strip() for p in EVENT_SEP.split(body or '')]
    parts = [_uncapitalise(p) for p in parts if p]
    if not lo <= len(parts) <= hi:
        return None
    if any(len(p.split()) > 8 or any(c.isdigit() for c in p) or NUMBERY.search(p) for p in parts):
        return None
    return parts


def sample_segments(events, shape, rng, fill=FILL, even=EVEN, dur_range=None, clip=CLIP_SECONDS):
    if shape == 'durations':
        lo, hi = dur_range or (0.5, 3.0)
        durations = [rng.uniform(lo, hi) for _ in events]
        if sum(durations) - clip * 0.9 > 0:
            durations = [d * (clip * 0.9 / sum(durations)) for d in durations]
        spans = packed_spans(durations, rng, clip)
        return [(a, b, event) for (a, b), event in zip(spans, events)]
    if shape == 'overlap':
        onsets = sorted(rng.uniform(0.0, clip * 0.8) for _ in events)
        return [(start, start + (clip - start) * rng.uniform(*fill), event)
                for start, event in zip(onsets, events)]
    onsets = slot_onsets(len(events), rng, even, clip)
    spans = fill_spans(onsets, rng, fill, clip)
    return [(a, b, event) for (a, b), event in zip(spans, events)]


def native_from_pool_item(body, shape, rng, decimals=1, **kwargs):
    events = split_events(body)
    if not events:
        return None
    try:
        segments = sample_segments(events, shape, rng, **kwargs)
        return render(snap(segments, decimals, kwargs.get('clip', CLIP_SECONDS)), decimals)
    except ValueError:
        return None


def _phrase(label):
    return label.strip().split(',')[0].strip().lower()


def segments_from_audiotime(event, mode, rng, fill=FILL, occ=OCC, clip=CLIP_SECONDS):
    if not event:
        raise ValueError('empty event')
    items = [(_phrase(label), values) for label, values in event.items()]
    if any(not phrase for phrase, _ in items):
        raise ValueError(f'unusable event label in {list(event)}')
    if mode == 'spans':
        return [(float(span[0]), float(span[1]), phrase) for phrase, spans in items for span in spans]
    if mode == 'durations':
        durations = [float(values[0]) for _, values in items]
        spans = packed_spans(durations, rng, clip)
        return [(a, b, phrase) for (a, b), (phrase, _) in zip(spans, items)]
    if mode == 'onsets':
        segments = []
        for phrase, onsets in items:
            spans = occ_spans([float(o) for o in onsets], rng, occ, fill, clip)
            segments += [(a, b, phrase) for a, b in spans]
        return segments
    raise ValueError(f'unknown mode {mode!r}')
