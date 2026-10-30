"""Model-independent candidate policy: pause boundaries and honest speaker attribution.

No sample names, reference phrases, expected speaker counts or role labels enter
these decisions. The policy is not enabled by the production entry point.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import math

from contextual_asr import model_evidence
from transcript_segments import build_segments
from full_pipeline import srt

RATE = 16000
UNKNOWN = 'SPEAKER_UNKNOWN'
POLICY = {'min_pause_seconds': .35, 'min_window_seconds': 2.,
          'max_window_seconds': 15., 'max_word_seconds': 2.,
          'speaker_min_coverage': .65, 'speaker_min_margin': .2}


def finite(x):
    return type(x) in (int, float) and math.isfinite(x)


def validate_timeline(timeline, duration):
    if not isinstance(timeline, list):
        raise ValueError('Expected a speaker timeline')
    for row in timeline:
        if (not isinstance(row, dict) or not all(finite(row.get(k)) for k in ('start', 'end'))
                or not 0 <= row['start'] < row['end'] <= duration + .05
                or not isinstance(row.get('speaker'), str) or not row['speaker']):
            raise ValueError('Invalid speaker timeline')


def merge_intervals(intervals):
    merged = []
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def pause_windows(timeline, frames):
    """Partition every frame once, cutting at mid-pause where possible.

    Fixed cuts are explicitly reported; no padding or duplicate ownership is
    hidden in the final transcript. Short tail is included in the final window.
    """
    if type(frames) is not int or not 0 < frames <= RATE * 7200:
        raise ValueError('Expected bounded mono 16 kHz audio')
    duration = frames / RATE
    validate_timeline(timeline, duration)
    active = merge_intervals((r['start'], min(duration, r['end'])) for r in timeline)
    pauses = [round((left[1] + right[0]) * RATE / 2)
              for left, right in zip(active, active[1:])
              if right[0] - left[1] >= POLICY['min_pause_seconds']]
    minimum, maximum = int(RATE * POLICY['min_window_seconds']), int(RATE * POLICY['max_window_seconds'])
    boundaries, reasons = [0], []
    while frames - boundaries[-1] > maximum:
        start = boundaries[-1]
        candidates = [p for p in pauses if start + minimum <= p <= start + maximum and frames - p >= minimum]
        end = candidates[-1] if candidates else start + maximum
        if not candidates and frames - end < minimum:
            end = (start + frames) // 2
        boundaries.append(end)
        reasons.append('pause' if candidates else 'forced_maximum')
    # Additional natural pauses create shorter alignment contexts, but never
    # tiny isolated fragments. This depends only on the acoustic timeline.
    coarse = boundaries + [frames]
    result = []
    for index, (left, right) in enumerate(zip(coarse, coarse[1:])):
        cuts = [left]
        for p in pauses:
            if p - cuts[-1] >= minimum and right - p >= minimum:
                cuts.append(p)
        cuts.append(right)
        for a, b in zip(cuts, cuts[1:]):
            result.append({'index': len(result), 'start_frame': a, 'end_frame': b,
                           'end_reason': 'end_of_file' if b == frames else
                           ('pause' if b in pauses else 'forced_maximum')})
    return result


def global_alignment(aligned, offset):
    result = model_evidence(deepcopy(aligned))
    for segment in result.get('segments', []):
        for row in [segment] + segment.get('words', []):
            for key in ('start', 'end'):
                if finite(row.get(key)):
                    row[key] += offset
    # A separate word_segments list may alias the same dicts. Rebuild it once
    # from the canonical segments so no timestamp gets a second offset.
    result['word_segments'] = [w for segment in result.get('segments', []) for w in segment.get('words', [])]
    return result


def _plain(segments):
    return ''.join(''.join(s['text'].split()) for s in segments)


def attribute(raw, aligned, timeline, duration):
    """Keep uncertain attribution explicit, without overwriting model evidence."""
    validate_timeline(timeline, duration)
    raw, aligned = model_evidence(raw), model_evidence(aligned)
    if _plain(raw['segments']) != _plain(aligned['segments']):
        raise ValueError('Alignment changed ASR text')
    original = deepcopy(aligned)
    issues = Counter()
    for si, segment in enumerate(aligned['segments']):
        segment['speaker'] = UNKNOWN
        start, end = segment.get('start'), segment.get('end')
        if not all(finite(v) for v in (start, end)) or not 0 <= start < end <= duration + .05:
            raise ValueError('Invalid alignment segment bounds')
        previous = start
        if not segment.get('words'):
            issues['no_aligned_words'] += 1
        for word in segment.get('words', []):
            word['speaker'] = UNKNOWN
            a, b = word.get('start'), word.get('end')
            reason = None
            if not all(finite(v) for v in (a, b)) or not start <= a < b <= end or a < previous:
                reason = 'unreliable_word_time'
            else:
                previous = a
                if b - a > POLICY['max_word_seconds']:
                    reason = 'long_word'
            if reason:
                issues[reason] += 1
                continue
            coverage = defaultdict(list)
            for row in timeline:
                low, high = max(a, row['start']), min(b, row['end'])
                if low < high:
                    coverage[row['speaker']].append((low, high))
            ranked = sorted(((sum(y - x for x, y in merge_intervals(parts)), label)
                             for label, parts in coverage.items()), reverse=True)
            best = ranked[0][0] / (b - a) if ranked else 0.
            second = ranked[1][0] / (b - a) if len(ranked) > 1 else 0.
            if best < POLICY['speaker_min_coverage']:
                issues['insufficient_speaker_coverage'] += 1
            elif best - second < POLICY['speaker_min_margin']:
                issues['ambiguous_speaker_overlap'] += 1
            else:
                word['speaker'] = ranked[0][1]
    segments, export_stats = build_segments(aligned['segments'])
    if _plain(raw['segments']) != _plain(segments):
        raise ValueError('Export lost ASR text')
    # Reject zero-length SRT cues rather than fabricating precise word times.
    if any(s['startSec'] >= s['endSec'] or s['startSec'] < 0 or s['endSec'] > duration + .06 for s in segments):
        raise ValueError('Unusable export timestamps')
    issues.update(export_stats['fallbackReasons'])
    return {'raw': raw, 'aligned': original, 'attributed': aligned, 'timeline': deepcopy(timeline),
            'segments': segments, 'rawText': '\n\n'.join(s['speaker'] + ': ' + s['text'] for s in segments),
            'srtContent': srt(segments), 'rawAsrText': ' '.join(s['text'].strip() for s in raw['segments']),
            'quality': {'status': 'review_required' if issues else 'automated_checks_passed',
                        'issues': dict(issues), 'human_quality_validated': False},
            'policy': deepcopy(POLICY), 'exportStats': export_stats}
