"""Isolated WhisperX experiment: continuous speech runs on each full channel.

The cached VAD supplies speech intervals, never reference words or human times.
Only gaps below 400 ms are joined. Padding cannot cross the midpoint of a gap.
ASR is batched by WhisperX; CTC alignment sees only its own run's audio.
"""
from copy import deepcopy
import inspect
import math
import os

import channel_asr as base
from asr_comparison import finite
from contextual_asr import model_evidence
from transcript_candidate import global_alignment

PROFILE = 'channel-runs-v1'
RATE = base.RATE
CHANNELS = base.CHANNELS
POLICY = {'vad_onset': .5, 'vad_offset': .363, 'join_gap_seconds': .4,
          'pad_seconds': .12, 'max_run_seconds': 30, 'max_runs_per_channel': 160,
          'min_speech_seconds': 0., 'reference_hints': False}


def prepare(manifest):
    plan = base.prepare(manifest, expected_profile=PROFILE)
    plan['speech_run_policy'] = deepcopy(POLICY)
    return plan


def windows(speech, frames):
    """Deterministic sample-aligned windows; no silence concatenation or loss.

    VAD may have a small edge overshoot from its receptive field. Clamp that
    edge only; invalid, reversed, overlapping or unordered evidence is rejected.
    Long uninterrupted speech is capped at 30 s, with an explicit split marker.
    """
    if type(frames) is not int or not 0 < frames <= RATE * base.MAX_SECONDS:
        raise ValueError('Invalid full channel duration')
    groups, previous = [], -1.
    for row in speech:
        if not isinstance(row, dict) or set(row) != {'start', 'end'}:
            raise ValueError('Unexpected VAD evidence')
        a, b = row['start'], row['end']
        if (not all(finite(x) for x in (a, b)) or not 0 <= a < b <= frames/RATE + .1
                or a < previous):
            raise ValueError('Invalid VAD timeline')
        previous = b
        start, end = max(0, math.floor(a*RATE)), min(frames, math.ceil(b*RATE))
        if start >= end:
            raise ValueError('VAD region beyond audio')
        if groups and start - groups[-1][1] < round(POLICY['join_gap_seconds']*RATE):
            groups[-1][1] = end
        else:
            groups.append([start, end])
    pad, cap = round(POLICY['pad_seconds']*RATE), round(POLICY['max_run_seconds']*RATE)
    result = []
    for i, (a, b) in enumerate(groups):
        start = max(0, a-pad, (groups[i-1][1]+a)//2 if i else 0)
        end = min(frames, b+pad, (b+groups[i+1][0])//2 if i+1 < len(groups) else frames)
        for part, left in enumerate(range(start, end, cap)):
            result.append({'start_frame': left, 'end_frame': min(left+cap, end),
                           'speech_group': i, 'part': part, 'duration_cap_split': end-start > cap})
    if len(result) > POLICY['max_runs_per_channel']:
        raise ValueError('Too many speech runs for the bounded test')
    return result


def verify_api():
    """CPU CI checks the pinned upstream API before any paid run."""
    from whisperx.vads.pyannote import Binarize
    from whisperx.vads.vad import Vad
    from whisperx.asr import FasterWhisperPipeline
    if not {'onset', 'offset', 'max_duration'} <= set(inspect.signature(Binarize).parameters):
        raise ValueError('VAD API changed')
    if not hasattr(Vad, 'preprocess_audio') or not hasattr(Vad, 'merge_chunks'):
        raise ValueError('VAD wrapper API changed')
    if not {'audio', 'chunk_size'} <= set(inspect.signature(FasterWhisperPipeline.transcribe).parameters):
        raise ValueError('WhisperX API changed')


def callbacks(model, vad):
    """Use the existing public VAD and the unchanged batched WhisperX decoder."""
    import torch
    from whisperx.vads.pyannote import Binarize
    from whisperx.vads.vad import Vad
    verify_api()

    class FixedRuns(Vad):
        def __init__(self, rows):
            super().__init__(POLICY['vad_onset'])
            self.rows = rows

        @staticmethod
        def preprocess_audio(audio):
            return audio

        def __call__(self, audio):
            return [{'start': r['start_frame']/RATE, 'end': r['end_frame']/RATE} for r in self.rows]

        @staticmethod
        def merge_chunks(segments, chunk_size, **kwargs):
            # Deliberately never call Vad.merge_chunks: it joins remote turns.
            return segments

    def detect(audio):
        scores = vad({'waveform': torch.from_numpy(audio).unsqueeze(0), 'sample_rate': RATE})
        annotation = Binarize(onset=POLICY['vad_onset'], offset=POLICY['vad_offset'])(scores)
        return [{'start': float(s.start), 'end': float(s.end)} for s in annotation.get_timeline().support()]

    def transcribe(audio, rows):
        if not rows:
            return {'language': 'ru', 'segments': []}
        original = model.vad_model
        try:
            model.vad_model = FixedRuns(rows)
            return model.transcribe(audio, **base.config_for('wx_ru')['transcribe'])
        finally:
            model.vad_model = original

    return detect, transcribe


def assemble(evidence, frames, confirmation):
    """Replay every offset from retained local evidence; never repair word times."""
    if [r.get('channel') for r in evidence] != list(CHANNELS):
        raise ValueError('Missing channel evidence')
    channel_rows, local_issues = [], []
    for item in evidence:
        planned = windows(item['speech'], frames)
        if [r['window'] for r in item['runs']] != planned:
            raise ValueError('Speech run plan changed')
        raw_all, aligned_all = [], []
        for ri, run in enumerate(item['runs']):
            win, raw, aligned = run['window'], run['raw_local'], run['aligned_local']
            duration = (win['end_frame']-win['start_frame'])/RATE
            if raw.get('language') != 'ru' or len(raw['segments']) != 1:
                raise ValueError('One ASR segment per speech run required')
            seg = raw['segments'][0]
            if seg.get('start') != 0 or seg.get('end') != duration:
                raise ValueError('Alignment must use the complete local run')
            decoder = run['decoder_segment_global']
            if ({k: v for k, v in decoder.items() if k not in ('start', 'end')}
                    != {k: v for k, v in seg.items() if k not in ('start', 'end')}
                    or any(not finite(decoder.get(k)) or abs(decoder[k]-v) > .0006
                           for k, v in (('start', win['start_frame']/RATE), ('end', win['end_frame']/RATE)))):
                raise ValueError('Local text or boundary differs from original decoder evidence')
            # Strict local bounds catch a time escaping its run even if global
            # bounds would still be valid. Evidence remains available for review.
            try:
                base._tokens({'raw': raw, 'aligned': aligned, 'channel': item['channel']}, duration)
            except (ValueError, KeyError, TypeError) as exc:
                local_issues.append({'channel': item['channel'], 'run': ri,
                    'reason': str(exc) if isinstance(exc, ValueError) else 'invalid_local_alignment'})
            offset = win['start_frame']/RATE
            raw_all.extend(global_alignment(raw, offset)['segments'])
            aligned_all.extend(global_alignment(aligned, offset)['segments'])
        channel_rows.append({'channel': item['channel'], 'raw': {'language': 'ru', 'segments': raw_all},
                             'aligned': {'segments': aligned_all}})
    result = base.merge_channels(channel_rows, frames)
    result.update(profile=PROFILE, speech_run_policy=deepcopy(POLICY), run_evidence=model_evidence(deepcopy(evidence)),
                  confirmation=deepcopy(confirmation), full_call_speaker_identity_verified=False)
    if local_issues:
        result['issues'].extend(local_issues)
        result.update(status='review_required', candidate_segments=None)
    return result


def decode_channels(left, right, *, detect, transcribe, align, confirmation):
    if os.getenv('CHANNEL_ASR_TEST') != '1':
        raise ValueError('Explicit isolated channel test flag required')
    frames = len(left)
    if frames != len(right) or not 0 < frames <= RATE*base.MAX_SECONDS:
        raise ValueError('Equal complete channels required')
    base._confirmation(confirmation, frames/RATE)
    evidence = []
    for name, audio in zip(CHANNELS, (left, right)):
        speech = model_evidence(detect(audio))
        planned = windows(speech, frames)
        raw = model_evidence(deepcopy(transcribe(audio, planned)))
        if raw.get('language') != 'ru' or len(raw.get('segments', [])) != len(planned):
            raise ValueError('WhisperX lost or merged a speech run')
        runs = []
        for win, segment in zip(planned, raw['segments']):
            a, b = win['start_frame'], win['end_frame']
            if any(not finite(segment.get(k)) or abs(segment[k]-v) > .0006
                   for k, v in (('start', a/RATE), ('end', b/RATE))):
                raise ValueError('WhisperX changed a run boundary')
            local = {**deepcopy(segment), 'start': 0, 'end': (b-a)/RATE}
            aligned = model_evidence(align([deepcopy(local)], audio[a:b]))
            runs.append({'window': deepcopy(win), 'raw_local': {'language': 'ru', 'segments': [local]},
                         'decoder_segment_global': deepcopy(segment), 'aligned_local': aligned})
        evidence.append({'channel': name, 'speech': speech, 'runs': runs})
    return assemble(evidence, frames, confirmation)


def rebuild(result, plan):
    return assemble(result['run_evidence'], plan['frames_per_channel'], plan['confirmation'])
