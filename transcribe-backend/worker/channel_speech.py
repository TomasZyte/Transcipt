"""Test-only contextual WhisperX decoding, CTC on original speech runs.

ASR context and alignment audio are deliberately separate. Native timestamps
only propose ownership; ambiguous or uncovered speech is never silently lost.
"""
from copy import deepcopy

import channel_repair as repair
from contextual_asr import model_evidence

base, runs, RATE = repair.base, repair.runs, repair.RATE
PROFILE = 'channel-speech-v1'
POLICY = {'alignment_audio': 'complete_original_vad_run', 'context_policy': deepcopy(repair.POLICY),
          'assignment': 'unique_intersected_run', 'manual_times': False, 'reference_hints': False}
callbacks, verify_api = repair.callbacks, repair.verify_api


def prepare(manifest):
    plan = base.prepare(manifest, expected_profile=PROFILE)
    plan.update(speech_run_policy=deepcopy(runs.POLICY), speech_alignment_policy=deepcopy(POLICY),
                recovery_decoder=deepcopy(repair.DECODER))
    return plan


def assign(region, native, original):
    """Never split decoded text among runs using a guessed word boundary."""
    if native.get('language') != 'ru' or not isinstance(native.get('segments'), list):
        raise ValueError('Invalid contextual recognition')
    mapping = {j: [] for j in region['replaced_runs']}
    unassigned = []
    count = region['end_frame']-region['start_frame']
    for si, segment in enumerate(native['segments']):
        # Validate native range but do not use its padded range for alignment.
        repair.alignment_window(segment, count)
        a, b = segment['start']+region['start_frame']/RATE, segment['end']+region['start_frame']/RATE
        eligible = [j for j in mapping if original['runs'][j]['window']['start_frame']/RATE < b
                    and original['runs'][j]['window']['end_frame']/RATE > a]
        if len(eligible) == 1:
            mapping[eligible[0]].append(si)
        else:
            unassigned.append({'segment': si, 'intersected_runs': eligible,
                               'reason': 'ambiguous_speech_assignment' if eligible else 'context_only_hypothesis'})
    return mapping, unassigned


def local_input(original, index, native, assigned):
    window = original['runs'][index]['window']
    duration = (window['end_frame']-window['start_frame'])/RATE
    text = ''.join(native['segments'][i]['text'] for i in assigned)
    return {'language': 'ru', 'segments': [{'start': 0., 'end': duration, 'text': text}]} if text.strip() else None


def assemble(control, recoveries, frames, confirmation, *, region_planner=repair.region_plan):
    if control != runs.assemble(control['run_evidence'], frames, confirmation):
        raise ValueError('Original speech evidence changed')
    planned = region_planner(control, frames)
    if [r['region'] for r in recoveries] != planned:
        raise ValueError('Recovery regions differ from automatic plan')
    channels, issues, review = [], [], []
    for original in control['run_evidence']:
        channel = original['channel']
        replacements = {}
        for ri, recovery in enumerate(r for r in recoveries if r['region']['channel'] == channel):
            region, native = recovery['region'], recovery['native']
            mapping, unassigned = assign(region, native, original)
            if recovery['unassigned'] != unassigned or [r['run'] for r in recovery['alignments']] != list(mapping):
                raise ValueError('Speech ownership changed or a run was dropped')
            issues.extend({'channel': channel, 'region': ri, **x} for x in unassigned)
            for alignment in recovery['alignments']:
                index = alignment['run']
                window = original['runs'][index]['window']
                raw = local_input(original, index, native, mapping[index])
                if (alignment['window'] != window or alignment['source_segments'] != mapping[index]
                        or alignment['raw_local'] != raw):
                    raise ValueError('CTC must use the complete original speech run and exact decoder text')
                reasons = []
                if raw is None:
                    if alignment['aligned_local'] is not None:
                        raise ValueError('Alignment supplied without recognition')
                    reasons.append('unresolved_speech')
                else:
                    duration = (window['end_frame']-window['start_frame'])/RATE
                    try:
                        base._tokens({'channel': channel, 'raw': raw, 'aligned': alignment['aligned_local']},
                                     duration, time_tolerance=.0005)
                    except (ValueError, KeyError, TypeError) as exc:
                        reasons.append(str(exc) if isinstance(exc, ValueError) else 'invalid_speech_alignment')
                    old_text = original['runs'][index]['raw_local']['segments'][0]['text']
                    if old_text.split() != raw['segments'][0]['text'].split():
                        reasons.append('recognition_disagreement')
                    for si in mapping[index]:
                        s = native['segments'][si]
                        p, lp = s.get('no_speech_prob'), s.get('avg_logprob')
                        if not base.finite(p) or not 0 <= p <= 1 or not base.finite(lp):
                            raise ValueError('Missing recognition confidence')
                        if p > repair.POLICY['no_speech_review_threshold'] or lp < repair.POLICY['logprob_review_threshold']:
                            reasons.append('uncertain_recognition')
                        try:
                            base._tokens({'channel': channel, 'raw': {'language': 'ru', 'segments': [s]},
                                          'aligned': {'segments': [s]}},
                                         (region['end_frame']-region['start_frame'])/RATE, time_tolerance=.0005)
                        except (ValueError, KeyError, TypeError):
                            reasons.append('native_timing_unreliable')
                    replacements[index] = (raw, alignment['aligned_local'])
                reasons = sorted(set(reasons))
                issues.extend({'channel': channel, 'region': ri, 'run': index, 'reason': reason} for reason in reasons)
                review.append({'channel': channel, 'run': index, 'startSec': window['start_frame']/RATE,
                    'endSec': window['end_frame']/RATE, 'timing_kind': 'vad_speech_window',
                    'status': 'unresolved_speech' if raw is None else 'hypothesis_requires_review' if reasons else 'aligned_unverified',
                    'hypothesis': None if raw is None else raw['segments'][0]['text'],
                    'reasons': reasons, 'human_verified': False})
        raw_all, aligned_all = [], []
        for index, row in enumerate(original['runs']):
            # An unresolved run keeps its original evidence for audit. It is
            # explicitly unresolved in review_spans and blocks final export.
            raw, aligned = replacements.get(index, (row['raw_local'], row['aligned_local']))
            offset = row['window']['start_frame']/RATE
            raw_all.extend(runs.global_alignment(raw, offset)['segments'])
            aligned_all.extend(runs.global_alignment(aligned, offset)['segments'])
        channels.append({'channel': channel, 'raw': {'language': 'ru', 'segments': raw_all},
                         'aligned': {'segments': aligned_all}})
    result = base.merge_channels(channels, frames, time_tolerance=.0005)
    result.update(profile=PROFILE, original=deepcopy(control), recovery_evidence=deepcopy(recoveries),
                  speech_alignment_policy=deepcopy(POLICY), recovery_decoder=deepcopy(repair.DECODER),
                  review_spans=review, confirmation=deepcopy(confirmation), full_call_speaker_identity_verified=False)
    result['issues'].extend(issues)
    if result['issues']: result.update(status='review_required', candidate_segments=None)
    return model_evidence(result)


def decode_channels(left, right, *, detect, transcribe, align, confirmation):
    return decode_with_plan(left, right, detect=detect, transcribe=transcribe, align=align,
                            confirmation=confirmation, region_planner=repair.region_plan,
                            assembler=assemble)


def decode_with_plan(left, right, *, detect, transcribe, align, confirmation, region_planner, assembler):
    """Shared execution; only code selects a planner, never an API payload."""
    original = runs.decode_channels(left, right, detect=detect, transcribe=transcribe,
                                    align=align, confirmation=confirmation)
    recoveries = []
    for region in region_planner(original, len(left)):
        channel = next(r for r in original['run_evidence'] if r['channel'] == region['channel'])
        audio = left if region['channel'] == 'left' else right
        native = model_evidence(deepcopy(transcribe.recover(audio[region['start_frame']:region['end_frame']])))
        mapping, unassigned = assign(region, native, channel)
        alignments = []
        for index, assigned in mapping.items():
            window = channel['runs'][index]['window']
            raw = local_input(channel, index, native, assigned)
            aligned = None if raw is None else model_evidence(align(deepcopy(raw['segments']),
                audio[window['start_frame']:window['end_frame']]))
            alignments.append({'run': index, 'window': deepcopy(window), 'source_segments': assigned,
                               'raw_local': raw, 'aligned_local': aligned})
        recoveries.append({'region': region, 'native': native, 'unassigned': unassigned, 'alignments': alignments})
    return assembler(original, recoveries, len(left), confirmation)


def rebuild(result, plan):
    return assemble(result['original'], result['recovery_evidence'], plan['frames_per_channel'], plan['confirmation'])
