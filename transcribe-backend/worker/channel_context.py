"""Offline-prepared WhisperX experiment: contextual ASR, speech-bounded CTC.

One ASR pass over contiguous same-channel blocks, including original pauses.
No production/cloud route, model loader, download or submission in this module.
Every run has one owner block; context-only words are retained as evidence.
"""
from copy import deepcopy
import os

import channel_repair as repair
from contextual_asr import model_evidence

runs, base, RATE = repair.runs, repair.base, repair.RATE
PROFILE = 'channel-context-v1'
POLICY = {'core_seconds': 24., 'context_seconds': 3., 'max_block_seconds': 30.,
          'max_blocks_per_channel': 32, 'max_total_audio_seconds': 600.,
          'audio': 'contiguous_original_channel', 'assignment': 'unique_word_run_overlap',
          'alignment_audio': 'complete_original_vad_run', 'reference_hints': False,
          'amplitude_gate': False, 'phrase_blacklist': False, 'retry_passes': 0}
# Keep the v2 decoder fixed to isolate the effect of audio segmentation.
# Retain probable silence for audit. High no_speech is a review flag, not an
# automatic deletion: the known final thanks also had high no_speech evidence.
DECODER = deepcopy(repair.DECODER)


def require_test():
    from channel_pipeline import require_test as require_channel_test
    require_channel_test()
    if os.getenv('CHANNEL_CONTEXT_TEST') != '1':
        raise ValueError('Explicit isolated context preparation flag required')


def callbacks(model, vad):
    """Same already-loaded WhisperX model as v2, without tiny batched crops."""
    require_test()
    detect, transcribe = repair.callbacks(model, vad)
    return detect, transcribe.recover


def blocks(speech, frames):
    windows = runs.windows(speech, frames)
    groups = []
    for index, window in enumerate(windows):
        if (groups and window['end_frame'] - windows[groups[-1][0]]['start_frame']
                <= round(POLICY['core_seconds'] * RATE)):
            groups[-1].append(index)
        else:
            groups.append([index])
    result = []
    for group in groups:
        a, b = windows[group[0]]['start_frame'], windows[group[-1]]['end_frame']
        # A single capped 30 s speech run is kept whole, without padding.
        pad = min(round(POLICY['context_seconds']*RATE),
                  (round(POLICY['max_block_seconds']*RATE)-(b-a))//2)
        if pad < 0:
            raise ValueError('Speech run exceeds context block limit')
        result.append({'start_frame': max(0, a-pad), 'end_frame': min(frames, b+pad),
                       'owned_runs': group})
    if len(result) > POLICY['max_blocks_per_channel']:
        raise ValueError('Too many contextual blocks')
    return result


def plan_channels(speech, frames):
    if set(speech) != set(runs.CHANNELS):
        raise ValueError('Exactly two channel timelines required')
    planned = {name: blocks(speech[name], frames) for name in runs.CHANNELS}
    seconds = sum(b['end_frame']-b['start_frame'] for rows in planned.values() for b in rows)/RATE
    if seconds > POLICY['max_total_audio_seconds']:
        raise ValueError('Context exceeds bounded compute allowance')
    return planned


def assign(block, native, windows):
    """Preserve every decoded word; ambiguous ownership never guesses a split."""
    if native.get('language') != 'ru' or not isinstance(native.get('segments'), list):
        raise ValueError('Invalid contextual recognition')
    mapping = {i: [] for i in block['owned_runs']}
    notes = []
    duration = (block['end_frame']-block['start_frame'])/RATE
    offset = block['start_frame']/RATE
    previous_end = -1.
    for si, segment in enumerate(native['segments']):
        repair.alignment_window(segment, block['end_frame']-block['start_frame'])
        p, lp = segment.get('no_speech_prob'), segment.get('avg_logprob')
        if not base.finite(p) or not 0 <= p <= 1 or not base.finite(lp):
            raise ValueError('Missing native decoder confidence')
        words = segment.get('words')
        if (not isinstance(words, list) or not words
                or any(not isinstance(w, dict) or not isinstance(w.get('word'), str) for w in words)
                or ''.join(''.join(w['word'].split()) for w in words) != ''.join(segment['text'].split())):
            notes.append({'segment': si, 'word': None, 'reason': 'missing_native_word_evidence',
                          'affected_runs': list(mapping)})
            continue
        for wi, word in enumerate(words):
            a, b = word.get('start'), word.get('end')
            probability = word.get('probability')
            if (not all(base.finite(v) for v in (a, b, probability)) or not 0 <= probability <= 1
                    or not 0 <= a < b <= duration + .02 or a < previous_end
                    or a < segment['start']-.02 or b > segment['end']+.02
                    or not word['word'].strip()):
                notes.append({'segment': si, 'word': wi, 'reason': 'invalid_native_word_time',
                              'affected_runs': list(mapping)})
                continue
            previous_end = b
            eligible = [i for i, w in enumerate(windows)
                        if w['start_frame']/RATE < b+offset and w['end_frame']/RATE > a+offset]
            owned = [i for i in eligible if i in mapping]
            if len(eligible) == 1 and owned:
                mapping[owned[0]].append([si, wi])
            else:
                reason = ('ambiguous_word_assignment' if len(eligible) > 1 else
                          'context_only_word' if eligible else 'word_outside_detected_speech')
                notes.append({'segment': si, 'word': wi, 'reason': reason,
                              'affected_runs': owned, 'intersected_runs': eligible})
    return mapping, notes


def local_input(window, native, references):
    text = ''.join(native['segments'][s]['words'][w]['word'] for s, w in references)
    return ({'language': 'ru', 'segments': [{'start': 0.,
            'end': (window['end_frame']-window['start_frame'])/RATE, 'text': text}]}
            if text.strip() else None)


def assemble(evidence, frames, confirmation):
    base._confirmation(confirmation, frames/RATE)
    if [c.get('channel') for c in evidence] != list(runs.CHANNELS):
        raise ValueError('Missing or reordered channel evidence')
    planned = plan_channels({c['channel']: c['speech'] for c in evidence}, frames)
    review, unassigned = [], []
    for channel in evidence:
        name = channel['channel']
        windows = runs.windows(channel['speech'], frames)
        if [b['window'] for b in channel['blocks']] != planned[name]:
            raise ValueError('Context block ownership changed')
        for item in channel['blocks']:
            block, native = item['window'], item['native']
            mapping, notes = assign(block, native, windows)
            if item['unassigned'] != notes or [a['run'] for a in item['alignments']] != list(mapping):
                raise ValueError('Dropped speech run or changed word ownership')
            unassigned.extend({'channel': name, 'block_start_frame': block['start_frame'], **n} for n in notes)
            for alignment in item['alignments']:
                index = alignment['run']
                window = windows[index]
                raw = local_input(window, native, mapping[index])
                if (alignment['window'] != window or alignment['word_refs'] != mapping[index]
                        or alignment['raw_local'] != raw):
                    raise ValueError('Alignment text or audio differs from the owned speech run')
                reasons = {n['reason'] for n in notes if index in n['affected_runs']}
                aligned = alignment['aligned_local']
                if raw is None:
                    if aligned is not None:
                        raise ValueError('Alignment without recognised text')
                    reasons.add('unresolved_speech')
                else:
                    for si, wi in mapping[index]:
                        segment = native['segments'][si]
                        word = segment['words'][wi]
                        if segment['no_speech_prob'] > .6:
                            reasons.add('high_no_speech_probability')
                        if segment['avg_logprob'] < -1 or word['probability'] < .15:
                            reasons.add('uncertain_recognition')
                        if word['end']-word['start'] > 2:
                            reasons.add('native_timing_unreliable')
                    try:
                        base._tokens({'raw': raw, 'aligned': aligned, 'channel': name},
                            (window['end_frame']-window['start_frame'])/RATE, time_tolerance=.0005)
                    except (ValueError, TypeError, KeyError) as exc:
                        reasons.add(str(exc) if isinstance(exc, ValueError) else 'invalid_alignment')
                    if isinstance(aligned, dict) and any(base.finite(w.get('score')) and w['score'] < .1
                            for s in aligned.get('segments', []) for w in s.get('words', [])):
                        reasons.add('low_alignment_score')
                review.append({'id': f'{name}-{index}', 'channel': name, 'run': index,
                    'startSec': window['start_frame']/RATE, 'endSec': window['end_frame']/RATE,
                    'text': None if raw is None else raw['segments'][0]['text'],
                    'reasons': sorted(reasons), 'human_verified': False,
                    'status': 'unresolved_speech' if raw is None else 'hypothesis_requires_review',
                    'aligned_segments': [] if aligned is None else runs.global_alignment(aligned,
                        window['start_frame']/RATE).get('segments', [])})
    review.sort(key=lambda r: (r['startSec'], r['channel'], r['run']))
    return model_evidence({'profile': PROFILE, 'status': 'review_required',
        'frames_per_channel': frames, 'confirmation': deepcopy(confirmation),
        'policy': deepcopy(POLICY), 'decoder': deepcopy(DECODER),
        'speech_run_policy': deepcopy(runs.POLICY), 'evidence': deepcopy(evidence),
        'review_spans': review, 'unassigned': unassigned,
        'asr_calls': sum(len(c['blocks']) for c in evidence),
        'asr_audio_seconds': sum(b['end_frame']-b['start_frame'] for bs in planned.values() for b in bs)/RATE,
        'candidate_segments': None, 'quality_validated': False, 'release_ready': False,
        'full_call_speaker_identity_verified': False})


def decode_channels(left, right, *, detect, transcribe, align, confirmation):
    require_test()
    frames = len(left)
    if len(right) != frames or type(frames) is not int or not 0 < frames <= RATE*base.MAX_SECONDS:
        raise ValueError('Equal complete bounded channels required')
    base._confirmation(confirmation, frames/RATE)
    speech = {name: model_evidence(detect(audio)) for name, audio in zip(runs.CHANNELS, (left, right))}
    planned = plan_channels(speech, frames)  # Fail limits before the first ASR call.
    evidence = []
    for name, audio in zip(runs.CHANNELS, (left, right)):
        windows = runs.windows(speech[name], frames)
        decoded = []
        for block in planned[name]:
            native = model_evidence(deepcopy(transcribe(audio[block['start_frame']:block['end_frame']])))
            mapping, notes = assign(block, native, windows)
            alignments = []
            for index, refs in mapping.items():
                window = windows[index]
                raw = local_input(window, native, refs)
                aligned = None if raw is None else model_evidence(align(deepcopy(raw['segments']),
                    audio[window['start_frame']:window['end_frame']]))
                alignments.append({'run': index, 'window': deepcopy(window), 'word_refs': refs,
                                   'raw_local': raw, 'aligned_local': aligned})
            decoded.append({'window': deepcopy(block), 'native': native,
                            'unassigned': notes, 'alignments': alignments})
        evidence.append({'channel': name, 'speech': speech[name], 'blocks': decoded})
    return assemble(evidence, frames, confirmation)


def validate_result(result):
    rebuilt = assemble(result['evidence'], result['frames_per_channel'], result['confirmation'])
    if rebuilt != result:
        raise ValueError('Context result differs from original evidence')


def prepare_from_saved(state, source_plan):
    """Use saved VAD only to estimate work; never pass saved text to decoding."""
    import channel_pipeline
    channel_pipeline.validate_result(state, state['input_plan'])
    if state['profile'] != 'channel-speech-v2':
        raise ValueError('Expected the retained v2 result')
    for key in ('source_sha256', 'channels', 'frames_per_channel', 'confirmation'):
        if state['input_plan'][key] != source_plan[key]:
            raise ValueError('Saved evidence belongs to different source audio')
    speech = {c['channel']: deepcopy(c['speech']) for c in state['result']['original']['run_evidence']}
    frames = source_plan['frames_per_channel']
    planned = plan_channels(speech, frames)
    return {'profile': PROFILE, 'status': 'prepared_not_run', 'network_calls': 0,
            'paid_jobs': 0, 'cloud_runner_connected': False, 'asr_model': 'WhisperX large-v3',
            'source_sha256': source_plan['source_sha256'], 'channels': deepcopy(source_plan['channels']),
            'frames_per_channel': frames, 'policy': deepcopy(POLICY), 'decoder': deepcopy(DECODER),
            'saved_vad_for_planning_only': speech, 'blocks': planned,
            'estimated_asr_calls': sum(len(bs) for bs in planned.values()),
            'estimated_asr_audio_seconds': sum(b['end_frame']-b['start_frame']
                for bs in planned.values() for b in bs)/RATE,
            'recognition_performed': False, 'quality_validated': False, 'release_ready': False}


if __name__ == '__main__':
    import argparse
    from asr_comparison import digest, read_json, write_new
    parser = argparse.ArgumentParser(description='Prepare local context blocks; never run a model or cloud job.')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--saved-result', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source_plan = base.prepare(args.manifest, expected_profile='channel-speech-v2')
    plan = prepare_from_saved(read_json(args.saved_result), source_plan)
    plan.update(saved_result_sha256=digest(args.saved_result), manifest_sha256=digest(args.manifest))
    write_new(args.output, plan)
    print('Prepared only: no model calls, cloud runner or paid jobs.')
