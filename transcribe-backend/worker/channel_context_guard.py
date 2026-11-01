"""Offline-prepared local verification of contextual WhisperX recognition.

Test-only route, no model loader or text hints. Historical context-v1 stays intact.
The same decoder gets one bounded, independent acoustic view of uncovered runs
and uncertain negations/numbers. Conflicts never silently replace primary text.
"""
from copy import deepcopy
import os
import re

import channel_context as context
from contextual_asr import model_evidence

RATE, runs, base = context.RATE, context.runs, context.base
PROFILE = 'channel-context-guard-v1'
POLICY = {'context_seconds': 2., 'critical_word_probability': .5,
          'max_checks': 32, 'max_check_seconds': 30., 'max_check_audio_seconds': 180.,
          'max_total_asr_audio_seconds': 600., 'check_passes': 1,
          'neighbour_speech': 'excluded', 'reference_hints': False,
          'automatic_conflict_resolution': False, 'phrase_blacklist': False}
NEGATIONS = frozenset(('не', 'нет', 'ни', 'нельзя', 'без'))
DECODER = deepcopy(context.DECODER)
verify_api = context.verify_api


def prepare(manifest):
    prepared = base.prepare(manifest, expected_profile=PROFILE)
    prepared.update(speech_run_policy=deepcopy(runs.POLICY),
                    context_policy=deepcopy(context.POLICY),
                    context_decoder=deepcopy(DECODER), guard_policy=deepcopy(POLICY))
    return prepared


def normalized(text):
    return tuple(re.findall(r'\w+', (text or '').casefold().replace('ё', 'е')))


def critical(word):
    tokens = normalized(word)
    return any(t in NEGATIONS or any(ch.isdigit() for ch in t) for t in tokens)


def plan(primary):
    """Only machine evidence selects runs. No owner text, row IDs or times."""
    context.validate_result(primary)
    frames = primary['frames_per_channel']
    rows = {(r['channel'], r['run']): r for r in primary['review_spans']}
    selected = []
    for channel in primary['evidence']:
        name = channel['channel']
        windows = runs.windows(channel['speech'], frames)
        for block in channel['blocks']:
            for alignment in block['alignments']:
                index = alignment['run']
                row = rows[name, index]
                reasons = []
                if row['text'] is None:
                    reasons.append('uncovered_speech')
                weak = []
                for si, wi in alignment['word_refs']:
                    word = block['native']['segments'][si]['words'][wi]
                    if critical(word['word']) and word['probability'] < POLICY['critical_word_probability']:
                        weak.append([si, wi])
                if weak:
                    reasons.append('uncertain_critical_word')
                if not reasons:
                    continue
                window = windows[index]
                pad = round(POLICY['context_seconds']*RATE)
                # Keep the target complete. Extra context contains no partial
                # neighbouring utterance and never joins two separate checks.
                a = max(0, window['start_frame']-pad,
                        windows[index-1]['end_frame'] if index else 0)
                b = min(frames, window['end_frame']+pad,
                        windows[index+1]['start_frame'] if index+1 < len(windows) else frames)
                selected.append({'channel': name, 'run': index,
                    'window': {'start_frame': a, 'end_frame': b, 'owned_runs': [index]},
                    'reasons': reasons, 'primary_block_start_frame': block['window']['start_frame'],
                    'critical_word_refs': weak})
    seconds = sum(s['window']['end_frame']-s['window']['start_frame'] for s in selected)/RATE
    if (len(selected) > POLICY['max_checks']
            or any((s['window']['end_frame']-s['window']['start_frame'])/RATE > POLICY['max_check_seconds'] for s in selected)
            or seconds > POLICY['max_check_audio_seconds']
            or seconds+primary['asr_audio_seconds'] > POLICY['max_total_asr_audio_seconds']):
        raise ValueError('Local verification exceeds bounded compute allowance')
    return {'profile': PROFILE, 'policy': deepcopy(POLICY), 'decoder': deepcopy(DECODER),
            'checks': selected, 'check_audio_seconds': seconds,
            'total_asr_audio_seconds': seconds+primary['asr_audio_seconds'],
            'max_extra_asr_calls': len(selected), 'reference_hints': False}


def _check_evidence(primary, planned, native):
    channel = next(c for c in primary['evidence'] if c['channel'] == planned['channel'])
    windows = runs.windows(channel['speech'], primary['frames_per_channel'])
    index = planned['run']
    mapping, notes = context.assign(planned['window'], native, windows)
    raw = context.local_input(windows[index], native, mapping[index])
    return windows[index], mapping[index], notes, raw


def _review_check(primary, planned, item):
    win, refs, notes, raw = _check_evidence(primary, planned, item['native'])
    if (item['alignment_window'] != win or item['word_refs'] != refs
            or item['unassigned'] != notes or item['raw_local'] != raw):
        raise ValueError('Local verification text, timing or ownership changed')
    aligned = item['aligned_local']
    issues = {n['reason'] for n in notes if planned['run'] in n['affected_runs']}
    text = None if raw is None else raw['segments'][0]['text']
    if raw is None:
        if aligned is not None:
            raise ValueError('Alignment without locally recognised text')
        issues.add('local_speech_unresolved')
    else:
        for si, wi in refs:
            s = item['native']['segments'][si]
            w = s['words'][wi]
            if s['no_speech_prob'] > .6:
                issues.add('high_no_speech_probability')
            if s['avg_logprob'] < -1 or w['probability'] < .15:
                issues.add('uncertain_recognition')
            if critical(w['word']) and w['probability'] < POLICY['critical_word_probability']:
                issues.add('uncertain_critical_word')
            if w['end']-w['start'] > 2:
                issues.add('native_timing_unreliable')
        try:
            base._tokens({'raw': raw, 'aligned': aligned, 'channel': planned['channel']},
                         (win['end_frame']-win['start_frame'])/RATE, time_tolerance=.0005)
        except (ValueError, TypeError, KeyError) as exc:
            issues.add(str(exc) if isinstance(exc, ValueError) else 'invalid_alignment')
        if isinstance(aligned, dict) and any(base.finite(w.get('score')) and w['score'] < .1
                for s in aligned.get('segments', []) for w in s.get('words', [])):
            issues.add('low_alignment_score')
    return {'text': text, 'reasons': sorted(issues),
            'aligned_segments': [] if not isinstance(aligned, dict) else
                runs.global_alignment(aligned, win['start_frame']/RATE).get('segments', []),
            'human_verified': False}


def assemble(primary, checks):
    prepared = plan(primary)
    if [c['planned'] for c in checks] != prepared['checks']:
        raise ValueError('Every selected run needs exactly one local check')
    alternatives = {(i['planned']['channel'], i['planned']['run']):
        _review_check(primary, i['planned'], i) for i in checks}
    selected = {(p['channel'], p['run']): p for p in prepared['checks']}
    review = []
    for original in primary['review_spans']:
        key = original['channel'], original['run']
        local = alternatives.get(key)
        row = deepcopy(original)
        row.update(primary_text=original['text'], local_alternative=deepcopy(local),
                   text_source='context', local_check_performed=local is not None)
        if local is not None:
            reasons = set(original['reasons']+selected[key]['reasons']+local['reasons'])
            if original['text'] is None and local['text'] is not None:
                # Preserve a recovered candidate, not a verified transcription.
                row.update(text=local['text'], aligned_segments=deepcopy(local['aligned_segments']),
                    text_source='local_unverified', status='recovered_candidate_requires_review')
                reasons.add('unverified_recovery')
            elif original['text'] is not None and normalized(original['text']) != normalized(local['text']):
                # Neither score nor CTC is a truth oracle; a changed negation
                # must not become a single apparently reliable statement.
                row.update(text=None, aligned_segments=[], text_source=None,
                           status='recognition_conflict')
                reasons.add('recognition_disagreement')
            elif local['text'] is not None:
                row['status'] = 'agreement_requires_review'
                reasons.add('agreement_is_not_verification')
            row['reasons'] = sorted(reasons)
        review.append(row)
    return model_evidence({'profile': PROFILE, 'status': 'review_required',
        'plan': prepared, 'primary': deepcopy(primary), 'checks': deepcopy(checks),
        'review_spans': review, 'candidate_segments': None, 'release_ready': False,
        'quality_validated': False, 'human_quality_validated': False,
        'asr_calls': primary['asr_calls']+len(checks),
        'asr_audio_seconds': prepared['total_asr_audio_seconds']})


def validate_result(result):
    if result != assemble(result['primary'], result['checks']):
        raise ValueError('Guard result differs from retained recognition evidence')


def rebuild(result, source_plan):
    # Bind both passes to the actual complete channel frames and confirmation.
    primary = context.rebuild(result['primary'], source_plan)
    if primary != result['primary']:
        raise ValueError('Primary evidence belongs to different channel input')
    return assemble(primary, result['checks'])


def require_test():
    context.require_test()
    if os.getenv('CHANNEL_CONTEXT_GUARD_TEST') != '1':
        raise ValueError('Explicit local-verification preparation flag required')


def callbacks(model, vad):
    require_test()
    return context.callbacks(model, vad)


def decode_channels(left, right, *, detect, transcribe, align, confirmation):
    """One fresh primary pass and one bounded check phase in the same job."""
    require_test()
    primary = context.decode_channels(left, right, detect=detect, transcribe=transcribe,
                                      align=align, confirmation=confirmation)
    return verify_channels(left, right, primary, transcribe=transcribe, align=align)


def verify_channels(left, right, primary, *, transcribe, align):
    """Dependency-injected future inference, never called by the offline CLI."""
    require_test()
    if len(left) != len(right) or len(left) != primary['frames_per_channel']:
        raise ValueError('Original complete channels required')
    prepared = plan(primary)  # All bounds before the first additional ASR call.
    checks = []
    for selected in prepared['checks']:
        audio = left if selected['channel'] == 'left' else right
        w = selected['window']
        native = model_evidence(deepcopy(transcribe(audio[w['start_frame']:w['end_frame']])))
        win, refs, notes, raw = _check_evidence(primary, selected, native)
        aligned = None if raw is None else model_evidence(align(deepcopy(raw['segments']),
            audio[win['start_frame']:win['end_frame']]))
        checks.append({'planned': deepcopy(selected), 'native': native,
            'alignment_window': win, 'word_refs': refs, 'unassigned': notes,
            'raw_local': raw, 'aligned_local': aligned})
    return assemble(primary, checks)


def prepare_saved(state):
    import channel_pipeline
    channel_pipeline.validate_result(state, state['input_plan'])
    if state['profile'] != context.PROFILE:
        raise ValueError('Expected immutable contextual recognition')
    prepared = plan(state['result'])
    prepared.update(status='prepared_not_run', network_calls=0, paid_jobs=0,
        source_sha256=state['input_plan']['source_sha256'],
        channels=deepcopy(state['input_plan']['channels']),
        saved_primary_for_planning_only=True, cloud_runner_connected=False,
        recognition_performed=False, quality_validated=False)
    return prepared


if __name__ == '__main__':
    import argparse
    from asr_comparison import digest, read_json, write_new
    parser = argparse.ArgumentParser(description='Offline verification plan; no model or network calls.')
    parser.add_argument('--saved-result', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    prepared = prepare_saved(read_json(args.saved_result))
    prepared['saved_result_sha256'] = digest(args.saved_result)
    write_new(args.output, prepared)
    print('Prepared only: no recognition, cloud route or paid job.')
