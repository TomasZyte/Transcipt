"""Bounded contextual recovery of invalid WhisperX speech-run alignments.

No reference text/times, phrase blacklist, word stretching or invented timing.
The original run evidence survives alongside every alternative hypothesis.
"""
from copy import deepcopy
from dataclasses import asdict
import math

import channel_runs as runs
from contextual_asr import model_evidence
from timed_turns import DECODER as TIMED_DECODER, verify_decoder_api

base = runs.base
RATE = runs.RATE
PROFILE = 'channel-repair-v1'
POLICY = {'context_seconds': 2., 'alignment_pad_seconds': .1,
          'max_regions': 8, 'max_region_seconds': 30., 'max_total_seconds': 60.,
          'reference_hints': False, 'no_speech_review_threshold': .6,
          'logprob_review_threshold': -1.}
# Retain hypotheses even when silence is likely: exclusion is never silent.
DECODER = {**TIMED_DECODER, 'no_speech_threshold': None,
           'log_prob_threshold': None, 'compression_ratio_threshold': None}


def prepare(manifest):
    plan = base.prepare(manifest, expected_profile=PROFILE)
    plan.update(speech_run_policy=deepcopy(runs.POLICY), repair_policy=deepcopy(POLICY),
                recovery_decoder=deepcopy(DECODER))
    return plan


def verify_api():
    runs.verify_api()
    verify_decoder_api()
    import inspect
    from whisperx.asr import WhisperModel
    if not set(DECODER) <= set(inspect.signature(WhisperModel.transcribe).parameters):
        raise ValueError('Recovery decoder API changed')


def callbacks(model, vad):
    verify_api()
    detect, transcribe = runs.callbacks(model, vad)
    def recover(audio):
        segments, info = model.model.transcribe(audio, **deepcopy(DECODER))
        return {'language': info.language, 'segments': [asdict(s) for s in segments]}
    # One loaded WhisperX model serves both passes. No extra model/download.
    transcribe.recover = recover
    return detect, transcribe


def region_plan(control, frames):
    """Expand failed local runs, including intersected neighbours in full."""
    result = []
    for item in control['run_evidence']:
        rows = item['runs']
        seeds = []
        for i, row in enumerate(rows):
            duration = (row['window']['end_frame']-row['window']['start_frame'])/RATE
            try:
                base._tokens({'channel': item['channel'], 'raw': row['raw_local'],
                              'aligned': row['aligned_local']}, duration, time_tolerance=.0005)
            except (ValueError, TypeError, KeyError):
                seeds.append(i)
        areas = []
        for i in seeds:
            w = rows[i]['window']
            a = max(0, w['start_frame']-round(POLICY['context_seconds']*RATE))
            b = min(frames, w['end_frame']+round(POLICY['context_seconds']*RATE))
            # Fixed point includes entire neighbours, never just half a word.
            while True:
                covered = [j for j, r in enumerate(rows)
                           if r['window']['start_frame'] < b and r['window']['end_frame'] > a]
                left = min([a] + [rows[j]['window']['start_frame'] for j in covered])
                right = max([b] + [rows[j]['window']['end_frame'] for j in covered])
                if (left, right) == (a, b): break
                a, b = left, right
            areas.append([a, b])
        merged = []
        for a, b in sorted(areas):
            if merged and a <= merged[-1][1]: merged[-1][1] = max(b, merged[-1][1])
            else: merged.append([a, b])
        for a, b in merged:
            result.append({'channel': item['channel'], 'start_frame': a, 'end_frame': b,
                           'replaced_runs': [j for j, r in enumerate(rows)
                               if r['window']['start_frame'] < b and r['window']['end_frame'] > a],
                           'trigger_runs': [j for j in seeds if a <= rows[j]['window']['start_frame'] < b]})
    if (len(result) > POLICY['max_regions'] or any((r['end_frame']-r['start_frame'])/RATE
            > POLICY['max_region_seconds'] for r in result)
            or sum(r['end_frame']-r['start_frame'] for r in result)/RATE > POLICY['max_total_seconds']):
        raise ValueError('Recovery exceeds bounded compute allowance')
    return result


def alignment_window(segment, frames):
    """Keep native timestamps as evidence; CTC sees the bounded local phrase."""
    a, b = segment.get('start'), segment.get('end')
    if (not all(base.finite(x) for x in (a, b)) or not 0 <= a < b <= frames/RATE + .02
            or not isinstance(segment.get('text'), str) or not segment['text'].strip()):
        raise ValueError('Invalid native phrase boundary')
    pad = POLICY['alignment_pad_seconds']
    left, right = max(0, math.floor((a-pad)*RATE)), min(frames, math.ceil((b+pad)*RATE))
    if left >= right: raise ValueError('Native phrase outside recovery audio')
    return {'start_frame': left, 'end_frame': right}


def assemble(control, recoveries, frames, confirmation, *, assembly_version=2):
    if type(assembly_version) is not int or assembly_version not in (1, 2):
        raise ValueError('Unknown recovery assembly version')
    checked = runs.assemble(control['run_evidence'], frames, confirmation)
    if control != checked:
        raise ValueError('Original run evidence changed')
    planned = region_plan(control, frames)
    if [r['region'] for r in recoveries] != planned:
        raise ValueError('Recovery must cover every automatically selected region exactly once')
    channels, issues, changes = [], [], []
    for channel in runs.CHANNELS:
        original = next(r for r in control['run_evidence'] if r['channel'] == channel)
        selected = [r for r in recoveries if r['region']['channel'] == channel]
        replaced = {i for r in selected for i in r['region']['replaced_runs']}
        pieces = [(r['window']['start_frame'], r['raw_local'], r['aligned_local'])
                  for i, r in enumerate(original['runs']) if i not in replaced]
        for ri, recovery in enumerate(selected):
            region, native = recovery['region'], recovery['native']
            count = region['end_frame']-region['start_frame']
            if native.get('language') != 'ru' or len(recovery['phrases']) != len(native['segments']):
                raise ValueError('Recovery language or phrase count changed')
            if not native['segments']:
                issues.append({'channel': channel, 'region': ri, 'reason': 'recovery_without_transcript'})
            before = ''.join(original['runs'][i]['raw_local']['segments'][0]['text'] for i in region['replaced_runs'])
            after = ''.join(s['text'] for s in native['segments'])
            changes.append({'channel': channel, 'region': ri, 'before': before, 'after': after,
                            'text_changed': before.split() != after.split(), 'human_verified': False})
            # CTC is forced alignment, not an independent check of recognition.
            # A fluent alternative with good-looking CTC times can still be a
            # hallucination. Preserve disagreement and native timing failures.
            if assembly_version == 2 and before.split() != after.split():
                issues.append({'channel': channel, 'region': ri, 'reason': 'recognition_disagreement'})
            covered = []
            for si, (segment, phrase) in enumerate(zip(native['segments'], recovery['phrases'])):
                win = alignment_window(segment, count)
                duration = (win['end_frame']-win['start_frame'])/RATE
                local = {**deepcopy(segment), 'start': 0., 'end': duration}
                raw = {'language': 'ru', 'segments': [local]}
                if phrase['window'] != win or phrase['raw_local'] != raw:
                    raise ValueError('Recovered text or crop differs from native evidence')
                if assembly_version == 2:
                    try:
                        base._tokens({'channel': channel,
                            'raw': {'language': 'ru', 'segments': [segment]},
                            'aligned': {'segments': [segment]}}, count/RATE, time_tolerance=.0005)
                    except (ValueError, TypeError, KeyError) as exc:
                        issues.append({'channel': channel, 'region': ri, 'phrase': si,
                            'reason': 'native_' + (str(exc) if isinstance(exc, ValueError) else 'invalid_word_evidence')})
                try:
                    tokens = base._tokens({'channel': channel, 'raw': raw, 'aligned': phrase['aligned_local']},
                                          duration, time_tolerance=.0005)
                    offset = (region['start_frame']+win['start_frame'])/RATE
                    covered.extend((offset+t['startSec'], offset+t['endSec']) for t in tokens)
                except (ValueError, TypeError, KeyError) as exc:
                    issues.append({'channel': channel, 'region': ri, 'phrase': si,
                                   'reason': str(exc) if isinstance(exc, ValueError) else 'invalid_recovery_alignment'})
                p, lp = segment.get('no_speech_prob'), segment.get('avg_logprob')
                if not base.finite(p) or not 0 <= p <= 1 or not base.finite(lp):
                    raise ValueError('Missing decoder confidence evidence')
                if p > POLICY['no_speech_review_threshold'] or lp < POLICY['logprob_review_threshold']:
                    issues.append({'channel': channel, 'region': ri, 'phrase': si,
                                   'reason': 'uncertain_recognition'})
                pieces.append((region['start_frame']+win['start_frame'], raw, phrase['aligned_local']))
            if assembly_version == 2:
                # Context can decode only a neighbour and omit the failed run.
                # Do not report that omission as a successful timing repair.
                for j in region['replaced_runs']:
                    w = original['runs'][j]['window']
                    if not any(a < w['end_frame']/RATE and b > w['start_frame']/RATE for a, b in covered):
                        issues.append({'channel': channel, 'region': ri, 'run': j,
                                       'reason': 'recovery_does_not_cover_original_run'})
        raw_all, aligned_all = [], []
        for offset, raw, aligned in sorted(pieces, key=lambda p: p[0]):
            raw_all.extend(runs.global_alignment(raw, offset/RATE)['segments'])
            aligned_all.extend(runs.global_alignment(aligned, offset/RATE)['segments'])
        channels.append({'channel': channel, 'raw': {'language': 'ru', 'segments': raw_all},
                         'aligned': {'segments': aligned_all}})
    result = base.merge_channels(channels, frames, time_tolerance=.0005)
    result.update(profile=PROFILE, original=deepcopy(control), recovery_evidence=deepcopy(recoveries),
                  repair_policy=deepcopy(POLICY), recovery_decoder=deepcopy(DECODER), changes=changes,
                  confirmation=deepcopy(confirmation), full_call_speaker_identity_verified=False)
    result['issues'].extend(issues)
    if assembly_version == 2:
        result['assembly_version'] = 2
    if result['issues']: result.update(status='review_required', candidate_segments=None)
    return model_evidence(result)


def decode_channels(left, right, *, detect, transcribe, align, confirmation):
    original = runs.decode_channels(left, right, detect=detect, transcribe=transcribe,
                                    align=align, confirmation=confirmation)
    planned = region_plan(original, len(left))
    recoveries = []
    for region in planned:
        audio = left if region['channel'] == 'left' else right
        chunk = audio[region['start_frame']:region['end_frame']]
        native = model_evidence(deepcopy(transcribe.recover(chunk)))
        if native.get('language') != 'ru': raise ValueError('Unexpected recovery language')
        phrases = []
        for segment in native['segments']:
            win = alignment_window(segment, len(chunk))
            a, b = win['start_frame'], win['end_frame']
            local = {**deepcopy(segment), 'start': 0., 'end': (b-a)/RATE}
            aligned = model_evidence(align([deepcopy(local)], chunk[a:b]))
            phrases.append({'window': win, 'raw_local': {'language': 'ru', 'segments': [local]},
                            'aligned_local': aligned})
        recoveries.append({'region': region, 'native': native, 'phrases': phrases})
    return assemble(original, recoveries, len(left), confirmation)


def rebuild(result, plan):
    return assemble(result['original'], result['recovery_evidence'], plan['frames_per_channel'], plan['confirmation'],
                    assembly_version=result.get('assembly_version', 1))
