"""Test-only extension: review weak recognition even when timing is valid.

CTC confidence is a retry cue, never proof of ASR accuracy. No phrase blacklist,
owner corrections, reference times, extra model or unbounded recovery loop.
Historical channel-speech-v1 remains reproducible with its original policy.
"""
from collections import Counter
from copy import deepcopy
import re

import channel_speech as speech

base, runs, repair = speech.base, speech.runs, speech.repair
PROFILE = 'channel-speech-v2'
SELECTION = {'min_phrase_words': 2, 'low_word_score': .1, 'weak_scored_fraction': .5,
             'min_scored_fraction': 1., 'repeated_phrase_count': 3,
             'reference_hints': False, 'phrase_blacklist': False}
POLICY = {**deepcopy(speech.POLICY), 'recognition_selection': deepcopy(SELECTION)}
callbacks, verify_api = speech.callbacks, speech.verify_api


def prepare(manifest):
    plan = base.prepare(manifest, expected_profile=PROFILE)
    plan.update(speech_run_policy=deepcopy(runs.POLICY), speech_alignment_policy=deepcopy(POLICY),
                recovery_decoder=deepcopy(repair.DECODER))
    return plan


def quality_triggers(control):
    """All scored lexical words participate; missing scores are not low scores."""
    normalize = lambda text: tuple(re.findall(r'\w+', text.casefold().replace('ё', 'е')))
    phrases = Counter(normalize(r['raw_local']['segments'][0]['text'])
                      for c in control['run_evidence'] for r in c['runs'])
    triggers = []
    for channel in control['run_evidence']:
        for index, run in enumerate(channel['runs']):
            phrase = normalize(run['raw_local']['segments'][0]['text'])
            words = [w for s in run['aligned_local']['segments'] for w in s.get('words', [])
                     if normalize(w.get('word', ''))]
            scores = [w['score'] for w in words if base.finite(w.get('score')) and 0 <= w['score'] <= 1]
            if (len(phrase) < SELECTION['min_phrase_words'] or not words
                    or len(scores)/len(words) < SELECTION['min_scored_fraction']):
                continue
            weak = sum(score < SELECTION['low_word_score'] for score in scores)
            if weak / len(scores) < SELECTION['weak_scored_fraction']:
                continue
            reasons = ['weak_phrase_alignment']
            if phrases[phrase] >= SELECTION['repeated_phrase_count']:
                reasons.append('repeated_low_confidence_phrase')
            triggers.append({'channel': channel['channel'], 'run': index,
                             'scored_words': len(scores), 'weak_words': weak,
                             'occurrences': phrases[phrase], 'reasons': sorted(reasons)})
    return triggers


def region_plan(control, frames):
    return repair.region_plan(control, frames,
        extra_seeds=[(t['channel'], t['run']) for t in quality_triggers(control)])


def assemble(control, recoveries, frames, confirmation):
    result = speech.assemble(control, recoveries, frames, confirmation, region_planner=region_plan)
    triggers = quality_triggers(control)
    result.update(profile=PROFILE, speech_alignment_policy=deepcopy(POLICY),
                  quality_trigger_evidence=deepcopy(triggers))
    by_run = {(t['channel'], t['run']): t for t in triggers}
    for row in result['review_spans']:
        trigger = by_run.get((row['channel'], row['run']))
        if trigger:
            row['reasons'] = sorted(set(row['reasons'] + trigger['reasons']))
            if row['hypothesis'] is not None:
                row['status'] = 'hypothesis_requires_review'
    for trigger in triggers:
        result['issues'].extend({'channel': trigger['channel'], 'run': trigger['run'], 'reason': reason}
                                for reason in trigger['reasons'])
    if result['issues']:
        result.update(status='review_required', candidate_segments=None)
    return result


def decode_channels(left, right, *, detect, transcribe, align, confirmation):
    return speech.decode_with_plan(left, right, detect=detect, transcribe=transcribe, align=align,
        confirmation=confirmation, region_planner=region_plan, assembler=assemble)


def rebuild(result, plan):
    return assemble(result['original'], result['recovery_evidence'], plan['frames_per_channel'], plan['confirmation'])
