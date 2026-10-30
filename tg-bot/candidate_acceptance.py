"""Offline quality gates using independent references, never inference hints.

Reads a saved candidate result and private evaluation annotations. Does not
import API clients, submit work, alter model output or enable a release.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path


def tokens(value):
    return re.findall(r'\w+', value.casefold().replace('ё', 'е'))


def phrase_matches(words, phrase):
    expanded = [(token, word) for word in words for token in tokens(word.get('word', ''))]
    wanted = tokens(phrase)
    if not wanted:
        raise ValueError('Empty reference phrase')
    return [[pair[1] for pair in expanded[i:i+len(wanted)]]
            for i in range(len(expanded)-len(wanted)+1)
            if [p[0] for p in expanded[i:i+len(wanted)]] == wanted]


def evaluate(result, references):
    if (result.get('status') != 'completed' or result.get('profile') not in ('release-candidate-v1', 'timed-turns-v1', 'acoustic-turns-v1', 'pause-context-v1')
            or references.get('purpose') != 'evaluation_only'):
        raise ValueError('Completed candidate and independent evaluation references required')
    samples = result['samples']
    if len({s['id'] for s in samples}) != len(samples):
        raise ValueError('Duplicate samples')
    if set(references['samples']) != {s['id'] for s in samples}:
        raise ValueError('Every sample needs an evaluation reference')
    rows = []
    for sample in samples:
        ref = references['samples'][sample['id']]
        if ref['audio_sha256'] != sample['sha256']:
            raise ValueError('Reference belongs to different audio')
        for mode, output in sample['outputs'].items():
            words = [w for s in output['attributed']['segments'] for w in s.get('words', [])]
            checks = []
            for phrase in ref.get('phrases', []):
                matches = phrase_matches(words, phrase['text'])
                low, high = phrase['seconds']
                if not 0 <= low < high <= sample['duration_seconds']:
                    raise ValueError('Invalid reference window')
                timed = [match for match in matches if all(
                    type(w.get('start')) in (float, int) and type(w.get('end')) in (float, int)
                    and low <= w['start'] < w['end'] <= high for w in match)]
                by_mode = phrase.get('forbidden_speakers_by_mode')
                if by_mode is not None and set(by_mode) != set(sample['outputs']):
                    raise ValueError('Every variant needs its own speaker label mapping')
                forbidden = set(by_mode[mode] if by_mode is not None else phrase.get('forbidden_speakers', []))
                known = [match for match in matches if all(
                    w.get('speaker') and w['speaker'] != 'SPEAKER_UNKNOWN'
                    and w['speaker'] not in forbidden for w in match)]
                accepted = [match for match in timed if match in known]
                checks.append({'id': phrase['id'], 'text_present': bool(matches),
                               'timing_in_reference_window': bool(timed),
                               'timing_precision': phrase.get('timing_precision', 'unspecified'),
                               'separated_from_forbidden_speakers': bool(known),
                               'observed_matches': [[{k: w.get(k) for k in ('word', 'start', 'end', 'speaker', 'score')}
                                                     for w in match] for match in matches],
                               'pass': bool(accepted)})
            diar_key = result['modes'][mode]['diarization']
            speaker_count = len({r['speaker'] for r in sample['diarization'][diar_key]['regular']})
            expected = ref.get('exact_speaker_count')
            if expected is not None:
                checks.append({'id': 'independent_speaker_count', 'expected': expected,
                               'observed': speaker_count, 'pass': expected == speaker_count})
            # A control transcript is a comparison baseline, not ground truth.
            control = sample['outputs'].get('control30')
            changed = tokens(output['rawAsrText']) != tokens(control['rawAsrText']) if control else None
            rows.append({'sample': sample['id'], 'mode': mode, 'checks': checks,
                         'reference_checks_pass': bool(checks) and all(x['pass'] for x in checks),
                         'text_changed_from_control': changed,
                         'model_issues': output['quality']['issues'],
                         'manual_review_required': True})
    return {'status': 'review_required', 'release_ready': False,
            'reason': 'A bounded benchmark never grants deployment approval or full quality certification',
            'rows': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', required=True, type=Path)
    parser.add_argument('--references', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    data = args.result.read_bytes()
    references = json.loads(args.references.read_text(encoding='utf-8'))
    # Speaker label numbers only have meaning for this exact saved output.
    if references.get('result_sha256') != hashlib.sha256(data).hexdigest():
        raise ValueError('Evaluation labels are not bound to this result')
    report = evaluate(json.loads(data), references)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print('Offline quality gates saved; no network calls or release changes.')


if __name__ == '__main__': main()
