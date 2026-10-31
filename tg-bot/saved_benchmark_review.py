"""Replay saved sound1-4 evidence without loading models or API clients."""
from copy import deepcopy
import importlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'transcribe-backend'/'worker'))
from asr_comparison import digest, read_json, config_for
from channel_review import tokens
from candidate_acceptance import evaluate, phrase_matches
from full_pipeline import format_output
from full_pipeline_review import text_changes
from quality_diagnostics import audit_result
from prepare_full_assets import SOURCES

PROFILES = {
    'full-pipeline-v1': ('full_pipeline', None),
    'context-pipeline-v1': ('context_pipeline', None),
    'release-candidate-v1': ('candidate_pipeline', 'release-candidate-2026-10-30'),
    'timed-turns-v1': ('timed_turns', 'timed-turns-2026-10-30'),
    'acoustic-turns-v1': ('acoustic_turns', 'acoustic-turns-2026-10-30'),
    'pause-context-v1': ('pause_context', 'pause-context-2026-10-30'),
}
IDS = ['sound1', 'sound2', 'sound3', 'sound4']
PHRASES = ['а здесь есть горка', 'папа смотри горка']
REVISION = 'edaa852ec7e145841d8ffdb056a99866b5f0a478'


def provenance(result):
    sources = read_json(SOURCES); runtime = result['runtime']; assets = runtime['asr']
    if result['source_manifest_sha256'] != digest(SOURCES) or assets['revision'] != REVISION:
        raise ValueError('Unexpected model provenance')
    for name, version in config_for('wx_auto')['packages'].items():
        if assets['packages'].get(name) != version:
            raise ValueError('Unexpected WhisperX package version')
    for kind in ('alignment', 'diarization'):
        for key in ('repository', 'revision'):
            if runtime[kind][key] != sources[kind][key]:
                raise ValueError('Unexpected secondary model provenance')


def measure(output, *, timeline=None):
    if 'benchmark' in output:
        words = [w for s in output['benchmark']['alignedSegments'] for w in s.get('words', [])]
        timeline = output['benchmark'].get('diarizationTimeline', {}).get('segments')
        text = output.get('rawAsrText') or ' '.join(s['text'] for s in output['benchmark'].get('asrSegments', output['benchmark']['alignedSegments']))
        issues = audit_result(output)
    else:
        words = [w for s in output['attributed']['segments'] for w in s.get('words', [])]
        text, issues = output['rawAsrText'], output['quality']['issues']
    return {'text': text, 'lexical_word_count': len(tokens(text)),
        'timeline_labels': None if timeline is None else len({s['speaker'] for s in timeline}),
        'export_labels': len({s['speaker'] for s in output['segments']}),
        'known_word_labels': len({w['speaker'] for w in words if w.get('speaker') and w['speaker']!='SPEAKER_UNKNOWN'}),
        'unknown_words': sum(not w.get('speaker') or w['speaker']=='SPEAKER_UNKNOWN' for w in words),
        'long_words': sum(isinstance(w.get('start'), (float, int)) and
                          isinstance(w.get('end'), (float, int)) and w['end']-w['start'] > 2 for w in words),
        'phrase_present': [bool(phrase_matches(words, p)) for p in PHRASES],
        'issues': deepcopy(issues), 'human_quality_validated': False}


def review(quality, runtime):
    quality, runtime = Path(quality), Path(runtime)
    inputs = {}; rows = []; checks = []; acceptances = {}
    def read(path):
        inputs[str(path.resolve())] = digest(path)
        return read_json(path)
    manifest = read(quality/'manifest.json')
    audio = {r['id']: r for r in manifest['samples']}
    if list(audio) != IDS:
        raise ValueError('Exactly sound1-4 required')
    import wave
    for name, row in audio.items():
        path = quality/(name+'.wav'); inputs[str(path.resolve())] = digest(path)
        with wave.open(str(path), 'rb') as wav:
            if ((wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000)
                    or abs(wav.getnframes()/16000 - row['duration_seconds']) > .0005
                    or digest(path) != row['wav_sha256']):
                raise ValueError('Benchmark audio changed')
            row['duration_seconds'] = wav.getnframes()/16000
    for profile, (module_name, ref_dir) in PROFILES.items():
        path = runtime/profile/'jobs'/('result-'+profile+'.json'); result = read(path)
        if result['status'] != 'completed' or result['profile'] != profile:
            raise ValueError('Incomplete saved result')
        provenance(result); module = importlib.import_module(module_name)
        if result['modes'] != module.MODES or [s['id'] for s in result['samples']] != IDS:
            raise ValueError('Saved modes or samples changed')
        if ref_dir:
            refs = read(quality/ref_dir/'references.json')
            if refs['result_sha256'] != digest(path):
                raise ValueError('Speaker references not bound to exact output')
            acceptances[profile] = evaluate(result, refs)
        for sample in result['samples']:
            name = sample['id']; duration = sample.get('duration_seconds', sample.get('durationSeconds'))
            if sample['sha256'] != audio[name]['wav_sha256'] or duration != audio[name]['duration_seconds']:
                raise ValueError('Saved result belongs to different audio')
            if set(sample['outputs']) != set(module.MODES):
                raise ValueError('Missing saved variant')
            error = None
            try:
                if profile == 'full-pipeline-v1':
                    for mode, output in sample['outputs'].items():
                        b = output['benchmark']
                        rebuilt = format_output({'language': output['languageDetected'], 'segments': b['asrSegments']},
                            {'segments': b['alignedSegments']}, b['diarizationTimeline'], mode, duration)
                        for key in ('segments', 'rawText', 'rawAsrText', 'srtContent', 'audit', 'wordCount'):
                            if output[key] != rebuilt[key]:
                                raise ValueError('Export does not replay')
                elif profile == 'context-pipeline-v1':
                    module.validate_sample(sample['outputs'], round(duration*16000), duration)
                else:
                    module.validate_sample(sample)
            except (ValueError, KeyError, TypeError) as exc:
                error = type(exc).__name__ + ': ' + str(exc)
            checks.append({'profile': profile, 'sample': name, 'replay_pass': error is None, 'error': error})
            control_name = 'standard30' if profile == 'full-pipeline-v1' else 'standard30_ru' if profile == 'context-pipeline-v1' else 'control30' if profile == 'release-candidate-v1' else None
            control_text = sample['outputs'][control_name]['rawAsrText'] if control_name else None
            for mode, output in sample['outputs'].items():
                if profile == 'context-pipeline-v1' and mode == 'context12_pad2':
                    # There is no valid full transcript when reconciliation fails.
                    row = {'profile': profile, 'sample': name, 'mode': mode,
                           'status': output['status'], 'complete_candidate': output['candidate'] is not None,
                           'window_count': len(output['window_evidence']), 'human_quality_validated': False}
                else:
                    timeline = None if 'benchmark' in output else sample['diarization'][module.MODES[mode]['diarization']]['regular']
                    row = {'profile': profile, 'sample': name, 'mode': mode, **measure(output, timeline=timeline)}
                    row['text_changes_vs_control'] = text_changes(control_text, row['text']) if control_text is not None else None
                row['replay_pass'] = error is None
                rows.append(row)
    for profile in ('baseline', 'community'):
        for name in IDS:
            data = read(quality/profile/(name+'.json')); audit = audit_result(data)
            passed = all(audit[k] for k in ('exportMatchesRebuild', 'exportTextMatchesAlignedIgnoringWhitespace'))
            limitation = 'Legacy evidence: export replay only; no raw ASR, full timeline or embedded audio hash.'
            checks.append({'profile': profile, 'sample': name, 'replay_pass': passed, 'scope': limitation,
                           'error': None if passed else 'Export mismatch'})
            rows.append({'profile': profile, 'sample': name, 'mode': profile, 'replay_pass': passed,
                         'evidence_limitations': limitation, **measure(data)})
    if any(digest(p) != sha for p, sha in inputs.items()):
        raise ValueError('Historical evidence changed during review')
    return {'status': 'review_required', 'release_ready': False, 'recognition_performed': False,
            'paid_jobs_submitted': 0, 'technical_replay_pass': all(c['replay_pass'] for c in checks),
            'checks': checks, 'rows': rows, 'acceptance': acceptances, 'input_sha256': inputs,
            'limitations': 'Saved outputs replayed; no new inference, independent listening, WER or DER.'}
