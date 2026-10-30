"""Bounded sound1-4 release-candidate comparison; no production enablement."""
import argparse
from copy import deepcopy
import gc
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from asr_comparison import config_for, digest, offline_models, read_json, verify_assets, write_new
from contextual_asr import model_evidence
from full_pipeline import IDS, validate_audio_rows
from prepare_full_assets import SOURCES, prepare_diarization, verify_bundle, verify_nltk
from quality_batch_handler import validate_request, fetch_sample
from short_turn_comparison import short_turn_filter, validate_runtime_pipeline, PARAMETERS
from speaker_comparison import annotation_rows
from transcript_candidate import RATE, POLICY, pause_windows, global_alignment, attribute

PROFILE = 'release-candidate-v1'
DIARIZATION = {'default': {'ratio': .2, 'threshold': .6},
               'short': {'ratio': .1, 'threshold': .6},
               'merged': {'ratio': .1, 'threshold': .7}}
MODES = {'control30': {'asr': 'control', 'diarization': 'default'},
         'control30_short': {'asr': 'control', 'diarization': 'short'},
         'pause_short': {'asr': 'pause', 'diarization': 'short'},
         'pause_merged': {'asr': 'pause', 'diarization': 'merged'}}


def require_test():
    expected = {'FULL_PIPELINE_IMAGE': '1', 'FULL_PIPELINE_TEST': '1', 'RELEASE_CANDIDATE_TEST': '1',
                'SHORT_TURN_COMPARISON_TEST': '0', 'SPEAKER_COMPARISON_TEST': '0',
                'CONTEXT_ASR_TEST': '0', 'ASR_COMPARISON_TEST': '0'}
    if any(os.getenv(k, '0') != v for k, v in expected.items()):
        raise ValueError('Dedicated release-candidate test flags required')


def validate_input(inp):
    require_test()
    if (not isinstance(inp, dict) or set(inp) != {'profile', 'samples'} or inp['profile'] != PROFILE
            or not isinstance(inp['samples'], list) or [s.get('id') for s in inp['samples']] != IDS
            or any(set(s) != {'id', 'sha256', 'media_url'} for s in inp['samples'])):
        raise ValueError('Only sound1-4 without hints, references or options are accepted')
    validate_request({'profile': 'community', 'samples': inp['samples']}, os.getenv('QUALITY_R2_HOST', ''))
    return inp['samples']


def decode_pause(audio, timeline, *, transcribe, align):
    windows = pause_windows(timeline, len(audio))
    evidence, raw_all, aligned_all = [], [], []
    for win in windows:
        start, end = win['start_frame'], win['end_frame']
        chunk = audio[start:end]
        raw = model_evidence(transcribe(chunk))
        if raw.get('language') != 'ru' or not isinstance(raw.get('segments'), list):
            raise ValueError('Unexpected language or ASR structure')
        aligned = model_evidence(align(deepcopy(raw['segments']), chunk))
        offset = start / RATE
        evidence.append({'window': deepcopy(win), 'raw_local': raw, 'aligned_local': aligned})
        raw_all.extend(global_alignment(raw, offset)['segments'])
        aligned_all.extend(global_alignment(aligned, offset)['segments'])
    return {'language': 'ru', 'segments': raw_all}, {'segments': aligned_all}, evidence


def validate_sample(sample):
    duration = sample['duration_seconds']
    if set(sample['diarization']) != set(DIARIZATION) or set(sample['outputs']) != set(MODES):
        raise ValueError('Incomplete candidate comparison')
    if set(sample['asr']) != {'control', 'pause'} or set(sample['alignment']) != {'control', 'pause'}:
        raise ValueError('Both complete ASR/alignment paths required')
    for key, setting in DIARIZATION.items():
        observed = sample['diarization'][key]
        params = deepcopy(PARAMETERS); params['clustering']['threshold'] = setting['threshold']
        diag = observed.get('filter_diagnostics')
        if (observed.get('parameters') != params or not isinstance(diag, list) or len(diag) != 1
                or diag[0].get('min_active_ratio') != setting['ratio']
                or type(diag[0].get('selected_embeddings')) is not int
                or type(diag[0].get('chunk_speaker_pairs')) is not int
                or not 0 < diag[0]['selected_embeddings'] <= diag[0]['chunk_speaker_pairs']):
            raise ValueError('Diarization configuration was not verified')
        from transcript_candidate import validate_timeline
        validate_timeline(observed['regular'], duration); validate_timeline(observed['exclusive'], duration)
        if any(a['end'] > b['start'] + 1e-6 for a, b in zip(observed['exclusive'], observed['exclusive'][1:])):
            raise ValueError('Exclusive timeline overlaps')
    windows = pause_windows(sample['diarization']['short']['exclusive'], round(duration * RATE))
    if [e['window'] for e in sample['windows']] != windows:
        raise ValueError('Acoustic window plan changed')
    raw_segments, aligned_segments = [], []
    for item in sample['windows']:
        offset = item['window']['start_frame'] / RATE
        raw_segments.extend(global_alignment(item['raw_local'], offset)['segments'])
        aligned_segments.extend(global_alignment(item['aligned_local'], offset)['segments'])
    if (sample['asr']['pause'] != {'language': 'ru', 'segments': raw_segments}
            or sample['alignment']['pause'] != {'segments': aligned_segments}):
        raise ValueError('Window offsets or ASR evidence changed')
    for key, mode in MODES.items():
        rebuilt = attribute(sample['asr'][mode['asr']], sample['alignment'][mode['asr']],
                            sample['diarization'][mode['diarization']]['regular'], duration)
        if rebuilt != sample['outputs'][key]:
            raise ValueError('Candidate output does not replay from source evidence')


def execute(manifest, output_path):
    require_test()
    rows = read_json(manifest)['samples']; validate_audio_rows(rows)
    state = {'profile': PROFILE, 'status': 'started', 'phase': 'assets', 'samples': [],
             'source_manifest_sha256': digest(SOURCES), 'modes': MODES, 'policy': POLICY,
             'diarization_configs': DIARIZATION, 'human_quality_validated': False}
    begun = time.perf_counter()
    try:
        config = config_for('wx_auto')
        assets = verify_assets('/opt/asr/model-lock.json', config)
        alignment = verify_bundle('/opt/full/alignment', 'alignment')
        nltk_hash = verify_nltk('/opt/full')
        diar_root = Path(manifest).parent / 'diarization'
        diar_assets = prepare_diarization(diar_root, os.getenv('HF_TOKEN', ''))
        state['runtime'] = {'asr': assets, 'alignment': alignment, 'diarization': diar_assets,
                            'nltk_lock_sha256': nltk_hash}
        with offline_models():
            import torch
            import whisperx
            from importlib.metadata import version
            from pyannote.audio import Pipeline
            from whisperx.vads.pyannote import load_vad_model
            state['runtime']['packages'] = {p: version(p) for p in ('torch', 'whisperx', 'pyannote.audio')}
            vad = load_vad_model('cuda', model_fp=assets['vad_path'], vad_onset=.5, vad_offset=.363)
            model = whisperx.load_model(assets['root'], device='cuda', local_files_only=True,
                                       vad_model=vad, **deepcopy(config['load']))
            am, metadata = whisperx.load_align_model('ru', 'cuda', model_name='/opt/full/alignment', model_cache_only=True)
            diarizer = Pipeline.from_pretrained(str(diar_root), token=False).to(torch.device('cuda'))
            validate_runtime_pipeline(diarizer, version('pyannote.audio'))
            for row in rows:
                state['phase'] = row['id'] + ':diarization'
                audio = whisperx.load_audio(row['path'])
                sample = {'id': row['id'], 'sha256': row['sha256'], 'duration_seconds': row['duration_seconds'],
                          'diarization': {}, 'asr': {}, 'alignment': {}, 'outputs': {}}
                state['samples'].append(sample)
                for name, setting in DIARIZATION.items():
                    params = deepcopy(PARAMETERS); params['clustering']['threshold'] = setting['threshold']
                    diarizer.instantiate(params)
                    torch.manual_seed(0)
                    diagnostics = []
                    with short_turn_filter(diarizer.clustering, setting['ratio'], diagnostics):
                        output = diarizer({'waveform': torch.from_numpy(audio[None, :]), 'sample_rate': RATE})
                    sample['diarization'][name] = {
                        'regular': annotation_rows(output.speaker_diarization, row['duration_seconds'], False),
                        'exclusive': annotation_rows(output.exclusive_speaker_diarization, row['duration_seconds'], True),
                        'parameters': model_evidence(diarizer.parameters(instantiated=True)),
                        'filter_diagnostics': diagnostics}
                state['phase'] = row['id'] + ':asr'
                if model.detect_language(audio) != 'ru':
                    raise ValueError('Russian-only candidate benchmark requires detected Russian')
                transcribe = lambda chunk: model.transcribe(chunk, language='ru', batch_size=8, task='transcribe', chunk_size=30)
                align = lambda segments, chunk: whisperx.align(segments, am, metadata, chunk, 'cuda', return_char_alignments=False)
                control = model_evidence(transcribe(audio))
                sample['asr']['control'] = control
                sample['alignment']['control'] = model_evidence(align(deepcopy(control['segments']), audio))
                pause_raw, pause_aligned, windows = decode_pause(audio, sample['diarization']['short']['exclusive'],
                                                               transcribe=transcribe, align=align)
                sample['asr']['pause'], sample['alignment']['pause'], sample['windows'] = pause_raw, pause_aligned, windows
                for name, mode in MODES.items():
                    sample['outputs'][name] = attribute(sample['asr'][mode['asr']], sample['alignment'][mode['asr']],
                        sample['diarization'][mode['diarization']]['regular'], row['duration_seconds'])
                validate_sample(sample)
                gc.collect(); torch.cuda.empty_cache()
        state.update(status='completed', phase='complete')
    except Exception as exc:
        state.update(status='failed', error_type=type(exc).__name__)
    state['elapsed_seconds'] = round(time.perf_counter() - begun, 3)
    write_new(output_path, state)
    return state


def handler(job):
    stage = 'validate'
    try:
        samples = validate_input(job.get('input'))
        with tempfile.TemporaryDirectory(prefix='candidate-') as directory:
            root = Path(directory); rows = []
            stage = 'download_audio'
            for sample in samples:
                path = root / (sample['id'] + '.wav'); fetch_sample(sample, path)
                rows.append({'id': sample['id'], 'sha256': sample['sha256'], 'path': str(path)})
            validate_audio_rows(rows)
            manifest, result_path = root / 'manifest.json', root / 'result.json'
            write_new(manifest, {'samples': rows})
            stage = 'child'
            child = subprocess.run([sys.executable, str(Path(__file__)), '--manifest', str(manifest),
                                    '--output', str(result_path)], timeout=270, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, check=False)
            if child.returncode or not result_path.is_file():
                raise RuntimeError('Candidate produced no evidence')
            return read_json(result_path)
    except Exception as exc:
        return {'profile': PROFILE, 'status': 'failed', 'phase': stage, 'error_type': type(exc).__name__}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--manifest', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); execute(args.manifest, args.output)
