"""Isolated pitch-normalization probe; original WhisperX text and clock retained.

This is a diagnostic comparison, not an automatic production enhancement.
No age, gender, reference text, reference times or speaker counts enter inference.
"""
import argparse
from copy import deepcopy
import gc
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import timed_turns as base

PROFILE = 'acoustic-turns-v1'
TRANSFORMS = {'original': 0, 'lower3': -3, 'lower6': -6}
PITCH = {'sr': 16000, 'bins_per_octave': 12, 'res_type': 'soxr_hq',
         'scale': False, 'n_fft': 1024, 'hop_length': 256}
MODES = {'original': {'alignment': 'original', 'diarization': 'original'}}
for _key in ('lower3', 'lower6'):
    MODES[_key + '_align'] = {'alignment': _key, 'diarization': 'original'}
    MODES[_key + '_diar'] = {'alignment': 'original', 'diarization': _key}
    MODES[_key + '_both'] = {'alignment': _key, 'diarization': _key}


def require_test():
    expected = {'FULL_PIPELINE_IMAGE': '1', 'FULL_PIPELINE_TEST': '1', 'ACOUSTIC_TURNS_TEST': '1',
                'TIMED_TURNS_TEST': '0', 'RELEASE_CANDIDATE_TEST': '0',
                'SHORT_TURN_COMPARISON_TEST': '0', 'SPEAKER_COMPARISON_TEST': '0',
                'CONTEXT_ASR_TEST': '0', 'ASR_COMPARISON_TEST': '0'}
    if any(os.getenv(k, '0') != v for k, v in expected.items()):
        raise ValueError('Dedicated acoustic test flags required')


def validate_input(inp):
    require_test()
    if (not isinstance(inp, dict) or set(inp) != {'profile', 'samples'} or inp['profile'] != PROFILE
            or not isinstance(inp['samples'], list) or len(inp['samples']) != 4
            or any(not isinstance(s, dict) or set(s) != {'id', 'sha256', 'media_url'} for s in inp['samples'])
            or [s['id'] for s in inp['samples']] != base.IDS):
        raise ValueError('Only four audio files; no inference hints')
    base.validate_request({'profile': 'community', 'samples': inp['samples']}, os.getenv('QUALITY_R2_HOST', ''))
    return inp['samples']


def pitch_audio(audio, name):
    import librosa
    import numpy as np
    if name not in TRANSFORMS or audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError('Invalid pitch input')
    if librosa.__version__ != '0.11.0':
        raise ValueError('Unreviewed pitch implementation')
    shifted = audio.copy() if name == 'original' else librosa.effects.pitch_shift(
        audio, n_steps=TRANSFORMS[name], **PITCH)
    shifted = np.ascontiguousarray(shifted, dtype=np.float32)
    if shifted.shape != audio.shape or not np.isfinite(shifted).all():
        raise ValueError('Pitch transform changed the audio clock')
    return shifted


def verify_transform():
    """CPU build check: unchanged clock, finite samples, no large onset shift."""
    import numpy as np
    frames = base.RATE * 3
    x = np.zeros(frames, dtype=np.float32)
    t = np.arange(base.RATE // 2) / base.RATE
    x[base.RATE:base.RATE + len(t)] = np.sin(2 * np.pi * 400 * t) * np.hanning(len(t))
    before = x.copy()
    for name in TRANSFORMS:
        y = pitch_audio(x, name)
        assert len(y) == frames and np.isfinite(y).all() and np.array_equal(x, before)
        active = np.flatnonzero(abs(y) > .2)
        assert active.size and .9 < active[0] / base.RATE < 1.3
        assert 1.2 < active[-1] / base.RATE < 1.7
        spectrum = abs(np.fft.rfft(y[base.RATE:base.RATE + len(t)]))
        peak = np.fft.rfftfreq(len(t), 1 / base.RATE)[spectrum.argmax()]
        assert abs(peak - 400 * 2 ** (TRANSFORMS[name] / 12)) < 12


def align_windows(records, audio, align):
    segments, local = [], []
    for record in records:
        win = record['window']; a, b = win['start_frame'], win['end_frame']
        aligned = base.model_evidence(align(deepcopy(record['bounds']), audio[a:b]))
        local.append({'window': deepcopy(win), 'aligned_local': aligned})
        segments.extend(base.global_alignment(aligned, a / base.RATE)['segments'])
    return {'segments': segments}, local


def validate_sample(sample):
    duration = sample['duration_seconds']
    if (set(sample['diarization']) != set(TRANSFORMS) or set(sample['alignments']) != set(TRANSFORMS)
            or set(sample['transformed_audio']) != set(TRANSFORMS) or set(sample['outputs']) != set(MODES)):
        raise ValueError('Missing acoustic variants')
    plan = base.pause_windows(sample['diarization']['original']['exclusive'], round(duration * base.RATE))
    if [r['window'] for r in sample['windows']] != plan: raise ValueError('Original window plan changed')
    raw, original = [], []
    for record in sample['windows']:
        win = record['window']; offset = win['start_frame'] / base.RATE
        bounds = base.sentence_bounds(record['native'], (win['end_frame'] - win['start_frame']) / base.RATE)
        if bounds != record['bounds']: raise ValueError('Decoder bounds changed')
        raw.extend(base.global_alignment({'segments': bounds}, offset)['segments'])
        original.extend(base.global_alignment(record['aligned_local'], offset)['segments'])
    if sample['asr'] != {'language': 'ru', 'segments': raw}: raise ValueError('Original ASR changed')
    for key in TRANSFORMS:
        observed = sample['diarization'][key]
        diag = observed['filter_diagnostics']
        if (observed['parameters'] != base.PARAMETERS or len(diag) != 1
                or diag[0]['min_active_ratio'] != .1
                or not 0 < diag[0]['selected_embeddings'] <= diag[0]['chunk_speaker_pairs']):
            raise ValueError('Unexpected diarization configuration')
        for kind in ('regular', 'exclusive'): base.validate_timeline(observed[kind], duration)
        evidence = sample['transformed_audio'][key]
        if (evidence['frames'] != round(duration * base.RATE) or evidence['sample_rate'] != base.RATE
                or evidence['n_steps'] != TRANSFORMS[key] or len(evidence['pcm_sha256']) != 64
                or any(c not in '0123456789abcdef' for c in evidence['pcm_sha256'])):
            raise ValueError('Invalid transformed clock or identity')
        local = sample['alignment_windows'][key]
        if [r['window'] for r in local] != plan: raise ValueError('Alignment window changed')
        rebuilt = [s for r in local for s in base.global_alignment(
            r['aligned_local'], r['window']['start_frame'] / base.RATE)['segments']]
        if sample['alignments'][key] != {'segments': rebuilt}: raise ValueError('Alignment offset changed')
    if sample['alignments']['original'] != {'segments': original}: raise ValueError('Control changed')
    for name, mode in MODES.items():
        rebuilt = base.attribute(sample['asr'], sample['alignments'][mode['alignment']],
            sample['diarization'][mode['diarization']]['regular'], duration)
        if sample['outputs'][name] != rebuilt: raise ValueError('Cannot reproduce acoustic output')


def execute(manifest, output_path):
    require_test()
    rows = base.read_json(manifest)['samples']; base.validate_audio_rows(rows)
    state = {'profile': PROFILE, 'status': 'started', 'phase': 'assets', 'samples': [],
             'source_manifest_sha256': base.digest(base.SOURCES), 'modes': MODES,
             'policy': base.POLICY, 'decoder': base.DECODER, 'transforms': TRANSFORMS,
             'pitch_config': PITCH, 'human_quality_validated': False}
    begun = time.perf_counter()
    try:
        config = base.config_for('wx_auto'); assets = base.verify_assets('/opt/asr/model-lock.json', config)
        alignment = base.verify_bundle('/opt/full/alignment', 'alignment'); nltk = base.verify_nltk('/opt/full')
        diar_root = Path(manifest).parent / 'diarization'
        diar_assets = base.prepare_diarization(diar_root, os.getenv('HF_TOKEN', ''))
        state['runtime'] = {'asr': assets, 'alignment': alignment, 'diarization': diar_assets, 'nltk_lock_sha256': nltk}
        with base.offline_models():
            import torch, whisperx
            from importlib.metadata import version
            from pyannote.audio import Pipeline
            from whisperx.vads.pyannote import load_vad_model
            base.verify_decoder_api()
            state['runtime']['packages'] = {p: version(p) for p in ('torch', 'whisperx', 'pyannote.audio', 'faster-whisper', 'librosa')}
            vad = load_vad_model('cuda', model_fp=assets['vad_path'], vad_onset=.5, vad_offset=.363)
            model = whisperx.load_model(assets['root'], device='cuda', local_files_only=True, vad_model=vad, **deepcopy(config['load']))
            am, metadata = whisperx.load_align_model('ru', 'cuda', model_name='/opt/full/alignment', model_cache_only=True)
            pipeline = Pipeline.from_pretrained(str(diar_root), token=False).to(torch.device('cuda'))
            base.validate_runtime_pipeline(pipeline, version('pyannote.audio'))
            align = lambda segments, audio: whisperx.align(segments, am, metadata, audio, 'cuda', return_char_alignments=False)
            for row in rows:
                source = whisperx.load_audio(row['path'])
                sample = {'id': row['id'], 'sha256': row['sha256'], 'duration_seconds': row['duration_seconds'],
                          'diarization': {}, 'alignments': {}, 'alignment_windows': {}, 'transformed_audio': {}, 'outputs': {}}
                state['samples'].append(sample)
                for name in TRANSFORMS:
                    state['phase'] = row['id'] + ':' + name
                    audio = pitch_audio(source, name)
                    sample['transformed_audio'][name] = {'frames': len(audio), 'sample_rate': base.RATE,
                        'n_steps': TRANSFORMS[name], 'pcm_sha256': hashlib.sha256(audio.astype('<f4').tobytes()).hexdigest()}
                    diagnostics = []; torch.manual_seed(0)
                    with base.short_turn_filter(pipeline.clustering, .1, diagnostics):
                        diar = pipeline({'waveform': torch.from_numpy(audio[None, :]), 'sample_rate': base.RATE})
                    sample['diarization'][name] = {'parameters': base.model_evidence(pipeline.parameters(instantiated=True)),
                        'filter_diagnostics': diagnostics,
                        'regular': base.annotation_rows(diar.speaker_diarization, row['duration_seconds'], False),
                        'exclusive': base.annotation_rows(diar.exclusive_speaker_diarization, row['duration_seconds'], True)}
                    if name == 'original':
                        windows = base.pause_windows(sample['diarization'][name]['exclusive'], len(audio))
                        sample['asr'], aligned, sample['windows'] = base.decode(audio, windows, model, align)
                        local = [{'window': deepcopy(r['window']), 'aligned_local': deepcopy(r['aligned_local'])} for r in sample['windows']]
                    else:
                        aligned, local = align_windows(sample['windows'], audio, align)
                    sample['alignments'][name] = aligned; sample['alignment_windows'][name] = local
                for name, mode in MODES.items():
                    sample['outputs'][name] = base.attribute(sample['asr'], sample['alignments'][mode['alignment']],
                        sample['diarization'][mode['diarization']]['regular'], row['duration_seconds'])
                validate_sample(sample)
                gc.collect(); torch.cuda.empty_cache()
        state.update(status='completed', phase='complete')
    except Exception as exc: state.update(status='failed', error_type=type(exc).__name__)
    state['elapsed_seconds'] = round(time.perf_counter() - begun, 3)
    base.write_new(output_path, state)
    return state


def handler(job):
    stage = 'validate'
    try:
        samples = validate_input(job.get('input'))
        with tempfile.TemporaryDirectory(prefix='acoustic-turns-') as directory:
            root = Path(directory); rows = []; stage = 'download_audio'
            for sample in samples:
                path = root / (sample['id'] + '.wav'); base.fetch_sample(sample, path)
                rows.append({'id': sample['id'], 'sha256': sample['sha256'], 'path': str(path)})
            base.validate_audio_rows(rows)
            manifest, result_path = root / 'manifest.json', root / 'result.json'
            base.write_new(manifest, {'samples': rows}); stage = 'child'
            child = subprocess.run([sys.executable, str(Path(__file__)), '--manifest', str(manifest),
                '--output', str(result_path)], timeout=270, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            if child.returncode or not result_path.is_file(): raise RuntimeError('No acoustic evidence')
            return base.read_json(result_path)
    except Exception as exc: return {'profile': PROFILE, 'status': 'failed', 'phase': stage, 'error_type': type(exc).__name__}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--manifest', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); execute(args.manifest, args.output)
