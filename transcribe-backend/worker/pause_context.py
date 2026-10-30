"""Test-only alignment ablation: decoder bounds versus the full acoustic window.

Keep WhisperX text and original audio common to every mode. No reference words,
speaker roles, expected counts or manual timestamps are accepted by this worker.
"""
import argparse
from copy import deepcopy
import gc
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import timed_turns as base

PROFILE = 'pause-context-v1'
ALIGNMENTS = ('control', 'extend_tail', 'joint_window')
DIARIZATION = {key: deepcopy(base.DIARIZATION[key]) for key in ('short', 'dense')}
MODES = {a + '_' + d: {'alignment': a, 'diarization': d}
         for a in ALIGNMENTS for d in DIARIZATION}


def require_test():
    expected = {'FULL_PIPELINE_IMAGE': '1', 'FULL_PIPELINE_TEST': '1', 'PAUSE_CONTEXT_TEST': '1',
                'ACOUSTIC_TURNS_TEST': '0', 'TIMED_TURNS_TEST': '0', 'RELEASE_CANDIDATE_TEST': '0',
                'SHORT_TURN_COMPARISON_TEST': '0', 'SPEAKER_COMPARISON_TEST': '0',
                'CONTEXT_ASR_TEST': '0', 'ASR_COMPARISON_TEST': '0'}
    if any(os.getenv(key, '0') != value for key, value in expected.items()):
        raise ValueError('Dedicated pause-context test flags required')


def validate_input(inp):
    require_test()
    if (not isinstance(inp, dict) or set(inp) != {'profile', 'samples'} or inp['profile'] != PROFILE
            or not isinstance(inp['samples'], list) or len(inp['samples']) != 4
            or any(not isinstance(s, dict) or set(s) != {'id', 'sha256', 'media_url'} for s in inp['samples'])
            or [s['id'] for s in inp['samples']] != base.IDS):
        raise ValueError('Only four complete audio files without inference hints')
    base.validate_request({'profile': 'community', 'samples': inp['samples']}, os.getenv('QUALITY_R2_HOST', ''))
    return inp['samples']


def alignment_bounds(native, duration, mode):
    """Never copy a human boundary or re-decode text to fit an expected answer.

    extend_tail isolates the last sentence's right-bound restriction; joint_window
    gives CTC all ordered text in a single, at most 15-second acoustic context.
    A single joined segment avoids aligning repeated phrases independently into
    the same audio occurrence. The historical control remains byte-for-byte.
    """
    if mode not in ALIGNMENTS or not base.finite(duration) or not 0 < duration <= base.POLICY['max_window_seconds']:
        raise ValueError('Invalid bounded alignment context')
    bounds = base.sentence_bounds(native, duration)
    if mode == 'control' or not bounds:
        return bounds
    if mode == 'extend_tail':
        bounds[-1]['end'] = duration
        return bounds
    text = ''.join(s['text'] for s in bounds)
    return [{'start': 0., 'end': duration, 'text': text}]


def realign(records, audio, align, mode):
    segments, local = [], []
    for record in records:
        win = record['window']; a, b = win['start_frame'], win['end_frame']
        bounds = alignment_bounds(record['native'], (b - a) / base.RATE, mode)
        aligned = (deepcopy(record['aligned_local']) if mode == 'control' else
                   base.model_evidence(align(deepcopy(bounds), audio[a:b])))
        local.append({'window': deepcopy(win), 'bounds': bounds, 'aligned_local': aligned})
        segments.extend(base.global_alignment(aligned, a / base.RATE)['segments'])
    return {'segments': segments}, local


def validate_sample(sample):
    duration = sample['duration_seconds']
    if (set(sample['diarization']) != set(DIARIZATION) or set(sample['alignments']) != set(ALIGNMENTS)
            or set(sample['alignment_windows']) != set(ALIGNMENTS) or set(sample['outputs']) != set(MODES)):
        raise ValueError('Incomplete context comparison')
    for name, settings in DIARIZATION.items():
        evidence = sample['diarization'][name]
        if evidence['settings'] != {**settings, 'step_seconds': 10 * settings['step_ratio']} or evidence['parameters'] != base.PARAMETERS:
            raise ValueError('Diarization changed')
        diag = evidence['filter_diagnostics']
        if (len(diag) != 1 or diag[0]['min_active_ratio'] != .1
                or not 0 < diag[0]['selected_embeddings'] <= diag[0]['chunk_speaker_pairs']):
            raise ValueError('Missing embedding selection evidence')
        for kind in ('regular', 'exclusive'):
            base.validate_timeline(evidence[kind], duration)
    plan = base.pause_windows(sample['diarization']['short']['exclusive'], round(duration * base.RATE))
    if [w['window'] for w in sample['windows']] != plan:
        raise ValueError('Acoustic windows changed')
    raw = []
    for record in sample['windows']:
        win = record['window']; offset = win['start_frame'] / base.RATE
        bounds = alignment_bounds(record['native'], (win['end_frame'] - win['start_frame']) / base.RATE, 'control')
        if record['bounds'] != bounds:
            raise ValueError('Native control changed')
        raw.extend(base.global_alignment({'segments': bounds}, offset)['segments'])
    if sample['asr'] != {'language': 'ru', 'segments': raw}:
        raise ValueError('Common WhisperX text changed')
    for mode in ALIGNMENTS:
        records = sample['alignment_windows'][mode]
        if len(records) != len(plan):
            raise ValueError('Missing alignment window')
        global_segments = []
        for original, record in zip(sample['windows'], records):
            win = original['window']; a, b = win['start_frame'], win['end_frame']
            expected = alignment_bounds(original['native'], (b - a) / base.RATE, mode)
            if record['window'] != win or record['bounds'] != expected:
                raise ValueError('Context bounds cannot be reproduced')
            if mode == 'control' and record['aligned_local'] != original['aligned_local']:
                raise ValueError('Control alignment changed')
            aligned = record['aligned_local']
            if base.plain(''.join(s['text'] for s in aligned['segments'])) != base.plain(''.join(s['text'] for s in expected)):
                raise ValueError('Alignment changed ordered text')
            for segment in aligned['segments']:
                start, end = segment.get('start'), segment.get('end')
                if (not base.finite(start) or not base.finite(end) or not 0 <= start < end <= (b-a)/base.RATE + .001):
                    raise ValueError('Alignment escaped the acoustic window')
            global_segments.extend(base.global_alignment(aligned, a / base.RATE)['segments'])
        if sample['alignments'][mode] != {'segments': global_segments}:
            raise ValueError('Audio offset changed')
    for name, mode in MODES.items():
        rebuilt = base.attribute(sample['asr'], sample['alignments'][mode['alignment']],
                                 sample['diarization'][mode['diarization']]['regular'], duration)
        if sample['outputs'][name] != rebuilt:
            raise ValueError('Export cannot be reproduced')


def execute(manifest, output_path):
    require_test()
    rows = base.read_json(manifest)['samples']; base.validate_audio_rows(rows)
    state = {'profile': PROFILE, 'status': 'started', 'phase': 'assets', 'samples': [],
             'source_manifest_sha256': base.digest(base.SOURCES), 'modes': MODES, 'policy': base.POLICY,
             'diarization_configs': DIARIZATION, 'alignment_configs': list(ALIGNMENTS),
             'decoder': base.DECODER, 'human_quality_validated': False}
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
            state['runtime']['packages'] = {p: version(p) for p in ('torch', 'whisperx', 'pyannote.audio', 'faster-whisper')}
            vad = load_vad_model('cuda', model_fp=assets['vad_path'], vad_onset=.5, vad_offset=.363)
            model = whisperx.load_model(assets['root'], device='cuda', local_files_only=True, vad_model=vad, **deepcopy(config['load']))
            am, metadata = whisperx.load_align_model('ru', 'cuda', model_name='/opt/full/alignment', model_cache_only=True)
            pipeline = Pipeline.from_pretrained(str(diar_root), token=False).to(torch.device('cuda'))
            base.validate_runtime_pipeline(pipeline, version('pyannote.audio'))
            if pipeline._segmentation.model.specifications.duration != 10:
                raise ValueError('Unexpected segmentation duration')
            for row in rows:
                audio = whisperx.load_audio(row['path'])
                sample = {'id': row['id'], 'sha256': row['sha256'], 'duration_seconds': row['duration_seconds'],
                          'diarization': {}, 'alignments': {}, 'alignment_windows': {}, 'outputs': {}}
                state['samples'].append(sample)
                for name, settings in DIARIZATION.items():
                    state['phase'] = row['id'] + ':' + name
                    diagnostics = []; torch.manual_seed(0)
                    with base.acoustic_settings(pipeline, settings) as actual, base.short_turn_filter(pipeline.clustering, .1, diagnostics):
                        result = pipeline({'waveform': torch.from_numpy(audio[None, :]), 'sample_rate': base.RATE})
                    sample['diarization'][name] = {'settings': actual, 'parameters': base.model_evidence(pipeline.parameters(instantiated=True)),
                        'filter_diagnostics': diagnostics,
                        'regular': base.annotation_rows(result.speaker_diarization, row['duration_seconds'], False),
                        'exclusive': base.annotation_rows(result.exclusive_speaker_diarization, row['duration_seconds'], True)}
                windows = base.pause_windows(sample['diarization']['short']['exclusive'], len(audio))
                align = lambda segments, chunk: whisperx.align(segments, am, metadata, chunk, 'cuda', return_char_alignments=False)
                state['phase'] = row['id'] + ':decode'
                sample['asr'], _, sample['windows'] = base.decode(audio, windows, model, align)
                for name in ALIGNMENTS:
                    state['phase'] = row['id'] + ':' + name
                    sample['alignments'][name], sample['alignment_windows'][name] = realign(sample['windows'], audio, align, name)
                for name, mode in MODES.items():
                    sample['outputs'][name] = base.attribute(sample['asr'], sample['alignments'][mode['alignment']],
                        sample['diarization'][mode['diarization']]['regular'], row['duration_seconds'])
                validate_sample(sample)
                gc.collect(); torch.cuda.empty_cache()
        state.update(status='completed', phase='complete')
    except Exception as exc:
        state.update(status='failed', error_type=type(exc).__name__)
    state['elapsed_seconds'] = round(time.perf_counter() - begun, 3)
    base.write_new(output_path, state)
    return state


def handler(job):
    stage = 'validate'
    try:
        samples = validate_input(job.get('input'))
        with tempfile.TemporaryDirectory(prefix='pause-context-') as directory:
            root = Path(directory); rows = []; stage = 'download_audio'
            for sample in samples:
                path = root / (sample['id'] + '.wav'); base.fetch_sample(sample, path)
                rows.append({'id': sample['id'], 'sha256': sample['sha256'], 'path': str(path)})
            base.validate_audio_rows(rows)
            manifest, result_path = root / 'manifest.json', root / 'result.json'
            base.write_new(manifest, {'samples': rows}); stage = 'child'
            child = subprocess.run([sys.executable, str(Path(__file__)), '--manifest', str(manifest),
                '--output', str(result_path)], timeout=270, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            if child.returncode or not result_path.is_file():
                raise RuntimeError('No pause-context evidence')
            return base.read_json(result_path)
    except Exception as exc:
        return {'profile': PROFILE, 'status': 'failed', 'phase': stage, 'error_type': type(exc).__name__}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--manifest', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); execute(args.manifest, args.output)
