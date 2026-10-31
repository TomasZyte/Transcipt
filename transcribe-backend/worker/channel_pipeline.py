"""Isolated full-channel WhisperX worker; never imported by production."""
import argparse
from copy import deepcopy
import hashlib
from importlib.metadata import version
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.request

import channel_asr as adapter
from asr_comparison import config_for, digest, offline_models, read_json, verify_assets, write_new
from prepare_full_assets import SOURCES, verify_bundle, verify_nltk
from quality_batch_handler import NoRedirect, validate_request

PROFILE = adapter.PROFILE
MAX_BYTES = adapter.MAX_SECONDS * adapter.RATE * 2 + 4096
FLAGS = {'FULL_PIPELINE_IMAGE': '1', 'FULL_PIPELINE_TEST': '1', 'CHANNEL_ASR_TEST': '1',
         **{name: '0' for name in ('ASR_COMPARISON_TEST', 'CONTEXT_ASR_TEST',
              'SPEAKER_COMPARISON_TEST', 'SHORT_TURN_COMPARISON_TEST', 'RELEASE_CANDIDATE_TEST',
              'TIMED_TURNS_TEST', 'ACOUSTIC_TURNS_TEST', 'PAUSE_CONTEXT_TEST')}}


def adapter_for(profile):
    if profile == PROFILE:
        return adapter
    if profile == 'channel-runs-v1':
        import channel_runs
        return channel_runs
    raise ValueError('Unknown channel experiment')


def require_test():
    if any(os.getenv(k, '0') != v for k, v in FLAGS.items()):
        raise ValueError('Dedicated channel test image and flags required')


def validate_input(inp, storage_host=None):
    if (not isinstance(inp, dict) or set(inp) != {'profile', 'source_sha256', 'confirmation', 'samples'}
            or inp['profile'] not in (PROFILE, 'channel-runs-v1') or not isinstance(inp['samples'], list)
            or len(inp['samples']) != 2 or any(not isinstance(s, dict)
                or set(s) != {'id', 'sha256', 'media_url'} for s in inp['samples'])
            or [s['id'] for s in inp['samples']] != list(adapter.CHANNELS)
            or not isinstance(inp['source_sha256'], str) or not adapter.SHA.fullmatch(inp['source_sha256'])):
        raise ValueError('Exactly two channels, no inference hints')
    adapter._confirmation(inp['confirmation'], adapter.MAX_SECONDS)
    validate_request({'profile': 'community', 'samples': inp['samples']},
                     storage_host or os.getenv('QUALITY_R2_HOST', ''))
    return inp['samples']


def fetch_channel(sample, path):
    with urllib.request.build_opener(NoRedirect()).open(sample['media_url'], timeout=30) as response:
        data = response.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES or hashlib.sha256(data).hexdigest() != sample['sha256']:
        raise ValueError('Channel size or hash mismatch')
    with Path(path).open('xb') as stream:
        stream.write(data)


def validate_result(state, plan):
    selected = adapter_for(plan.get('profile', PROFILE))
    if (state.get('profile') != selected.PROFILE or state.get('status') != 'completed'
            or state.get('input_plan') != input_contract(plan) or state.get('source_manifest_sha256') != digest(SOURCES)
            or state.get('human_quality_validated') is not False):
        raise ValueError('Unexpected channel result identity')
    result = state['result']
    if selected is adapter:
        rebuilt = adapter.merge_channels(result['evidence'], plan['frames_per_channel'])
        rebuilt.update(confirmation=plan['confirmation'], full_call_speaker_identity_verified=False)
    else:
        rebuilt = selected.rebuild(result, plan)
    if result != rebuilt:
        raise ValueError('Channel export does not match original evidence')


def input_contract(plan):
    # Local CRLF and worker LF manifests have different byte hashes. All decoded
    # content, audio hashes and frame counts must still agree exactly.
    return {k: deepcopy(v) for k, v in plan.items() if k != 'manifest_sha256'}


def execute(manifest, output_path):
    require_test()
    if Path(output_path).exists():
        raise ValueError('Never overwrite a previous attempt')
    selected = adapter_for(read_json(manifest)['profile'])
    plan = selected.prepare(manifest)
    state = {'profile': selected.PROFILE, 'status': 'started', 'phase': 'assets', 'input_plan': input_contract(plan),
             'source_manifest_sha256': digest(SOURCES), 'human_quality_validated': False}
    started = time.perf_counter()
    try:
        config = config_for('wx_ru')
        assets = verify_assets('/opt/asr/model-lock.json', config)
        alignment = verify_bundle('/opt/full/alignment', 'alignment')
        state['runtime'] = {'asr': assets, 'alignment': alignment,
                            'nltk_lock_sha256': verify_nltk('/opt/full')}
        with offline_models():
            import torch
            import whisperx
            from whisperx.vads.pyannote import load_vad_model
            torch.manual_seed(0)
            state['runtime']['packages'] = {p: version(p) for p in config['packages']}
            state['phase'] = 'load_models'
            vad = load_vad_model('cuda', model_fp=assets['vad_path'], vad_onset=.5, vad_offset=.363)
            model = whisperx.load_model(assets['root'], device='cuda', local_files_only=True,
                                        vad_model=vad, **deepcopy(config['load']))
            am, metadata = whisperx.load_align_model('ru', 'cuda', model_name='/opt/full/alignment', model_cache_only=True)
            audio = [whisperx.load_audio(str(Path(manifest).parent / row['file'])) for row in plan['channels']]
            if any(len(a) != plan['frames_per_channel'] for a in audio):
                raise ValueError('Decoder changed the full channel duration')
            state['phase'] = 'transcribe_and_align'
            callbacks = {'transcribe': model.transcribe}
            if selected is not adapter:
                detect, transcribe = selected.callbacks(model, vad)
                callbacks = {'detect': detect, 'transcribe': transcribe}
            state['result'] = selected.decode_channels(*audio, **callbacks,
                align=lambda segments, chunk: whisperx.align(segments, am, metadata, chunk, 'cuda', return_char_alignments=False),
                confirmation=plan['confirmation'])
        state.update(status='completed', phase='complete')
        validate_result(state, plan)
    except Exception as exc:
        state.update(status='failed', error_type=type(exc).__name__)
    state['elapsed_seconds'] = round(time.perf_counter() - started, 3)
    write_new(output_path, state)
    return state


def handler(job):
    stage = 'validate'
    try:
        require_test()
        inp = job.get('input')
        samples = validate_input(inp)
        with tempfile.TemporaryDirectory(prefix='channel-test-') as directory:
            root = Path(directory)
            manifest, result_path = root/'input.json', root/'result.json'
            stage = 'download_audio'
            for sample in samples:
                fetch_channel(sample, root / (sample['id'] + '-full.wav'))
            selected = adapter_for(inp['profile'])
            write_new(manifest, {'profile': selected.PROFILE, 'source_sha256': inp['source_sha256'],
                'confirmation': inp['confirmation'], 'channels': [
                    {'channel': s['id'], 'file': s['id'] + '-full.wav', 'sha256': s['sha256'],
                     'source_start_seconds': 0} for s in samples]})
            selected.prepare(manifest)
            stage = 'child'
            child = subprocess.run([sys.executable, str(Path(__file__)), '--manifest', str(manifest),
                '--output', str(result_path)], timeout=270, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, check=False)
            if child.returncode or not result_path.is_file():
                raise RuntimeError('No channel evidence')
            return read_json(result_path)
    except Exception as exc:
        return {'profile': PROFILE, 'status': 'failed', 'phase': stage, 'error_type': type(exc).__name__}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    execute(args.manifest, args.output)
