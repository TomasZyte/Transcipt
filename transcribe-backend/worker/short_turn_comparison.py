"""Test-only short-turn filtering experiment. Frozen WhisperX words and timing."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import inspect
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import speaker_comparison as previous
from asr_comparison import digest, offline_models, read_json, SHA, write_new
from contextual_asr import model_evidence
from prepare_full_assets import prepare_diarization, SOURCES

PROFILE = 'speaker-short-turns-v1'
RATIOS = {'standard': 0.2, 'short_turns': 0.1}
MODES = tuple(name + '_' + kind for name in RATIOS for kind in ('regular', 'exclusive'))
PARAMETERS = {'segmentation': {'min_duration_off': 0.0},
              'clustering': {'threshold': 0.6, 'Fa': 0.07, 'Fb': 0.8}}


def require_test():
    expected = {'FULL_PIPELINE_IMAGE': '1', 'FULL_PIPELINE_TEST': '1',
                'SHORT_TURN_COMPARISON_TEST': '1', 'SPEAKER_COMPARISON_TEST': '0',
                'ASR_COMPARISON_TEST': '0', 'CONTEXT_ASR_TEST': '0'}
    if any(os.getenv(key, '0') != value for key, value in expected.items()):
        raise ValueError('Dedicated short-turn test flags required')


def validate_frozen(frozen):
    if (not isinstance(frozen, dict)
            or set(frozen) != {'source_result_sha256', 'asr_evidence', 'alignment_local'}
            or not SHA.fullmatch(frozen.get('source_result_sha256', ''))):
        raise ValueError('Expected saved speaker experiment evidence')
    previous.validate_evidence(frozen['asr_evidence'])
    local = frozen['alignment_local']
    if not isinstance(local, dict) or set(local) != {'segments', 'word_segments'}:
        raise ValueError('Unexpected frozen alignment structure')
    # No speaker labels, manual intervals, hints or arbitrary model options.
    words = []
    for segment in local['segments']:
        if set(segment) != {'start', 'end', 'text', 'words'}:
            raise ValueError('Frozen segments must not contain speaker hints')
        words.extend(segment['words'])
    if not words or local['word_segments'] != words:
        raise ValueError('Frozen word views disagree')
    for word in words:
        if (not isinstance(word, dict) or not {'word'} <= set(word)
                or not set(word) <= {'word', 'start', 'end', 'score'}
                or not isinstance(word['word'], str)):
            raise ValueError('Unexpected aligned word')
    return previous.global_alignment(local, frozen['asr_evidence'])


def validate_input(inp, host):
    if (not isinstance(inp, dict) or set(inp) != {'profile', 'samples', 'frozen'}
            or inp['profile'] != PROFILE):
        raise ValueError('Only the fixed short-turn comparison is accepted')
    previous.validate_request_input({'profile': previous.PROFILE, 'samples': inp['samples'],
        'asr_evidence': inp['frozen']['asr_evidence']}, host)
    validate_frozen(inp['frozen'])
    return inp['samples'][0]


@contextmanager
def short_turn_filter(cluster, ratio, diagnostics):
    """Temporarily override only this instance's filtering ratio, not VBx itself."""
    if ratio not in RATIOS.values():
        raise ValueError('Unregistered filter setting')
    original = cluster.filter_embeddings
    parameters = inspect.signature(original).parameters
    if ('min_active_ratio' not in parameters or parameters['min_active_ratio'].default != 0.2):
        raise ValueError('Unsupported pyannote filtering API')
    had_instance_method = 'filter_embeddings' in vars(cluster)
    saved_instance_method = vars(cluster).get('filter_embeddings')

    def filtered(embeddings, segmentations=None, min_active_ratio=0.2):
        if min_active_ratio != 0.2:
            raise ValueError('Unexpected filter caller override')
        output = original(embeddings, segmentations=segmentations, min_active_ratio=ratio)
        diagnostics.append({'min_active_ratio': ratio,
                            'selected_embeddings': len(output[0]),
                            'chunk_speaker_pairs': int(embeddings.shape[0] * embeddings.shape[1])})
        return output

    cluster.filter_embeddings = filtered
    try:
        yield
    finally:
        if had_instance_method:
            cluster.filter_embeddings = saved_instance_method
        else:
            del cluster.filter_embeddings


def validate_runtime_pipeline(pipeline, package_version):
    if (package_version != '4.0.3'
            or type(pipeline.clustering).__name__ != 'VBxClustering'
            or pipeline.parameters(instantiated=True) != PARAMETERS
            or pipeline.embedding_exclude_overlap is not True):
        raise ValueError('Unexpected model/package/settings; experiment must be reviewed')


def verify_filter_api():
    """CPU smoke check against the installed pyannote implementation, no models."""
    from importlib.metadata import version
    from types import SimpleNamespace
    import numpy as np
    from pyannote.audio.pipelines.clustering import VBxClustering
    if version('pyannote.audio') != '4.0.3':
        raise ValueError('Unexpected pyannote version')
    cluster = VBxClustering(plda=None)
    embeddings = np.ones((1, 2, 3))
    segments = np.zeros((1, 100, 2))
    segments[0, :90, 0] = 1
    segments[0, 90:, 1] = 1
    for ratio, expected in ((.2, 1), (.1, 2)):
        diagnostics = []
        with short_turn_filter(cluster, ratio, diagnostics):
            selected = cluster.filter_embeddings(embeddings, SimpleNamespace(data=segments))[0]
        if len(selected) != expected or diagnostics[0]['selected_embeddings'] != expected:
            raise ValueError('Installed filter does not implement the reviewed comparison')
    if 'filter_embeddings' in vars(cluster):
        raise ValueError('Filter override was not restored')


def execute(manifest_path, output_path):
    require_test()
    manifest = read_json(manifest_path)
    sample, frozen = manifest['sample'], manifest['frozen']
    aligned = validate_frozen(frozen)
    duration = previous.validate_audio(sample['path'], sample['sha256'], frozen['asr_evidence'])
    state = {'profile': PROFILE, 'status': 'started', 'phase': 'assets', 'outputs': {},
             'sample_id': 'sound2', 'audio_sha256': sample['sha256'], 'duration_seconds': duration,
             'frozen': frozen, 'frozen_sha256': previous.evidence_sha(frozen),
             'asr_rerun': False, 'alignment_rerun': False, 'quality_validated': False,
             'source_manifest_sha256': digest(SOURCES)}
    started = time.perf_counter()
    try:
        diar_root = Path(sample['path']).parent / 'diarization'
        assets = prepare_diarization(diar_root, os.getenv('HF_TOKEN', ''))
        with offline_models():
            import torch
            import whisperx
            import pandas as pd
            from importlib.metadata import version
            from pyannote.audio import Pipeline
            pipeline = Pipeline.from_pretrained(str(diar_root), token=False).to(torch.device('cuda'))
            validate_runtime_pipeline(pipeline, version('pyannote.audio'))
            state['runtime'] = {'diarization': assets, 'parameters': model_evidence(pipeline.parameters(instantiated=True)),
                'embedding_exclude_overlap': pipeline.embedding_exclude_overlap,
                'segmentation_window_seconds': float(pipeline._segmentation.model.specifications.duration),
                'packages': {p: version(p) for p in ('torch', 'torchaudio', 'whisperx', 'pyannote.audio')}}
            audio = whisperx.load_audio(sample['path'])
            for name, ratio in RATIOS.items():
                state['phase'] = name
                torch.manual_seed(0)
                diagnostics = []
                with short_turn_filter(pipeline.clustering, ratio, diagnostics):
                    # No num_speakers, manual timings, role labels or reference words.
                    output = pipeline({'waveform': torch.from_numpy(audio[None, :]), 'sample_rate': previous.RATE})
                if len(diagnostics) != 1:
                    raise ValueError('Filtering hook did not run exactly once')
                for kind, annotation in (('regular', output.speaker_diarization),
                                         ('exclusive', output.exclusive_speaker_diarization)):
                    timeline = previous.annotation_rows(annotation, duration, kind == 'exclusive')
                    assigned = whisperx.assign_word_speakers(pd.DataFrame(timeline), deepcopy(aligned), fill_nearest=False)
                    assigned = model_evidence(assigned)
                    if previous.text_key(assigned['segments']) != previous.text_key(frozen['asr_evidence']['segments']):
                        raise ValueError('Frozen text changed')
                    state['outputs'][name + '_' + kind] = {'min_active_ratio': ratio,
                        'num_speakers': None, 'filter_diagnostics': diagnostics, 'timeline': timeline,
                        'aligned': assigned, 'export': previous.export_result(assigned)}
        state.update(status='completed', phase='complete')
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
        sample = validate_input(inp, os.getenv('QUALITY_R2_HOST', ''))
        with tempfile.TemporaryDirectory(prefix='short-turns-') as directory:
            root = Path(directory)
            stage = 'download'
            audio = root / 'sound2.wav'
            previous.fetch_sample(sample, audio)
            previous.validate_audio(audio, sample['sha256'], inp['frozen']['asr_evidence'])
            manifest, output = root / 'manifest.json', root / 'result.json'
            write_new(manifest, {'sample': {'path': str(audio), 'sha256': sample['sha256']}, 'frozen': inp['frozen']})
            stage = 'child'
            child = subprocess.run([sys.executable, str(Path(__file__)), '--manifest', str(manifest),
                                    '--output', str(output)], timeout=270, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL, check=False)
            if child.returncode or not output.is_file():
                raise RuntimeError('Short-turn comparison produced no result')
            return read_json(output)
    except Exception as exc:
        return {'profile': PROFILE, 'status': 'failed', 'phase': stage, 'error_type': type(exc).__name__}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    execute(args.manifest, args.output)
