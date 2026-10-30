"""Dedicated test batch entry point; never copied into the production image."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from asr_comparison import read_json, write_new
from full_pipeline import PROFILE, IDS, MODES, validate_audio_rows
import context_pipeline
from quality_batch_handler import validate_request as validate_r2, fetch_sample

CHILD_TIMEOUT = 270


def validate(inp):
    if os.getenv('FULL_PIPELINE_IMAGE') != '1' or os.getenv('FULL_PIPELINE_TEST') != '1':
        raise ValueError('Isolated full-pipeline test mode is required')
    if not isinstance(inp,dict) or set(inp) != {'profile','samples'} or inp['profile'] not in (PROFILE,context_pipeline.PROFILE):
        raise ValueError('Only the fixed full pipeline profile is supported')
    if inp['profile'] == context_pipeline.PROFILE:
        context_pipeline.require_context()
    samples = inp['samples']
    if not isinstance(samples,list) or len(samples) != 4 or any(
            not isinstance(s,dict) or set(s) != {'id','sha256','media_url'} for s in samples):
        raise ValueError('No options, references or speaker counts may be supplied')
    if [s['id'] for s in samples] != IDS:
        raise ValueError('Expected sound1-4 in fixed order')
    validate_r2({'profile':'community','samples':samples},os.getenv('QUALITY_R2_HOST',''))
    return samples


def handler(job):
    if isinstance(job.get('input'),dict) and job['input'].get('profile') == 'release-candidate-v1':
        import candidate_pipeline
        return candidate_pipeline.handler(job)
    if isinstance(job.get('input'),dict) and job['input'].get('profile') == 'speaker-short-turns-v1':
        import short_turn_comparison
        return short_turn_comparison.handler(job)
    if isinstance(job.get('input'),dict) and job['input'].get('profile') == 'speaker-children-v1':
        import speaker_comparison
        return speaker_comparison.handler(job)
    stage = 'validate'
    try:
        samples = validate(job.get('input'))
        profile = job['input']['profile']
        with tempfile.TemporaryDirectory(prefix='full-pipeline-') as directory:
            root = Path(directory)
            rows = []
            stage = 'download_audio'
            for sample in samples:
                path = root/(sample['id']+'.wav')
                fetch_sample(sample,path)
                rows.append({'id':sample['id'],'sha256':sample['sha256'],'path':str(path)})
            validate_audio_rows(rows)
            manifest, output = root/'manifest.json', root/'result.json'
            write_new(manifest,{'samples':rows})
            stage = 'child'
            child = subprocess.run([sys.executable,str(Path(__file__).with_name('full_pipeline.py')),
                '--manifest',str(manifest),'--output',str(output),'--profile',profile],timeout=CHILD_TIMEOUT,
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=False)
            if child.returncode != 0 or not output.is_file():
                raise RuntimeError('Full pipeline child did not produce evidence')
            result = read_json(output)
            expected_modes = context_pipeline.MODES if profile == context_pipeline.PROFILE else MODES
            if result.get('profile') != profile or result.get('modes') != expected_modes:
                raise ValueError('Unexpected full pipeline child output')
            return result
    except Exception as exc:
        return {'profile':PROFILE,'status':'failed','phase':stage,'error_type':type(exc).__name__}


if __name__ == '__main__':
    import runpod
    runpod.serverless.start({'handler':handler})
