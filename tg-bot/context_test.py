"""Context experiment: offline plan by default; a fresh bounded campaign to submit."""
import argparse
import asyncio
import json
import logging
from pathlib import Path
import sys
import wave

import full_pipeline_test as full
from contextual_asr import plan_windows, RATE, MODE
from context_pipeline import PROFILE, MODES, validate_sample
from integration_guard import save_json, SecretFormatter


def history_files(runtime):
    found = full.history_files(runtime)
    folder = runtime/full.CAMPAIGN/'jobs'
    if any(folder.glob('*.tmp')):
        raise ValueError('Full pipeline journal contains an incomplete write')
    for path in folder.glob('*.json'):
        if path.name.startswith(('message-','retry-','failed-')):
            found[path.relative_to(runtime).as_posix()] = path
    return found


class ContextRunner(full.FullRunner):
    campaign_name = output_profile = PROFILE
    modes = MODES
    validated_key = 'context_pipeline_validated'
    history_files = staticmethod(history_files)
    additional_preflight = {'context_asr_enabled':True}

    def validate_output(self, result):
        assets = self.validate_provenance(result)
        samples = result.get('samples',[])
        if len(samples) != len(self.rows):
            raise ValueError('All four context comparisons are required')
        for sample,row in zip(samples,self.rows):
            if (sample.get('id') != row['id'] or sample.get('sha256') != row['sha256']
                    or sample.get('durationSeconds') != row['duration_seconds']):
                raise ValueError('Sample identity mismatch')
            with wave.open(row['path'],'rb') as wav:
                frames = wav.getnframes()
            validate_sample(sample['outputs'],frames,row['duration_seconds'])
        self.state.update(context_pipeline_validated=True,quality_validated=False,
            model_lock_sha256=assets['lock_sha256'],
            context_status={s['id']:s['outputs'][MODE]['status'] for s in samples})
        save_json(self.path,self.state)


def offline_plan(folder):
    rows = [{'id':row['id'],'sha256':row['wav_sha256'],'path':str(path)}
            for row,path in full.load_samples(folder)]
    full.validate_audio_rows(rows)
    for row in rows:
        with wave.open(row['path'],'rb') as wav:
            row['windows'] = plan_windows(wav.getnframes())
    return {'profile':PROFILE,'status':'prepared_not_run','modes':MODES,'samples':rows,
        'network_calls':0,'quality_validated':False,'paid_run_ready':False,
        'next_required':'Fresh campaign, cumulative budget and enabled preflight'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',type=Path)
    parser.add_argument('--env-file',type=Path)
    parser.add_argument('--submit',action='store_true')
    args = parser.parse_args()
    try:
        if not args.submit:
            if args.corpus is None:
                raise ValueError('Offline planning requires --corpus')
            print(json.dumps(offline_plan(args.corpus),ensure_ascii=True))
            return 0
        if args.env_file is None or args.corpus is not None:
            raise ValueError('Submission requires only --env-file and --submit')
        settings = full.TestSettings.load(args.env_file)
        runner = ContextRunner(settings)
        logging.basicConfig(level=logging.INFO)
        for handler in logging.getLogger().handlers:
            handler.setFormatter(SecretFormatter(settings.secrets.values()))
        logging.getLogger('httpx').setLevel(logging.WARNING)
        asyncio.run(runner.execute())
        print('Context evidence validated. Quality remains unvalidated; disable test flags and Max workers.')
        return 0
    except Exception as exc:
        print('Context experiment stopped ('+type(exc).__name__+'); no automatic retry.')
        return 1


if __name__ == '__main__':
    sys.exit(main())
