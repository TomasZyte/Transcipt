"""One GigaAM comparison on the saved sound2 clip; no API calls by default."""
import argparse
import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
import json
import logging
from pathlib import Path
import re
import sys

from asr_test import AsrRunner, ENDPOINT, BUCKET, CLIENT_SECONDS, money
from asr_comparison import prepare, digest, read_json, normalize_result, SHA
from context_test import history_files as previous_history
from integration_config import TestSettings, RESERVE_USD
from integration_guard import OneJobRunner, SecretFormatter, save_json, TERMINAL

CAMPAIGN = 'gigaam-children-v1'
VARIANT = 'gigaam_v3'
REVISION = '7447938d791c4f3e643386ee22c33777004293a5'
COMMIT = '2bd8f040f0bfa6f38c79fb241f634c73869022a2'
IMAGE = 'registry.runpod.net/tomaszyte-transcipt-codex-asr-build-check-transcribe-backend-worker-dockerfile-asr-gigaam:' + COMMIT[:9]


def history_files(runtime):
    found = previous_history(runtime)
    folder = runtime/'context-pipeline-v1'/'jobs'
    if any(folder.glob('*.tmp')):
        raise ValueError('Context journal has an incomplete write')
    for path in folder.glob('*.json'):
        if path.name.startswith(('message-', 'retry-', 'failed-')):
            found[path.relative_to(runtime).as_posix()] = path
    return found


class GigaamRunner(AsrRunner):
    def __init__(self, settings, **kwargs):
        if settings.endpoint != ENDPOINT or settings.bucket != BUCKET:
            raise ValueError('Dedicated test resources required')
        self.base_settings = settings
        root = settings.runtime/CAMPAIGN
        if root.resolve() != root or (root/'jobs').resolve() != root/'jobs':
            raise ValueError('Test runtime must not redirect through links')
        self.campaign_path = root/'campaign.json'
        c = self.campaign = read_json(self.campaign_path)
        self.campaign_sha = digest(self.campaign_path)
        expected = {'schema':1, 'campaign':CAMPAIGN, 'endpoint':ENDPOINT,
                    'bucket':BUCKET, 'variant':VARIANT, 'image':IMAGE,
                    'commit':COMMIT, 'model_revision':REVISION, 'clip_id':'context'}
        if any(type(c.get(k)) is not type(v) or c[k] != v for k,v in expected.items()):
            raise ValueError('Unexpected GigaAM campaign identity')
        budget, prior = money(c['budget_usd']), money(c['baseline_spend_usd'])
        # Previous context job is still inside the provider's billing lag. Keep its
        # reserved allowance rather than replacing it with the older displayed bill.
        previous = read_json(settings.runtime/'context-pipeline-v1/jobs/message-context-pipeline-v1.json')
        if (previous.get('status') not in TERMINAL
                or prior < max(settings.prior_spend, money(previous['planning_total_usd']))
                or not 0 < budget <= min(settings.budget, Decimal('2'))
                or prior+RESERVE_USD > budget):
            raise ValueError('Prior allowance or total budget is not covered')
        isolated = replace(settings, runtime=root, max_jobs=1, budget=budget, prior_spend=prior)
        OneJobRunner.__init__(self, isolated, CAMPAIGN, timeout_s=CLIENT_SECONDS, **kwargs)
        self.variant = VARIANT
        if digest(c['manifest']) != c['manifest_sha256']:
            raise ValueError('Audio manifest changed')
        self.plan = prepare(Path(c['manifest']), 'context', VARIANT)
        if self.plan['audio_sha256'] != c['audio_sha256']:
            raise ValueError('Unexpected comparison clip')
        self.state.update(campaign_sha256=self.campaign_sha, variant=VARIANT,
                          audio_sha256=self.plan['audio_sha256'], asr_validated=False)
        self.validate_history()

    def validate_history(self):
        if digest(self.campaign_path) != self.campaign_sha:
            raise ValueError('Campaign changed after preparation')
        actual, expected = history_files(self.base_settings.runtime), self.campaign['history_sha256']
        if not expected or set(actual) != set(expected):
            raise ValueError('Historical attempts changed; reconcile before proceeding')
        for name,path in actual.items():
            if digest(path) != expected[name] or read_json(path).get('status') not in TERMINAL:
                raise ValueError('History changed or contains an unresolved attempt')

    def check_available(self):
        self.validate_history()
        if (digest(self.campaign['manifest']) != self.campaign['manifest_sha256']
                or digest(self.plan['audio_file']) != self.plan['audio_sha256']):
            raise ValueError('Comparison audio changed')
        OneJobRunner.check_available(self)
        if any(self.path.parent.glob('*.json')) or any(self.path.parent.glob('*.tmp')):
            raise RuntimeError('One attempt only; no automatic retry or reset')

    def preflight(self, now=None):
        path = self.settings.runtime/'preflight.json'
        p = read_json(path)
        stamp = datetime.fromisoformat(p['verified_at_utc'].replace('Z','+00:00'))
        now = now or datetime.now(timezone.utc)
        if stamp.tzinfo is None or not 0 <= (now-stamp).total_seconds() <= 900:
            raise ValueError('Refresh UI settings within 15 minutes')
        expected = {'endpoint':ENDPOINT,'image':IMAGE,'paid_run_authorized':True,
            'active_workers':0,'max_workers':1,'gpu_count':1,'execution_timeout_seconds':120,
            'idle_timeout_seconds':5,'flashboot':False,'asr_execution_enabled':True,
            'full_pipeline_enabled':False,'context_asr_enabled':False,
            'campaign_sha256':self.campaign_sha,'variant':VARIANT}
        if any(type(p.get(k)) is not type(v) or p[k] != v for k,v in expected.items()):
            raise ValueError('Preflight does not confirm isolated GigaAM settings')
        if not 0 < money(p['gpu_rate_usd_per_hour']) <= Decimal('0.75'):
            raise ValueError('Unexpected GPU rate')
        used = max(self.settings.prior_spend, money(p['test_billed_usd']))
        if used+RESERVE_USD > self.settings.budget:
            raise RuntimeError('Cumulative budget exhausted')
        self.state.update(preflight_sha256=digest(path),planning_total_usd=str(used+RESERVE_USD))
        return p

    def validate_output(self, result):
        o = result.get('output', {})
        p = self.plan
        expected = {'status':'completed','variant_id':VARIANT,'clip_id':p['clip_id'],
            'audio_sha256':p['audio_sha256'],'duration_seconds':p['duration_seconds'],
            'source_start_seconds':p['source_start_seconds'],'source_end_seconds':p['source_end_seconds'],
            'config':p['config'],'timestamp_kind':'unavailable','language_detected':None,
            'network_submission':False}
        if (result.get('profile') != 'asr-comparison' or result.get('sample_id') != p['clip_id']
                or not isinstance(o,dict) or any(o.get(k) != v for k,v in expected.items())):
            raise ValueError('Result does not match the prepared GigaAM comparison')
        rebuilt = normalize_result(p,o['raw_engine_output'])
        if any(o.get(k) != rebuilt[k] for k in ('raw_asr_text','raw_asr_segments','timestamp_kind')):
            raise ValueError('Transcript changed from its raw evidence')
        runtime = o.get('runtime', {})
        lock = runtime.get('lock_sha256','')
        packages = runtime.get('packages',{})
        if isinstance(packages,dict):
            packages = {re.sub(r'[-_.]+','-',str(k)).lower():v for k,v in packages.items()}
        if (runtime.get('revision') != REVISION or not isinstance(lock,str) or not SHA.fullmatch(lock)
                or not isinstance(packages,dict) or not {'gigaam','torch','torchaudio','omegaconf','hydra-core'} <= set(packages)
                or any(not isinstance(packages[k],str) or packages[k].split('+')[0] != '2.8.0'
                       for k in ('torch','torchaudio'))):
            raise ValueError('Model provenance missing or changed')
        self.state.update(asr_validated=True,quality_validated=False,model_lock_sha256=lock)
        save_json(self.path,self.state)

    def summary(self):
        return {'status':'prepared_not_run','variant':VARIANT,'duration_seconds':self.plan['duration_seconds'],
            'audio_sha256':self.plan['audio_sha256'],'max_attempts':1,'baseline_usd':str(self.settings.prior_spend),
            'planning_total_usd':str(self.settings.prior_spend+RESERVE_USD),'budget_usd':str(self.settings.budget),
            'network_calls':0,'reservation_created':False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file',type=Path,required=True)
    parser.add_argument('--submit',action='store_true')
    args = parser.parse_args()
    try:
        settings = TestSettings.load(args.env_file)
        runner = GigaamRunner(settings)
        runner.check_available()
        if not args.submit:
            print(json.dumps(runner.summary()))
            return 0
        logging.basicConfig(level=logging.INFO)
        for handler in logging.getLogger().handlers:
            handler.setFormatter(SecretFormatter(settings.secrets.values()))
        logging.getLogger('httpx').setLevel(logging.WARNING)
        asyncio.run(runner.execute())
        print('GigaAM evidence saved and verified; disable test flag and Max workers.')
        return 0
    except Exception as exc:
        print(json.dumps({'status':'stopped','error_type':type(exc).__name__,'automatic_retry':False}))
        return 1


if __name__ == '__main__':
    sys.exit(main())
