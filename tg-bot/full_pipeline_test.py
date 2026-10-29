"""One bounded sound1-4 batch on test resources. Default is offline preparation."""
import argparse
import asyncio
import base64
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import logging
from pathlib import Path
import sys

from integration_config import TestSettings, RESERVE_USD
from integration_guard import OneJobRunner, SecretFormatter, TERMINAL, process_lock, save_json
from asr_test import AsrRunner, money, ENDPOINT, BUCKET, REVISION
from quality_test import load_samples
from asr_comparison import digest, read_json, SHA
from full_pipeline import IDS, MODES, PROFILE, validate_audio_rows, format_output
from prepare_full_assets import SOURCES
from quality_batch_handler import validate_request

CAMPAIGN = PROFILE
EXECUTION_MS, TTL_MS, CLIENT_SECONDS = 300_000, 420_000, 450
IMAGE_PREFIX = 'registry.runpod.net/tomaszyte-transcipt-codex-full-pipeline-test-transcribe-backend-worker-dockerfile-asr-full:'


def history_files(runtime):
    found = {}
    for folder in (runtime/'jobs', runtime/'asr-children-v1'/'jobs'):
        if any(folder.glob('*.tmp')):
            raise ValueError('Historical journal contains an incomplete write')
        for path in folder.glob('*.json'):
            if path.name.startswith(('message-', 'retry-', 'failed-')):
                found[path.relative_to(runtime).as_posix()] = path
    return found


class FullRunner(OneJobRunner):
    health = AsrRunner.health
    campaign_name, output_profile, modes = CAMPAIGN, PROFILE, MODES
    validated_key = 'full_pipeline_validated'
    history_files = staticmethod(history_files)
    additional_preflight = {}

    def __init__(self, settings, **kwargs):
        if settings.endpoint != ENDPOINT or settings.bucket != BUCKET:
            raise ValueError('Dedicated test resources required')
        self.base_settings = settings
        root = settings.runtime/self.campaign_name
        if root.resolve() != root or (root/'jobs').resolve() != root/'jobs':
            raise ValueError('Runtime must not redirect through links')
        self.campaign_path = root/'campaign.json'
        c = self.campaign = read_json(self.campaign_path)
        self.campaign_sha = digest(self.campaign_path)
        commit = c.get('commit', '')
        if (c.get('schema') != 1 or c.get('campaign') != self.campaign_name
                or c.get('endpoint') != ENDPOINT or c.get('bucket') != BUCKET
                or c.get('modes') != self.modes or len(commit) != 40
                or any(ch not in '0123456789abcdef' for ch in commit)
                or c.get('image') != IMAGE_PREFIX + commit[:9]
                or c.get('source_manifest_sha256') != digest(SOURCES)):
            raise ValueError('Unexpected full pipeline campaign identity')
        budget, prior = money(c['budget_usd']), money(c['baseline_spend_usd'])
        if (not 0 < budget <= min(settings.budget, Decimal('2'))
                or prior < settings.prior_spend or prior + RESERVE_USD > budget):
            raise ValueError('The reserved batch exceeds the cumulative budget')
        isolated = replace(settings, runtime=root, max_jobs=1, budget=budget, prior_spend=prior)
        super().__init__(isolated, self.campaign_name, timeout_s=CLIENT_SECONDS, **kwargs)
        manifest = Path(c['manifest'])
        if digest(manifest) != c['manifest_sha256']:
            raise ValueError('Audio manifest changed')
        self.rows = [{'id':row['id'], 'sha256':row['wav_sha256'], 'path':str(path)}
                     for row,path in load_samples(manifest.parent)]
        validate_audio_rows(self.rows)
        if {r['id']:r['sha256'] for r in self.rows} != c['audio_sha256']:
            raise ValueError('Audio identity changed')
        self.state.update(campaign_sha256=self.campaign_sha,
                          audio_sha256=c['audio_sha256'], source_manifest_sha256=digest(SOURCES))
        self.state[self.validated_key] = False
        self.validate_history()

    def validate_history(self):
        if digest(self.campaign_path) != self.campaign_sha:
            raise ValueError('Campaign changed after preparation')
        actual, expected = self.history_files(self.base_settings.runtime), self.campaign.get('history_sha256')
        if not expected or set(actual) != set(expected):
            raise ValueError('History changed; reconcile billing first')
        for name,path in actual.items():
            if digest(path) != expected[name] or read_json(path).get('status') not in TERMINAL:
                raise ValueError('Historical attempt changed or remains unresolved')

    def check_available(self):
        self.validate_history()
        validate_audio_rows(self.rows)
        super().check_available()
        if any(self.path.parent.glob('*.json')) or any(self.path.parent.glob('*.tmp')):
            raise RuntimeError('This campaign permits one attempt and no reset')

    def preflight(self, now=None):
        path = self.settings.runtime/'preflight.json'
        p = read_json(path)
        stamp = datetime.fromisoformat(p['verified_at_utc'].replace('Z','+00:00'))
        now = now or datetime.now(timezone.utc)
        if stamp.tzinfo is None or not 0 <= (now-stamp).total_seconds() <= 900:
            raise ValueError('Refresh the UI observation within 15 minutes')
        expected = {'endpoint':ENDPOINT, 'image':self.campaign['image'], 'paid_run_authorized':True,
                    'active_workers':0, 'max_workers':1, 'gpu_count':1,
                    'execution_timeout_seconds':300, 'idle_timeout_seconds':5,
                    'flashboot':False, 'full_pipeline_enabled':True, 'asr_comparison_enabled':False,
                    'campaign_sha256':self.campaign_sha}
        expected.update(self.additional_preflight)
        if any(type(p.get(k)) is not type(v) or p[k] != v for k,v in expected.items()):
            raise ValueError('Preflight does not confirm the bounded test setup')
        if not 0 < money(p['gpu_rate_usd_per_hour']) <= Decimal('0.75'):
            raise ValueError('Unexpected GPU rate')
        used = max(self.settings.prior_spend, money(p['test_billed_usd']))
        if used+RESERVE_USD > self.settings.budget:
            raise RuntimeError('Current billing exceeds the reserved cumulative budget')
        self.state.update(preflight_sha256=digest(path),planning_total_usd=str(used+RESERVE_USD))
        return p

    def payload(self, urls):
        if len(urls) != 4:
            raise ValueError('Exactly four full files required')
        return {'profile':self.output_profile, 'samples':[{'id':r['id'],'sha256':r['sha256'],'media_url':url}
                                             for r,url in zip(self.rows,urls)]}

    def request_body(self, payload):
        urls = [s.get('media_url') for s in payload.get('samples',[])]
        if payload != self.payload(urls):
            raise ValueError('No options, speaker counts or reference text may be sent')
        validate_request({'profile':'community','samples':payload['samples']},self.settings.storage_host)
        return {'input':payload,'policy':{'executionTimeout':EXECUTION_MS,'ttl':TTL_MS}}

    def validate_provenance(self, result):
        if (result.get('profile') != self.output_profile or result.get('status') != 'completed'
                or result.get('modes') != self.modes or result.get('source_manifest_sha256') != digest(SOURCES)):
            raise ValueError('Incomplete or mismatching full-pipeline result')
        sources, runtime = read_json(SOURCES), result.get('runtime',{})
        assets = runtime.get('asr',{})
        if (assets.get('revision') != REVISION or not SHA.fullmatch(assets.get('lock_sha256',''))
                or not assets.get('packages') or not SHA.fullmatch(runtime.get('nltk_lock_sha256',''))):
            raise ValueError('Missing model/package provenance')
        for kind in ('alignment','diarization'):
            if (runtime.get(kind,{}).get('revision') != sources[kind]['revision']
                    or runtime[kind].get('repository') != sources[kind]['repository']):
                raise ValueError('Unexpected model revision')
        return assets

    def validate_output(self, result):
        assets = self.validate_provenance(result)
        samples = result.get('samples',[])
        if len(samples) != 4:
            raise ValueError('Expected eight outputs for four files')
        for sample,row in zip(samples,self.rows):
            if (sample.get('id') != row['id'] or sample.get('sha256') != row['sha256']
                    or sample.get('durationSeconds') != row['duration_seconds']
                    or set(sample.get('outputs',{})) != set(MODES)):
                raise ValueError('Sample identity or mode mismatch')
            timelines = []
            for mode,output in sample['outputs'].items():
                b = output['benchmark']
                timelines.append(b['diarizationTimeline'])
                rebuilt = format_output({'segments':b['asrSegments'],'language':output['languageDetected']},
                    {'segments':b['alignedSegments']},b['diarizationTimeline'],mode,row['duration_seconds'])
                for key in ('segments','rawText','rawAsrText','srtContent','audit','wordCount'):
                    if output[key] != rebuilt[key]:
                        raise ValueError('Export does not match its ASR/alignment evidence')
                for key,value in rebuilt['processing'].items():
                    if output['processing'].get(key) != value:
                        raise ValueError('Unexpected processing mode')
            if timelines[0] != timelines[1] or timelines[0].get('status') != 'ok':
                raise ValueError('The two modes must share a complete full-file speaker timeline')
        self.state.update(full_pipeline_validated=True,model_lock_sha256=assets['lock_sha256'])
        save_json(self.path,self.state)

    async def execute(self):
        with process_lock(self.base_settings.runtime):
            self.check_available()
            self.preflight()
            await self.health()
            urls = await asyncio.to_thread(upload_samples,self.settings,self.rows,campaign=self.campaign_name)
            self.check_available()
            self.preflight()
            await self.health()
            result = await super().__call__(self.payload(urls))
            self.validate_output(result)
            return result

    async def __call__(self, *args, **kwargs):
        raise RuntimeError('Use execute() to enforce preflight and the shared process lock')


def upload_samples(settings, rows, *, campaign=CAMPAIGN):
    import boto3
    from botocore.config import Config
    validate_audio_rows(rows)
    s3 = boto3.client('s3',endpoint_url='https://'+settings.storage_host,
        aws_access_key_id=settings.secrets['S3_ACCESS_KEY'],aws_secret_access_key=settings.secrets['S3_SECRET_KEY'],
        region_name='auto',config=Config(signature_version='s3v4',connect_timeout=5,read_timeout=20,
                                       retries={'total_max_attempts':1}))
    urls = []
    for row in rows:
        data = Path(row['path']).read_bytes()
        if hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError('Audio changed before upload')
        key = 'quality/'+campaign+'/'+row['sha256']+'/'+row['id']+'.wav'
        s3.put_object(Bucket=BUCKET,Key=key,Body=data,ContentType='audio/wav',
                      ContentMD5=base64.b64encode(hashlib.md5(data).digest()).decode('ascii'))
        stream = s3.get_object(Bucket=BUCKET,Key=key)['Body']
        try:
            received = stream.read(2*1024*1024+1)
        finally:
            stream.close()
        if hashlib.sha256(received).hexdigest() != row['sha256']:
            raise ValueError('R2 round-trip checksum mismatch')
        urls.append(s3.generate_presigned_url('get_object',Params={'Bucket':BUCKET,'Key':key},ExpiresIn=1200))
    return urls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file',type=Path,required=True)
    parser.add_argument('--submit',action='store_true')
    args = parser.parse_args()
    try:
        settings = TestSettings.load(args.env_file)
        runner = FullRunner(settings)
        runner.check_available()
        if not args.submit:
            print(json.dumps({'status':'prepared_not_run','samples':IDS,'modes':MODES,
                'attempts':1,'prior_allowance_usd':str(runner.settings.prior_spend),
                'planning_total_usd':str(runner.settings.prior_spend+RESERVE_USD),'network_calls':0}))
            return 0
        logging.basicConfig(level=logging.INFO)
        for handler in logging.getLogger().handlers:
            handler.setFormatter(SecretFormatter(settings.secrets.values()))
        logging.getLogger('httpx').setLevel(logging.WARNING)
        asyncio.run(runner.execute())
        print('Four full-file comparisons completed and validated. Return test Max workers and flag to 0.')
        return 0
    except Exception as exc:
        print('Full pipeline stopped ('+type(exc).__name__+'); inspect the private journal before any further action.')
        return 1


if __name__ == '__main__':
    sys.exit(main())
