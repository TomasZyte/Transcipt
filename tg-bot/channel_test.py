"""One split-channel test after fresh billing reconciliation; offline by default."""
import argparse
import asyncio
import base64
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
import logging
from pathlib import Path

import pause_context_test as previous
import channel_pipeline as worker
from integration_guard import OneJobRunner, SecretFormatter, TERMINAL, process_lock, save_json
from prepare_short_turn_test import stamp

full = previous.full


def history_files(runtime):
    found = previous.history_files(runtime)
    folder = runtime / previous.worker.PROFILE / 'jobs'
    if any(folder.glob('*.tmp')):
        raise ValueError('Unfinished prior journal')
    for path in folder.glob('*.json'):
        if path.name.startswith(('message-', 'failed-', 'retry-')):
            found[path.relative_to(runtime).as_posix()] = path
    return found


def validate_billing(billing, now=None):
    now = now or datetime.now(timezone.utc)
    if (billing.get('endpoint') != full.ENDPOINT or billing.get('period_ui') != 'окт. 1 31'
            or billing.get('period_start') != '2026-10-01' or billing.get('period_end') != '2026-10-31'
            or billing.get('period_timezone') != 'UTC' or billing.get('group') != 'Serverless / Endpoint'
            or billing.get('billing_lag_hours') != 1
            or not 0 <= (now - stamp(billing['observed_at_utc'])).total_seconds() <= 900):
        raise ValueError('Fresh whole-period test billing required')
    lines = billing['table'].splitlines()
    index = lines[0].split('\t').index(full.ENDPOINT)
    amounts = [full.money(line.split('\t')[index].replace('$', '').replace(',', '.'))
               for line in lines[1:] if line.strip()]
    total = sum(amounts, Decimal(0))
    if not amounts or total != full.money(billing['displayed_total_test_usd']):
        raise ValueError('Endpoint column disagrees with total')
    return total, Decimal('.001') * sum(value != 0 for value in amounts)


def reconcile(billing, health, shutdown, journals, now=None):
    now = now or datetime.now(timezone.utc)
    billed, rounding = validate_billing(billing, now)
    if (health.get('endpoint') != full.ENDPOINT or shutdown.get('endpoint') != full.ENDPOINT
            or not 0 <= (now - stamp(health['observed_at_utc'])).total_seconds() <= 900
            or health['health'] != shutdown['health'] or shutdown.get('max_workers') != 0
            or billing.get('max_workers') != 0 or billing.get('active_workers') != 0
            or billing.get('endpoint_idle_observed_this_turn') is not True):
        raise ValueError('Fresh idle endpoint must match previous shutdown')
    current = health['health']
    if (current['jobs']['inQueue'] or current['jobs']['inProgress'] or not current['workers']
            or any(current['workers'].values()) or len(journals) != 17
            or any(j.get('status') not in TERMINAL for j in journals)):
        raise ValueError('All seventeen historical attempts must be resolved')
    cutoff = stamp(shutdown['observed_at_utc']).timestamp()
    if stamp(billing['observed_at_utc']).timestamp() - cutoff < 4500:
        raise ValueError('Billing lag plus margin has not elapsed')
    for j in journals:
        if float(j.get('submitted_at') or 0) + float(j.get('elapsed_sec') or 0) > cutoff:
            raise ValueError('Journal newer than the prior shutdown')
    latest = max(journals, key=lambda j: float(j.get('submitted_at') or 0))
    if latest.get('pause_context_validated') is not True or latest.get('planning_total_usd') != '1.75':
        raise ValueError('Expected the completed previous context attempt')
    settled = max(Decimal('.25'), (billed + rounding + Decimal('.10')).quantize(Decimal('.01'), rounding=ROUND_CEILING))
    if settled + full.RESERVE_USD > Decimal('2'):
        raise ValueError('Cumulative test budget exceeded')
    return {'kind': 'closed_test_allowance_reconciliation', 'endpoint': full.ENDPOINT,
            'historical_journals': 17, 'previous_planning_total_usd': '1.75',
            'observed_billed_usd': str(billed), 'rounding_allowance_usd': str(rounding),
            'baseline_spend_usd': str(settled), 'next_reserve_usd': str(full.RESERVE_USD),
            'planning_total_usd': str(settled + full.RESERVE_USD), 'budget_usd': '2',
            'old_journals_modified': False}


class ChannelRunner(full.FullRunner):
    campaign_name = output_profile = worker.PROFILE
    history_files = staticmethod(history_files)
    health = previous.PauseRunner.health
    additional_preflight = {'channel_asr_enabled': True, 'pause_context_enabled': False,
        'acoustic_turns_enabled': False, 'timed_turns_enabled': False, 'release_candidate_enabled': False,
        'short_turn_comparison_enabled': False, 'speaker_comparison_enabled': False, 'context_asr_enabled': False}

    def __init__(self, settings, **kwargs):
        if settings.endpoint != full.ENDPOINT or settings.bucket != full.BUCKET:
            raise ValueError('Dedicated test resources required')
        self.base_settings = settings
        root = settings.runtime / self.campaign_name
        if root.resolve() != root or (root/'jobs').resolve() != root/'jobs':
            raise ValueError('Runtime cannot redirect')
        self.campaign_path = root/'campaign.json'
        self.campaign = c = full.read_json(self.campaign_path)
        self.campaign_sha = full.digest(self.campaign_path)
        commit = c.get('commit', '')
        if (c.get('schema') != 1 or c.get('campaign') != self.campaign_name
                or c.get('endpoint') != full.ENDPOINT or c.get('bucket') != full.BUCKET
                or len(commit) != 40 or any(ch not in '0123456789abcdef' for ch in commit)
                or c.get('image') != full.IMAGE_PREFIX + commit[:9]
                or c.get('source_manifest_sha256') != full.digest(full.SOURCES)):
            raise ValueError('Unexpected channel campaign identity')
        budget, prior = full.money(c['budget_usd']), full.money(c['baseline_spend_usd'])
        if not 0 < budget <= min(settings.budget, Decimal('2')) or prior < settings.prior_spend or prior + full.RESERVE_USD > budget:
            raise ValueError('Insufficient cumulative allowance')
        isolated = replace(settings, runtime=root, max_jobs=1, budget=budget, prior_spend=prior)
        OneJobRunner.__init__(self, isolated, self.campaign_name, timeout_s=full.CLIENT_SECONDS, **kwargs)
        self.manifest = Path(c['manifest'])
        self.plan = worker.adapter.prepare(self.manifest)
        if self.plan['manifest_sha256'] != c['manifest_sha256']:
            raise ValueError('Channel manifest changed')
        self.rows = [{'id': r['channel'], 'path': str(self.manifest.parent / r['file']), 'sha256': r['sha256']}
                     for r in self.plan['channels']]
        self.state.update(campaign_sha256=self.campaign_sha, channel_asr_validated=False,
                          audio_sha256={r['id']: r['sha256'] for r in self.rows})
        self.validate_history()
        prior_runner = previous.PauseRunner(settings)
        if c.get('previous_campaign_sha256') != prior_runner.campaign_sha:
            raise ValueError('Prior campaign changed')
        self.budget_paths = {key: Path(c[key + '_file']) for key in ('billing', 'health', 'shutdown', 'reconciliation')}
        self.budget_hashes = {key: c[key + '_sha256'] for key in self.budget_paths}
        self.validate_budget_files()
        observations = {key: full.read_json(path) for key, path in self.budget_paths.items()}
        if observations['shutdown']['health']['jobs'] != {'completed': 17, 'failed': 1, 'inProgress': 0, 'inQueue': 0, 'retried': 2}:
            raise ValueError('Prior shutdown history changed')
        if observations['shutdown'].get('release') != 47:
            raise ValueError('Expected the recorded context shutdown')
        at = max(stamp(observations[k]['observed_at_utc']) for k in ('billing', 'health'))
        if at > datetime.now(timezone.utc):
            raise ValueError('Future evidence')
        verified = reconcile(observations['billing'], observations['health'], observations['shutdown'],
                             [full.read_json(p) for p in history_files(settings.runtime).values()], now=at)
        record = observations['reconciliation']
        if record.get('history_sha256') != c['history_sha256'] or any(record.get(k) != v for k, v in verified.items()):
            raise ValueError('Reconciliation cannot be reproduced')
        if prior < full.money(verified['baseline_spend_usd']):
            raise ValueError('Prior spend omitted')
        self.expected_health = observations['health']['health']

    def validate_budget_files(self):
        if any(full.digest(path) != self.budget_hashes[k] for k, path in self.budget_paths.items()):
            raise ValueError('Budget evidence changed')

    def check_available(self):
        self.validate_history()
        self.validate_budget_files()
        if worker.adapter.prepare(self.manifest) != self.plan:
            raise ValueError('Input changed')
        OneJobRunner.check_available(self)
        if any(self.path.parent.glob('*.json')) or any(self.path.parent.glob('*.tmp')):
            raise RuntimeError('One attempt, never reset a campaign')

    def preflight(self, now=None):
        p = super().preflight(now)
        if full.digest(p['billing_file']) != p['billing_sha256']:
            raise ValueError('Fresh billing evidence changed')
        billed, _ = validate_billing(full.read_json(p['billing_file']), now)
        if billed != full.money(p['test_billed_usd']):
            raise ValueError('Preflight billing differs from observation')
        return p

    def payload(self, urls):
        if len(urls) != 2:
            raise ValueError('Exactly two channel URLs required')
        return {'profile': worker.PROFILE, 'source_sha256': self.plan['source_sha256'],
                'confirmation': self.plan['confirmation'], 'samples': [
                    {'id': r['id'], 'sha256': r['sha256'], 'media_url': url} for r, url in zip(self.rows, urls)]}

    def request_body(self, payload):
        if payload != self.payload([r.get('media_url') for r in payload.get('samples', [])]):
            raise ValueError('Unexpected input options')
        worker.validate_input(payload, self.settings.storage_host)
        return {'input': payload, 'policy': {'executionTimeout': full.EXECUTION_MS, 'ttl': full.TTL_MS}}

    def validate_output(self, result):
        worker.validate_result(result, self.plan)
        runtime = result['runtime']; assets = runtime['asr']
        config = worker.config_for('wx_ru'); source = full.read_json(full.SOURCES)['alignment']
        if assets.get('revision') != full.REVISION or not full.SHA.fullmatch(assets.get('lock_sha256', '')):
            raise ValueError('Unexpected ASR assets')
        if any(runtime['alignment'].get(k) != source[k] for k in ('repository', 'revision')):
            raise ValueError('Unexpected alignment assets')
        files = runtime['alignment'].get('files', {})
        if set(files) != set(source['files']) or any(not full.SHA.fullmatch(v) for v in files.values()):
            raise ValueError('Missing alignment hashes')
        if any(row.get('sha256') and files[k] != row['sha256'] for k, row in source['files'].items()):
            raise ValueError('Alignment weights changed')
        if runtime.get('packages') != config['packages'] or not full.SHA.fullmatch(runtime.get('nltk_lock_sha256', '')):
            raise ValueError('Unexpected runtime')
        self.state.update(channel_asr_validated=True, quality_validated=False, model_lock_sha256=assets['lock_sha256'])
        save_json(self.path, self.state)

    async def execute(self):
        with process_lock(self.base_settings.runtime):
            self.check_available(); self.preflight(); await self.health()
            urls = await asyncio.to_thread(upload_channels, self.settings, self.rows)
            self.check_available(); self.preflight(); await self.health()
            result = await OneJobRunner.__call__(self, self.payload(urls))
            self.validate_output(result)
            return result


def upload_channels(settings, rows):
    import boto3
    from botocore.config import Config
    if settings.endpoint != full.ENDPOINT or settings.bucket != full.BUCKET:
        raise ValueError('Only dedicated test resources')
    s3 = boto3.client('s3', endpoint_url='https://' + settings.storage_host,
        aws_access_key_id=settings.secrets['S3_ACCESS_KEY'], aws_secret_access_key=settings.secrets['S3_SECRET_KEY'],
        region_name='auto', config=Config(signature_version='s3v4', connect_timeout=5, read_timeout=20,
                                         retries={'total_max_attempts': 1}))
    urls = []
    for row in rows:
        data = Path(row['path']).read_bytes()
        if len(data) > worker.MAX_BYTES or hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError('Channel changed before upload')
        key = 'quality/' + worker.PROFILE + '/' + row['sha256'] + '/' + row['id'] + '.wav'
        s3.put_object(Bucket=full.BUCKET, Key=key, Body=data, ContentType='audio/wav',
                      ContentMD5=base64.b64encode(hashlib.md5(data).digest()).decode('ascii'))
        stream = s3.get_object(Bucket=full.BUCKET, Key=key)['Body']
        try:
            received = stream.read(worker.MAX_BYTES + 1)
        finally:
            stream.close()
        if hashlib.sha256(received).hexdigest() != row['sha256']:
            raise ValueError('R2 round-trip mismatch')
        urls.append(s3.generate_presigned_url('get_object', Params={'Bucket': full.BUCKET, 'Key': key}, ExpiresIn=1200))
    return urls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file', type=Path, required=True)
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    try:
        runner = ChannelRunner(full.TestSettings.load(args.env_file)); runner.check_available()
        if not args.submit:
            print(json.dumps({'status': 'prepared_not_run', 'network_calls': 0,
                'planning_total_usd': str(runner.settings.prior_spend + full.RESERVE_USD)}))
            return 0
        logging.basicConfig(level=logging.INFO)
        for handler in logging.getLogger().handlers:
            handler.setFormatter(SecretFormatter(runner.settings.secrets.values()))
        logging.getLogger('httpx').setLevel(logging.WARNING)
        asyncio.run(runner.execute())
        print('Channel evidence verified; stop test flags and Max workers. Quality review required.')
        return 0
    except Exception as exc:
        print(json.dumps({'status': 'stopped', 'error_type': type(exc).__name__, 'automatic_retry': False}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
