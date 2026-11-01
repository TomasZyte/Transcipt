"""One guarded contextual WhisperX test; fresh billing and all prior reserves retained."""
import argparse
import asyncio
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
import json
import logging
from pathlib import Path
import re

import channel_context_test as previous
import channel_test as base
from integration_guard import SecretFormatter, TERMINAL
from prepare_short_turn_test import stamp

full = previous.full
PROFILE = 'channel-context-guard-v1'
PERIOD = previous.PERIOD
EXPECTED_JOBS = {'completed': 23, 'failed': 1, 'inProgress': 0, 'inQueue': 0, 'retried': 2}


def history_files(runtime):
    found = previous.history_files(runtime)
    folder = runtime/previous.PROFILE/'jobs'
    if any(folder.glob('*.tmp')):
        raise ValueError('Unfinished prior journal')
    for path in folder.glob('*.json'):
        if path.name.startswith(('message-', 'failed-', 'retry-')):
            found[path.relative_to(runtime).as_posix()] = path
    return found


def reconcile(billing, health, shutdown, journals, now=None):
    now = now or datetime.now(timezone.utc)
    billed, rounding = base.validate_billing(billing, now, period=PERIOD)
    if (health.get('endpoint') != full.ENDPOINT or shutdown.get('endpoint') != full.ENDPOINT
            or not 0 <= (now-stamp(health['observed_at_utc'])).total_seconds() <= 900
            or health['health'] != shutdown['health'] or health['health']['jobs'] != EXPECTED_JOBS
            or shutdown.get('release') != 73 or shutdown.get('max_workers') != 0
            or shutdown.get('active_workers') != 0 or billing.get('max_workers') != 0
            or billing.get('active_workers') != 0 or billing.get('endpoint_idle_observed_this_turn') is not True
            or not health['health']['workers'] or any(health['health']['workers'].values())
            or len(journals) != 23 or any(j.get('status') not in TERMINAL for j in journals)):
        raise ValueError('All twenty-three prior attempts must match the stopped endpoint')
    cutoff = stamp(shutdown['observed_at_utc']).timestamp()
    if cutoff > now.timestamp() or any(float(j.get('submitted_at') or 0)
            + float(j.get('elapsed_sec') or 0) > cutoff for j in journals):
        raise ValueError('Unsettled job after shutdown')
    latest = max(journals, key=lambda j: float(j.get('submitted_at') or 0))
    if latest.get('channel_asr_validated') is not True or latest.get('planning_total_usd') != '1.33':
        raise ValueError('Expected the completed context attempt')
    # Billing may still lag the latest job. Do not release ANY previous reserve.
    prior_allowance = full.money(latest['planning_total_usd'])
    baseline = max(prior_allowance, (billed+rounding+Decimal('.10')).quantize(Decimal('.01'), rounding=ROUND_CEILING))
    if baseline+full.RESERVE_USD > Decimal('2'):
        raise ValueError('Cumulative budget exhausted')
    return {'kind': 'fresh_billing_with_all_prior_reserves_retained', 'endpoint': full.ENDPOINT,
            'historical_journals': 23, 'previous_planning_total_usd': '1.33',
            'observed_billed_usd': str(billed), 'rounding_allowance_usd': str(rounding),
            'baseline_spend_usd': str(baseline), 'next_reserve_usd': str(full.RESERVE_USD),
            'planning_total_usd': str(baseline+full.RESERVE_USD), 'budget_usd': '2',
            'released_prior_reserve_usd': '0', 'old_journals_modified': False}


class GuardRunner(base.ChannelRunner):
    campaign_name = output_profile = PROFILE
    history_files = staticmethod(history_files)
    additional_preflight = {**base.ChannelRunner.additional_preflight, 'channel_context_enabled': True, 'channel_context_guard_enabled': True}

    def bind_prior(self, settings):
        prior = previous.ContextRunner(settings)
        c = self.campaign
        if c['previous_campaign_sha256'] != prior.campaign_sha:
            raise ValueError('Prior campaign changed')
        self.budget_paths = {k: Path(c[k+'_file']) for k in ('billing', 'health', 'shutdown', 'reconciliation')}
        self.budget_hashes = {k: c[k+'_sha256'] for k in self.budget_paths}
        self.validate_budget_files()
        observations = {k: full.read_json(p) for k,p in self.budget_paths.items()}
        at = max(stamp(observations[k]['observed_at_utc']) for k in ('billing','health'))
        if at > datetime.now(timezone.utc):
            raise ValueError('Future evidence')
        verified = reconcile(observations['billing'],observations['health'],observations['shutdown'],
                             [full.read_json(p) for p in self.history_files(settings.runtime).values()],now=at)
        record = observations['reconciliation']
        if (record.get('history_sha256') != c['history_sha256']
                or any(record.get(k) != v for k,v in verified.items())
                or self.settings.prior_spend < full.money(verified['baseline_spend_usd'])):
            raise ValueError('Reconciliation cannot be reproduced')
        self.expected_health = observations['health']['health']

    def preflight(self, now=None):
        p = full.FullRunner.preflight(self, now)
        if full.digest(p['billing_file']) != p['billing_sha256']:
            raise ValueError('Fresh billing changed')
        billed, _ = base.validate_billing(full.read_json(p['billing_file']), now, period=PERIOD)
        if billed != full.money(p['test_billed_usd']):
            raise ValueError('Preflight bill differs')
        # RunPod adds a generated suffix to repository names. Keep both the
        # logical campaign tag and separately observed real image/digest.
        actual = p.get('observed_image', '')
        repository, separator, tag = actual.rpartition(':')
        expected_repo = full.IMAGE_PREFIX.rpartition(':')[0]
        if (not separator or not (repository == expected_repo or repository.startswith(expected_repo+'_'))
                or tag != self.campaign['commit'][:9]
                or not re.fullmatch(r'sha256:[0-9a-f]{64}', p.get('observed_image_digest', ''))):
            raise ValueError('Actual test image and digest must be observed')
        return p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file',type=Path,required=True)
    parser.add_argument('--submit',action='store_true')
    args = parser.parse_args()
    try:
        runner = GuardRunner(full.TestSettings.load(args.env_file))
        runner.check_available()
        if not args.submit:
            print(json.dumps({'status':'prepared_not_run','network_calls':0,'max_attempts':1,
                              'planning_total_usd':str(runner.settings.prior_spend+full.RESERVE_USD)}))
            return 0
        logging.basicConfig(level=logging.INFO)
        for handler in logging.getLogger().handlers:
            handler.setFormatter(SecretFormatter(runner.settings.secrets.values()))
        logging.getLogger('httpx').setLevel(logging.WARNING)
        asyncio.run(runner.execute())
        print('Context evidence verified; stop guard/context/channel/full flags and Max workers. Quality is unverified.')
        return 0
    except Exception as exc:
        print(json.dumps({'status':'stopped','error_type':type(exc).__name__,'automatic_retry':False}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
