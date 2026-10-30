"""One acoustic comparison after an append-only, lag-aware budget reconciliation."""
import argparse
import asyncio
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
import json
import logging
from pathlib import Path

import timed_turn_test as previous
import acoustic_turns as worker
from integration_guard import save_json, SecretFormatter, TERMINAL
from prepare_short_turn_test import stamp, dollars

full = previous.full


def history_files(runtime):
    found = previous.history_files(runtime)
    folder = runtime / previous.worker.PROFILE / 'jobs'
    if any(folder.glob('*.tmp')): raise ValueError('Unfinished historical write')
    for path in folder.glob('*.json'):
        if path.name.startswith(('message-', 'failed-', 'retry-')):
            found[path.relative_to(runtime).as_posix()] = path
    return found


def reconcile(billing, health, shutdown, closed_shutdown, closed, recent, now=None):
    """Close only lag-covered allowances; carry the latest job's full $0.50."""
    now = now or datetime.now(timezone.utc)
    observed = stamp(billing['observed_at_utc'])
    if any(not 0 <= (now - stamp(v['observed_at_utc'])).total_seconds() <= 900 for v in (billing, health)):
        raise ValueError('Fresh billing and health required')
    if (billing.get('endpoint') != full.ENDPOINT or health.get('endpoint') != full.ENDPOINT
            or shutdown.get('endpoint') != full.ENDPOINT or closed_shutdown.get('endpoint') != full.ENDPOINT
            or billing.get('period_ui') != 'окт. 1 30' or billing.get('period_timezone') != 'UTC'
            or billing.get('group') != 'Serverless / Endpoint' or billing.get('billing_lag_hours') != 1
            or billing.get('active_workers') != 0 or billing.get('max_workers') != 0
            or billing.get('endpoint_idle_observed_this_turn') is not True):
        raise ValueError('Whole-period test endpoint observation required')
    current = health['health']
    if (current != shutdown['health'] or shutdown.get('max_workers') != 0
            or current['jobs']['inQueue'] or current['jobs']['inProgress']
            or not current['workers'] or any(current['workers'].values())):
        raise ValueError('Latest job or worker remains active')
    if len(closed) != 14 or len(recent) != 1 or any(j.get('status') not in TERMINAL for j in closed):
        raise ValueError('All fifteen original terminal journals required')
    cutoff = stamp(closed_shutdown['observed_at_utc']).timestamp()
    if any(float(j.get('submitted_at') or 0) + float(j.get('elapsed_sec') or 0) > cutoff for j in closed):
        raise ValueError('An old journal is newer than its shutdown')
    if observed.timestamp() - cutoff < 3600 + 900:
        raise ValueError('Prior billing lag plus margin has not elapsed')
    last = recent[0]
    if (last.get('status') != 'COMPLETED' or last.get('timed_turns_validated') is not True
            or float(last.get('submitted_at') or 0) <= cutoff
            or float(last['submitted_at']) + float(last.get('elapsed_sec') or 0) > stamp(shutdown['observed_at_utc']).timestamp()):
        raise ValueError('Unresolved or mismatching latest attempt')
    expected_jobs = {**closed_shutdown['health']['jobs']}
    expected_jobs['completed'] += 1
    if current['jobs'] != expected_jobs: raise ValueError('Unjournaled cloud job or retry')
    lines = billing['table'].splitlines(); headings = lines[0].split('\t')
    index = headings.index(full.ENDPOINT)
    amounts = [dollars(line.split('\t')[index].replace('$', '').replace(',', '.')) for line in lines[1:] if line.strip()]
    billed = sum(amounts, Decimal(0))
    if not amounts or billed != dollars(billing['displayed_total_test_usd']):
        raise ValueError('Endpoint column and total disagree')
    rounding = Decimal('.001') * sum(v != 0 for v in amounts)
    settled = max(Decimal('.25'), (billed + rounding + Decimal('.10')).quantize(Decimal('.01'), rounding=ROUND_CEILING))
    carried, reserve = Decimal('.50'), full.RESERVE_USD
    if settled + carried + reserve > Decimal('2'): raise ValueError('Insufficient cumulative budget')
    return {'schema': 1, 'kind': 'partial_closed_allowance_reconciliation', 'endpoint': full.ENDPOINT,
            'observed_billed_usd': str(billed), 'settled_allowance_usd': str(settled),
            'rounding_allowance_usd': str(rounding), 'recent_full_reserve_usd': str(carried),
            'baseline_spend_usd': str(settled + carried), 'next_attempt_reserve_usd': str(reserve),
            'planning_total_usd': str(settled + carried + reserve), 'budget_usd': '2',
            'closed_journals': 14, 'recent_journals': 1, 'old_journals_modified': False,
            'prior_planning_total_usd': last['planning_total_usd']}


class AcousticRunner(full.FullRunner):
    campaign_name = output_profile = worker.PROFILE
    modes = worker.MODES
    validated_key = 'acoustic_turns_validated'
    history_files = staticmethod(history_files)
    health = previous.TimedRunner.health
    additional_preflight = {'acoustic_turns_enabled': True, 'timed_turns_enabled': False,
                            'release_candidate_enabled': False, 'short_turn_comparison_enabled': False,
                            'speaker_comparison_enabled': False, 'context_asr_enabled': False}

    def __init__(self, settings, **kwargs):
        super().__init__(settings, **kwargs)
        prior = previous.TimedRunner(settings); journal = full.read_json(prior.path)
        if (self.campaign.get('previous_campaign_sha256') != prior.campaign_sha
                or journal.get('campaign_sha256') != prior.campaign_sha
                or full.money(journal['planning_total_usd']) - full.money(prior.campaign['baseline_spend_usd']) != full.RESERVE_USD):
            raise ValueError('Latest campaign reservation changed')
        c = self.campaign
        record_path = Path(c['reconciliation_file']); record = full.read_json(record_path)
        self.budget_hashes = {str(record_path): c['reconciliation_sha256']}
        observations = {}
        for name in ('billing', 'health', 'shutdown', 'closed_shutdown'):
            path = Path(c[name + '_file'])
            self.budget_hashes[str(path)] = record['evidence_sha256'][name]
            observations[name] = full.read_json(path)
        if record.get('history_sha256') != c['history_sha256']:
            raise ValueError('Reconciliation history changed')
        if full.digest(c['closed_shutdown_file']) != prior.campaign['previous_shutdown_sha256']:
            raise ValueError('Closed history shutdown changed')
        closed_paths = previous.history_files(settings.runtime)
        if set(c['history_sha256']) != set(closed_paths) | {prior.path.relative_to(settings.runtime).as_posix()}:
            raise ValueError('Unexpected additional journal')
        observed = max(stamp(observations[n]['observed_at_utc']) for n in ('billing', 'health'))
        if observed > datetime.now(timezone.utc): raise ValueError('Future budget observation')
        verified = reconcile(**observations, closed=[full.read_json(p) for p in closed_paths.values()], recent=[journal], now=observed)
        if any(record.get(k) != v for k, v in verified.items()): raise ValueError('Reconciliation cannot be reproduced')
        if full.money(c['baseline_spend_usd']) < full.money(verified['baseline_spend_usd']):
            raise ValueError('Latest reserve not carried in full')
        self.expected_health = observations['health']['health']
        self.validate_budget_files()

    def validate_budget_files(self):
        if any(full.digest(path) != sha for path, sha in self.budget_hashes.items()):
            raise ValueError('Budget evidence changed')

    def check_available(self):
        self.validate_budget_files()
        super().check_available()

    def validate_output(self, result):
        assets = self.validate_provenance(result); sources = full.read_json(full.SOURCES)
        if (result.get('policy') != worker.base.POLICY or result.get('transforms') != worker.TRANSFORMS
                or result.get('pitch_config') != worker.PITCH or result.get('decoder') != worker.base.DECODER
                or result.get('human_quality_validated') is not False or len(result.get('samples', [])) != 4):
            raise ValueError('Unexpected acoustic experiment identity')
        for kind in ('alignment', 'diarization'):
            files = result['runtime'][kind].get('files', {})
            if set(files) != set(sources[kind]['files']): raise ValueError('Incomplete assets')
            for name, checksum in files.items():
                expected = sources[kind]['files'][name].get('sha256')
                if not full.SHA.fullmatch(checksum) or (expected and checksum != expected): raise ValueError('Changed asset')
        for name, version in {'whisperx': '3.8.6', 'pyannote.audio': '4.0.3', 'faster-whisper': '1.2.1', 'librosa': '0.11.0'}.items():
            if result['runtime']['packages'].get(name) != version: raise ValueError('Package changed')
        for sample, row in zip(result['samples'], self.rows):
            if any(sample.get(k) != row[k] for k in ('id', 'sha256', 'duration_seconds')): raise ValueError('Audio changed')
            worker.validate_sample(sample)
        self.state.update(acoustic_turns_validated=True, quality_validated=False, model_lock_sha256=assets['lock_sha256'])
        save_json(self.path, self.state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file', required=True, type=Path); parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    try:
        settings = full.TestSettings.load(args.env_file); runner = AcousticRunner(settings); runner.check_available()
        if not args.submit:
            print(json.dumps({'status': 'prepared_not_run', 'network_calls': 0, 'max_attempts': 1,
                'planning_total_usd': str(runner.settings.prior_spend + full.RESERVE_USD)})); return 0
        logging.basicConfig(level=logging.INFO)
        for h in logging.getLogger().handlers: h.setFormatter(SecretFormatter(settings.secrets.values()))
        logging.getLogger('httpx').setLevel(logging.WARNING)
        asyncio.run(runner.execute())
        print('Acoustic evidence verified; disable flags/Max workers. Human quality remains unverified.')
        return 0
    except Exception as exc:
        print(json.dumps({'status': 'stopped', 'error_type': type(exc).__name__, 'automatic_retry': False})); return 1


if __name__ == '__main__': raise SystemExit(main())
