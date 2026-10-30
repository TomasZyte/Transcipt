"""One context comparison, carrying every previous reservation without resetting it."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
import logging
from pathlib import Path

import acoustic_turn_test as previous
import pause_context as worker
from integration_guard import save_json, SecretFormatter
from prepare_short_turn_test import stamp, dollars

full = previous.full


def history_files(runtime):
    found = previous.history_files(runtime)
    folder = runtime / previous.worker.PROFILE / 'jobs'
    if any(folder.glob('*.tmp')):
        raise ValueError('Unfinished prior write')
    for path in folder.glob('*.json'):
        if path.name.startswith(('message-', 'failed-', 'retry-')):
            found[path.relative_to(runtime).as_posix()] = path
    return found


def validate_prior(campaign, prior_sha, journal, shutdown):
    if (journal.get('status') != 'COMPLETED' or journal.get('acoustic_turns_validated') is not True
            or journal.get('campaign_sha256') != prior_sha
            or campaign.get('previous_campaign_sha256') != prior_sha
            or full.money(campaign['baseline_spend_usd']) < full.money(journal['planning_total_usd'])):
        raise ValueError('Carry the full latest plan, including the last paid attempt')
    health = shutdown['health']
    if (shutdown.get('endpoint') != full.ENDPOINT or shutdown.get('max_workers') != 0
            or health['jobs']['inQueue'] or health['jobs']['inProgress']
            or not health['workers'] or any(health['workers'].values())):
        raise ValueError('Latest test has not been stopped')
    return health


def validate_billing(observation, now=None):
    now = now or datetime.now(timezone.utc)
    if (observation.get('endpoint') != full.ENDPOINT
            or observation.get('period_ui') != 'окт. 1 30'
            or observation.get('period_timezone') != 'UTC'
            or observation.get('group') != 'Serverless / Endpoint'
            or observation.get('billing_lag_hours') != 1
            or not 0 <= (now - stamp(observation['observed_at_utc'])).total_seconds() <= 900):
        raise ValueError('Fresh, whole-period test billing required')
    lines = observation['table'].splitlines()
    index = lines[0].split('\t').index(full.ENDPOINT)
    amounts = [dollars(line.split('\t')[index].replace('$', '').replace(',', '.'))
               for line in lines[1:] if line.strip()]
    total = sum(amounts, full.money('0'))
    if not amounts or total != full.money(observation['displayed_total_test_usd']):
        raise ValueError('Endpoint billing sum disagrees')
    return total


class PauseRunner(full.FullRunner):
    campaign_name = output_profile = worker.PROFILE
    modes = worker.MODES
    validated_key = 'pause_context_validated'
    history_files = staticmethod(history_files)
    health = previous.AcousticRunner.health
    additional_preflight = {'pause_context_enabled': True, 'acoustic_turns_enabled': False,
                            'timed_turns_enabled': False, 'release_candidate_enabled': False,
                            'short_turn_comparison_enabled': False, 'speaker_comparison_enabled': False,
                            'context_asr_enabled': False}

    def __init__(self, settings, **kwargs):
        super().__init__(settings, **kwargs)
        prior = previous.AcousticRunner(settings); journal = full.read_json(prior.path)
        c = self.campaign
        self.shutdown_path = Path(c['previous_shutdown_file'])
        self.shutdown_sha = c['previous_shutdown_sha256']
        if full.digest(self.shutdown_path) != self.shutdown_sha:
            raise ValueError('Previous shutdown changed')
        self.expected_health = validate_prior(c, prior.campaign_sha, journal, full.read_json(self.shutdown_path))

    def check_available(self):
        if full.digest(self.shutdown_path) != self.shutdown_sha:
            raise ValueError('Previous shutdown changed')
        super().check_available()

    def preflight(self, now=None):
        p = super().preflight(now)
        path = Path(p['billing_file'])
        if full.digest(path) != p['billing_sha256']:
            raise ValueError('Billing evidence changed')
        if validate_billing(full.read_json(path), now) != full.money(p['test_billed_usd']):
            raise ValueError('Preflight billing differs from the UI observation')
        return p

    def validate_output(self, result):
        assets = self.validate_provenance(result); sources = full.read_json(full.SOURCES)
        if (result.get('policy') != worker.base.POLICY or result.get('diarization_configs') != worker.DIARIZATION
                or result.get('alignment_configs') != list(worker.ALIGNMENTS)
                or result.get('decoder') != worker.base.DECODER
                or result.get('human_quality_validated') is not False or len(result.get('samples', [])) != 4):
            raise ValueError('Unexpected context experiment identity')
        for kind in ('alignment', 'diarization'):
            files = result['runtime'][kind].get('files', {})
            if set(files) != set(sources[kind]['files']):
                raise ValueError('Incomplete model assets')
            for name, checksum in files.items():
                expected = sources[kind]['files'][name].get('sha256')
                if not full.SHA.fullmatch(checksum) or (expected and checksum != expected):
                    raise ValueError('Model asset changed')
        for name, version in {'whisperx': '3.8.6', 'pyannote.audio': '4.0.3', 'faster-whisper': '1.2.1'}.items():
            if result['runtime']['packages'].get(name) != version:
                raise ValueError('Package changed')
        for sample, row in zip(result['samples'], self.rows):
            if any(sample.get(k) != row[k] for k in ('id', 'sha256', 'duration_seconds')):
                raise ValueError('Audio changed')
            worker.validate_sample(sample)
        self.state.update(pause_context_validated=True, quality_validated=False, model_lock_sha256=assets['lock_sha256'])
        save_json(self.path, self.state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file', required=True, type=Path); parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    try:
        settings = full.TestSettings.load(args.env_file); runner = PauseRunner(settings); runner.check_available()
        if not args.submit:
            print(json.dumps({'status': 'prepared_not_run', 'network_calls': 0, 'max_attempts': 1,
                              'planning_total_usd': str(runner.settings.prior_spend + full.RESERVE_USD)}))
            return 0
        logging.basicConfig(level=logging.INFO)
        for h in logging.getLogger().handlers:
            h.setFormatter(SecretFormatter(settings.secrets.values()))
        logging.getLogger('httpx').setLevel(logging.WARNING)
        asyncio.run(runner.execute())
        print('Context evidence verified; disable flags/Max. Human quality is not yet verified.')
        return 0
    except Exception as exc:
        print(json.dumps({'status': 'stopped', 'error_type': type(exc).__name__, 'automatic_retry': False}))
        return 1


if __name__ == '__main__': raise SystemExit(main())
