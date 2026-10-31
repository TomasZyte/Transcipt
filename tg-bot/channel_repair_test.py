"""One contextual recovery test; carry the full previous $1.29 allowance."""
import argparse
import asyncio
import json
import logging
from pathlib import Path

import channel_runs_test as previous
from integration_guard import SecretFormatter

full = previous.full
PROFILE = 'channel-repair-v1'


def history_files(runtime):
    found = previous.history_files(runtime)
    folder = runtime / previous.PROFILE / 'jobs'
    if any(folder.glob('*.tmp')):
        raise ValueError('Unfinished prior journal')
    for path in folder.glob('*.json'):
        if path.name.startswith(('message-', 'failed-', 'retry-')):
            found[path.relative_to(runtime).as_posix()] = path
    return found


def validate_prior(campaign, prior_sha, journal, shutdown):
    if (journal.get('status') != 'COMPLETED' or journal.get('channel_asr_validated') is not True
            or journal.get('campaign_sha256') != prior_sha
            or campaign.get('previous_campaign_sha256') != prior_sha
            or full.money(campaign['baseline_spend_usd']) < full.money(journal['planning_total_usd'])):
        raise ValueError('Carry the entire previous reservation, including its paid job')
    h = shutdown['health']
    if (shutdown.get('endpoint') != full.ENDPOINT or shutdown.get('max_workers') != 0
            or shutdown.get('active_workers') != 0 or shutdown.get('full_pipeline_enabled') is not False
            or shutdown.get('channel_asr_enabled') is not False
            or h['jobs'] != {'completed': 19, 'failed': 1, 'inProgress': 0, 'inQueue': 0, 'retried': 2}
            or not h['workers'] or any(h['workers'].values())):
        raise ValueError('Expected stopped speech-run test and unchanged historical counters')
    return h


class RepairRunner(previous.previous.ChannelRunner):
    campaign_name = output_profile = PROFILE
    history_files = staticmethod(history_files)

    def bind_prior(self, settings):
        prior = previous.RunsRunner(settings)
        c = self.campaign
        self.budget_paths = {'shutdown': Path(c['previous_shutdown_file']), 'prior_journal': prior.path,
                             'prior_campaign': prior.campaign_path}
        self.budget_hashes = {'shutdown': c['previous_shutdown_sha256'],
                             'prior_journal': c['previous_journal_sha256'],
                             'prior_campaign': c['previous_campaign_sha256']}
        self.validate_budget_files()
        self.expected_health = validate_prior(c, prior.campaign_sha, full.read_json(prior.path),
                                              full.read_json(self.budget_paths['shutdown']))
        if len(self.history_files(settings.runtime)) != 19:
            raise ValueError('All nineteen prior attempts must be accounted for')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file', type=Path, required=True)
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    try:
        runner = RepairRunner(full.TestSettings.load(args.env_file)); runner.check_available()
        if not args.submit:
            print(json.dumps({'status': 'prepared_not_run', 'network_calls': 0, 'max_attempts': 1,
                              'planning_total_usd': str(runner.settings.prior_spend + full.RESERVE_USD)}))
            return 0
        logging.basicConfig(level=logging.INFO)
        for handler in logging.getLogger().handlers:
            handler.setFormatter(SecretFormatter(runner.settings.secrets.values()))
        logging.getLogger('httpx').setLevel(logging.WARNING)
        asyncio.run(runner.execute())
        print('Recovery evidence verified; stop test flags/Max. Human review still required.')
        return 0
    except Exception as exc:
        print(json.dumps({'status': 'stopped', 'error_type': type(exc).__name__, 'automatic_retry': False}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
