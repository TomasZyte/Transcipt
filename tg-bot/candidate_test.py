"""One bounded full-corpus candidate test. Offline checks unless --submit."""
import argparse
import asyncio
import json
import logging
from pathlib import Path
import sys

import full_pipeline_test as full
import short_turn_test as previous
import candidate_pipeline as candidate
from integration_guard import save_json, SecretFormatter


def history_files(runtime):
    found = previous.preparation.history_files(runtime)
    folder = runtime / previous.CAMPAIGN / 'jobs'
    if any(folder.glob('*.tmp')):
        raise ValueError('Previous attempt has an unfinished write')
    for path in folder.glob('*.json'):
        if path.name.startswith(('message-', 'failed-', 'retry-')):
            found[path.relative_to(runtime).as_posix()] = path
    return found


class CandidateRunner(full.FullRunner):
    campaign_name = output_profile = candidate.PROFILE
    modes = candidate.MODES
    validated_key = 'candidate_technical_validated'
    history_files = staticmethod(history_files)
    health = previous.ShortTurnRunner.health
    additional_preflight = {'release_candidate_enabled': True, 'short_turn_comparison_enabled': False,
                            'speaker_comparison_enabled': False, 'context_asr_enabled': False}

    def __init__(self, settings, **kwargs):
        super().__init__(settings, **kwargs)
        # The latest campaign already reconciled closed earlier allowances.
        # Keep its FULL reservation, even if delayed billing has not caught up.
        prior = previous.ShortTurnRunner(settings)
        journal_path = prior.path
        journal = full.read_json(journal_path)
        if (journal.get('status') != 'COMPLETED' or journal.get('short_turn_validated') is not True
                or journal.get('campaign_sha256') != prior.campaign_sha
                or full.money(self.campaign['baseline_spend_usd']) < full.money(journal['planning_total_usd'])
                or self.campaign.get('previous_campaign_sha256') != prior.campaign_sha):
            raise ValueError('Carry the full previous allowance; do not reset the budget')
        shutdown = Path(self.campaign['previous_shutdown_file'])
        if full.digest(shutdown) != self.campaign['previous_shutdown_sha256']:
            raise ValueError('Previous shutdown evidence changed')
        observed = full.read_json(shutdown)
        self.expected_health = observed['health']
        if (observed.get('endpoint') != full.ENDPOINT or observed.get('max_workers') != 0
                or self.expected_health['jobs']['inQueue'] or self.expected_health['jobs']['inProgress']
                or any(self.expected_health['workers'].values())):
            raise ValueError('Previous campaign must be stopped')

    def validate_output(self, result):
        assets = self.validate_provenance(result)
        sources = full.read_json(full.SOURCES)
        for kind in ('alignment', 'diarization'):
            files = result['runtime'][kind].get('files', {})
            if set(files) != set(sources[kind]['files']):
                raise ValueError('Incomplete model asset provenance')
            for name, checksum in files.items():
                expected = sources[kind]['files'][name].get('sha256')
                if not full.SHA.fullmatch(checksum) or (expected and checksum != expected):
                    raise ValueError('Model asset hash changed')
        if (result.get('policy') != candidate.POLICY or result.get('diarization_configs') != candidate.DIARIZATION
                or result.get('human_quality_validated') is not False
                or result['runtime']['packages'].get('pyannote.audio') != '4.0.3'
                or result['runtime']['packages'].get('whisperx') != '3.8.6'
                or len(result.get('samples', [])) != 4):
            raise ValueError('Incomplete or unexpected candidate settings')
        for sample, row in zip(result['samples'], self.rows):
            if (sample.get('id') != row['id'] or sample.get('sha256') != row['sha256']
                    or sample.get('duration_seconds') != row['duration_seconds']):
                raise ValueError('Candidate audio identity changed')
            candidate.validate_sample(sample)
        self.state.update(candidate_technical_validated=True, quality_validated=False,
                          model_lock_sha256=assets['lock_sha256'])
        save_json(self.path, self.state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file', required=True, type=Path)
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    try:
        settings = full.TestSettings.load(args.env_file)
        runner = CandidateRunner(settings); runner.check_available()
        if not args.submit:
            print(json.dumps({'status': 'prepared_not_run', 'network_calls': 0, 'max_attempts': 1,
                              'planning_total_usd': str(runner.settings.prior_spend + full.RESERVE_USD)}))
            return 0
        logging.basicConfig(level=logging.INFO)
        for h in logging.getLogger().handlers:
            h.setFormatter(SecretFormatter(settings.secrets.values()))
        logging.getLogger('httpx').setLevel(logging.WARNING)
        asyncio.run(runner.execute())
        print('Candidate evidence verified. Disable test flags and Max workers; quality review remains required.')
        return 0
    except Exception as exc:
        print(json.dumps({'status': 'stopped', 'error_type': type(exc).__name__, 'automatic_retry': False}))
        return 1


if __name__ == '__main__': sys.exit(main())
