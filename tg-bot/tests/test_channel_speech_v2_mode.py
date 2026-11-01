from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import channel_speech_v2_test as r
import test_channel_mode as fixture


class SpeechV2BudgetTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 11, 1, 15, tzinfo=timezone.utc)
        self.bill = {'endpoint': r.full.ENDPOINT, 'period_start': r.PERIOD[0], 'period_end': r.PERIOD[1],
            'period_ui': r.PERIOD[2], 'period_timezone': 'UTC', 'group': 'Serverless / Endpoint',
            'billing_lag_hours': 1, 'observed_at_utc': self.now.isoformat(),
            'table': 'Date\t'+r.full.ENDPOINT+'\n31\t$0,038\n30\t$0,056\n29\t$0,053\n28\t$0,075',
            'displayed_total_test_usd': '.222', 'max_workers': 0, 'active_workers': 0,
            'endpoint_idle_observed_this_turn': True}
        self.health = {'observed_at_utc': self.now.isoformat(), 'endpoint': r.full.ENDPOINT,
            'health': {'jobs': deepcopy(r.EXPECTED_JOBS), 'workers': {'idle': 0, 'running': 0}}}
        self.stop = {**deepcopy(self.health), 'observed_at_utc': (self.now-timedelta(hours=8)).isoformat(),
                     'max_workers': 0, 'active_workers': 0, 'release': 63}
        self.journals = [{'status': 'COMPLETED', 'submitted_at': i} for i in range(21)]
        self.journals[-1].update(channel_asr_validated=True, planning_total_usd='0.82')

    def reconcile(self):
        return r.reconcile(self.bill, self.health, self.stop, self.journals, self.now)

    def test_all_prior_spend_and_rounding_margin_count_without_rewriting_journals(self):
        before = deepcopy(self.journals)
        result = self.reconcile()
        self.assertEqual(result['baseline_spend_usd'], '0.33')
        self.assertEqual(result['planning_total_usd'], '0.83')
        self.assertEqual(result['historical_journals'], 21)
        self.assertEqual(self.journals, before)

    def test_stale_partial_bill_production_and_inconsistent_total_block(self):
        original = deepcopy(self.bill)
        for change in ({'period_start': '2026-11-01'}, {'endpoint': 'production'},
                       {'displayed_total_test_usd': '0'}, {'period_ui': 'Today'},
                       {'observed_at_utc': (self.now-timedelta(minutes=16)).isoformat()}):
            self.bill = {**original, **change}
            with self.assertRaises(ValueError): self.reconcile()

    def test_unsettled_or_changed_history_workers_shutdown_and_budget_block(self):
        for change in (lambda: self.journals.pop(),
                       lambda: self.journals[0].update(status='IN_PROGRESS'),
                       lambda: self.health['health']['jobs'].update(completed=22),
                       lambda: self.health['health']['workers'].update(running=1),
                       lambda: self.stop.update(release=62),
                       lambda: self.stop.update(observed_at_utc=(self.now-timedelta(minutes=74)).isoformat()),
                       lambda: self.journals[-1].update(planning_total_usd='1.79'),
                       lambda: self.bill.update(displayed_total_test_usd='1.5', table='Date\t'+r.full.ENDPOINT+'\n31\t$1.5')):
            self.setUp(); change()
            with self.assertRaises(ValueError): self.reconcile()

    def test_payload_has_no_references_and_no_network_before_preflight(self):
        harness = fixture.ChannelBudgetTests()
        runner = harness.runner(); runner.__class__ = r.SpeechV2Runner
        urls = [f'https://{runner.settings.storage_host}/poslovnica-test/quality/{n}.wav?sig=x' for n in ('left', 'right')]
        body = runner.request_body(runner.payload(urls))
        self.assertEqual(body['input']['profile'], r.PROFILE)
        self.assertEqual(body['policy']['executionTimeout'], 300000)
        with self.assertRaises(ValueError): runner.request_body({**body['input'], 'reference': 'private'})
        harness.test_no_network_before_budget_preflight()

    def test_new_history_includes_only_prior_submission_journals(self):
        root = Path('/isolated')
        prior = {'old/message-first.json': root/'old/message-first.json'}
        journal = root/r.previous.PROFILE/'jobs/message-channel-speech-v1.json'
        result = journal.with_name('result-channel-speech-v1.json')
        with patch.object(r.previous, 'history_files', return_value=prior.copy()), \
             patch.object(Path, 'glob', side_effect=[[], [journal, result]]):
            found = r.history_files(root)
        self.assertEqual(len(found), 2)
        self.assertIn(journal.relative_to(root).as_posix(), found)
        self.assertNotIn(result.relative_to(root).as_posix(), found)

    def test_default_cli_cannot_submit_or_access_network(self):
        runner = Mock()
        runner.settings.prior_spend = r.Decimal('.33')
        with patch('sys.argv', ['channel_speech_v2_test.py', '--env-file', 'fixture.env']), \
             patch.object(r.full.TestSettings, 'load', return_value=object()), \
             patch.object(r, 'SpeechV2Runner', return_value=runner), patch('builtins.print'):
            self.assertEqual(r.main(), 0)
        runner.check_available.assert_called_once()
        runner.execute.assert_not_called()


if __name__ == '__main__': unittest.main()
