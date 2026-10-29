from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_short_turn_test as p


class ReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 10, 30, tzinfo=timezone.utc)
        health = {'jobs': {'completed': 12, 'failed': 1, 'retried': 2, 'inProgress': 0, 'inQueue': 0},
                  'workers': {'idle': 0, 'running': 0, 'initializing': 0}}
        self.health = {'endpoint': p.previous.ENDPOINT, 'observed_at_utc': self.now.isoformat(), 'health': health}
        self.shutdown = {'observed_at_utc': (self.now - timedelta(hours=2)).isoformat(), 'health': deepcopy(health)}
        self.billing = {'endpoint': p.previous.ENDPOINT, 'observed_at_utc': self.now.isoformat(),
            'period_ui': 'окт. 1 30', 'period_timezone': 'UTC', 'group': 'Serverless / Endpoint',
            'active_workers': 0, 'max_workers': 0, 'endpoint_idle_observed_this_turn': True,
            'billing_lag_hours': 1, 'displayed_total_test_usd': '0.121',
            'table': 'Start date\tTotal\tl20o867nueti1d\ttt17em69to6yua\n29 окт.\t$9\t$8,954\t$0,046\n28 окт.\t$10\t$9,925\t$0,075'}
        self.journals = [{'status': 'COMPLETED', 'submitted_at': (self.now - timedelta(hours=3)).timestamp(),
                          'elapsed_sec': 25, 'planning_total_usd': '1.616'} for _ in range(12)]

    def check(self):
        return p.reconcile(self.billing, self.health, self.shutdown, self.journals, self.now)

    def test_only_test_column_counts_and_history_is_not_rewritten(self):
        before = deepcopy(self.journals)
        result = self.check()
        self.assertEqual(result['observed_billed_usd'], '0.121')
        self.assertEqual(result['baseline_spend_usd'], '0.25')
        self.assertEqual(result['planning_total_usd'], '0.75')
        self.assertEqual(result['previous_planning_total_usd'], '1.616')
        self.assertFalse(result['paid_run_ready'])
        self.assertEqual(before, self.journals)

    def test_in_progress_or_new_cloud_job_blocks_release(self):
        self.health['health']['jobs']['completed'] += 1
        with self.assertRaises(ValueError): self.check()
        self.health['health'] = deepcopy(self.shutdown['health'])
        self.health['health']['workers']['running'] = 1
        with self.assertRaises(ValueError): self.check()

    def test_billing_must_be_fresh_and_after_lag_margin(self):
        self.billing['observed_at_utc'] = (self.now - timedelta(minutes=16)).isoformat()
        with self.assertRaises(ValueError): self.check()
        self.billing['observed_at_utc'] = self.now.isoformat()
        self.shutdown['observed_at_utc'] = (self.now - timedelta(minutes=70)).isoformat()
        with self.assertRaises(ValueError): self.check()

    def test_unresolved_recent_or_missing_local_job_blocks_release(self):
        self.journals[0]['status'] = 'IN_QUEUE'
        with self.assertRaises(ValueError): self.check()
        self.journals[0]['status'] = 'COMPLETED'
        self.journals[0]['submitted_at'] = self.now.timestamp() - 120
        with self.assertRaises(ValueError): self.check()
        self.journals.pop()
        with self.assertRaises(ValueError): self.check()

    def test_wrong_endpoint_or_truncated_period_rejected(self):
        self.billing['endpoint'] = 'l20o867nueti1d'
        with self.assertRaises(ValueError): self.check()
        self.billing['endpoint'] = p.previous.ENDPOINT
        self.billing['period_ui'] = 'окт. 29 30'
        with self.assertRaises(ValueError): self.check()

    def test_billing_disagreement_and_over_budget_rejected(self):
        self.billing['displayed_total_test_usd'] = '0.001'
        with self.assertRaises(ValueError): self.check()
        self.billing['table'] = self.billing['table'].replace('$0,046', '$1,546')
        self.billing['displayed_total_test_usd'] = '1.621'
        with self.assertRaises(ValueError): self.check()


if __name__ == '__main__': unittest.main()
