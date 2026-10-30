from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import acoustic_turn_test as a


class AcousticBudgetTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 10, 30, 2, 30, tzinfo=timezone.utc)
        jobs = {'completed': 14, 'failed': 1, 'retried': 2, 'inQueue': 0, 'inProgress': 0}
        self.old = {'endpoint': a.full.ENDPOINT, 'observed_at_utc': (self.now - timedelta(hours=2)).isoformat(),
                    'max_workers': 0, 'health': {'jobs': jobs, 'workers': {'idle': 0, 'running': 0}}}
        self.shutdown = deepcopy(self.old); self.shutdown['health']['jobs']['completed'] += 1
        self.shutdown['observed_at_utc'] = (self.now - timedelta(minutes=10)).isoformat()
        self.health = deepcopy(self.shutdown); self.health['observed_at_utc'] = self.now.isoformat()
        self.billing = {'endpoint': a.full.ENDPOINT, 'observed_at_utc': self.now.isoformat(),
            'period_ui': 'окт. 1 30', 'period_timezone': 'UTC', 'group': 'Serverless / Endpoint',
            'billing_lag_hours': 1, 'active_workers': 0, 'max_workers': 0,
            'endpoint_idle_observed_this_turn': True, 'displayed_total_test_usd': '.143',
            'table': 'Date\t' + a.full.ENDPOINT + '\nOct 30\t$0.015\nOct 29\t$0.053\nOct 28\t$0.075'}
        self.closed = [{'status': 'COMPLETED', 'submitted_at': (self.now - timedelta(hours=3)).timestamp(), 'elapsed_sec': 60}] * 14
        self.recent = [{'status': 'COMPLETED', 'timed_turns_validated': True,
            'submitted_at': (self.now - timedelta(minutes=20)).timestamp(), 'elapsed_sec': 60,
            'planning_total_usd': '1.75'}]

    def run_reconcile(self):
        return a.reconcile(self.billing, self.health, self.shutdown, self.old, self.closed, self.recent, now=self.now)

    def test_carry_unbilled_full_reserve_and_do_not_modify_history(self):
        before = deepcopy((self.closed, self.recent))
        r = self.run_reconcile()
        self.assertEqual(r['observed_billed_usd'], '0.143')
        self.assertEqual(r['recent_full_reserve_usd'], '0.50')
        self.assertEqual(r['planning_total_usd'], '1.25')
        self.assertEqual((self.closed, self.recent), before)

    def test_stale_or_not_lag_covered_billing_cannot_free_old_allowance(self):
        self.billing['observed_at_utc'] = (self.now - timedelta(minutes=16)).isoformat()
        with self.assertRaises(ValueError): self.run_reconcile()
        self.billing['observed_at_utc'] = self.now.isoformat()
        self.old['observed_at_utc'] = (self.now - timedelta(minutes=70)).isoformat()
        with self.assertRaises(ValueError): self.run_reconcile()

    def test_unknown_jobs_retries_or_worker_prevent_reconciliation(self):
        for field in ('completed', 'retried', 'inProgress'):
            with self.subTest(field=field):
                self.health['health']['jobs'][field] += 1
                with self.assertRaises(ValueError): self.run_reconcile()
                self.health['health']['jobs'][field] -= 1
        self.health['health']['workers']['running'] = 1
        with self.assertRaises(ValueError): self.run_reconcile()

    def test_missing_history_unresolved_recent_or_wrong_bucket_totals_block(self):
        self.closed.pop()
        with self.assertRaises(ValueError): self.run_reconcile()
        self.setUp(); self.recent[0]['status'] = 'IN_PROGRESS'
        with self.assertRaises(ValueError): self.run_reconcile()
        self.setUp(); self.billing['displayed_total_test_usd'] = '.1'
        with self.assertRaises(ValueError): self.run_reconcile()
        self.setUp(); self.billing['endpoint'] = 'production'
        with self.assertRaises(ValueError): self.run_reconcile()

    def test_budget_includes_rounding_contingency_recent_and_next_reserve(self):
        self.billing['table'] = 'Date\t' + a.full.ENDPOINT + '\nOct 30\t$1.001'
        self.billing['displayed_total_test_usd'] = '1.001'
        with self.assertRaises(ValueError): self.run_reconcile()


if __name__ == '__main__': unittest.main()
