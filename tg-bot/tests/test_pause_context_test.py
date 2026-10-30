from copy import deepcopy
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pause_context_test as p


class PauseBudgetTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 10, 30, 12, 30, tzinfo=timezone.utc)
        self.c = {'previous_campaign_sha256': 'abc', 'baseline_spend_usd': '1.25'}
        self.j = {'status': 'COMPLETED', 'acoustic_turns_validated': True,
                  'campaign_sha256': 'abc', 'planning_total_usd': '1.25'}
        self.shutdown = {'endpoint': p.full.ENDPOINT, 'max_workers': 0, 'health': {
            'jobs': {'completed': 16, 'failed': 1, 'retried': 2, 'inQueue': 0, 'inProgress': 0},
            'workers': {'idle': 0, 'running': 0}}}
        self.bill = {'endpoint': p.full.ENDPOINT, 'observed_at_utc': self.now.isoformat(),
                     'period_ui': 'окт. 1 30', 'period_timezone': 'UTC', 'group': 'Serverless / Endpoint',
                     'billing_lag_hours': 1, 'displayed_total_test_usd': '.172',
                     'table': 'Start date\tproduction\t' + p.full.ENDPOINT + '\n30 Oct\t$5\t$0,044\n29 Oct\t$4\t$0,053\n28 Oct\t$1\t$0,075'}

    def test_full_old_plan_carried_not_replaced_with_small_billed_value(self):
        before = deepcopy((self.c, self.j, self.shutdown))
        self.assertEqual(p.validate_prior(self.c, 'abc', self.j, self.shutdown), self.shutdown['health'])
        self.assertEqual((self.c, self.j, self.shutdown), before)
        self.c['baseline_spend_usd'] = '.172'
        with self.assertRaises(ValueError): p.validate_prior(self.c, 'abc', self.j, self.shutdown)

    def test_unresolved_prior_or_running_workers_block(self):
        for key, value in (('status', 'IN_PROGRESS'), ('acoustic_turns_validated', False), ('campaign_sha256', 'wrong')):
            journal = {**self.j, key: value}
            with self.assertRaises(ValueError): p.validate_prior(self.c, 'abc', journal, self.shutdown)
        for section, key in (('jobs', 'inProgress'), ('jobs', 'inQueue'), ('workers', 'idle'), ('workers', 'running')):
            observed = deepcopy(self.shutdown); observed['health'][section][key] = 1
            with self.assertRaises(ValueError): p.validate_prior(self.c, 'abc', self.j, observed)

    def test_billing_reads_test_column_and_rejects_wrong_period_or_total(self):
        self.assertEqual(p.validate_billing(self.bill, self.now), p.full.money('.172'))
        for key, value in (('period_ui', 'today'), ('endpoint', 'production'), ('displayed_total_test_usd', '.1')):
            with self.assertRaises(ValueError): p.validate_billing({**self.bill, key: value}, self.now)

    def test_stale_or_future_billing_not_reused(self):
        for delta in (timedelta(minutes=16), timedelta(seconds=-1)):
            with self.assertRaises(ValueError): p.validate_billing(self.bill, self.now + delta)


if __name__ == '__main__': unittest.main()
