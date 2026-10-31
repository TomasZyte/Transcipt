from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import channel_test as c


class ChannelBudgetTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 10, 31, 9, tzinfo=timezone.utc)
        self.billing = {'endpoint': c.full.ENDPOINT, 'period_ui': 'окт. 1 31', 'period_start': '2026-10-01',
            'period_end': '2026-10-31', 'period_timezone': 'UTC', 'group': 'Serverless / Endpoint',
            'billing_lag_hours': 1, 'observed_at_utc': self.now.isoformat(),
            'displayed_total_test_usd': '.184', 'table': 'Date\t' + c.full.ENDPOINT + '\n30\t$0,056\n29\t$0,053\n28\t$0,075',
            'max_workers': 0, 'active_workers': 0, 'endpoint_idle_observed_this_turn': True}
        self.health = {'endpoint': c.full.ENDPOINT, 'observed_at_utc': self.now.isoformat(),
            'health': {'jobs': {'completed': 17, 'failed': 1, 'inQueue': 0, 'inProgress': 0, 'retried': 2},
                       'workers': {'idle': 0, 'running': 0}}}
        self.shutdown = {**deepcopy(self.health), 'max_workers': 0,
                         'observed_at_utc': (self.now-timedelta(hours=20)).isoformat()}
        self.journals = [{'status': 'COMPLETED', 'submitted_at': 1} for _ in range(16)] + [
            {'status': 'COMPLETED', 'submitted_at': 2, 'pause_context_validated': True, 'planning_total_usd': '1.75'}]

    def reconcile(self):
        return c.reconcile(self.billing, self.health, self.shutdown, self.journals, self.now)

    def test_reconciliation_closes_only_lag_covered_history_and_keeps_margin(self):
        before = deepcopy(self.journals)
        result = self.reconcile()
        self.assertEqual(result['observed_billed_usd'], '0.184')
        self.assertEqual(result['baseline_spend_usd'], '0.29')
        self.assertEqual(result['planning_total_usd'], '0.79')
        self.assertEqual(self.journals, before)
        self.assertFalse(result['old_journals_modified'])

    def test_stale_partial_period_wrong_endpoint_and_inconsistent_sum_rejected(self):
        original = deepcopy(self.billing)
        for change in ({'period_ui': 'окт. 30 31'}, {'period_start': '2026-10-30'},
                       {'endpoint': 'production'}, {'displayed_total_test_usd': '0'},
                       {'observed_at_utc': (self.now-timedelta(minutes=16)).isoformat()},
                       {'observed_at_utc': (self.now+timedelta(minutes=1)).isoformat()}):
            self.billing = {**original, **change}
            with self.assertRaises(ValueError): self.reconcile()

    def test_lag_unresolved_job_new_job_and_workers_block_reconciliation(self):
        self.shutdown['observed_at_utc'] = (self.now-timedelta(minutes=74)).isoformat()
        with self.assertRaises(ValueError): self.reconcile()
        self.setUp(); self.journals[0]['status'] = 'reserved'
        with self.assertRaises(ValueError): self.reconcile()
        self.setUp(); self.health['health']['jobs']['completed'] += 1
        with self.assertRaises(ValueError): self.reconcile()
        self.setUp(); self.health['health']['workers']['running'] = 1
        with self.assertRaises(ValueError): self.reconcile()
        self.setUp(); self.journals.pop()
        with self.assertRaises(ValueError): self.reconcile()

    def test_budget_overflow_blocks_new_attempt(self):
        self.billing.update(displayed_total_test_usd='1.5', table='Date\t'+c.full.ENDPOINT+'\n30\t$1.5')
        with self.assertRaises(ValueError): self.reconcile()

    def runner(self):
        r = c.ChannelRunner.__new__(c.ChannelRunner)
        r.settings = SimpleNamespace(storage_host='a'*32+'.r2.cloudflarestorage.com')
        r.plan = {'source_sha256': 'f'*64, 'confirmation': {'kind': 'owner_preview', 'start_seconds': 0, 'end_seconds': 32}}
        r.rows = [{'id': name, 'sha256': str(i)*64} for i, name in enumerate(('left', 'right'))]
        return r

    def test_payload_has_exact_channels_no_hints_and_bounded_execution(self):
        r = self.runner()
        urls = [f'https://{r.settings.storage_host}/poslovnica-test/quality/{n}.wav?sig=x' for n in ('left', 'right')]
        payload = r.payload(urls)
        self.assertEqual(r.request_body(payload)['policy'], {'executionTimeout': 300000, 'ttl': 420000})
        with self.assertRaises(ValueError): r.request_body({**payload, 'prompt': 'reference'})
        payload['samples'][0]['media_url'] = payload['samples'][0]['media_url'].replace('poslovnica-test', 'transcribe')
        with self.assertRaises(ValueError): r.request_body(payload)

    def test_no_network_before_budget_preflight(self):
        import asyncio
        from contextlib import nullcontext
        r = self.runner(); r.base_settings = SimpleNamespace(runtime=Path('.'))
        r.check_available = lambda: None
        r.preflight = lambda: (_ for _ in ()).throw(ValueError('No fresh billing'))
        with patch.object(c, 'process_lock', return_value=nullcontext()), patch.object(c, 'upload_channels') as upload:
            with self.assertRaises(ValueError): asyncio.run(r.execute())
            upload.assert_not_called()


if __name__ == '__main__': unittest.main()
