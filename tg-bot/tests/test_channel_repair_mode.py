from copy import deepcopy
import unittest

import channel_repair_test as r
import test_channel_mode as fixture


class RepairBudgetTests(unittest.TestCase):
    def test_full_previous_reserve_and_shutdown_are_required(self):
        c = {'baseline_spend_usd': '1.29', 'previous_campaign_sha256': 'a'*64}
        j = {'status': 'COMPLETED', 'channel_asr_validated': True,
             'campaign_sha256': 'a'*64, 'planning_total_usd': '1.29'}
        stop = {'endpoint': r.full.ENDPOINT, 'max_workers': 0, 'active_workers': 0,
                'full_pipeline_enabled': False, 'channel_asr_enabled': False,
                'health': {'jobs': {'completed': 19, 'failed': 1, 'inProgress': 0, 'inQueue': 0, 'retried': 2},
                           'workers': {'idle': 0, 'running': 0}}}
        r.validate_prior(c, 'a'*64, j, stop)
        self.assertEqual(r.full.money(c['baseline_spend_usd'])+r.full.RESERVE_USD, r.full.money('1.79'))
        for changed in ({**c, 'baseline_spend_usd': '.192'}, {**c, 'previous_campaign_sha256': 'b'*64}):
            with self.assertRaises(ValueError): r.validate_prior(changed, 'a'*64, j, stop)
        for key in ('max_workers', 'full_pipeline_enabled', 'channel_asr_enabled'):
            with self.assertRaises(ValueError): r.validate_prior(c, 'a'*64, j, {**stop, key: 1})
        bad = deepcopy(stop); bad['health']['jobs']['completed'] = 20
        with self.assertRaises(ValueError): r.validate_prior(c, 'a'*64, j, bad)

    def test_bounded_policy_and_no_hints_before_network(self):
        harness = fixture.ChannelBudgetTests()
        runner = harness.runner(); runner.__class__ = r.RepairRunner
        urls = [f'https://{runner.settings.storage_host}/poslovnica-test/quality/{n}.wav?sig=x' for n in ('left', 'right')]
        body = runner.request_body(runner.payload(urls))
        self.assertEqual(body['input']['profile'], r.PROFILE)
        self.assertEqual(body['policy']['executionTimeout'], 300000)
        with self.assertRaises(ValueError): runner.request_body({**body['input'], 'reference': 'private'})
        harness.test_no_network_before_budget_preflight()


if __name__ == '__main__': unittest.main()
