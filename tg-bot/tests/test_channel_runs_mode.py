from copy import deepcopy
import unittest

import channel_runs_test as r
import test_channel_mode as fixture


class RunsBudgetTests(unittest.TestCase):
    def test_previous_reserve_cannot_be_replaced_with_lower_stale_bill(self):
        campaign = {'baseline_spend_usd': '.79', 'previous_campaign_sha256': 'a'*64}
        journal = {'status': 'COMPLETED', 'channel_asr_validated': True,
                   'campaign_sha256': 'a'*64, 'planning_total_usd': '.79'}
        shutdown = {'endpoint': r.full.ENDPOINT, 'max_workers': 0, 'active_workers': 0,
                    'health': {'jobs': {'completed': 18, 'failed': 1, 'inProgress': 0, 'inQueue': 0, 'retried': 2},
                               'workers': {'idle': 0, 'running': 0}}}
        r.validate_prior(campaign, 'a'*64, journal, shutdown)
        self.assertEqual(r.full.money(campaign['baseline_spend_usd'])+r.full.RESERVE_USD, r.full.money('1.29'))
        for change in ({'baseline_spend_usd': '.184'}, {'previous_campaign_sha256': 'b'*64}):
            with self.assertRaises(ValueError): r.validate_prior({**campaign, **change}, 'a'*64, journal, shutdown)
        for change in ({'status': 'IN_PROGRESS'}, {'channel_asr_validated': False}):
            with self.assertRaises(ValueError): r.validate_prior(campaign, 'a'*64, {**journal, **change}, shutdown)
        bad = deepcopy(shutdown); bad['health']['workers']['idle'] = 1
        with self.assertRaises(ValueError): r.validate_prior(campaign, 'a'*64, journal, bad)

    def test_profile_payload_is_separate_and_preflight_is_required_before_network(self):
        harness = fixture.ChannelBudgetTests()
        runner = harness.runner()
        runner.__class__ = r.RunsRunner
        urls = [f'https://{runner.settings.storage_host}/poslovnica-test/quality/{n}.wav?sig=x' for n in ('left', 'right')]
        body = runner.request_body(runner.payload(urls))
        self.assertEqual(body['input']['profile'], r.PROFILE)
        self.assertEqual(body['policy']['executionTimeout'], 300000)
        with self.assertRaises(ValueError): runner.request_body({**body['input'], 'reference': 'private'})
        harness.test_no_network_before_budget_preflight()


if __name__ == '__main__': unittest.main()
