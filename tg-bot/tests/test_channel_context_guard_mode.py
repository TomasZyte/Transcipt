from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import channel_context_guard_test as r
import test_channel_mode as fixture
import test_channel_speech_v2_mode as previous_fixture


class GuardBudgetTests(unittest.TestCase):
    def setUp(self):
        old=previous_fixture.SpeechV2BudgetTests();old.setUp()
        self.now=old.now
        self.bill=deepcopy(old.bill)
        self.health=deepcopy(old.health)
        self.health['health']['jobs']=deepcopy(r.EXPECTED_JOBS)
        self.health['health']['workers']={k:0 for k in ('idle','initializing','ready','running','throttled','unhealthy')}
        self.stop={**deepcopy(self.health),'observed_at_utc':(self.now-timedelta(minutes=20)).isoformat(),
                   'max_workers':0,'active_workers':0,'release':73}
        self.journals=[{'status':'COMPLETED','submitted_at':i} for i in range(23)]
        self.journals[-1].update(channel_asr_validated=True,planning_total_usd='1.33')

    def reconcile(self):
        return r.reconcile(self.bill,self.health,self.stop,self.journals,self.now)

    def test_recent_job_keeps_every_prior_reserve_despite_lagging_bill(self):
        before=deepcopy(self.journals)
        value=self.reconcile()
        self.assertEqual(value['baseline_spend_usd'],'1.33')
        self.assertEqual(value['planning_total_usd'],'1.83')
        self.assertEqual(value['released_prior_reserve_usd'],'0')
        self.assertEqual(self.journals,before)

    def test_higher_fresh_bill_takes_precedence_and_overflow_blocks(self):
        self.bill.update(table='Date\t'+r.full.ENDPOINT+'\n1\t$1.3',displayed_total_test_usd='1.3')
        self.assertEqual(self.reconcile()['planning_total_usd'],'1.91')
        self.bill.update(table='Date\t'+r.full.ENDPOINT+'\n1\t$1.5',displayed_total_test_usd='1.5')
        with self.assertRaises(ValueError): self.reconcile()

    def test_stale_partial_or_inconsistent_bill_rejected(self):
        for update in ({'endpoint':'production'},{'period_ui':'Today'},
                       {'observed_at_utc':(self.now-timedelta(minutes=16)).isoformat()},
                       {'displayed_total_test_usd':'0'}):
            self.setUp();self.bill.update(update)
            with self.assertRaises(ValueError):self.reconcile()

    def test_changed_history_active_worker_or_post_shutdown_job_blocks(self):
        for mutate in (lambda:self.journals.pop(), lambda:self.journals[0].update(status='IN_QUEUE'),
                       lambda:self.health['health']['workers'].update(running=1),
                       lambda:self.stop.update(release=66),
                       lambda:self.journals[-1].update(planning_total_usd='0.33'),
                       lambda:self.journals[-1].update(submitted_at=self.now.timestamp())):
            self.setUp();mutate()
            with self.assertRaises(ValueError):self.reconcile()

    def test_one_payload_has_no_reference_and_unchanged_timeout(self):
        harness=fixture.ChannelBudgetTests();runner=harness.runner();runner.__class__=r.GuardRunner
        urls=[f'https://{runner.settings.storage_host}/poslovnica-test/quality/{n}.wav?sig=x' for n in ('left','right')]
        body=runner.request_body(runner.payload(urls))
        self.assertEqual(body['input']['profile'],r.PROFILE)
        self.assertEqual(body['policy'],{'executionTimeout':300000,'ttl':420000})
        with self.assertRaises(ValueError):runner.request_body({**body['input'],'prompt':'reference'})
        harness.test_no_network_before_budget_preflight()

    def test_preflight_requires_actual_image_digest_and_fresh_bill(self):
        harness=fixture.ChannelBudgetTests();runner=harness.runner();runner.__class__=r.GuardRunner
        runner.campaign={'commit':'a'*40}
        p={'billing_file':'fixture','billing_sha256':'b'*64,'test_billed_usd':'.222',
           'observed_image':r.full.IMAGE_PREFIX[:-1]+'_build:aaaaaaaaa',
           'observed_image_digest':'sha256:'+'c'*64}
        with patch.object(r.full.FullRunner,'preflight',return_value=p), \
             patch.object(r.full,'digest',return_value='b'*64),patch.object(r.full,'read_json',return_value=self.bill):
            self.assertEqual(runner.preflight(self.now),p)
            p['observed_image']='registry.runpod.net/production:aaaaaaaaa'
            with self.assertRaises(ValueError):runner.preflight(self.now)

    def test_default_cli_does_not_submit(self):
        runner=Mock();runner.settings.prior_spend=r.Decimal('1.33')
        with patch('sys.argv',['context','--env-file','fixture']),patch.object(r.full.TestSettings,'load'), \
             patch.object(r,'GuardRunner',return_value=runner),patch('builtins.print'):
            self.assertEqual(r.main(),0)
        runner.execute.assert_not_called()

    def test_capacity_wait_is_allowed_without_changing_job_history(self):
        expected=deepcopy(self.health['health'])
        for changes in ({},{'idle':1,'ready':1},{'throttled':1}):
            health=deepcopy(expected);health['workers'].update(changes)
            r.validate_pre_submit_health(health,expected)
        self.assertEqual(expected,self.health['health'])

    def test_busy_initializing_extra_workers_and_changed_counts_still_block(self):
        expected=deepcopy(self.health['health'])
        for changes in ({'throttled':2},{'initializing':1},{'running':1},
                        {'unhealthy':1},{'throttled':1,'idle':1,'ready':1},
                        {'throttled':True},{'ready':1}):
            health=deepcopy(expected);health['workers'].update(changes)
            with self.assertRaises(RuntimeError):r.validate_pre_submit_health(health,expected)
        for changes in ({'inQueue':1},{'inProgress':1},{'completed':24}):
            health=deepcopy(expected);health['jobs'].update(changes)
            with self.assertRaises(RuntimeError):r.validate_pre_submit_health(health,expected)


if __name__=='__main__':unittest.main()
