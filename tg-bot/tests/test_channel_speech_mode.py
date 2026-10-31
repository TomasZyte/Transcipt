from copy import deepcopy
from datetime import datetime, timedelta, timezone
import unittest
import channel_speech_test as r
import test_channel_mode as fixture


class SpeechBudgetTests(unittest.TestCase):
    def setUp(self):
        self.now=datetime(2026,11,1,tzinfo=timezone.utc)
        self.bill={'endpoint':r.full.ENDPOINT,'period_start':r.PERIOD[0],'period_end':r.PERIOD[1],
            'period_ui':r.PERIOD[2],'period_timezone':'UTC','group':'Serverless / Endpoint','billing_lag_hours':1,
            'observed_at_utc':self.now.isoformat(),'table':'Date\t'+r.full.ENDPOINT+'\n31\t$0,028\n30\t$0,056\n29\t$0,053\n28\t$0,075',
            'displayed_total_test_usd':'.212','max_workers':0,'active_workers':0,'endpoint_idle_observed_this_turn':True}
        self.health={'observed_at_utc':self.now.isoformat(),'endpoint':r.full.ENDPOINT,
            'health':{'jobs':deepcopy(r.EXPECTED_JOBS),'workers':{'idle':0,'running':0}}}
        self.stop={**deepcopy(self.health),'observed_at_utc':(self.now-timedelta(hours=8)).isoformat(),
                   'max_workers':0,'active_workers':0}
        self.journals=[{'status':'COMPLETED','submitted_at':i} for i in range(20)]
        self.journals[-1].update(channel_asr_validated=True,planning_total_usd='1.79')

    def reconcile(self):return r.reconcile(self.bill,self.health,self.stop,self.journals,self.now)

    def test_settled_bill_replaces_closed_reserve_without_resetting_history(self):
        before=deepcopy(self.journals); result=self.reconcile()
        self.assertEqual(result['baseline_spend_usd'],'0.32')
        self.assertEqual(result['planning_total_usd'],'0.82')
        self.assertEqual(self.journals,before)

    def test_new_month_only_partial_bill_wrong_endpoint_and_lag_are_rejected(self):
        original=deepcopy(self.bill)
        for change in ({'period_start':'2026-11-01'},{'endpoint':'production'},
                       {'displayed_total_test_usd':'0'},{'period_ui':'Today'},
                       {'observed_at_utc':(self.now-timedelta(minutes=16)).isoformat()}):
            self.bill={**original,**change}
            with self.assertRaises(ValueError):self.reconcile()
        self.bill=original; self.stop['observed_at_utc']=(self.now-timedelta(minutes=74)).isoformat()
        with self.assertRaises(ValueError):self.reconcile()

    def test_pending_missing_new_jobs_or_budget_overflow_block_new_campaign(self):
        self.journals.pop()
        with self.assertRaises(ValueError):self.reconcile()
        self.setUp(); self.journals[0]['status']='IN_PROGRESS'
        with self.assertRaises(ValueError):self.reconcile()
        self.setUp(); self.health['health']['jobs']['completed']+=1
        with self.assertRaises(ValueError):self.reconcile()
        self.setUp(); self.health['health']['workers']['idle']=1
        with self.assertRaises(ValueError):self.reconcile()
        self.setUp(); self.bill.update(displayed_total_test_usd='1.5',table='Date\t'+r.full.ENDPOINT+'\n31\t$1.5')
        with self.assertRaises(ValueError):self.reconcile()

    def test_no_network_without_fresh_preflight_and_no_reference_payload(self):
        harness=fixture.ChannelBudgetTests(); runner=harness.runner();runner.__class__=r.SpeechRunner
        urls=[f'https://{runner.settings.storage_host}/poslovnica-test/quality/{n}.wav?sig=x' for n in ('left','right')]
        body=runner.request_body(runner.payload(urls))
        self.assertEqual(body['input']['profile'],r.PROFILE)
        with self.assertRaises(ValueError):runner.request_body({**body['input'],'text':'private'})
        harness.test_no_network_before_budget_preflight()


if __name__=='__main__':unittest.main()
