from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import sys,unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import independent_switch_test as c

NOW=datetime(2026,11,11,21,tzinfo=timezone.utc)
HEALTH={'jobs':dict(c.EXPECTED_JOBS),'workers':{k:0 for k in ('idle','ready','initializing','running','throttled','unhealthy')}}

def evidence():
    billing={'endpoint':c.full.ENDPOINT,'period_start':'2026-10-13','period_end':'2026-11-11',
        'period_timezone':'UTC','observed_at_utc':NOW.isoformat(),'billing_lag_hours':1,'observed_billed_usd':'0.299'}
    rows=[{'Endpoint':c.full.ENDPOINT,'Date':'2026-11-01T00:00:00Z','Amount':'0.299'}]
    journals=[{'status':'COMPLETED','endpoint_id':c.full.ENDPOINT,'bucket':c.full.BUCKET,
        'submitted_at':1793550000+i,'elapsed_sec':60,'job_id':'old'+str(i)} for i in range(24)]
    journals[-1]['job_id']='13a77045-e4c8-46ab-8972-ad73336a43c0-e2'
    return billing,rows,journals,deepcopy(HEALTH)

class IndependentClientTests(unittest.TestCase):
    def test_rejects_registry_push_of_cancelled_build_artifact(self):
        current='c'*64
        good='exporting config sha256:'+current+' done\n'+'c'*12+'  config 10.7 KB'
        self.assertEqual(c.validate_build_publication(good),current)
        for text in (good.replace('c'*12+'  config','f'*12+'  config'),
                     good.split('\n')[0],good+'\n'+good):
            with self.assertRaises(ValueError): c.validate_build_publication(text)

    def test_completed_provider_job_does_not_mean_probe_success(self):
        runner=c.IndependentRunner.__new__(c.IndependentRunner)
        runner.lock=c.full.read_json(Path(c.probe.__file__).with_name('independent-probe-lock.json'))
        runner.path=Path('unused.json')
        runner.state={'status':'COMPLETED','independent_probe_validated':False}
        # Actual failure shape received from the old full-pipeline handler.
        result={'profile':'full-pipeline-v1','status':'failed','phase':'validate','error_type':'ValueError'}
        with patch.object(c,'save_json') as save,self.assertRaises(ValueError):
            runner.accept_result(result)
        self.assertEqual(runner.state['status'],'COMPLETED')
        self.assertEqual(runner.state['probe_status'],'rejected_output')
        self.assertFalse(runner.state['independent_probe_validated'])
        save.assert_called_once()

    def test_settled_billing_uses_only_test_endpoint_and_reserves_fifty_cents(self):
        values=evidence(); values[1].append({'Endpoint':'production','Date':'2026-11-11T00:00:00Z','Amount':'100'})
        self.assertEqual(str(c.validate_budget(*values,now=NOW)),'0.42')

    def test_rejects_stale_incomplete_or_over_budget_evidence(self):
        for kind in ('stale','period','timezone','total','duplicate','budget','unresolved','recent','other_endpoint','missing_journal','different_latest'):
            b,rows,j,h=evidence()
            if kind=='stale': b['observed_at_utc']='2026-11-11T20:44:59Z'
            elif kind=='period': b['period_start']='2026-10-29'
            elif kind=='timezone': b['period_timezone']='local'
            elif kind=='total': b['observed_billed_usd']='.1'
            elif kind=='duplicate': rows.append(deepcopy(rows[0]))
            elif kind=='budget': b['observed_billed_usd']=rows[0]['Amount']='1.50'
            elif kind=='unresolved': j[0]['status']='IN_QUEUE'
            elif kind=='recent': j[-1]['submitted_at']=NOW.timestamp()-100
            elif kind=='other_endpoint': j[0]['endpoint_id']='production'
            elif kind=='missing_journal': j.pop(0)
            else: j[-1]['job_id']='different'
            with self.subTest(kind=kind),self.assertRaises(ValueError): c.validate_budget(b,rows,j,h,now=NOW)

    def test_health_accepts_only_stopped_or_one_known_capacity_worker(self):
        c.validate_health(HEALTH)
        for variant in ({'initializing':1},{'throttled':1},{'idle':1,'ready':1}):
            h=deepcopy(HEALTH);h['workers'].update(variant);c.validate_health(h)
        for variant in ({'running':1},{'idle':2,'ready':2},{'unhealthy':1},{'initializing':1,'throttled':1}):
            h=deepcopy(HEALTH);h['workers'].update(variant)
            with self.assertRaises(ValueError): c.validate_health(h)
        h=deepcopy(HEALTH);h['jobs']['completed']+=1
        with self.assertRaises(ValueError): c.validate_health(h)

    def test_legacy_snapshot_requires_matching_scoped_terminal_journal(self):
        b,rows,j,h=evidence()
        j[1]['status']='FAILED'
        j[0]={'id':j[1]['job_id'],'status':'FAILED'}
        self.assertEqual(str(c.validate_budget(b,rows,j,h,now=NOW)),'0.42')
        j[0]['id']='unrelated'
        with self.assertRaises(ValueError): c.validate_budget(b,rows,j,h,now=NOW)

    def test_exact_audio_only_batch_has_timeout_and_no_production_urls(self):
        runner=c.IndependentRunner.__new__(c.IndependentRunner)
        runner.lock={'samples':[{'id':k,'sha256':'a'*64} for k in c.probe.IDS]}
        runner.settings=SimpleNamespace(storage_host='a'*32+'.r2.cloudflarestorage.com')
        payload={'profile':c.PROFILE,'samples':[dict(r,media_url='https://'+runner.settings.storage_host+'/poslovnica-test/quality/'+r['id']+'.wav?sig=test') for r in runner.lock['samples']]}
        self.assertEqual(runner.request_body(payload)['policy'],{'executionTimeout':300000,'ttl':420000})
        for kind in ('extra','hash','production','count'):
            bad=deepcopy(payload)
            if kind=='extra': bad['samples'][0]['text']='hint'
            elif kind=='hash': bad['samples'][0]['sha256']='b'*64
            elif kind=='count': bad['samples'].pop()
            else: bad['samples'][0]['media_url']=bad['samples'][0]['media_url'].replace('poslovnica-test','transcribe')
            with self.assertRaises(ValueError):runner.request_body(bad)

if __name__=='__main__': unittest.main()
