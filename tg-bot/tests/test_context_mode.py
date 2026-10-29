import asyncio
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock, AsyncMock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import context_test as c
import context_pipeline as pipeline
import test_full_pipeline_mode as fixture
from integration_guard import save_json


class ContextClient(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve()
        self.settings=fixture.fixture(self.root)
        self.old_result=fixture.output(c.full.FullRunner(self.settings))
        self.old=c.full.read_json(self.root/c.full.CAMPAIGN/'campaign.json')
        (self.root/c.full.CAMPAIGN/'jobs').mkdir()
        save_json(self.root/c.full.CAMPAIGN/'jobs/message-full-pipeline-v1.json',{'status':'COMPLETED'})
        self.dest=self.root/c.PROFILE
        self.dest.mkdir()
        campaign={**self.old,'campaign':c.PROFILE,'modes':c.MODES,
                  'history_sha256':{n:c.full.digest(p) for n,p in c.history_files(self.root).items()}}
        save_json(self.dest/'campaign.json',campaign)
        pre=c.full.read_json(self.root/c.full.CAMPAIGN/'preflight.json')
        save_json(self.dest/'preflight.json',{**pre,'context_asr_enabled':True,
                   'campaign_sha256':c.full.digest(self.dest/'campaign.json')})
        self.runner=c.ContextRunner(self.settings)

    def test_default_offline_plan_does_not_read_secrets_or_create_campaign(self):
        before=sorted(p.relative_to(self.root).as_posix() for p in self.root.rglob('*'))
        with patch.object(sys,'argv',['context_test.py','--corpus',str(self.root)]), \
             patch.object(c.full.TestSettings,'load',side_effect=AssertionError('secret access')), \
             patch.object(c.asyncio,'run',side_effect=AssertionError('network')), \
             patch('sys.stdout',new_callable=io.StringIO) as stream:
            self.assertEqual(c.main(),0)
            result=json.loads(stream.getvalue())
            self.assertFalse(result['paid_run_ready'])
            self.assertEqual(result['network_calls'],0)
        self.assertEqual(before,sorted(p.relative_to(self.root).as_posix() for p in self.root.rglob('*')))

    def test_current_conservative_budget_stops_new_half_dollar_reservation(self):
        path=self.dest/'campaign.json'
        save_json(path,{**c.full.read_json(path),'baseline_spend_usd':'1.616'})
        with self.assertRaises(ValueError): c.ContextRunner(self.settings)

    def test_previous_full_job_is_pinned_and_cannot_be_removed_or_changed(self):
        path=self.root/c.full.CAMPAIGN/'jobs/message-full-pipeline-v1.json'
        save_json(path,{'status':'FAILED'})
        with self.assertRaises(ValueError): self.runner.check_available()

    def test_context_preflight_required_before_upload(self):
        path=self.dest/'preflight.json'
        save_json(path,{**c.full.read_json(path),'context_asr_enabled':False})
        with patch.object(c.full,'upload_samples',side_effect=AssertionError('must not upload')):
            with self.assertRaises(ValueError): asyncio.run(self.runner.execute())

    def result(self):
        result=deepcopy(self.old_result)
        result.update(profile=c.PROFILE,modes=c.MODES)
        raw={'language':'ru','segments':[{'start':0.,'end':.9,'text':'Тест'}]}
        aligned={'segments':[{'start':0.,'end':.9,'text':'Тест',
                              'words':[{'word':'Тест','start':0.,'end':.9,'score':.8}]}]}
        model=Mock()
        model.detect_language.return_value='ru'
        model.transcribe.side_effect=lambda *a,**k:deepcopy(raw)
        def assign(value):
            for segment in value['segments']:
                segment['speaker']='SPEAKER_00'
                for word in segment['words']: word['speaker']='SPEAKER_00'
            return value
        with patch.dict(os.environ,{'FULL_PIPELINE_IMAGE':'1','FULL_PIPELINE_TEST':'1','CONTEXT_ASR_TEST':'1'}):
            for sample in result['samples']:
                timeline=sample['outputs']['standard30']['benchmark']['diarizationTimeline']
                sample['outputs']=pipeline.execute_sample([0.]*16000,model=model,
                    align=lambda *a:deepcopy(aligned),assign=assign,timeline=timeline,duration=1.)
        return result

    def test_one_request_keeps_window_evidence_and_consumes_campaign(self):
        result=self.result()
        requests=[]
        class Client:
            def __init__(self,**kwargs): pass
            async def __aenter__(self): return self
            async def __aexit__(self,*args): pass
            async def post(self,url,**kwargs):
                requests.append(kwargs['json'])
                return fixture.response({'id':'fixture-context-job'})
            async def get(self,url,**kwargs): return fixture.response({'status':'COMPLETED','output':result})
        self.runner.client_factory=Client
        self.runner.poll_s=0
        urls=['https://'+self.settings.storage_host+'/poslovnica-test/quality/'+s+'.wav?sig=fixture' for s in c.full.IDS]
        with patch.object(self.runner,'health',new=AsyncMock()),patch.object(c.full,'upload_samples',return_value=urls):
            asyncio.run(self.runner.execute())
            with self.assertRaises(RuntimeError): asyncio.run(self.runner.execute())
        self.assertEqual(len(requests),1)
        self.assertEqual(requests[0]['input']['profile'],c.PROFILE)
        self.assertTrue(c.full.read_json(self.runner.path)['context_pipeline_validated'])
        self.assertFalse(c.full.read_json(self.runner.path)['quality_validated'])
        bad=deepcopy(result)
        bad['runtime']['asr']['revision']='wrong'
        with self.assertRaises(ValueError): self.runner.validate_output(bad)


if __name__=='__main__': unittest.main()
