"""Offline protection against extra spending and mixed full-pipeline evidence."""
import asyncio
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, AsyncMock
import wave

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import full_pipeline_test as f
from integration_config import TestSettings
from integration_guard import save_json, process_lock
from test_asr_mode import response


def fixture(root):
    settings = TestSettings(123456789,'fixturebot',1234,f.ENDPOINT,f.BUCKET,
        '0'*32+'.r2.cloudflarestorage.com',root,2,Decimal('2'),Decimal('.116'),
        {'RUNPOD_API_KEY':'offline-key','S3_ACCESS_KEY':'offline-id','S3_SECRET_KEY':'offline-secret'})
    rows = []
    for name in f.IDS:
        path = root/(name+'.wav')
        with wave.open(str(path),'wb') as audio:
            audio.setparams((1,2,16000,0,'NONE','not compressed'))
            audio.writeframes(b'\0\0'*16000)
        rows.append({'id':name,'wav':path.name,'wav_sha256':f.digest(path)})
    manifest = root/'manifest.json'
    save_json(manifest,{'samples':rows})
    for folder,name in ((root/'jobs','message-old'),(root/'asr-children-v1'/'jobs','message-asr-wx_auto'),
                        (root/'asr-children-v1'/'jobs','message-asr-fw_v3')):
        folder.mkdir(parents=True,exist_ok=True)
        save_json(folder/(name+'.json'),{'status':'COMPLETED','reserved_usd':'.50'})
    target = root/f.CAMPAIGN
    target.mkdir()
    commit = 'a'*40
    campaign = {'schema':1,'campaign':f.CAMPAIGN,'endpoint':f.ENDPOINT,'bucket':f.BUCKET,
        'modes':f.MODES,'commit':commit,'image':f.IMAGE_PREFIX+commit[:9],
        'budget_usd':'2','baseline_spend_usd':'1.116','manifest':str(manifest),
        'manifest_sha256':f.digest(manifest),'source_manifest_sha256':f.digest(f.SOURCES),
        'audio_sha256':{r['id']:r['wav_sha256'] for r in rows},
        'history_sha256':{n:f.digest(p) for n,p in f.history_files(root).items()}}
    save_json(target/'campaign.json',campaign)
    save_json(target/'preflight.json',{'verified_at_utc':datetime.now(timezone.utc).isoformat(),
        'endpoint':f.ENDPOINT,'image':campaign['image'],'paid_run_authorized':True,
        'active_workers':0,'max_workers':1,'gpu_count':1,'execution_timeout_seconds':300,
        'idle_timeout_seconds':5,'flashboot':False,'full_pipeline_enabled':True,
        'asr_comparison_enabled':False,'campaign_sha256':f.digest(target/'campaign.json'),
        'gpu_rate_usd_per_hour':'.75','test_billed_usd':'.075'})
    return settings


def output(runner):
    sources = f.read_json(f.SOURCES)
    runtime = {'asr':{'revision':f.REVISION,'lock_sha256':'b'*64,'packages':{'whisperx':'3.8.6'}},
               'nltk_lock_sha256':'c'*64}
    runtime.update({k:{'repository':sources[k]['repository'],'revision':sources[k]['revision']} for k in ('alignment','diarization')})
    raw = {'language':'ru','segments':[{'start':0.,'end':.9,'text':'Тест'}]}
    aligned = {'segments':[{'start':0.,'end':.9,'text':'Тест','speaker':'SPEAKER_00',
                            'words':[{'word':'Тест','start':0.,'end':.9,'speaker':'SPEAKER_00'}]}]}
    timeline = {'status':'ok','segments':[{'start':0.,'end':.9,'speaker':'SPEAKER_00'}]}
    return {'profile':f.PROFILE,'status':'completed','modes':f.MODES,'runtime':runtime,
        'source_manifest_sha256':f.digest(f.SOURCES),'samples':[{'id':r['id'],'sha256':r['sha256'],
        'durationSeconds':r['duration_seconds'],'outputs':{m:f.format_output(raw,aligned,timeline,m,1.) for m in f.MODES}}
        for r in runner.rows]}


class FullOffline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.settings = fixture(self.root)
        self.runner = f.FullRunner(self.settings)
        self.urls = ['https://'+self.settings.storage_host+'/poslovnica-test/quality/'+s+'.wav?signature=fixture' for s in f.IDS]

    def test_offline_default_never_uploads_or_reserves(self):
        with patch.object(sys,'argv',['full_pipeline_test.py','--env-file','unused']), \
             patch.object(f.TestSettings,'load',return_value=self.settings), \
             patch.object(f.asyncio,'run',side_effect=AssertionError('network forbidden')), \
             patch('sys.stdout',new_callable=io.StringIO) as stdout:
            self.assertEqual(f.main(),0)
            self.assertEqual(json.loads(stdout.getvalue())['planning_total_usd'],'1.616')
        self.assertFalse(self.runner.path.exists())

    def test_rejects_production_and_changed_prior_history(self):
        for changes in ({'endpoint':'l20o867nueti1d'},{'bucket':'transcribe'}):
            with self.assertRaises(ValueError): f.FullRunner(replace(self.settings,**changes))
        save_json(self.root/'asr-children-v1/jobs/message-asr-fw_v3.json',{'status':'FAILED'})
        with self.assertRaises(ValueError): self.runner.check_available()

    def test_budget_typed_config_and_freshness_fail_closed(self):
        path = self.runner.settings.runtime/'preflight.json'
        original = f.read_json(path)
        for changes in ({'test_billed_usd':'1.501'},{'gpu_rate_usd_per_hour':'.76'},
                        {'max_workers':True},{'active_workers':1},{'full_pipeline_enabled':False},
                        {'asr_comparison_enabled':True},{'execution_timeout_seconds':600},
                        {'verified_at_utc':(datetime.now(timezone.utc)-timedelta(minutes=16)).isoformat()}):
            save_json(path,{**original,**changes})
            with self.assertRaises((ValueError,RuntimeError)): self.runner.preflight()

    def test_exact_four_full_files_in_one_request_without_hints(self):
        payload = self.runner.payload(self.urls)
        body = self.runner.request_body(payload)
        self.assertEqual(body['policy'],{'executionTimeout':300000,'ttl':420000})
        self.assertEqual(len(body['input']['samples']),4)
        for key in ('prompt','speaker_count','options'):
            changed = deepcopy(payload)
            changed['samples'][0][key] = 'forbidden'
            with self.assertRaises(ValueError): self.runner.request_body(changed)

    def test_result_rebuild_catches_text_loss_mixed_audio_and_model(self):
        self.runner.path.parent.mkdir(parents=True)
        result = output(self.runner)
        self.runner.validate_output(result)
        self.assertTrue(self.runner.state['full_pipeline_validated'])
        for mutate in (lambda r:r['samples'].pop(),
                       lambda r:r['samples'][0].update(sha256='a'*64),
                       lambda r:r['runtime']['alignment'].update(revision='wrong'),
                       lambda r:r['samples'][0]['outputs']['short12'].update(rawText='lost')):
            bad = deepcopy(result)
            mutate(bad)
            with self.assertRaises(ValueError): self.runner.validate_output(bad)

    def test_parent_lock_and_health_prevent_upload(self):
        with patch.object(f,'upload_samples',side_effect=AssertionError('must not upload')):
            with process_lock(self.root):
                with self.assertRaises(FileExistsError): asyncio.run(self.runner.execute())
            with patch.object(self.runner,'health',new=AsyncMock(side_effect=RuntimeError('busy'))):
                with self.assertRaises(RuntimeError): asyncio.run(self.runner.execute())
        self.assertFalse(self.runner.path.exists())

    def test_one_post_persists_all_eight_results_and_blocks_repeat(self):
        result = output(self.runner)
        requests = []
        class Client:
            def __init__(inner,**kwargs): pass
            async def __aenter__(inner): return inner
            async def __aexit__(inner,*args): pass
            async def post(inner,url,**kwargs):
                requests.append(kwargs['json'])
                return response({'id':'fixture-job'})
            async def get(inner,url,**kwargs): return response({'status':'COMPLETED','output':result})
        self.runner.client_factory = Client
        self.runner.poll_s = 0
        before = {n:p.read_bytes() for n,p in f.history_files(self.root).items()}
        with patch.object(self.runner,'health',new=AsyncMock()),patch.object(f,'upload_samples',return_value=self.urls):
            asyncio.run(self.runner.execute())
            with self.assertRaises(RuntimeError): asyncio.run(self.runner.execute())
        self.assertEqual(len(requests),1)
        self.assertTrue(f.read_json(self.runner.path)['full_pipeline_validated'])
        self.assertEqual(before,{n:p.read_bytes() for n,p in f.history_files(self.root).items()})

    def test_ambiguous_submission_consumes_the_only_attempt(self):
        class Client:
            def __init__(inner,**kwargs): pass
            async def __aenter__(inner): return inner
            async def __aexit__(inner,*args): pass
            async def post(inner,*args,**kwargs): raise TimeoutError('ambiguous')
        self.runner.client_factory = Client
        with patch.object(self.runner,'health',new=AsyncMock()),patch.object(f,'upload_samples',return_value=self.urls):
            with self.assertRaises(TimeoutError): asyncio.run(self.runner.execute())
        self.assertEqual(f.read_json(self.runner.path)['status'],'reserved')
        with self.assertRaises(RuntimeError): self.runner.check_available()


if __name__ == '__main__': unittest.main()
