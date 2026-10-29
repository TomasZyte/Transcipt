"""Offline regression tests for a new one-shot model and unreconciled costs."""
from copy import deepcopy
from dataclasses import replace
from datetime import datetime,timedelta,timezone
from decimal import Decimal
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock,Mock,patch

import httpx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import gigaam_test as g
import asr_test as a
from integration_guard import save_json
from test_asr_mode import fixture as old_fixture, response


def fixture(root):
    settings=old_fixture(root)
    old=root/'context-pipeline-v1/jobs/message-context-pipeline-v1.json'
    old.parent.mkdir(parents=True)
    save_json(old,{'status':'COMPLETED','planning_total_usd':'0.616','context_pipeline_validated':False})
    folder=root/g.CAMPAIGN
    folder.mkdir()
    campaign={'schema':1,'campaign':g.CAMPAIGN,'endpoint':g.ENDPOINT,'bucket':g.BUCKET,
        'variant':g.VARIANT,'image':g.IMAGE,'commit':g.COMMIT,'model_revision':g.REVISION,
        'clip_id':'context','budget_usd':'2','baseline_spend_usd':'0.616',
        'manifest':str(root/'manifest.json'),'manifest_sha256':g.digest(root/'manifest.json'),
        'audio_sha256':g.digest(root/'context.wav'),
        'history_sha256':{k:g.digest(p) for k,p in g.history_files(root).items()}}
    save_json(folder/'campaign.json',campaign)
    preflight={'verified_at_utc':datetime.now(timezone.utc).isoformat(),'endpoint':g.ENDPOINT,
        'image':g.IMAGE,'paid_run_authorized':True,'variant':g.VARIANT,
        'active_workers':0,'max_workers':1,'gpu_count':1,'execution_timeout_seconds':120,
        'idle_timeout_seconds':5,'flashboot':False,'asr_execution_enabled':True,
        'full_pipeline_enabled':False,'context_asr_enabled':False,
        'campaign_sha256':g.digest(folder/'campaign.json'),
        'gpu_rate_usd_per_hour':'0.75','test_billed_usd':'0.098'}
    save_json(folder/'preflight.json',preflight)
    return settings


def output(runner):
    raw={'text':'неподсказанный тест','segments':[{'text':'неподсказанный тест','start':None,'end':None}],
         'language_detected':None}
    result=g.normalize_result(runner.plan,raw)
    result['runtime']={'revision':g.REVISION,'lock_sha256':'b'*64,'packages':{
        'gigaam':'0.1.0','torch':'2.8.0+cu128','torchaudio':'2.8.0+cu128','hydra-core':'1.3.2','omegaconf':'2.3.0'}}
    return {'profile':'asr-comparison','sample_id':'context','output':result}


class GigaamOffline(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve()
        self.settings=fixture(self.root)
        self.runner=g.GigaamRunner(self.settings)

    def test_default_is_offline_and_preserves_previous_failed_quality_evidence(self):
        before={k:p.read_bytes() for k,p in g.history_files(self.root).items()}
        with patch.object(sys,'argv',['gigaam_test.py','--env-file','offline.env']), \
             patch.object(g.TestSettings,'load',return_value=self.settings), \
             patch.object(a,'upload_audio',side_effect=AssertionError('network forbidden')), \
             patch.object(g.asyncio,'run',side_effect=AssertionError('network forbidden')), \
             patch('sys.stdout',new_callable=io.StringIO) as stdout:
            self.assertEqual(g.main(),0)
            self.assertEqual(json.loads(stdout.getvalue())['planning_total_usd'],'1.116')
        self.assertEqual(before,{k:p.read_bytes() for k,p in g.history_files(self.root).items()})
        self.assertFalse(self.runner.path.parent.exists())

    def test_delayed_bill_cannot_erase_context_reserve(self):
        path=self.runner.campaign_path
        original=g.read_json(path)
        for baseline in ('0.116','1.501'):
            save_json(path,{**original,'baseline_spend_usd':baseline})
            with self.assertRaises(ValueError):g.GigaamRunner(self.settings)

    def test_other_resources_and_model_are_rejected(self):
        for changes in ({'endpoint':'l20o867nueti1d'},{'bucket':'transcribe'}):
            with self.assertRaises(ValueError):g.GigaamRunner(replace(self.settings,**changes))
        c=g.read_json(self.runner.campaign_path)
        save_json(self.runner.campaign_path,{**c,'variant':'fw_v3'})
        with self.assertRaises(ValueError):g.GigaamRunner(self.settings)

    def test_changed_context_history_blocks_new_attempt(self):
        p=self.root/'context-pipeline-v1/jobs/message-context-pipeline-v1.json'
        save_json(p,{'status':'IN_PROGRESS','planning_total_usd':'0.616'})
        with self.assertRaises(ValueError):self.runner.check_available()

    def test_preflight_enforces_flags_rate_current_bill_and_age(self):
        p=self.runner.settings.runtime/'preflight.json'
        original=g.read_json(p)
        changes=[{'max_workers':2},{'max_workers':True},{'active_workers':1},
            {'image':'other'},{'asr_execution_enabled':False},{'full_pipeline_enabled':True},
            {'context_asr_enabled':True},{'execution_timeout_seconds':300},
            {'test_billed_usd':'1.501'},{'gpu_rate_usd_per_hour':'0.751'},
            {'campaign_sha256':'a'*64}, {'paid_run_authorized':False},
            {'verified_at_utc':(datetime.now(timezone.utc)-timedelta(minutes=16)).isoformat()}]
        for change in changes:
            save_json(p,{**original,**change})
            with self.assertRaises((ValueError,RuntimeError)):self.runner.preflight()

    def test_only_one_fixed_clip_without_reference_text_can_be_sent(self):
        url='https://'+self.settings.storage_host+'/poslovnica-test/quality/context.wav?signature=fixture'
        payload=self.runner.payload(url)
        self.assertEqual(self.runner.request_body(payload)['policy'],{'executionTimeout':120000,'ttl':240000})
        for bad in ({**payload,'prompt':'ожидаемая фраза'},{**payload,'variant':'fw_v3'},
                    self.runner.payload(url.replace('poslovnica-test','transcribe'))):
            with self.assertRaises(ValueError):self.runner.request_body(bad)

    def test_raw_text_provenance_and_timestamps_cannot_be_fabricated(self):
        original=output(self.runner)
        self.runner.path.parent.mkdir()
        for mutation in (
            lambda o:o['output'].update(raw_asr_text='подсказанная фраза'),
            lambda o:o['output'].update(timestamp_kind='word_times'),
            lambda o:o['output']['runtime'].update(revision='0'*40),
            lambda o:o['output']['runtime']['packages'].update(torch='2.9.0')):
            bad=deepcopy(original);mutation(bad)
            with self.assertRaises(ValueError):self.runner.validate_output(bad)
        self.runner.validate_output(original)
        self.assertTrue(g.read_json(self.runner.path)['asr_validated'])
        self.assertFalse(g.read_json(self.runner.path)['quality_validated'])


class GigaamCloudFakes(unittest.IsolatedAsyncioTestCase):
    async def test_one_attempt_records_before_post_and_cannot_repeat(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve()
            settings=fixture(root)
            client=AsyncMock();client.__aenter__.return_value=client
            runner=g.GigaamRunner(settings,poll_s=0,client_factory=Mock(return_value=client))
            async def post(url,**kwargs):
                self.assertEqual(g.read_json(runner.path)['status'],'reserved')
                self.assertTrue((root/'runner.lock').exists())
                return response({'id':'gigaam-fixture'})
            async def get(url,**kwargs):
                if url.endswith('/health'):
                    return response({'jobs':{'inProgress':0,'inQueue':0},'workers':{'idle':0,'running':0}})
                return response({'status':'COMPLETED','output':output(runner)})
            client.post.side_effect=post;client.get.side_effect=get
            url='https://'+settings.storage_host+'/poslovnica-test/quality/context.wav?signature=fixture'
            with patch.object(a,'upload_audio',return_value=url):
                await runner.execute()
                with self.assertRaises(RuntimeError):await runner.execute()
            self.assertEqual(client.post.await_count,1)
            state=g.read_json(runner.path)
            self.assertEqual(state['planning_total_usd'],'1.116')
            self.assertEqual(state['reserved_usd'],'0.50')
            self.assertTrue(state['asr_validated'])
            self.assertFalse((root/'runner.lock').exists())


if __name__=='__main__':unittest.main()
