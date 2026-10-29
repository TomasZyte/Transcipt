from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch
import wave

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import speaker_test as s
from test_asr_mode import fixture as old_fixture, response


def write_wav(path,seconds):
    with wave.open(str(path),'wb') as wav:
        wav.setparams((1,2,16000,0,'NONE','not compressed'))
        wav.writeframes(b'\0\0'*int(seconds*16000))


def fixture(root):
    settings=old_fixture(root)
    old=root/'gigaam-children-v1/jobs/message-gigaam-children-v1.json'
    old.parent.mkdir(parents=True)
    s.save_json(old,{'status':'COMPLETED','planning_total_usd':'1.116'})
    write_wav(root/'sound2.wav',36)
    write_wav(root/'context.wav',11)
    source=root/'source.json'
    s.save_json(source,{'profile':'asr-comparison','output':{'status':'completed','variant_id':'wx_auto',
        'duration_seconds':11,'config':{'engine':'whisperx','model':'large-v3'},
        'language_detected':'ru','audio_sha256':s.digest(root/'context.wav'),
        'source_start_seconds':8.5,'source_end_seconds':19.5,
        'raw_engine_output':{'segments':[{'start':0.,'end':11.,'text':'Тестовая речь.'}]}}})
    target=root/s.CAMPAIGN;target.mkdir()
    campaign={'schema':1,'campaign':s.CAMPAIGN,'endpoint':s.ENDPOINT,'bucket':s.BUCKET,
        'commit':'c'*40,'image':s.IMAGE_PREFIX+'c'*9,'source_manifest_sha256':s.digest(s.SOURCES),
        'budget_usd':'2','baseline_spend_usd':'1.116','audio_file':str(root/'sound2.wav'),
        'audio_sha256':s.digest(root/'sound2.wav'),'source_result':str(source),
        'asr_evidence_sha256':s.comparison.evidence_sha(s.frozen_evidence(source)),
        'history_sha256':{k:s.digest(p) for k,p in s.history_files(root).items()}}
    s.save_json(target/'campaign.json',campaign)
    preflight={'verified_at_utc':datetime.now(timezone.utc).isoformat(),'endpoint':s.ENDPOINT,
        'image':campaign['image'],'paid_run_authorized':True,'active_workers':0,'max_workers':1,
        'gpu_count':1,'execution_timeout_seconds':300,'idle_timeout_seconds':5,'flashboot':False,
        'full_pipeline_enabled':True,'asr_comparison_enabled':False,'context_asr_enabled':False,
        'speaker_comparison_enabled':True,'campaign_sha256':s.digest(target/'campaign.json'),
        'gpu_rate_usd_per_hour':'0.75','test_billed_usd':'0.110'}
    s.save_json(target/'preflight.json',preflight)
    return settings


def output(runner):
    sources=s.read_json(s.SOURCES)
    runtime={k:{'repository':v['repository'],'revision':v['revision'],
        'files':{n:r.get('sha256','d'*64) for n,r in v['files'].items()}} for k,v in sources.items()}
    runtime['packages']={'whisperx':'3.8.6','pyannote.audio':'4.0.3'}
    local={'segments':[{'start':0.,'end':11.,'text':'Тестовая речь.',
                       'words':[{'word':'Тестовая','start':1.,'end':2.},
                                {'word':'речь.','start':2.,'end':3.}]}]}
    aligned=s.comparison.global_alignment(local,runner.evidence)
    assigned=deepcopy(aligned)
    assigned['segments'][0]['speaker']='A'
    for word in assigned['segments'][0]['words']:word['speaker']='A'
    outputs={mode:{'num_speakers':4 if mode.startswith('four') else None,
        'timeline':[{'start':0.,'end':36.,'speaker':'A'}], 'aligned':deepcopy(assigned),
        'export':s.comparison.export_result(assigned)} for mode in s.comparison.MODES}
    return {'profile':s.CAMPAIGN,'status':'completed','sample_id':'sound2','asr_rerun':False,
        'audio_sha256':runner.campaign['audio_sha256'],'duration_seconds':36,
        'asr_evidence':runner.evidence,'asr_evidence_sha256':s.comparison.evidence_sha(runner.evidence),
        'source_manifest_sha256':s.digest(s.SOURCES),'runtime':runtime,
        'alignment_local':local,'alignment_global':aligned,'outputs':outputs}


class SpeakerOffline(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.settings=fixture(self.root)
        self.runner=s.SpeakerRunner(self.settings)

    def test_default_is_offline_without_budget_reservation(self):
        with patch.object(sys,'argv',['speaker_test.py','--env-file','fixture']), \
             patch.object(s.TestSettings,'load',return_value=self.settings), \
             patch.object(s.asyncio,'run',side_effect=AssertionError('network')), \
             patch('sys.stdout',new_callable=io.StringIO) as stdout:
            self.assertEqual(s.main(),0)
            self.assertEqual(json.loads(stdout.getvalue())['planning_total_usd'],'1.616')
        self.assertFalse(self.runner.path.parent.exists())

    def test_production_and_lowered_prior_allowance_rejected(self):
        for changes in ({'endpoint':'l20o867nueti1d'},{'bucket':'transcribe'}):
            with self.assertRaises(ValueError):s.SpeakerRunner(replace(self.settings,**changes))
        c=s.read_json(self.runner.campaign_path)
        for prior in ('0.116','1.501'):
            s.save_json(self.runner.campaign_path,{**c,'baseline_spend_usd':prior})
            with self.assertRaises(ValueError):s.SpeakerRunner(self.settings)

    def test_frozen_asr_or_history_changes_block_submission(self):
        source=Path(self.runner.campaign['source_result']);data=s.read_json(source)
        data['output']['raw_engine_output']['segments'][0]['text']='Изменённый текст'
        s.save_json(source,data)
        with self.assertRaises(ValueError):self.runner.check_available()

    def test_preflight_requires_new_flag_and_cumulative_bill(self):
        path=self.runner.settings.runtime/'preflight.json';original=s.read_json(path)
        for change in ({'speaker_comparison_enabled':False},{'context_asr_enabled':True},
                       {'test_billed_usd':'1.501'},{'max_workers':2}):
            s.save_json(path,{**original,**change})
            with self.assertRaises((ValueError,RuntimeError)):self.runner.preflight()

    def test_frozen_source_payload_and_bounds(self):
        url='https://'+self.settings.storage_host+'/poslovnica-test/quality/sound2.wav?signature=x'
        payload=self.runner.payload(url)
        self.assertEqual(self.runner.request_body(payload)['policy'],{'executionTimeout':300000,'ttl':420000})
        bad=deepcopy(payload);bad['asr_evidence']['segments'][0]['text']='подсказка'
        with self.assertRaises(ValueError):self.runner.request_body(bad)

    def test_result_rejects_modified_text_times_provenance_and_exports(self):
        self.runner.path.parent.mkdir()
        result=output(self.runner)
        self.runner.validate_output(result)
        self.assertTrue(s.read_json(self.runner.path)['speaker_comparison_validated'])
        for change in ('time','export','model','offset'):
            bad=deepcopy(result)
            if change=='time':bad['outputs']['four_regular']['aligned']['segments'][0]['words'][0]['start']=99
            if change=='export':bad['outputs']['auto_exclusive']['export']['text']='fake'
            if change=='model':bad['runtime']['diarization']['revision']='0'*40
            if change=='offset':bad['alignment_global']['segments'][0]['start']=0
            with self.assertRaises(ValueError):self.runner.validate_output(bad)


class SpeakerCloudFakes(unittest.IsolatedAsyncioTestCase):
    async def test_single_submission_and_parent_lock_with_all_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();settings=fixture(root)
            client=AsyncMock();client.__aenter__.return_value=client
            runner=s.SpeakerRunner(settings,poll_s=0,client_factory=Mock(return_value=client))
            async def post(url,**kwargs):
                self.assertTrue((root/'runner.lock').exists())
                self.assertEqual(s.read_json(runner.path)['status'],'reserved')
                return response({'id':'speaker-fixture'})
            async def get(url,**kwargs):
                if url.endswith('/health'):return response({'jobs':{'inProgress':0,'inQueue':0},'workers':{'idle':0,'running':0}})
                return response({'status':'COMPLETED','output':output(runner)})
            client.post.side_effect=post;client.get.side_effect=get
            url='https://'+settings.storage_host+'/poslovnica-test/quality/sound2.wav?signature=x'
            with patch.object(s,'upload_sound2',return_value=url):
                await runner.execute()
                with self.assertRaises(RuntimeError):await runner.execute()
            self.assertEqual(client.post.await_count,1)
            self.assertEqual(s.read_json(runner.path)['planning_total_usd'],'1.616')
            self.assertFalse((root/'runner.lock').exists())


if __name__=='__main__':unittest.main()
