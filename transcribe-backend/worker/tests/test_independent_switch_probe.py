from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import independent_switch_probe as p

ENV={'FULL_PIPELINE_IMAGE':'1','FULL_PIPELINE_TEST':'1','INDEPENDENT_SWITCH_TEST':'1',
     'QUALITY_R2_HOST':'a'*32+'.r2.cloudflarestorage.com'}


def lock():
    return {'schema':1,'profile':p.PROFILE,
            'algorithm':{'version':p.repair.VERSION,'policy':deepcopy(p.repair.POLICY),'code_sha256':dict(p.FROZEN_CODES)},
            'model_sources_sha256':p.base.digest(p.base.SOURCES),
            'asr_sources_sha256':p.base.digest(Path(p.__file__).with_name('asr_model_sources.json')),
            'decoder':deepcopy(p.base.DECODER),
            'samples':[{'id':name,'sha256':'a'*64,'frames':96000} for name in p.IDS]}


@dataclass
class Word:
    word: str
    start: float
    end: float


@dataclass
class Segment:
    text: str
    words: list


def collect(row, tail_seconds=0.):
    calls={'diarize':[],'decode':[],'align':[]}
    duration=row['duration_seconds']
    class Model:
        def transcribe(self,audio,**kwargs):
            calls['decode'].append((len(audio),deepcopy(kwargs)))
            values=[Segment('Вопрос?',[Word('Вопрос?',.3,.7)]),
                    Segment('Да!',[Word('Да!',1.04,1.16)]),
                    Segment('Дальше.',[Word('Дальше.',1.5,2.)])]
            return iter(values),SimpleNamespace(language='ru')
    def diarize(audio,settings):
        calls['diarize'].append(deepcopy(settings))
        timeline=[{'start':0.,'end':1.,'speaker':'A'},
                  {'start':1.,'end':1.2,'speaker':'B'},
                  {'start':1.2,'end':duration+tail_seconds,'speaker':'A'}]
        return {'settings':dict(settings,step_seconds=10*settings['step_ratio']),
                'parameters':deepcopy(p.base.PARAMETERS),
                'filter_diagnostics':[{'min_active_ratio':.1,'selected_embeddings':2,'chunk_speaker_pairs':2}],
                'regular':deepcopy(timeline),'exclusive':deepcopy(timeline)}
    def align(bounds,audio):
        calls['align'].append((deepcopy(bounds),len(audio)))
        segments=[]
        for bound in bounds:
            tokens=bound['text'].split()
            step=(bound['end']-bound['start'])/len(tokens)
            words=[{'word':w,'start':bound['start']+i*step+.001,
                    'end':bound['start']+(i+1)*step-.001,'score':.8} for i,w in enumerate(tokens)]
            segments.append({**bound,'words':words})
        return {'segments':segments}
    def decode(audio,windows,alignment):
        return p.base.decode(audio,windows,SimpleNamespace(model=Model()),alignment)
    result=p.collect_sample(row,[0]*round(duration*p.base.RATE),diarize=diarize,decode=decode,align=align)
    return result,calls


def result_for(locked):
    full=p.base.read_json(p.base.SOURCES)
    source=p.base.read_json(Path(p.__file__).with_name('asr_model_sources.json'))['large-v3']
    state={'profile':p.PROFILE,'status':'completed','modes':deepcopy(p.pause.MODES),
           'decoder':deepcopy(p.base.DECODER),'diarization_configs':deepcopy(p.pause.DIARIZATION),
           'alignment_configs':list(p.pause.ALIGNMENTS),'algorithm':deepcopy(locked['algorithm']),
           'source_manifest_sha256':locked['model_sources_sha256'],'input_samples':deepcopy(locked['samples']),
           'asr_sources_sha256':locked['asr_sources_sha256'],
           'runtime':{'asr':{'revision':source['revision'],'packages':p.base.config_for('wx_auto')['packages']},
                      **{k:{name:full[k][name] for name in ('repository','revision')} for k in ('alignment','diarization')}},
           'samples':[]}
    for row in locked['samples']:
        entry,_=collect(dict(row,duration_seconds=row['frames']/p.base.RATE))
        state['samples'].append(entry)
    return state


class IndependentProbeTests(unittest.TestCase):
    def test_observed_padded_tails_are_clipped_without_losing_original_model_times(self):
        # Exact CPU observations from the approved 5.5 s / 5 s clips, not hints.
        for duration,end in ((5.5,5.599718750000001),(5.,5.076593750000001)):
            with self.subTest(duration=duration):
                rows=[{'start':1.97159375,'end':end,'speaker':'SPEAKER_00'}]
                before={'regular':deepcopy(rows),'exclusive':deepcopy(rows)}
                result=p.bound_diarization(before,duration)
                self.assertEqual(result['regular'],[dict(rows[0],end=duration)])
                self.assertEqual(result['raw_timelines'],before)
                self.assertEqual(before['regular'],rows)
                self.assertEqual(result['audio_bounds']['changes']['regular'][0]['before']['end'],end)
                p.validate_diarization_bounds(result,duration)

    def test_padded_tail_no_longer_aborts_sample_and_real_short_answer_is_preserved(self):
        row={'id':'call-check-01','sha256':'a'*64,'duration_seconds':5.5}
        result,calls=collect(row,tail_seconds=.09971875)
        baseline,candidate=result['baseline'],result['candidate']
        self.assertEqual(len(calls['decode']),1)
        self.assertEqual(result['repair_audit']['applied_pairs'],0)
        for track in p.pause.DIARIZATION:
            evidence=baseline['diarization'][track]
            self.assertEqual(evidence['regular'][-1]['end'],5.5)
            self.assertGreater(evidence['raw_timelines']['regular'][-1]['end'],5.5)
            self.assertEqual(evidence['regular'][1],{'start':1.,'end':1.2,'speaker':'B'})
            self.assertEqual(candidate['diarization'][track],evidence)
        p.pause.validate_sample(baseline)

    def test_crop_drops_only_turns_wholly_outside_audio_and_retains_overlapping_speech(self):
        rows=[{'start':0.,'end':1.,'speaker':'A'},
              {'start':.8,'end':1.2,'speaker':'B'},
              {'start':1.2,'end':2.1,'speaker':'A'},
              {'start':2.1,'end':2.3,'speaker':'B'}]
        exclusive=[dict(rows[0],end=.8)]+deepcopy(rows[1:])
        evidence={'regular':rows,'exclusive':exclusive}
        result=p.bound_diarization(evidence,2.)
        self.assertEqual(result['regular'],rows[:2]+[dict(rows[2],end=2.)])
        self.assertEqual(result['raw_timelines'],evidence)
        self.assertIsNone(result['audio_bounds']['changes']['regular'][-1]['after'])
        p.validate_diarization_bounds(result,2.)

    def test_crop_rejects_bad_model_data_instead_of_hiding_it_in_padding(self):
        good={'start':0.,'end':1.,'speaker':'A'}
        for bad in (dict(good,start=-.1),dict(good,end=float('nan')),
                    dict(good,end=float('inf')),dict(good,start=2.),
                    dict(good,speaker=''),dict(good,end=True)):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                p.bound_diarization({'regular':[bad],'exclusive':[bad]},1.)
        for rows in ([],[{'start':2.,'end':3.,'speaker':'A'}]):
            with self.subTest(rows=rows),self.assertRaises(ValueError):
                p.bound_diarization({'regular':rows,'exclusive':rows},1.)
        # Invalid exclusive overlap outside the file is still an error.
        overlap=[good,{'start':2.,'end':4.,'speaker':'A'},{'start':3.,'end':5.,'speaker':'B'}]
        with self.assertRaises(ValueError):
            p.bound_diarization({'regular':overlap,'exclusive':overlap},1.)

    def test_boundary_audit_rejects_changed_turn_label_or_original_evidence(self):
        rows=[{'start':0.,'end':5.6,'speaker':'A'}]
        evidence=p.bound_diarization({'regular':rows,'exclusive':rows},5.5)
        for target in ('bounded','raw','audit','missing'):
            changed=deepcopy(evidence)
            if target=='bounded': changed['regular'][0]['speaker']='B'
            elif target=='raw': changed['raw_timelines']['regular'][0]['end']=5.7
            elif target=='audit': changed['audio_bounds']['changes']['regular']=[]
            else: del changed['raw_timelines']
            with self.subTest(target=target),self.assertRaises(ValueError):
                p.validate_diarization_bounds(changed,5.5)
        with self.assertRaises(ValueError): p.bound_diarization(evidence,5.5)

    def test_handler_preserves_failed_child_stage_without_validating_success_or_leaking_data(self):
        child={'status':'failed','phase':'call-check-02','error_type':'RuntimeError',
               'error':'hf_private secret https://private/?signature=token',
               'samples':[{'text':'private conversation'}],
               'failure_frames':[{'module':'timed_turns.py','line':96,'locals':'private'},
                                 {'module':'private-account-name.py','line':123}]}
        def run_child(args,**kwargs):
            target=Path(args[args.index('--output')+1])
            p.base.write_new(target,child)
            return SimpleNamespace(returncode=0)
        with tempfile.TemporaryDirectory() as directory:
            locked=Path(directory)/'lock.json';p.base.write_new(locked,lock())
            with patch.dict(p.os.environ,{**ENV,'INDEPENDENT_SWITCH_LOCK':str(locked)},clear=True),\
                 patch.object(p,'validate_input',return_value=[]),patch.object(p,'validate_local',return_value=[]),\
                 patch.object(p.subprocess,'run',side_effect=run_child),patch.object(p,'validate_result') as validate:
                response=p.handler({'input':{}})
        self.assertEqual(response['status'],'failed')
        self.assertEqual(response['child_phase'],'call-check-02')
        self.assertEqual(response['error_type'],'RuntimeError')
        self.assertEqual(response['completed_samples'],1)
        self.assertEqual(response['failure_frames'],[{'module':'timed_turns.py','line':96}])
        self.assertNotIn('private',str(response));validate.assert_not_called()

    def test_execute_records_asset_failure_phase_without_exception_message(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);lp=root/'lock.json';mp=root/'input.json';op=root/'output.json'
            p.base.write_new(lp,lock());p.base.write_new(mp,{})
            with patch.dict(p.os.environ,ENV,clear=True),patch.object(p,'validate_local',return_value=[]),\
                 patch.object(p.base,'verify_assets',side_effect=ValueError('private-token-and-url')):
                result=p.execute(mp,lp,op)
            self.assertEqual(result['phase'],'verify_asr_assets')
            self.assertEqual(result['status'],'failed')
            self.assertEqual(result['error_type'],'ValueError')
            self.assertEqual(p.base.read_json(op),result)
            self.assertNotIn('private-token',str(result))
            self.assertTrue(any(f['module']=='independent_switch_probe.py' for f in result['failure_frames']))

    def test_startup_registers_dedicated_handler_and_prints_identity(self):
        registered=[]
        sdk=SimpleNamespace(serverless=SimpleNamespace(start=lambda options:registered.append(options)))
        locked=Path(p.__file__).with_name('independent-probe-lock.json')
        with patch.dict(p.os.environ,ENV,clear=True),patch.dict(sys.modules,{'runpod':sdk}),\
             patch.object(sys,'argv',['probe','--serve','--lock',str(locked)]),\
             patch('builtins.print') as announce:
            p.main()
        self.assertEqual(registered,[{'handler':p.handler}])
        message=p.json.loads(announce.call_args.args[0])
        self.assertEqual(message['event'],'INDEPENDENT_PROBE_HANDLER_READY')
        self.assertEqual(message['profile'],p.PROFILE)
        self.assertEqual(message['lock_sha256'],p.base.digest(locked))

    def test_collection_calls_two_settings_and_three_alignments_without_hints(self):
        entry,calls=collect({'id':p.IDS[0],'sha256':'a'*64,'duration_seconds':6.})
        self.assertEqual(calls['diarize'],list(p.pause.DIARIZATION.values()))
        self.assertEqual(len(calls['decode']),1)
        self.assertEqual(calls['decode'][0][1],p.base.DECODER)
        self.assertEqual(len(calls['align']),3)  # native control, extend_tail, joint_window
        baseline=entry['baseline']
        self.assertEqual(set(baseline['outputs']),set(p.pause.MODES))
        self.assertEqual(entry['repair_audit']['applied_pairs'],0)  # preserve the short "Да!"
        for mode in p.pause.MODES:
            self.assertEqual(baseline['outputs'][mode],entry['candidate']['outputs'][mode])
        p.pause.validate_sample(baseline)
        self.assertEqual(baseline['alignment_windows']['control'][0]['aligned_local'],
                         baseline['windows'][0]['aligned_local'])

    def test_disabled_or_conflicting_mode_blocks_execution_before_assets_or_network(self):
        for env in ({},{**ENV,'INDEPENDENT_SWITCH_TEST':'0'},{**ENV,'CHANNEL_ASR_TEST':'1'},
                    {**ENV,'FULL_PIPELINE_IMAGE':'0'}):
            with self.subTest(env=env),patch.dict(p.os.environ,env,clear=True),patch.object(p.base,'verify_assets') as assets:
                with self.assertRaises(ValueError): p.execute('absent','absent','absent')
                assets.assert_not_called()
                with patch.object(p.base,'fetch_sample') as fetch:
                    response=p.handler({'input':{}})
                    self.assertEqual(response['status'],'failed')
                    fetch.assert_not_called()

    def test_request_accepts_only_locked_test_bucket_audio_and_no_human_hints(self):
        locked=lock()
        inp={'profile':p.PROFILE,'samples':[{'id':r['id'],'sha256':r['sha256'],
              'media_url':'https://'+ENV['QUALITY_R2_HOST']+'/poslovnica-test/quality/'+r['id']+'.wav?signature=private'}
              for r in locked['samples']]}
        with patch.dict(p.os.environ,ENV,clear=True):
            self.assertEqual(p.validate_input(inp,locked),inp['samples'])
            for key in ('text','speaker_count','num_speakers','start','reference'):
                bad=deepcopy(inp); bad['samples'][0][key]='hint'
                with self.assertRaises(ValueError): p.validate_input(bad,locked)
            for edit in ('production','other_hash','duplicate','top_hint'):
                bad=deepcopy(inp)
                if edit=='production': bad['samples'][0]['media_url']=bad['samples'][0]['media_url'].replace('poslovnica-test','transcribe')
                elif edit=='other_hash': bad['samples'][0]['sha256']='b'*64
                elif edit=='duplicate': bad['samples'][1]=bad['samples'][0]
                else: bad['reference']='hint'
                with self.assertRaises(ValueError): p.validate_input(bad,locked)

    def test_lock_rejects_changed_policy_code_models_or_extra_clips(self):
        p.check_lock(lock())
        for edit in ('policy','code','model','asr','decoder','clip','large'):
            changed=lock()
            if edit=='policy': changed['algorithm']['policy']['max_island_seconds']=.6
            elif edit=='code': changed['algorithm']['code_sha256']['pause_context.py']='b'*64
            elif edit=='model': changed['model_sources_sha256']='b'*64
            elif edit=='asr': changed['asr_sources_sha256']='b'*64
            elif edit=='decoder': changed['decoder']['initial_prompt']='hint'
            elif edit=='clip': changed['samples'].append(changed['samples'][0])
            else: changed['samples'][0]['frames']=31*p.base.RATE
            with self.assertRaises(ValueError): p.check_lock(changed)
        with patch.object(p.base,'digest',return_value='different'):
            with self.assertRaises(ValueError): p.check_lock(lock())

    def test_local_preflight_validates_actual_pcm_and_does_not_mutate_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'audio.wav'
            with wave.open(str(path),'wb') as wav:
                wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(p.base.RATE);wav.writeframes(b'\0'*192000)
            locked=lock()
            for row in locked['samples']: row['sha256']=p.base.digest(path)
            manifest={'profile':p.PROFILE,'samples':[{'id':r['id'],'sha256':r['sha256'],'path':str(path)} for r in locked['samples']]}
            before=deepcopy(manifest)
            self.assertEqual(len(p.validate_local(manifest,locked)),4)
            self.assertEqual(manifest,before)
            manifest['samples'][0]['speaker_count']=2
            with self.assertRaises(ValueError): p.validate_local(manifest,locked)
            manifest=deepcopy(before)
            with path.open('ab') as f: f.write(b'changed')
            with self.assertRaises(ValueError): p.validate_local(manifest,locked)

    def test_result_replays_baseline_and_candidate_and_rejects_substitution(self):
        locked=lock(); result=result_for(locked)
        p.validate_result(result,locked)
        for edit in ('missing_diarization','candidate','model','audio','partial'):
            changed=deepcopy(result)
            if edit=='missing_diarization': del changed['samples'][0]['baseline']['diarization']['dense']
            elif edit=='candidate': changed['samples'][0]['candidate']['asr']['segments'][0]['text']='replacement'
            elif edit=='model': changed['runtime']['asr']['revision']='other'
            elif edit=='audio': changed['samples'][0]['baseline']['sha256']='b'*64
            else: changed['samples'].pop()
            with self.assertRaises(ValueError): p.validate_result(changed,locked)

    def test_handler_does_not_echo_transport_secrets(self):
        with patch.dict(p.os.environ,ENV,clear=True),patch.object(p.base,'read_json',side_effect=ValueError('private-secret-url')):
            response=p.handler({'input':{}})
        self.assertNotIn('private-secret-url',str(response))
        self.assertEqual(response['phase'],'validate')


if __name__=='__main__':
    unittest.main()
