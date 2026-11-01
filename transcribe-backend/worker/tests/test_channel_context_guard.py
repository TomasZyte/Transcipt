from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import channel_context_guard as g
import channel_pipeline as pipeline
import test_channel_context as previous

FLAGS = {**previous.FLAGS, 'CHANNEL_CONTEXT_GUARD_TEST': '1'}


class GuardTests(unittest.TestCase):
    def primary(self, kind='missing'):
        first = [(' Ответ.', 1., 1.3)] if kind == 'missing' else [(' Не', 1., 1.15), (' вернули.', 1.15, 1.4)]
        words = first if kind == 'missing' else first+[(' Позже.',4.7,4.9)]
        native = previous.native(words)
        if kind != 'missing':
            native['segments'][0]['words'][0]['probability'] = .418
        return previous.ContextTests().decode(decoded=native, frames=5*g.RATE,
            speech=[{'start':1.,'end':1.5},{'start':4.7,'end':5.}])

    def verify(self, primary, text='Коротко.', *, no_speech=.1, empty=False, bad_time=False):
        prepared = g.plan(primary)
        self.calls, self.alignment_calls = [], []
        def transcribe(audio):
            selected = prepared['checks'][len(self.calls)]
            self.calls.append(audio)
            channel = next(c for c in primary['evidence'] if c['channel'] == selected['channel'])
            win = g.runs.windows(channel['speech'],len(range(5*g.RATE)))[selected['run']]
            a = (win['start_frame']-audio.start)/g.RATE+.03
            data = previous.native([(' '+w,a+.1*i,a+.1*i+.07) for i,w in enumerate(text.split())],no_speech=no_speech)
            if empty:data['segments']=[]
            if bad_time:data['segments'][0]['words'][0]['end']=a-.1
            return data
        def align(segments,audio):
            self.alignment_calls.append((deepcopy(segments),audio))
            return {'segments':[{'text':segments[0]['text'],'words':[
                {'word':w,'start':.03+.1*i,'end':.10+.1*i,'score':.8}
                for i,w in enumerate(segments[0]['text'].split())]}]}
        with patch.dict(os.environ,FLAGS,clear=True):
            return g.verify_channels(range(5*g.RATE),range(5*g.RATE),primary,
                                     transcribe=transcribe,align=align)

    def test_missing_final_speech_is_selected_and_retained_as_unverified_candidate(self):
        primary=self.primary()
        planned=g.plan(primary)
        self.assertEqual([(r['channel'],r['run']) for r in planned['checks']], [('left',1),('right',1)])
        for r in planned['checks']:
            self.assertEqual(r['window']['end_frame'],5*g.RATE)
            self.assertEqual(r['window']['start_frame'],round(2.58*g.RATE))
        result=self.verify(primary)
        recovered=[r for r in result['review_spans'] if r['text_source']=='local_unverified']
        self.assertEqual(len(recovered),2)
        self.assertTrue(all(r['text']==' Коротко.' and r['primary_text'] is None for r in recovered))
        self.assertTrue(all('unverified_recovery' in r['reasons'] for r in recovered))
        self.assertIsNone(result['candidate_segments'])
        self.assertFalse(result['release_ready'])

    def test_weak_negation_selected_despite_good_ctc_and_no_old_flag(self):
        primary=self.primary('negation')
        first=primary['review_spans'][0]
        self.assertEqual(first['reasons'],[])
        selected=g.plan(primary)['checks']
        self.assertEqual(len(selected),2)
        self.assertTrue(all(p['reasons']==['uncertain_critical_word'] for p in selected))
        self.assertTrue(all(p['critical_word_refs']==[[0,0]] for p in selected))

    def test_changed_negation_keeps_both_hypotheses_without_claiming_winner(self):
        primary=self.primary('negation')
        result=self.verify(primary,'Мне вернули.')
        conflicts=[r for r in result['review_spans'] if r['status']=='recognition_conflict']
        self.assertEqual(len(conflicts),2)
        for r in conflicts:
            self.assertIsNone(r['text']); self.assertEqual(r['aligned_segments'],[])
            self.assertEqual(r['primary_text'],' Не вернули.')
            self.assertEqual(r['local_alternative']['text'],' Мне вернули.')
            self.assertIn('recognition_disagreement',r['reasons'])
        self.assertEqual(result['primary'],primary)

    def test_agreement_is_not_human_verification(self):
        result=self.verify(self.primary('negation'),'НЕ вернули!')
        checked=[r for r in result['review_spans'] if r['local_check_performed']]
        self.assertTrue(all(r['status']=='agreement_requires_review' for r in checked))
        self.assertTrue(all(not r['human_verified'] for r in checked))
        self.assertFalse(result['quality_validated'])

    def test_empty_local_pass_never_approves_primary_critical_word(self):
        result=self.verify(self.primary('negation'),empty=True)
        self.assertTrue(all(r['status']=='recognition_conflict' for r in result['review_spans'] if r['local_check_performed']))
        self.assertEqual(self.alignment_calls,[])

    def test_empty_local_pass_preserves_unresolved_speech(self):
        result=self.verify(self.primary(),empty=True)
        self.assertTrue(all(r['text'] is None for r in result['review_spans'] if r['local_check_performed']))
        self.assertEqual(self.alignment_calls,[])

    def test_high_no_speech_does_not_discard_short_answer(self):
        result=self.verify(self.primary(),no_speech=.845)
        recovered=[r for r in result['review_spans'] if r['text_source']=='local_unverified']
        self.assertEqual(len(recovered),2)
        self.assertTrue(all('high_no_speech_probability' in r['reasons'] for r in recovered))

    def test_invalid_native_timing_preserved_not_stretched_into_run(self):
        result=self.verify(self.primary(),bad_time=True)
        self.assertEqual(self.alignment_calls,[])
        self.assertTrue(all(r['text'] is None for r in result['review_spans'] if r['local_check_performed']))
        self.assertEqual(result['checks'][0]['unassigned'][0]['reason'],'invalid_native_word_time')

    def test_ctc_receives_only_complete_original_run(self):
        self.verify(self.primary())
        for segments,audio in self.alignment_calls:
            self.assertEqual(audio.start,round(4.58*g.RATE))
            self.assertEqual(audio.stop,5*g.RATE)
            self.assertEqual(segments[0]['start'],0.)
            self.assertAlmostEqual(segments[0]['end'],.42)

    def test_neighbouring_speech_not_included_or_partially_cropped(self):
        primary=self.primary('negation')
        for c in g.plan(primary)['checks']:
            self.assertLessEqual(c['window']['end_frame'],round(4.58*g.RATE))
            self.assertLessEqual(c['window']['end_frame']-c['window']['start_frame'],30*g.RATE)

    def test_numbers_and_negation_have_generic_rules_not_reference_phrases(self):
        for word in ('НЕ,','Нет.','нельзя','без','1200','12,5%'):
            self.assertTrue(g.critical(word),word)
        for word in ('мне','никогдашний','настоятельной','спасибо','оплата'):
            self.assertFalse(g.critical(word),word)
        primary=self.primary('negation')
        # No threshold or target has been keyed to a private call or a word ID.
        text=json.dumps(g.plan(primary),ensure_ascii=False)
        self.assertNotIn('вернули',text)

    def test_all_limits_checked_before_additional_recognition(self):
        primary=self.primary()
        callback=Mock()
        for name in ('max_checks','max_check_audio_seconds','max_total_asr_audio_seconds','max_check_seconds'):
            with self.subTest(name=name),patch.dict(g.POLICY,{name:0}),patch.dict(os.environ,FLAGS,clear=True):
                with self.assertRaisesRegex(ValueError,'bounded compute'):
                    g.verify_channels(range(5*g.RATE),range(5*g.RATE),primary,transcribe=callback,align=callback)
        callback.assert_not_called()

    def test_no_selected_runs_no_additional_model_calls(self):
        primary=previous.ContextTests().decode()
        callback=Mock()
        with patch.dict(os.environ,FLAGS,clear=True):
            result=g.verify_channels(range(8*g.RATE),range(8*g.RATE),primary,transcribe=callback,align=callback)
        self.assertEqual(result['checks'],[])
        callback.assert_not_called()

    def test_route_requires_all_flags_and_rejects_other_profiles(self):
        self.assertIs(pipeline.adapter_for(g.PROFILE),g)
        callback=Mock()
        with patch.dict(os.environ,previous.FLAGS,clear=True),self.assertRaises(ValueError):
            g.verify_channels([],[],self.primary(),transcribe=callback,align=callback)
        callback.assert_not_called()
        self.assertEqual(g.DECODER,g.context.DECODER)
        self.assertIsNone(g.DECODER['initial_prompt'])
        with patch.dict(os.environ,FLAGS,clear=True):
            pipeline.require_profile(g.PROFILE)
            for profile in ('channel-context-v1','channel-speech-v2','channel-asr-v1'):
                with self.assertRaises(ValueError):pipeline.require_profile(profile)

    def test_complete_pass_recomputes_primary_without_saved_words(self):
        primary=self.primary()
        expected=self.verify(primary)
        audio=range(5*g.RATE)
        detect,transcribe,align=Mock(),Mock(),Mock()
        with patch.dict(os.environ,FLAGS,clear=True), \
             patch.object(g.context,'decode_channels',return_value=primary) as first, \
             patch.object(g,'verify_channels',return_value=expected) as second:
            result=g.decode_channels(audio,audio,detect=detect,transcribe=transcribe,
                                     align=align,confirmation=primary['confirmation'])
        self.assertIs(result,expected)
        first.assert_called_once_with(audio,audio,detect=detect,transcribe=transcribe,
                                      align=align,confirmation=primary['confirmation'])
        second.assert_called_once_with(audio,audio,primary,transcribe=transcribe,align=align)

    def test_replay_rejects_dropped_checks_fabricated_text_and_promoted_quality(self):
        result=self.verify(self.primary('negation'),'Мне вернули.')
        g.validate_result(result)
        for mutate in (lambda r:r['checks'].pop(),
                       lambda r:r['checks'][0].update(raw_local=None),
                       lambda r:r['review_spans'][0].update(text='Подменённое.'),
                       lambda r:r.update(quality_validated=True),
                       lambda r:r['checks'][0]['planned']['window'].update(start_frame=1)):
            bad=deepcopy(result);mutate(bad)
            with self.assertRaises((ValueError,TypeError,KeyError)):g.validate_result(bad)


class GuardIntegrationTests(unittest.TestCase):
    setUp = previous.input_fixture.ChannelInputTests.setUp

    def test_manifest_payload_and_gate_before_download(self):
        self.data['profile']=g.PROFILE
        self.manifest.write_text(json.dumps(self.data),encoding='utf-8')
        prepared=g.prepare(self.manifest)
        self.assertEqual(prepared['guard_policy'],g.POLICY)
        self.assertEqual(prepared['context_decoder'],g.context.DECODER)
        payload=previous.payload_fixture.ChannelWorkerTests().payload();payload['profile']=g.PROFILE
        pipeline.validate_input(payload,previous.payload_fixture.HOST)
        with patch.dict(os.environ,previous.FLAGS,clear=True),patch.object(pipeline,'fetch_channel') as fetch:
            self.assertEqual(previous.router.handler({'input':payload})['phase'],'validate')
            fetch.assert_not_called()
        payload['primary_text']='historical private text'
        with self.assertRaises(ValueError):pipeline.validate_input(payload,previous.payload_fixture.HOST)

    def test_worker_replay_binds_original_channel_frames_and_confirmation(self):
        fixture=GuardTests();result=fixture.verify(fixture.primary())
        source={'profile':g.PROFILE,'frames_per_channel':5*g.RATE,
                'confirmation':result['primary']['confirmation']}
        state={'profile':g.PROFILE,'status':'completed','input_plan':source,
               'source_manifest_sha256':pipeline.digest(pipeline.SOURCES),
               'human_quality_validated':False,'result':result}
        pipeline.validate_result(state,source)
        for update in ({'frames_per_channel':6*g.RATE},
                       {'confirmation':{**source['confirmation'],'end_seconds':4}}):
            wrong={**source,**update}
            with self.assertRaises(ValueError):pipeline.validate_result({**state,'input_plan':wrong},wrong)


if __name__=='__main__':unittest.main()
