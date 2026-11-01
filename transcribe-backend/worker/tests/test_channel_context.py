from copy import deepcopy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import channel_context as c
import channel_pipeline as pipeline
import test_channel_speech_v2 as previous
import test_channel_asr as input_fixture
import test_channel_pipeline as payload_fixture
import full_pipeline_handler as router

FLAGS = {**pipeline.FLAGS, 'CHANNEL_CONTEXT_TEST': '1'}


def native(words, *, no_speech=.1):
    return {'language': 'ru', 'segments': [{'start': words[0][1], 'end': words[-1][2],
        'text': ''.join(w[0] for w in words), 'avg_logprob': -.2, 'no_speech_prob': no_speech,
        'words': [{'word': text, 'start': a, 'end': b, 'probability': .8} for text,a,b in words]}]}


class ContextTests(unittest.TestCase):
    def decode(self, *, decoded=None, speech=None, frames=8*c.RATE):
        self.asr_calls, self.ctc_calls = [], []
        speech = speech if speech is not None else [{'start':1., 'end':1.5}, {'start':4.,'end':4.5}]
        audio = range(frames)
        def transcribe(chunk):
            self.asr_calls.append(chunk)
            return deepcopy(decoded if decoded is not None else native([(' Первый',1.,1.3),(' второй.',4.,4.3)]))
        def align(segments, chunk):
            self.ctc_calls.append((deepcopy(segments), chunk))
            s = deepcopy(segments[0])
            s['words'] = [{'word':w, 'start':.13+i*.1, 'end':.2+i*.1, 'score':.9}
                          for i,w in enumerate(s['text'].split())]
            return {'segments':[s]}
        with patch.dict(os.environ, FLAGS, clear=True):
            return c.decode_channels(audio, audio, detect=lambda a: deepcopy(speech), transcribe=transcribe,
                align=align, confirmation={'kind':'owner_preview','start_seconds':0,'end_seconds':frames/c.RATE})

    def test_asr_keeps_real_pause_and_ctc_gets_only_its_run(self):
        result = self.decode()
        self.assertEqual(len(self.asr_calls), 2)
        self.assertTrue(all(a.start == 0 and a.stop > 7*c.RATE for a in self.asr_calls))
        self.assertEqual(len(self.ctc_calls), 4)
        for (segments,a), start in zip(self.ctc_calls, [.88,3.88,.88,3.88]):
            self.assertEqual(a.start, round(start*c.RATE))
            self.assertLess(len(a), c.RATE)
            self.assertEqual(segments[0]['start'], 0.)
        self.assertEqual([r['text'] for r in result['review_spans']], [' Первый',' Первый',' второй.',' второй.'])
        word = result['review_spans'][-1]['aligned_segments'][0]['words'][0]
        self.assertAlmostEqual(word['start'], 4.01)
        self.assertFalse(result['release_ready'])
        self.assertIsNone(result['candidate_segments'])

    def test_context_blocks_own_each_run_once_and_never_cut_long_run(self):
        speech = [{'start':1.,'end':2.}, {'start':24.,'end':25.}, {'start':28.,'end':58.}]
        windows = c.runs.windows(speech, 60*c.RATE)
        blocks = c.blocks(speech, 60*c.RATE)
        self.assertEqual([i for b in blocks for i in b['owned_runs']], list(range(len(windows))))
        for b in blocks:
            self.assertLessEqual(b['end_frame']-b['start_frame'],30*c.RATE)
            for i in b['owned_runs']:
                self.assertLessEqual(b['start_frame'],windows[i]['start_frame'])
                self.assertGreaterEqual(b['end_frame'],windows[i]['end_frame'])

    def test_context_only_words_are_not_emitted_twice(self):
        speech = [{'start':1.,'end':2.}, {'start':23.,'end':24.}, {'start':26.,'end':27.}]
        windows = c.runs.windows(speech, 30*c.RATE)
        block = c.blocks(speech,30*c.RATE)[0]
        mapping, notes = c.assign(block, native([(' Один.',1.,1.2),(' Два.',23.,23.2),(' Контекст.',26.,26.2)]),windows)
        self.assertEqual(mapping,{0:[[0,0]],1:[[0,1]]})
        self.assertEqual(notes[0]['reason'],'context_only_word')
        self.assertEqual(notes[0]['intersected_runs'],[2])

    def test_word_crossing_two_runs_is_retained_unassigned(self):
        result = self.decode(decoded=native([(' Неразделимое.',1.,4.2)]))
        self.assertEqual(self.ctc_calls, [])
        self.assertTrue(all(r['text'] is None for r in result['review_spans']))
        self.assertEqual(result['unassigned'][0]['reason'],'ambiguous_word_assignment')
        self.assertTrue(all('ambiguous_word_assignment' in r['reasons'] for r in result['review_spans']))
        self.assertEqual(result['evidence'][0]['blocks'][0]['native']['segments'][0]['text'],' Неразделимое.')

    def test_high_no_speech_does_not_delete_real_short_answer(self):
        result = self.decode(decoded=native([(' Ответ.',1.,1.3)],no_speech=.85))
        nonempty = [r for r in result['review_spans'] if r['text']]
        self.assertEqual(len(nonempty),2)
        self.assertTrue(all('high_no_speech_probability' in r['reasons'] for r in nonempty))
        self.assertTrue(all(r['text']==' Ответ.' for r in nonempty))

    def test_missing_words_empty_and_invalid_times_cannot_be_forced_into_ctc(self):
        for kind in ('missing','empty','reversed','outside_segment','word_text_changed'):
            data = native([(' Текст.',1.,1.3)])
            if kind=='missing': data['segments'][0].pop('words')
            if kind=='empty': data['segments']=[]
            if kind=='reversed': data['segments'][0]['words'][0]['end']=.9
            if kind=='outside_segment': data['segments'][0]['words'][0]['end']=1.9
            if kind=='word_text_changed': data['segments'][0]['words'][0]['word']=' Иное.'
            with self.subTest(kind=kind):
                result=self.decode(decoded=data)
                self.assertEqual(self.ctc_calls,[])
                self.assertTrue(all(r['text'] is None for r in result['review_spans']))

    def test_legitimate_repeated_phrase_preserved_without_blacklist(self):
        result=self.decode(decoded=native([(' Ещё раз.',1.,1.3),(' Ещё раз.',4.,4.3)]))
        self.assertTrue(all(r['text']==' Ещё раз.' for r in result['review_spans']))

    def test_no_speech_means_no_asr_or_ctc_calls(self):
        result=self.decode(speech=[])
        self.assertEqual(self.asr_calls,[])
        self.assertEqual(self.ctc_calls,[])
        self.assertEqual(result['review_spans'],[])

    def test_compute_limit_rejected_before_asr(self):
        with patch.dict(c.POLICY, {'max_total_audio_seconds':1}), self.assertRaisesRegex(ValueError,'compute'):
            self.decode()
        self.assertEqual(self.asr_calls,[])

    def test_disabled_flags_reject_before_any_callback(self):
        callback=Mock()
        with patch.dict(os.environ,pipeline.FLAGS,clear=True), self.assertRaises(ValueError):
            c.decode_channels([0],[0],detect=callback,transcribe=callback,align=callback,confirmation={})
        callback.assert_not_called()

    def test_evidence_replay_rejects_ownership_and_text_tampering(self):
        result=self.decode()
        c.validate_result(result)
        for change in (lambda r:r['review_spans'][0].update(text='invented'),
                       lambda r:r['evidence'][0]['blocks'][0]['alignments'].pop(),
                       lambda r:r['evidence'][0]['blocks'][0]['window']['owned_runs'].reverse(),
                       lambda r:r.update(release_ready=True)):
            bad=deepcopy(result); change(bad)
            with self.assertRaises(ValueError):c.validate_result(bad)

    def test_test_route_requires_additional_flag_and_decoder_unchanged(self):
        self.assertIs(pipeline.adapter_for(c.PROFILE), c)
        with patch.dict(os.environ,pipeline.FLAGS,clear=True), self.assertRaises(ValueError):
            pipeline.require_profile(c.PROFILE)
        with patch.dict(os.environ,FLAGS,clear=True):
            pipeline.require_profile(c.PROFILE)
            with self.assertRaises(ValueError): pipeline.require_profile('channel-speech-v2')
        self.assertEqual(c.DECODER,c.repair.DECODER)
        self.assertIsNone(c.DECODER['initial_prompt'])
        self.assertFalse(c.DECODER['condition_on_previous_text'])

    def test_preparation_checks_identity_and_uses_no_old_words(self):
        fixture=previous.RecognitionRecoveryTests()
        result=fixture.decode()
        plan={'profile':'channel-speech-v2','frames_per_channel':9*c.RATE,
              'confirmation':result['confirmation'],'source_sha256':'a'*64,'channels':[]}
        state={'profile':'channel-speech-v2','status':'completed','input_plan':plan,
               'source_manifest_sha256':pipeline.digest(pipeline.SOURCES),
               'human_quality_validated':False,'result':result}
        prepared=c.prepare_from_saved(state,plan)
        self.assertEqual(prepared['paid_jobs'],0)
        self.assertFalse(prepared['cloud_runner_connected'])
        self.assertNotIn('Тихая',str(prepared))
        with self.assertRaises(ValueError): c.prepare_from_saved(state,{**plan,'source_sha256':'b'*64})


class ContextIntegrationTests(unittest.TestCase):
    setUp = input_fixture.ChannelInputTests.setUp

    def test_manifest_route_and_pre_download_gate(self):
        self.data['profile']=c.PROFILE
        self.manifest.write_text(json.dumps(self.data),encoding='utf-8')
        plan=c.prepare(self.manifest)
        self.assertEqual(plan['context_decoder'],c.DECODER)
        payload=payload_fixture.ChannelWorkerTests().payload();payload['profile']=c.PROFILE
        pipeline.validate_input(payload,payload_fixture.HOST)
        with patch.dict(os.environ,pipeline.FLAGS,clear=True),patch.object(pipeline,'fetch_channel') as fetch:
            self.assertEqual(router.handler({'input':payload})['phase'],'validate')
            fetch.assert_not_called()
        payload['reference']='private'
        with self.assertRaises(ValueError):pipeline.validate_input(payload,payload_fixture.HOST)

    def test_worker_envelope_rejects_different_frames_and_confirmation(self):
        result=ContextTests().decode()
        plan={'profile':c.PROFILE,'frames_per_channel':8*c.RATE,'confirmation':result['confirmation']}
        state={'profile':c.PROFILE,'status':'completed','input_plan':plan,
               'source_manifest_sha256':pipeline.digest(pipeline.SOURCES),
               'human_quality_validated':False,'result':result}
        pipeline.validate_result(state,plan)
        wrong={**plan,'confirmation':{**plan['confirmation'],'end_seconds':7}}
        bad={**state,'input_plan':wrong}
        with self.assertRaises(ValueError):pipeline.validate_result(bad,wrong)


if __name__=='__main__':unittest.main()
