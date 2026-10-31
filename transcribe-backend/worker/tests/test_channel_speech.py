from copy import deepcopy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import channel_speech as r
import channel_pipeline as worker
import test_channel_asr as fixture


class SpeechAlignmentTests(unittest.TestCase):
    def decode(self, empty=False, ambiguous=False, escape=False):
        audio = range(20*r.RATE)
        self.calls=[]
        def transcribe(audio, rows):
            return {'language':'ru','segments':[{'start':w['start_frame']/r.RATE,
                'end':w['end_frame']/r.RATE,'text':text} for w,text in zip(rows,('Начало.','Тихо.','Финал.'))]}
        def recover(chunk):
            end = 3. if ambiguous else 2.3
            return {'language':'ru','segments':[] if empty else [{'start':0.,'end':end,'text':'Ответ.',
                'avg_logprob':-.2,'no_speech_prob':.1,
                'words':[{'word':'Ответ.','start':0.,'end':end,'probability':.9}]}]}
        transcribe.recover=recover
        def align(segments, chunk):
            self.calls.append((deepcopy(segments),chunk))
            s=deepcopy(segments[0]); text=s['text']
            s['words']=[] if text=='Финал.' else [{'word':text,'start':.05,
                'end':2.2 if text=='Тихо.' else 1. if escape and text=='Ответ.' else .3}]
            return {'segments':[s],'word_segments':s['words']}
        with patch.dict(os.environ,{'CHANNEL_ASR_TEST':'1'}):
            return r.decode_channels(audio,audio,detect=lambda a:[{'start':1.,'end':2.},
                {'start':6.,'end':8.5},{'start':18.4,'end':18.7}],transcribe=transcribe,align=align,
                confirmation={'kind':'owner_preview','start_seconds':0,'end_seconds':20})

    def test_context_pause_never_reaches_ctc_and_offset_is_applied_once(self):
        result=self.decode()
        final=[(segs,chunk) for segs,chunk in self.calls if segs[0]['text']=='Ответ.'][-1]
        expected=result['original']['run_evidence'][1]['runs'][2]['window']
        self.assertEqual((final[1].start,final[1].stop),(expected['start_frame'],expected['end_frame']))
        self.assertLess(len(final[1])/r.RATE,.55)
        word=result['evidence'][1]['aligned']['segments'][-1]['words'][0]
        self.assertAlmostEqual(word['start'],expected['start_frame']/r.RATE+.05)
        self.assertAlmostEqual(word['end'],expected['start_frame']/r.RATE+.3)
        self.assertTrue(result['original']['issues'])
        self.assertTrue(any(i['reason']=='native_timing_unreliable' for i in result['issues']))
        self.assertFalse(result['release_ready'])

    def test_uncovered_speech_keeps_original_and_explicit_marker(self):
        result=self.decode(empty=True)
        self.assertEqual(result['evidence'],result['original']['evidence'])
        self.assertTrue(all(x['status']=='unresolved_speech' and x['hypothesis'] is None for x in result['review_spans']))
        self.assertIsNone(result['candidate_segments'])

    def test_ambiguous_and_context_only_hypotheses_do_not_guess_ownership(self):
        result=self.decode(); original=result['original']['run_evidence'][0]
        region={'start_frame':0,'end_frame':20*r.RATE,'replaced_runs':[0,1]}
        native={'language':'ru','segments':[{'text':'Одна фраза.','start':1.5,'end':6.5}]}
        assigned,unassigned=r.assign(region,native,original)
        self.assertEqual(assigned,{0:[],1:[]})
        self.assertEqual(unassigned[0]['reason'],'ambiguous_speech_assignment')
        native['segments'][0].update(start=10.,end=11.)
        self.assertEqual(r.assign(region,native,original)[1][0]['reason'],'context_only_hypothesis')

    def test_local_escape_still_blocks_export(self):
        result=self.decode(escape=True)
        self.assertIsNone(result['candidate_segments'])
        self.assertTrue(any(i['reason']=='unreliable_word_time' for i in result['issues']))

    def test_replay_rejects_context_crop_changed_text_or_dropped_run(self):
        result=self.decode(); plan={'frames_per_channel':20*r.RATE,'confirmation':result['confirmation']}
        self.assertEqual(r.rebuild(result,plan),result)
        for mutation in (lambda x:x['recovery_evidence'][0]['alignments'][0]['window'].update(start_frame=0),
                         lambda x:x['recovery_evidence'][0]['alignments'][0]['raw_local']['segments'][0].update(text='Подмена'),
                         lambda x:x['recovery_evidence'][0]['alignments'].clear(),
                         lambda x:x['recovery_evidence'][0]['alignments'][0].update(source_segments=[])):
            bad=deepcopy(result); mutation(bad)
            with self.assertRaises(ValueError):r.rebuild(bad,plan)

    def test_guard_precedes_audio_callbacks(self):
        cb=Mock()
        with patch.dict(os.environ,{},clear=True),self.assertRaises(ValueError):
            r.decode_channels([0],[0],detect=cb,transcribe=cb,align=cb,confirmation={})
        cb.assert_not_called()


class ManifestTests(unittest.TestCase):
    setUp=fixture.ChannelInputTests.setUp

    def test_new_profile_preserves_full_audio_contract_and_rejects_hints(self):
        import json
        self.data['profile']=r.PROFILE
        self.manifest.write_text(json.dumps(self.data),encoding='utf-8')
        plan=r.prepare(self.manifest)
        self.assertEqual(plan['speech_alignment_policy']['alignment_audio'],'complete_original_vad_run')
        self.assertIs(worker.adapter_for(r.PROFILE),r)
        self.data['reference']='Ответ.'
        self.manifest.write_text(json.dumps(self.data),encoding='utf-8')
        with self.assertRaises(ValueError):r.prepare(self.manifest)


if __name__=='__main__':unittest.main()
