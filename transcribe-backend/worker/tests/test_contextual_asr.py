from copy import deepcopy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import contextual_asr as c
from full_pipeline_review import word_support, text_changes, review_sample


def evidence(window, words, text=None):
    offset = window['read_start_frame']/c.RATE
    aligned_words = [{'word':text,'start':a-offset,'end':b-offset,'score':.8} for text,a,b in words]
    text = text if text is not None else ' '.join(w[0] for w in words)
    segments = [{'text':text,'start':aligned_words[0]['start'],'end':aligned_words[-1]['end'],
                 'words':aligned_words}] if words else []
    return {'window':deepcopy(window),'raw':{'segments':[{'text':s['text']} for s in segments]},
            'aligned':{'segments':segments}}


class ContextTests(unittest.TestCase):
    def test_plan_uses_exact_frames_and_bounded_padding(self):
        for frames in (1,12*c.RATE,12*c.RATE+1,36*c.RATE+7,60*c.RATE):
            plan = c.plan_windows(frames)
            self.assertEqual(plan[0]['core_start_frame'],0)
            self.assertEqual(plan[-1]['core_end_frame'],frames)
            self.assertEqual(sum(w['core_end_frame']-w['core_start_frame'] for w in plan),frames)
            for left,right in zip(plan,plan[1:]):
                self.assertEqual(left['core_end_frame'],right['core_start_frame'])
            for w in plan:
                self.assertTrue(0<=w['read_start_frame']<=w['core_start_frame']<w['core_end_frame']<=w['read_end_frame']<=frames)
                self.assertLessEqual(w['read_end_frame']-w['read_start_frame'],16*c.RATE)

    def test_rejects_wrong_duration_types(self):
        for frames in (0,-1,True,10.,60*c.RATE+1):
            with self.assertRaises(ValueError): c.plan_windows(frames)

    def test_short_tail_is_rebalanced_without_silence_padding_or_lost_frames(self):
        for frames in (12*c.RATE+1,36*c.RATE+1515,25*c.RATE):
            plan=c.plan_windows(frames)
            lengths=[w['core_end_frame']-w['core_start_frame'] for w in plan]
            self.assertEqual(sum(lengths),frames)
            self.assertTrue(all(c.MIN_TAIL <= n <= c.CORE for n in lengths))
            self.assertLessEqual(abs(lengths[-1]-lengths[-2]),1)

    def test_real_repetitions_survive_exact_boundary_ownership(self):
        frames=24*c.RATE
        plan=c.plan_windows(frames)
        words=[('Да,',11.5,11.8),('да!',12.1,12.4)]
        rows=[evidence(w,words) for w in plan]
        original=deepcopy(rows)
        r=c.reconcile(rows,frames)
        self.assertEqual(r['status'],'candidate_unvalidated')
        self.assertEqual([s['text'] for s in r['candidate_segments']],['Да,','да!'])
        self.assertEqual(rows,original)

    def test_token_at_boundary_owned_once_by_right_window(self):
        frames=24*c.RATE
        rows=[evidence(w,[('слово',11.9,12.1)]) for w in c.plan_windows(frames)]
        r=c.reconcile(rows,frames)
        self.assertEqual(len(r['candidate_segments']),1)
        self.assertEqual(r['candidate_segments'][0]['source_window'],1)

    def test_disagreement_or_changed_time_does_not_choose_a_favorite(self):
        frames=24*c.RATE
        plan=c.plan_windows(frames)
        for right in ([('другое',11.5,11.8)],[('слово',12.1,12.4)]):
            rows=[evidence(plan[0],[('слово',11.5,11.8)]),evidence(plan[1],right)]
            r=c.reconcile(rows,frames)
            self.assertEqual(r['status'],'review_required')
            self.assertIsNone(r['candidate_segments'])

    def test_missing_time_missing_text_long_word_and_unsorted_evidence_abstain(self):
        plan=c.plan_windows(5*c.RATE)
        original=[evidence(plan[0],[('раз',.2,.5),('два',1.,1.4)])]
        for change in (
            lambda e:e[0]['aligned']['segments'][0]['words'][0].pop('end'),
            lambda e:e[0]['aligned']['segments'][0]['words'][0].update(start=float('nan')),
            lambda e:e[0]['aligned']['segments'][0]['words'][0].update(end=3.),
            lambda e:e[0]['aligned']['segments'][0]['words'][1].update(start=.1),
            lambda e:e[0]['raw']['segments'][0].update(text='раз два три'),
            lambda e:e[0]['aligned']['segments'][0]['words'].pop()):
            rows=deepcopy(original)
            change(rows)
            r=c.reconcile(rows,5*c.RATE)
            self.assertEqual(r['status'],'review_required')
            self.assertIsNone(r['candidate_segments'])

    def test_punctuation_and_non_latin_text_preserved(self):
        plan=c.plan_windows(5*c.RATE)
        row=evidence(plan[0],[('你',.2,.5),('好',.6,.9)],text='«你好!»')
        r=c.reconcile([row],5*c.RATE)
        self.assertEqual(r['candidate_segments'][0]['text'],'«你好!»')

    def test_standalone_unmapped_punctuation_is_not_silently_discarded(self):
        window=c.plan_windows(c.RATE)[0]
        row={'window':window,'raw':{'segments':[{'text':'…'}]},
             'aligned':{'segments':[{'text':'…','words':[]}]}}
        r=c.reconcile([row],c.RATE)
        self.assertEqual(r['status'],'review_required')
        self.assertIsNone(r['candidate_segments'])

    def test_backwards_midpoints_cannot_select_noncontiguous_text(self):
        window=c.plan_windows(5*c.RATE)[0]
        row=evidence(window,[('раз',.2,2.1),('два',.6,.8)])
        r=c.reconcile([row],5*c.RATE)
        self.assertEqual(r['status'],'review_required')
        self.assertEqual(r['issues'][0]['reason'],'unordered_words')

    def test_missing_or_reordered_windows_fail(self):
        plan=c.plan_windows(24*c.RATE)
        rows=[evidence(w,[]) for w in plan]
        for bad in (rows[:1],list(reversed(rows))):
            with self.assertRaises(ValueError): c.reconcile(bad,24*c.RATE)

    def test_flag_and_language_checked_before_callbacks(self):
        callback=Mock(side_effect=AssertionError('must not execute'))
        with patch.dict(os.environ,{'CONTEXT_ASR_TEST':'0'}):
            with self.assertRaises(ValueError): c.decode_with_context([0.]*100,transcribe=callback,align=callback,language='ru')
        with patch.dict(os.environ,{'CONTEXT_ASR_TEST':'1'}):
            with self.assertRaises(ValueError): c.decode_with_context([0.]*100,transcribe=callback,align=callback,language='en')
        callback.assert_not_called()

    def test_callbacks_receive_context_without_reference_or_short_vad_recut(self):
        audio=list(range(24*c.RATE))
        calls=[]
        def transcribe(chunk,**kwargs):
            calls.append((chunk[0],chunk[-1],kwargs))
            offset=chunk[0]/c.RATE
            return {'language':'ru','segments':[{'start':11.5-offset,'end':12.4-offset,'text':'Да, да!'}]}
        def align(segments,chunk):
            offset=chunk[0]/c.RATE
            # Deliberately mutate caller data: raw evidence must remain intact.
            segments[0]['words']=[{'word':'Да,','start':11.5-offset,'end':11.8-offset},
                                  {'word':'да!','start':12.1-offset,'end':12.4-offset}]
            return {'segments':segments}
        with patch.dict(os.environ,{'CONTEXT_ASR_TEST':'1'}):
            r=c.decode_with_context(audio,transcribe=transcribe,align=align,language='ru')
        self.assertEqual([(a,b) for a,b,_ in calls],[(0,14*c.RATE-1),(10*c.RATE,24*c.RATE-1)])
        self.assertTrue(all(opts=={'language':'ru','batch_size':8,'task':'transcribe','chunk_size':30} for _,_,opts in calls))
        self.assertNotIn('words',r['window_evidence'][0]['raw']['segments'][0])
        self.assertFalse(r['quality_validated'])
        self.assertEqual(r['diarization'],'not_run')


class SupportTests(unittest.TestCase):
    def test_alignment_spread_explains_assignment_without_fixing_label(self):
        word={'word':'test','start':14.009,'end':17.431,'speaker':'wife_label','score':.345}
        timeline={'status':'ok','segments':[{'start':11.10096875,'end':16.02846875,'speaker':'wife_label'},
            {'start':16.02846875,'end':16.75409375,'speaker':'other_label'},
            {'start':17.42909375,'end':20.41596875,'speaker':'husband_label'}]}
        original=deepcopy(word)
        result=word_support(word,timeline)
        self.assertTrue(result['assignment_matches_overlap'])
        self.assertEqual(result['largest_overlap_labels'],['wife_label'])
        self.assertEqual(result['overlap_seconds']['other_label'],.725625)
        self.assertIn('long_word',result['flags'])
        self.assertEqual(word,original)

    def test_no_overlap_and_ties_are_not_invented_confidence(self):
        timeline={'status':'ok','segments':[{'start':0,'end':1,'speaker':'a'},{'start':0,'end':1,'speaker':'b'}]}
        r=word_support({'word':'x','start':.2,'end':.5,'speaker':'a'},timeline)
        self.assertEqual(r['largest_overlap_labels'],['a','b'])
        r=word_support({'word':'x','start':2,'end':3,'speaker':'a'},timeline)
        self.assertIsNone(r['assignment_matches_overlap'])
        r=word_support({'word':'x','start':True,'end':3},timeline)
        self.assertEqual(r['status'],'invalid_word_time')

    def test_low_alignment_score_is_a_review_flag_not_a_probability(self):
        r=word_support({'word':'x','start':.1,'end':.3,'speaker':'a','score':0},
                       {'status':'ok','segments':[{'start':0,'end':1,'speaker':'a'}]})
        self.assertIn('low_alignment_score',r['flags'])
        self.assertTrue(r['assignment_matches_overlap'])

    def test_disagreement_is_not_reference_based_error_rate(self):
        r=text_changes('Да, да! Слово.','да да другое')
        self.assertEqual(r['changes'][0]['control_tokens'],['слово'])
        self.assertEqual(r['changes'][0]['candidate_tokens'],['другое'])
        self.assertNotIn('wer',r)


if __name__=='__main__': unittest.main()
