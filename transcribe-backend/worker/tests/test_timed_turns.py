from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import timed_turns as t


def native():
    return [{'text':' Hello there! Another phrase.', 'words':[
        {'word':' Hello','start':.5,'end':.8},{'word':' there!','start':.9,'end':1.2},
        {'word':' Another','start':4.,'end':4.3},{'word':' phrase.','start':4.4,'end':4.8}]}]


class TimedTurnTests(unittest.TestCase):
    def test_sentence_boundary_preserves_silence_and_all_text(self):
        rows=native(); before=deepcopy(rows)
        bounds=t.sentence_bounds(rows,6)
        self.assertEqual(len(bounds),2)
        self.assertAlmostEqual(bounds[0]['end'],1.28)
        self.assertAlmostEqual(bounds[1]['start'],3.92)
        self.assertEqual(''.join(s['text'] for s in bounds), rows[0]['text'])
        self.assertEqual(rows,before)

    def test_no_padding_into_adjacent_sentence(self):
        rows=native(); rows[0]['words'][2]['start']=1.21
        bounds=t.sentence_bounds(rows,6)
        self.assertEqual(bounds[0]['end'],1.21)
        self.assertEqual(bounds[1]['start'],1.2)

    def test_bad_tokens_bounds_and_order_rejected(self):
        for field,value in [('word','wrong'),('start',-1),('end',7),('start',float('nan'))]:
            rows=native(); rows[0]['words'][0][field]=value
            with self.assertRaises(ValueError): t.sentence_bounds(rows,6)
        rows=native(); rows[0]['words'][2]['start']=.2
        with self.assertRaises(ValueError): t.sentence_bounds(rows,6)

    def test_silent_window_and_zero_duration_fragment(self):
        self.assertEqual(t.sentence_bounds([],6),[])
        rows=[{'text':'a','words':[{'word':'a','start':1.,'end':1.}]}]
        self.assertLess(t.sentence_bounds(rows,2)[0]['start'],1)
        self.assertGreater(t.sentence_bounds(rows,2)[0]['end'],1)

    def test_acoustic_overrides_restore_even_on_error(self):
        p=SimpleNamespace(segmentation_step=.1,embedding_exclude_overlap=True,
            _segmentation=SimpleNamespace(step=1.,model=SimpleNamespace(specifications=SimpleNamespace(duration=10))))
        with self.assertRaises(RuntimeError):
            with t.acoustic_settings(p,t.DIARIZATION['overlap']) as actual:
                self.assertEqual(actual,{'step_seconds':.5,'step_ratio':.05,'exclude_overlap':False})
                raise RuntimeError('stop')
        self.assertEqual((p.segmentation_step,p._segmentation.step,p.embedding_exclude_overlap),(.1,1.,True))
        with self.assertRaises(ValueError):
            with t.acoustic_settings(p,{'step_ratio':.01,'exclude_overlap':False}): pass

    def test_decoder_uses_same_model_no_reference_and_offsets_once(self):
        @dataclass
        class Segment:
            text:str
            words:list
        calls=[]
        def transcribe(audio,**kwargs):
            calls.append((len(audio),kwargs))
            return iter([Segment(**native()[0])]),SimpleNamespace(language='ru')
        def align(bounds,audio):
            return {'segments':[{**s,'words':[]} for s in bounds]}
        model=SimpleNamespace(model=SimpleNamespace(transcribe=transcribe))
        raw,aligned,evidence=t.decode([0]*t.RATE*8,[{'start_frame':2*t.RATE,'end_frame':8*t.RATE}],model,align)
        self.assertEqual(calls,[(6*t.RATE,t.DECODER)])
        self.assertAlmostEqual(raw['segments'][1]['start'],5.92)
        self.assertEqual(aligned['segments'][1]['start'],raw['segments'][1]['start'])
        self.assertAlmostEqual(evidence[0]['bounds'][1]['start'],3.92)

    def test_test_flags_and_no_options(self):
        with patch.dict(t.os.environ,{},clear=True):
            with self.assertRaises(ValueError): t.validate_input({})
        with patch.dict(t.os.environ,{'FULL_PIPELINE_IMAGE':'1','FULL_PIPELINE_TEST':'1','TIMED_TURNS_TEST':'1'},clear=True):
            t.require_test()
            with self.assertRaises(ValueError): t.validate_input({'profile':t.PROFILE,'samples':[None]*4})
            with self.assertRaises(ValueError): t.validate_input({'profile':t.PROFILE,'samples':[],'num_speakers':4})


if __name__=='__main__': unittest.main()
