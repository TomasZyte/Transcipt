from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import context_pipeline as c
import full_pipeline_handler as h
from contextual_asr import RATE

ENV={'FULL_PIPELINE_IMAGE':'1','FULL_PIPELINE_TEST':'1','CONTEXT_ASR_TEST':'1'}
TIMELINE={'status':'ok','segments':[{'start':0.,'end':24.,'speaker':'same_full_file_label'}]}


class ModelFloat(float):
    """JSON-serializable scalar subtype returned by an aligner."""


def compare(*,disagree=False,fail_align=False,sink=None,scalar=float):
    audio=list(range(24*RATE))
    model=Mock()
    model.detect_language.return_value='ru'
    def transcribe(chunk,**kwargs):
        offset=chunk[0]/RATE
        text='Да, другое!' if disagree and offset else 'Да, да!'
        return {'language':'ru','segments':[{'start':11.5-offset,'end':12.4-offset,'text':text}]}
    model.transcribe.side_effect=transcribe
    count=0
    def align(segments,chunk):
        nonlocal count
        count+=1
        if fail_align and count==3:
            raise RuntimeError('fixture')
        offset=chunk[0]/RATE
        segments[0]['words']=[{'word':'Да,','start':11.5-offset,'end':11.8-offset,'score':.8},
            {'word':'другое!' if disagree and offset else 'да!', 'start':12.1-offset,'end':12.4-offset,'score':.9}]
        for item in [segments[0]]+segments[0]['words']:
            for key in ('start','end','score'):
                if key in item: item[key]=scalar(item[key])
        return {'segments':segments}
    def assign(aligned):
        for segment in aligned['segments']:
            segment['speaker']='same_full_file_label'
            for word in segment.get('words',[]): word['speaker']='same_full_file_label'
        return aligned
    with patch.dict(os.environ,ENV):
        return c.execute_sample(audio,model=model,align=align,assign=assign,timeline=TIMELINE,
            duration=24.,output_sink=sink),model


class ContextComparison(unittest.TestCase):
    def test_model_scalar_types_keep_global_times_and_replay_after_json(self):
        result,_=compare(scalar=ModelFloat)
        c.validate_sample(result,24*RATE,24.)
        saved=json.loads(json.dumps(result,allow_nan=False))
        c.validate_sample(saved,24*RATE,24.)
        self.assertEqual(saved,result)
        ctx=result[c.MODE]
        self.assertEqual(ctx['status'],'candidate_unvalidated')
        self.assertEqual(ctx['window_evidence'][1]['aligned']['segments'][0]['words'][0]['start'],1.5)
        word=ctx['window_outputs'][1]['output']['benchmark']['alignedSegments'][0]['words'][0]
        self.assertEqual(word['start'],11.5)
        self.assertIs(type(word['start']),float)
        self.assertEqual(' '.join(s['text'] for s in ctx['candidate']['segments']),'Да, да!')

    def test_model_scalar_normalization_does_not_accept_invalid_timestamps(self):
        from contextual_asr import model_evidence, reconcile, plan_windows
        for start in (True,ModelFloat('nan'),ModelFloat('inf'),ModelFloat(-1)):
            item={'window':plan_windows(RATE)[0], 'raw':{'segments':[{'text':'x'}]},
                  'aligned':{'segments':[{'text':'x','words':[{'word':'x','start':start,'end':.5}]}]}}
            result=reconcile([model_evidence(item)],RATE)
            self.assertEqual(result['status'],'review_required')
            self.assertIsNone(result['candidate_segments'])
        with self.assertRaises(ValueError): model_evidence({'start':object()})

    def test_handler_requires_extra_flag_and_rejects_hints_before_any_work(self):
        host='0'*32+'.r2.cloudflarestorage.com'
        payload={'profile':c.PROFILE,'samples':[{'id':'sound'+str(i),'sha256':'a'*64,
            'media_url':'https://'+host+'/poslovnica-test/quality/x?signature=fixture'} for i in range(1,5)]}
        for key in ENV:
            with patch.dict(os.environ,{**ENV,'QUALITY_R2_HOST':host,key:'0'}):
                with self.assertRaises(ValueError): h.validate(payload)
        with patch.dict(os.environ,{**ENV,'QUALITY_R2_HOST':host}):
            h.validate(payload)
            for key in ('prompt','speaker_count','language'):
                with self.assertRaises(ValueError): h.validate({**payload,key:'forbidden'})

    def test_same_language_global_time_and_speakers_with_repeated_words_preserved(self):
        result,model=compare()
        c.validate_sample(result,24*RATE,24.)
        model.detect_language.assert_called_once()
        self.assertTrue(all(call.kwargs=={'language':'ru','batch_size':8,'task':'transcribe','chunk_size':30}
                            for call in model.transcribe.call_args_list))
        context=result[c.MODE]
        self.assertEqual(' '.join(s['text'] for s in context['candidate']['segments']),'Да, да!')
        second=context['window_outputs'][1]['output']['benchmark']['alignedSegments'][0]['words']
        self.assertEqual(second[0]['start'],11.5)
        self.assertFalse(context['quality_validated'])

    def test_disagreement_retains_all_windows_and_no_final_candidate(self):
        result,_=compare(disagree=True)
        c.validate_sample(result,24*RATE,24.)
        context=result[c.MODE]
        self.assertEqual(context['status'],'review_required')
        self.assertIsNone(context['candidate'])
        self.assertEqual(len(context['window_outputs']),2)
        self.assertIn('другое',context['window_outputs'][1]['output']['rawAsrText'])

    def test_failed_alignment_keeps_control_completed_windows_and_current_raw(self):
        sink={}
        with self.assertRaises(RuntimeError): compare(fail_align=True,sink=sink)
        self.assertIn(c.CONTROL,sink)
        rows=sink[c.MODE]['window_evidence']
        self.assertEqual(len(rows),2)
        self.assertIn('aligned',rows[0])
        self.assertIn('raw',rows[1])
        self.assertNotIn('aligned',rows[1])

    def test_validation_rejects_mutated_seams_time_exports_timeline_and_quality_claims(self):
        result,_=compare()
        for mutate in (
            lambda r:r[c.MODE]['window_evidence'].reverse(),
            lambda r:r[c.MODE]['reconciliation'].update(status='review_required'),
            lambda r:r[c.MODE]['window_outputs'][1]['output'].update(rawText='invented'),
            lambda r:r[c.MODE]['window_outputs'][1]['output']['benchmark']['alignedSegments'][0]['words'][0].update(start=1.5),
            lambda r:r[c.MODE]['diarizationTimeline']['segments'][0].update(speaker='different'),
            lambda r:r[c.MODE].update(quality_validated=True)):
            bad=deepcopy(result)
            mutate(bad)
            with self.assertRaises(ValueError): c.validate_sample(bad,24*RATE,24.)


if __name__=='__main__': unittest.main()
