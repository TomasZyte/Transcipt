from copy import deepcopy
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import speaker_comparison as s
import full_pipeline_handler as handler


def wav_bytes(seconds):
    stream = io.BytesIO()
    with wave.open(stream,'wb') as w:
        w.setparams((1,2,16000,0,'NONE','not compressed'))
        w.writeframes(b'\x00\x00'*int(seconds*16000))
    return stream.getvalue()


def evidence():
    return {'variant':'wx_auto','model':'large-v3','language':'ru',
        'source_result_sha256':'a'*64,'clip_sha256':s.hashlib.sha256(wav_bytes(11)).hexdigest(),
        'source_start_seconds':8.5,'source_end_seconds':19.5,
        'segments':[{'start':0.1,'end':10.9,'text':'Слова без подсказок.'}]}


class SpeakerComparisonTests(unittest.TestCase):
    def test_raw_evidence_has_no_prompt_or_options(self):
        e = evidence()
        s.validate_evidence(e)
        for change in ({'prompt':'child'}, {'variant':'fw_v3'}, {'model':'other'},
                       {'source_start_seconds':9}, {'segments':[{'start':False,'end':11,'text':'x'}]}):
            with self.assertRaises(ValueError):
                s.validate_evidence({**e,**change})

    def test_explicit_isolation_flags_required_before_download(self):
        good = {'FULL_PIPELINE_IMAGE':'1','FULL_PIPELINE_TEST':'1','SPEAKER_COMPARISON_TEST':'1',
                'ASR_COMPARISON_TEST':'0','CONTEXT_ASR_TEST':'0'}
        for key,value in good.items():
            wrong = {**good,key:'0' if value=='1' else '1'}
            with patch.dict(os.environ,wrong), patch.object(s,'fetch_sample',side_effect=AssertionError('network')):
                result = handler.handler({'input':{'profile':s.PROFILE}})
                self.assertEqual(result['phase'],'validate')
                self.assertEqual(result['status'],'failed')
        with patch.dict(os.environ,good): s.require_test()

    def test_only_sound2_and_test_bucket_permitted(self):
        host = 'a'*32+'.r2.cloudflarestorage.com'
        inp = {'profile':s.PROFILE,'samples':[{'id':'sound2','sha256':'a'*64,
            'media_url':'https://'+host+'/poslovnica-test/quality/sound2.wav?signature=x'}],
            'asr_evidence':evidence()}
        s.validate_request_input(inp,host)
        for change in ('bucket','extra','count','id'):
            bad=deepcopy(inp)
            if change=='bucket': bad['samples'][0]['media_url']=bad['samples'][0]['media_url'].replace('poslovnica-test','transcribe')
            if change=='extra': bad['options']={}
            if change=='count': bad['samples']*=2
            if change=='id': bad['samples'][0]['id']='sound1'
            with self.assertRaises(ValueError): s.validate_request_input(bad,host)

    def test_clip_is_exact_slice_of_full_waveform(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'sound2.wav';path.write_bytes(wav_bytes(36))
            e=evidence()
            self.assertEqual(s.validate_audio(path,s.digest(path),e),36)
            e['clip_sha256']='b'*64
            with self.assertRaises(ValueError):s.validate_audio(path,s.digest(path),e)

    def test_alignment_is_shifted_once_despite_aliases_and_numeric_subclasses(self):
        class Scalar(float): pass
        word={'word':'Слова','start':Scalar(.2),'end':Scalar(.7)}
        aligned={'segments':[{'text':'Слова без подсказок.','start':Scalar(.1),
                             'end':Scalar(10.9),'words':[word]}], 'word_segments':[word]}
        shifted=s.global_alignment(aligned,evidence())
        self.assertAlmostEqual(shifted['segments'][0]['words'][0]['start'],8.7)
        self.assertAlmostEqual(shifted['word_segments'][0]['start'],8.7)
        self.assertEqual(word['start'],.2)
        self.assertEqual(s.global_alignment(s.json.loads(s.json.dumps(aligned)),evidence()),shifted)

    def test_alignment_cannot_silently_change_text_or_create_times(self):
        for aligned in ({'segments':[{'text':'Новые слова','start':0,'end':1}]},
            {'segments':[{'text':'Слова без подсказок.','start':0,'end':99}]},
            {'segments':[{'text':'Слова без подсказок.','start':float('nan'),'end':1}]}):
            with self.assertRaises(ValueError):s.global_alignment(aligned,evidence())

    def test_regular_overlap_allowed_exclusive_overlap_rejected(self):
        rows=[{'start':0.,'end':2.,'speaker':'A'},{'start':1.,'end':3.,'speaker':'B'}]
        s.validate_timeline(rows,36)
        with self.assertRaises(ValueError):s.validate_timeline(rows,36,True)
        rows[1]['start']=2.
        s.validate_timeline(rows,36,True)
        with self.assertRaises(ValueError):s.validate_timeline(rows,1)

    def test_export_uses_word_speakers_without_changing_text(self):
        aligned={'segments':[{'text':'Привет. Пока.','start':9.,'end':11.,'speaker':'adult',
            'words':[{'word':'Привет.','start':9.,'end':10.,'speaker':'child'},
                     {'word':'Пока.','start':10.,'end':11.,'speaker':'adult'}]}]}
        result=s.export_result(aligned)
        self.assertEqual(len(result['segments']),2)
        self.assertEqual(s.text_key(result['segments']),'Привет.Пока.')
        self.assertIn('00:00:09,000',result['srt'])


if __name__=='__main__':unittest.main()
