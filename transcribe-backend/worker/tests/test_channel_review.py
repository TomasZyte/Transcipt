from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import channel_review as review
import channel_asr
from asr_comparison import digest
from prepare_full_assets import SOURCES
import test_channel_speech as fixture


def evidence():
    result = fixture.SpeechAlignmentTests().decode(empty=True)
    confirmation = result['confirmation']
    plan = {'profile':'channel-speech-v1','frames_per_channel':320000,'duration_seconds':20,
            'confirmation':confirmation,'channels':[],'source_sha256':'a'*64}
    state = {'profile':'channel-speech-v1','status':'completed','input_plan':plan,
        'source_manifest_sha256':digest(SOURCES),'human_quality_validated':False,'result':result}
    control_result=channel_asr.merge_channels(result['original']['evidence'],320000)
    control_result.update(confirmation=confirmation,full_call_speaker_identity_verified=False)
    control={**deepcopy(state),'profile':'channel-asr-v1','input_plan':{**plan,'profile':'channel-asr-v1'},'result':control_result}
    return state,control


class ReviewTests(unittest.TestCase):
    def test_all_windows_including_unresolved_preserve_evidence(self):
        state,control=evidence(); before=deepcopy((state,control))
        result=review.build(state,control)
        expected={(c['channel'],i):r for c in state['result']['original']['run_evidence'] for i,r in enumerate(c['runs'])}
        self.assertEqual(len(result['rows']),len(expected))
        for row in result['rows']:
            raw=expected[row['channel'],row['run']]
            self.assertEqual(row['original_text'],raw['raw_local']['segments'][0]['text'].strip())
            self.assertEqual(row['startSec'],raw['window']['start_frame']/16000)
        self.assertEqual((state,control),before)
        missing=[r for r in result['rows'] if r['status']=='unresolved_speech']
        self.assertEqual(len(missing),4)
        self.assertTrue(all(r['text'] is None and r['words']==[] for r in missing))
        self.assertFalse(result['production_export_allowed'])
        self.assertFalse(result['release_ready'])
        self.assertEqual(review.text_export(result).count('[НЕРАЗБОРЧИВАЯ РЕЧЬ'),4)

    def test_controls_cannot_mix_audio_or_forged_results(self):
        for target in ('audio','text'):
            state,control=evidence()
            if target=='audio':control['input_plan']['source_sha256']='b'*64
            else:state['result']['review_spans'][0]['hypothesis']='Invented'
            with self.assertRaises(ValueError):review.build(state,control)

    def test_control_only_words_are_flagged_without_inventing_text(self):
        rows=[{'run':0,'original_text':'Один два.'},{'run':1,'original_text':'Три четыре.'}]
        before=deepcopy(rows)
        self.assertEqual(review.disagreements(rows,'Один два. Пропущено. Три четыре.'),{1})
        self.assertEqual(rows,before)

    def test_export_cannot_be_promoted_by_flag(self):
        state,control=evidence(); result=review.build(state,control)
        result['production_export_allowed']=True
        with self.assertRaises(ValueError):review.text_export(result)


if __name__=='__main__':unittest.main()
