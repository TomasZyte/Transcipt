"""Full candidate contracts, text preservation and strict isolated requests."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
import wave

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import full_pipeline as f
import full_pipeline_handler as h
import prepare_full_assets as assets
import context_pipeline as context

HOST = '0'*32 + '.r2.cloudflarestorage.com'


def segments():
    return [{'start':0.,'end':.9,'text':'Один два.', 'speaker':'SPEAKER_00',
        'words':[{'word':'Один','start':0.,'end':.4,'speaker':'SPEAKER_00'},
                 {'word':'два.','start':.5,'end':.9,'speaker':'SPEAKER_01'}]}]


class Frame:
    def to_dict(self,*args):
        return [{'start':0.,'end':.45,'speaker':'SPEAKER_00'},
                {'start':.5,'end':1.,'speaker':'SPEAKER_01'}]
    def copy(self,**kwargs):
        return Frame()


class FullContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rows = []
        for name in f.IDS:
            path = self.root/(name+'.wav')
            with wave.open(str(path),'wb') as wav:
                wav.setparams((1,2,16000,0,'NONE','not compressed'))
                wav.writeframes(b'\0\0'*16000)
            self.rows.append({'id':name,'path':str(path),'sha256':f.digest(path)})
        self.inp = {'profile':f.PROFILE,'samples':[{'id':r['id'],'sha256':r['sha256'],
            'media_url':'https://'+HOST+'/poslovnica-test/quality/'+r['id']+'.wav?signature=fixture'} for r in self.rows]}
        self.env = {'FULL_PIPELINE_TEST':'1','FULL_PIPELINE_IMAGE':'1','QUALITY_R2_HOST':HOST,'HF_TOKEN':'fake'}

    def test_fixed_batch_requires_exact_ids_fields_and_test_mode(self):
        with patch.dict(os.environ,self.env):
            self.assertEqual(h.validate(self.inp),self.inp['samples'])
            for changes in ({'profile':'baseline'}, {'prompt':'expected text'}, {'samples':self.inp['samples'][:3]}):
                with self.assertRaises(ValueError):
                    h.validate({**self.inp,**changes})
            for extra in ('speaker_count','prompt','options'):
                payload = deepcopy(self.inp)
                payload['samples'][0][extra] = 4
                with self.assertRaises(ValueError):
                    h.validate(payload)
        for key in ('FULL_PIPELINE_TEST','FULL_PIPELINE_IMAGE'):
            with patch.dict(os.environ,{**self.env,key:'0'}),self.assertRaises(ValueError):
                h.validate(self.inp)

    def test_production_bucket_and_missing_signature_rejected(self):
        with patch.dict(os.environ,self.env):
            for url in ('https://'+HOST+'/transcribe/quality/x?sig=1',
                        'https://'+HOST+'/poslovnica-test/quality/x',
                        'https://example.com/poslovnica-test/quality/x?sig=1'):
                payload = deepcopy(self.inp)
                payload['samples'][0]['media_url'] = url
                with self.assertRaises(ValueError):
                    h.validate(payload)

    def test_audio_checks_hash_order_format_and_truncation(self):
        f.validate_audio_rows(self.rows)
        self.assertEqual([r['duration_seconds'] for r in self.rows],[1.]*4)
        with self.assertRaises(ValueError):
            f.validate_audio_rows(list(reversed(self.rows)))
        self.rows[0]['sha256'] = 'a'*64
        with self.assertRaises(ValueError):
            f.validate_audio_rows(self.rows)
        p = Path(self.rows[0]['path'])
        p.write_bytes(p.read_bytes()[:-2])
        self.rows[0]['sha256'] = f.digest(p)
        with self.assertRaises(ValueError):
            f.validate_audio_rows(self.rows)

    def test_export_preserves_text_and_word_speaker_turns(self):
        source = segments()
        raw = {'language':'ru','segments':f.snapshot_asr(source)}
        result = f.format_output(raw,{'segments':source},f.snapshot_diarization(Frame()),'short12',1.)
        self.assertEqual(len(result['segments']),2)
        self.assertEqual(result['rawAsrText'],'Один два.')
        self.assertIn('Спикер 2: два.',result['srtContent'])
        self.assertTrue(result['audit']['exportMatchesRebuild'])
        self.assertEqual(raw['segments'],f.snapshot_asr(source))

    def test_alignment_cannot_drop_or_rewrite_words(self):
        raw = {'language':'ru','segments':f.snapshot_asr(segments())}
        bad = segments()
        bad[0]['text'] = 'Один.'
        with self.assertRaises(ValueError):
            f.format_output(raw,{'segments':bad},f.snapshot_diarization(Frame()),'standard30',1.)

    def test_out_of_range_times_and_unknown_mode_rejected(self):
        for timing in ((-1.,.9),(0.,2.),(float('nan'),.9)):
            raw = {'language':'ru','segments':[{'start':timing[0],'end':timing[1],'text':'Один два.'}]}
            with self.assertRaises(ValueError):
                f.format_output(raw,{'segments':segments()},f.snapshot_diarization(Frame()),'short12',1.)
        with self.assertRaises(ValueError):
            f.format_output({'segments':segments()},{'segments':segments()},None,'arbitrary',1.)

    def test_srt_formats_millisecond_rounding_without_invalid_seconds(self):
        text = f.srt([{'startSec':59.9996,'endSec':60.1,'speaker':'S1','text':'x'}])
        self.assertIn('00:01:00,000 --> 00:01:00,100',text)

    def test_gated_asset_manifest_rejects_modified_file(self):
        target = self.root/'asset'
        target.write_bytes(b'model')
        row = {'size':5,'sha256':f.digest(target)}
        assets.verify_file(target,row)
        target.write_bytes(b'wrong')
        with self.assertRaises(ValueError):
            assets.verify_file(target,row)

    def test_image_has_separate_entrypoint_default_off_and_no_gated_build_download(self):
        root = Path(f.__file__).parent
        docker = (root/'Dockerfile.asr-full').read_text()
        self.assertIn('FULL_PIPELINE_TEST=0',docker)
        self.assertIn('"full_pipeline_handler.py"',docker)
        self.assertNotIn('ARG HF_TOKEN',docker)
        self.assertNotIn('full_pipeline', (root/'Dockerfile').read_text())

    def test_full_batch_uses_same_weights_two_chunk_sizes_and_one_global_diarization(self):
        self._fake_run()

    def test_context_profile_uses_fixed_language_and_one_global_diarization(self):
        self._fake_run(contextual=True)

    def test_failed_stage_keeps_partial_evidence_and_does_not_continue(self):
        self._fake_run(fail_align=True)

    def _fake_run(self,fail_align=False,contextual=False):
        manifest, output = self.root/'manifest.json',self.root/'result.json'
        f.write_new(manifest,{'samples':self.rows})
        model = Mock()
        model.detect_language.return_value = 'ru'
        model.transcribe.side_effect = lambda *args,**kwargs: {'language':'ru','segments':f.snapshot_asr(segments())}
        wx = types.ModuleType('whisperx')
        wx.load_model = Mock(return_value=model)
        wx.load_audio = Mock(return_value=[0.]*16000 if contextual else 'audio-array')
        wx.load_align_model = Mock(return_value=(None,{}))
        wx.align = Mock(side_effect=ValueError('fixture') if fail_align else None,
                        return_value={'segments':segments()})
        wx.assign_word_speakers = Mock(side_effect=lambda frame,result:deepcopy(result))
        vad = types.ModuleType('whisperx.vads.pyannote')
        vad.load_vad_model = Mock()
        diar = types.ModuleType('whisperx.diarize')
        diar_model = Mock(return_value=Frame())
        diar.DiarizationPipeline = Mock(return_value=diar_model)
        torch = types.ModuleType('torch')
        torch.manual_seed = Mock()
        torch.cuda = types.SimpleNamespace(empty_cache=Mock())
        with patch.dict(os.environ,{**self.env,'CONTEXT_ASR_TEST':'1'}),patch.dict(sys.modules,{'whisperx':wx,
             'whisperx.vads.pyannote':vad,'whisperx.diarize':diar,'torch':torch}), \
             patch.object(f,'verify_assets',return_value={'root':'asr','vad_path':'vad'}), \
             patch.object(f,'verify_bundle',return_value={}),patch.object(f,'verify_nltk',return_value='lock'), \
             patch.object(f,'prepare_diarization',return_value={}):
            result = f.execute(manifest,output,context.PROFILE if contextual else f.PROFILE)
        self.assertEqual(result['status'],'failed' if fail_align else 'completed')
        self.assertTrue(output.is_file())
        if fail_align:
            self.assertEqual(result['phase'],'sound1:standard30:alignment')
            self.assertEqual(model.transcribe.call_count,1)
        else:
            self.assertEqual(wx.load_model.call_count,1)
            self.assertEqual([c.kwargs['chunk_size'] for c in model.transcribe.call_args_list],[30]*8 if contextual else [30,12]*4)
            self.assertTrue(all(c.kwargs['language'] == ('ru' if contextual else None) for c in model.transcribe.call_args_list))
            if contextual:
                self.assertEqual(model.detect_language.call_count,4)
                self.assertEqual(result['modes'],context.MODES)
                self.assertTrue(all(s['outputs'][context.MODE]['quality_validated'] is False for s in result['samples']))
            self.assertEqual(diar_model.call_count,4)
            self.assertTrue(all(c.kwargs == {} for c in diar_model.call_args_list))
            self.assertEqual(wx.align.call_count,8)
        with patch.dict(os.environ,self.env),self.assertRaises(ValueError):
            f.execute(manifest,output)


if __name__ == '__main__':
    unittest.main()
