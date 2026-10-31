from copy import deepcopy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import channel_pipeline as c
import full_pipeline_handler as router
import test_channel_asr as fixture

sample = fixture.sample

HOST = 'a'*32 + '.r2.cloudflarestorage.com'


class ChannelWorkerTests(unittest.TestCase):
    def payload(self):
        return {'profile': c.PROFILE, 'source_sha256': 'a'*64,
                'confirmation': {'kind': 'owner_preview', 'start_seconds': 0, 'end_seconds': 32},
                'samples': [{'id': name, 'sha256': str(i)*64,
                    'media_url': f'https://{HOST}/poslovnica-test/quality/{name}.wav?sig=fixture'}
                    for i, name in enumerate(c.adapter.CHANNELS)]}

    def test_only_separate_test_flags_allow_execution(self):
        with patch.dict(os.environ, c.FLAGS, clear=True):
            c.require_test()
            for key, value in c.FLAGS.items():
                with patch.dict(os.environ, {key: '0' if value == '1' else '1'}), self.assertRaises(ValueError):
                    c.require_test()
        with patch.dict(os.environ, {}, clear=True), patch.object(c, 'fetch_channel') as fetch:
            result = router.handler({'input': self.payload()})
            self.assertEqual(result['phase'], 'validate')
            fetch.assert_not_called()

    def test_rejects_production_urls_options_counts_and_reversed_channels(self):
        original = self.payload()
        self.assertEqual(c.validate_input(original, HOST), original['samples'])
        for change in (
            lambda d: d.update(prompt='reference'),
            lambda d: d.update(source_sha256='invalid'),
            lambda d: d['samples'].reverse(),
            lambda d: d['samples'][0].update(speaker_count=2),
            lambda d: d['samples'][0].update(media_url=f'https://{HOST}/transcribe/quality/x?sig=1'),
            lambda d: d['samples'][0].update(media_url='https://example.com/x?sig=1'),
        ):
            data = deepcopy(original); change(data)
            with self.assertRaises(ValueError): c.validate_input(data, HOST)

    def test_fetch_channel_rejects_bad_hash_before_writing(self):
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.return_value = b'wrong'
        opener = Mock(); opener.open.return_value = response
        with patch.object(c.urllib.request, 'build_opener', return_value=opener), patch.object(Path, 'open') as opened:
            with self.assertRaises(ValueError): c.fetch_channel(self.payload()['samples'][0], Path('ignored.wav'))
            opened.assert_not_called()

    def test_replay_rejects_changed_channel_export_or_input_identity(self):
        plan = {'frames_per_channel': 5*c.adapter.RATE, 'confirmation': {'kind': 'owner_preview'}, 'manifest_sha256': 'a'*64}
        result = c.adapter.merge_channels(sample(), plan['frames_per_channel'])
        result.update(confirmation=plan['confirmation'], full_call_speaker_identity_verified=False)
        state = {'profile': c.PROFILE, 'status': 'completed', 'input_plan': c.input_contract(plan),
                 'source_manifest_sha256': c.digest(c.SOURCES), 'human_quality_validated': False, 'result': result}
        c.validate_result(state, plan)
        for change in (lambda s: s['result']['candidate_segments'][0].update(speaker='Спикер 2'),
                       lambda s: s['result']['candidate_segments'][0].update(startSec=0),
                       lambda s: s['input_plan'].update(frames_per_channel=10)):
            bad = deepcopy(state); change(bad)
            with self.assertRaises(ValueError): c.validate_result(bad, plan)

    def test_cpu_image_default_off_and_production_has_no_route(self):
        root = Path(c.__file__).parent
        docker = (root/'Dockerfile.asr-full').read_text()
        self.assertIn('CHANNEL_ASR_TEST=0', docker)
        self.assertIn('COPY channel_asr.py channel_pipeline.py', docker)
        for name in ('Dockerfile', 'handler.py', 'pipeline.py'):
            self.assertNotIn('channel_pipeline', (root/name).read_text(encoding='utf-8'))


class ChannelExecutionTests(unittest.TestCase):
    setUp = fixture.ChannelInputTests.setUp
    prepare = fixture.ChannelInputTests.prepare

    def test_complete_worker_uses_two_channels_without_gated_diarizer_or_llm(self):
        self.prepare()
        data = sample()
        for row in data:
            row['aligned']['segments'][0]['words'] = [{'word': 'Да.', 'start': .1, 'end': .3}]
            row['aligned']['segments'][0]['text'] = 'Да.'
            row['raw']['segments'] = [{'text': 'Да.'}]
        audio = [list(range(c.adapter.RATE)), list(range(c.adapter.RATE))]
        model = Mock(); model.transcribe.side_effect = [r['raw'] for r in data]
        wx = SimpleNamespace(load_model=Mock(return_value=model), load_align_model=Mock(return_value=(None, {})),
            load_audio=Mock(side_effect=audio), align=Mock(side_effect=[r['aligned'] for r in data]))
        modules = {'torch': SimpleNamespace(manual_seed=Mock()), 'whisperx': wx,
                   'whisperx.vads.pyannote': SimpleNamespace(load_vad_model=Mock())}
        with patch.dict(os.environ, c.FLAGS, clear=True), patch.dict(sys.modules, modules), \
             patch.object(c, 'verify_assets', return_value={'root': '/fake/asr', 'vad_path': '/fake/vad'}), \
             patch.object(c, 'verify_bundle', return_value={}), patch.object(c, 'verify_nltk', return_value='a'*64), \
             patch.object(c, 'version', side_effect=lambda name: c.config_for('wx_ru')['packages'][name]):
            result = c.execute(self.manifest, self.root/'result.json')
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(model.transcribe.call_count, 2)
        self.assertEqual(wx.load_model.call_count, 1)
        self.assertEqual(wx.align.call_count, 2)
        self.assertEqual(result['result']['diarization'], 'not_run')
        self.assertEqual(result['result']['summary'], 'not_run')
        self.assertFalse(result['human_quality_validated'])


if __name__ == '__main__': unittest.main()
