from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import short_turn_comparison as s
import full_pipeline_handler
from test_speaker_comparison import evidence, wav_bytes

FLAGS = {'FULL_PIPELINE_IMAGE': '1', 'FULL_PIPELINE_TEST': '1',
         'SHORT_TURN_COMPARISON_TEST': '1', 'SPEAKER_COMPARISON_TEST': '0',
         'ASR_COMPARISON_TEST': '0', 'CONTEXT_ASR_TEST': '0'}


def frozen():
    words = [{'word': text, 'start': float(i), 'end': i + .4, 'score': .8}
             for i, text in enumerate(('Слова', 'без', 'подсказок.'))]
    return {'source_result_sha256': 'f' * 64, 'asr_evidence': evidence(),
            'alignment_local': {'segments': [{'start': .1, 'end': 10.9,
                'text': 'Слова без подсказок.', 'words': words}], 'word_segments': deepcopy(words)}}


class VBxClustering:
    def filter_embeddings(self, embeddings, segmentations=None, min_active_ratio=.2):
        # Distinct test outputs prove the supplied ratio reaches the actual filter.
        return (['one'] if min_active_ratio == .2 else ['one', 'two'], [], [])


class ShortTurnTests(unittest.TestCase):
    def test_gate_rejects_before_download_for_every_wrong_flag(self):
        for key, value in FLAGS.items():
            with patch.dict(os.environ, {**FLAGS, key: '0' if value == '1' else '1'}), \
                 patch.object(s.previous, 'fetch_sample', side_effect=AssertionError('network')):
                result = full_pipeline_handler.handler({'input': {'profile': s.PROFILE}})
                self.assertEqual(result['status'], 'failed')
                self.assertEqual(result['phase'], 'validate')

    def test_request_cannot_supply_counts_roles_or_production_bucket(self):
        host = 'a' * 32 + '.r2.cloudflarestorage.com'
        inp = {'profile': s.PROFILE, 'samples': [{'id': 'sound2', 'sha256': 'a' * 64,
            'media_url': 'https://' + host + '/poslovnica-test/quality/sound2.wav?sig=x'}], 'frozen': frozen()}
        s.validate_input(inp, host)
        for key in ('num_speakers', 'roles', 'child_intervals'):
            with self.assertRaises(ValueError):
                s.validate_input({**inp, key: [11.5, 12.5]}, host)
        inp['samples'][0]['media_url'] = inp['samples'][0]['media_url'].replace('poslovnica-test', 'transcribe')
        with self.assertRaises(ValueError): s.validate_input(inp, host)

    def test_frozen_views_cannot_disagree_or_include_speaker_hints(self):
        original = frozen()
        shifted = s.validate_frozen(original)
        self.assertEqual(shifted['segments'][0]['words'][0]['start'], 8.5)
        self.assertEqual(original, frozen())
        for place in ('word', 'segment', 'time', 'text'):
            bad = frozen()
            if place == 'word': bad['alignment_local']['segments'][0]['words'][0]['speaker'] = 'child'
            if place == 'segment': bad['alignment_local']['segments'][0]['speaker'] = 'child'
            if place == 'time': bad['alignment_local']['word_segments'][0]['start'] = 5
            if place == 'text': bad['alignment_local']['segments'][0]['text'] = 'New text'
            with self.assertRaises(ValueError): s.validate_frozen(bad)

    def test_filter_is_local_and_restored_even_after_failure(self):
        cluster, other = VBxClustering(), VBxClustering()
        data = SimpleNamespace(shape=(2, 3, 4))
        records = []
        with self.assertRaises(RuntimeError):
            with s.short_turn_filter(cluster, .1, records):
                self.assertEqual(len(cluster.filter_embeddings(data)[0]), 2)
                self.assertEqual(len(other.filter_embeddings(data)[0]), 1)
                raise RuntimeError('inference failed')
        self.assertNotIn('filter_embeddings', vars(cluster))
        self.assertEqual(len(cluster.filter_embeddings(data)[0]), 1)
        self.assertEqual(records, [{'min_active_ratio': .1, 'selected_embeddings': 2, 'chunk_speaker_pairs': 6}])

    def test_unreviewed_ratio_and_changed_api_are_rejected(self):
        with self.assertRaises(ValueError):
            with s.short_turn_filter(VBxClustering(), .05, []): pass
        altered = SimpleNamespace(filter_embeddings=lambda embeddings, segmentations=None, min_active_ratio=.3: [])
        with self.assertRaises(ValueError):
            with s.short_turn_filter(altered, .1, []): pass
        with s.short_turn_filter(VBxClustering(), .1, []) as _:
            pass

    def test_runtime_settings_must_match_pinned_comparison(self):
        pipeline = SimpleNamespace(clustering=VBxClustering(), embedding_exclude_overlap=True,
                                   parameters=lambda **kwargs: deepcopy(s.PARAMETERS))
        s.validate_runtime_pipeline(pipeline, '4.0.3')
        with self.assertRaises(ValueError): s.validate_runtime_pipeline(pipeline, '4.0.4')
        pipeline.embedding_exclude_overlap = False
        with self.assertRaises(ValueError): s.validate_runtime_pipeline(pipeline, '4.0.3')

    def test_execute_keeps_asr_and_alignment_frozen_and_never_forces_count(self):
        class Audio:
            shape = (2, 3, 4)
            def __getitem__(self, item): return self
        annotation = SimpleNamespace(itertracks=lambda **kwargs: iter([(SimpleNamespace(start=0., end=36.), 0, 'A')]))
        calls = []
        class Pipeline:
            clustering = VBxClustering()
            embedding_exclude_overlap = True
            _segmentation = SimpleNamespace(model=SimpleNamespace(specifications=SimpleNamespace(duration=10)))
            @classmethod
            def from_pretrained(cls, *args, **kwargs): return cls()
            def to(self, device): return self
            def parameters(self, **kwargs): return deepcopy(s.PARAMETERS)
            def __call__(self, sample, **kwargs):
                calls.append(kwargs)
                self.clustering.filter_embeddings(Audio())
                return SimpleNamespace(speaker_diarization=annotation, exclusive_speaker_diarization=annotation)
        torch = SimpleNamespace(device=lambda x: x, from_numpy=lambda x: x, manual_seed=lambda x: None)
        whisperx = SimpleNamespace(load_audio=lambda x: Audio(),
            load_model=Mock(side_effect=AssertionError('ASR rerun')),
            align=Mock(side_effect=AssertionError('alignment rerun')),
            assign_word_speakers=lambda rows, aligned, **kwargs: aligned)
        modules = {'torch': torch, 'whisperx': whisperx, 'pandas': SimpleNamespace(DataFrame=lambda x: x),
                   'pyannote': SimpleNamespace(), 'pyannote.audio': SimpleNamespace(Pipeline=Pipeline)}
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            audio = base / 'sound2.wav'; audio.write_bytes(wav_bytes(36))
            source = frozen()
            manifest = base / 'manifest.json'
            manifest.write_text(json.dumps({'sample': {'path': str(audio), 'sha256': s.digest(audio)}, 'frozen': source}), encoding='utf-8')
            with patch.dict(os.environ, FLAGS), patch.dict(sys.modules, modules), \
                 patch.object(s, 'prepare_diarization', return_value={}), \
                 patch('importlib.metadata.version', return_value='4.0.3'):
                result = s.execute(manifest, base / 'output.json')
            self.assertEqual(result['status'], 'completed', result)
            self.assertEqual(calls, [{}, {}])
            self.assertEqual(result['frozen'], source)
            self.assertFalse(result['asr_rerun'])
            self.assertFalse(result['alignment_rerun'])
            self.assertEqual(set(result['outputs']), set(s.MODES))
            for mode, output in result['outputs'].items():
                self.assertEqual(output['aligned'], s.validate_frozen(source))
                self.assertIsNone(output['num_speakers'])
                count = 1 if mode.startswith('standard') else 2
                self.assertEqual(output['filter_diagnostics'][0]['selected_embeddings'], count)
            whisperx.load_model.assert_not_called()
            whisperx.align.assert_not_called()


if __name__ == '__main__': unittest.main()
