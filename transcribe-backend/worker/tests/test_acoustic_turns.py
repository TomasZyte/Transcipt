from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import acoustic_turns as a


def sample():
    timeline = [{'start': 0., 'end': 2., 'speaker': 'A'}]
    win = a.base.pause_windows(timeline, 32000)[0]
    native = [{'text': ' word.', 'words': [{'word': ' word.', 'start': .4, 'end': 1.1}]}]
    bounds = a.base.sentence_bounds(native, 2)
    local = {'segments': [{'text': ' word.', 'start': .4, 'end': 1.1,
                          'words': [{'word': 'word.', 'start': .4, 'end': 1.1, 'score': .8}]}]}
    s = {'duration_seconds': 2., 'windows': [{'window': win, 'native': native, 'bounds': bounds,
          'aligned_local': local}], 'asr': {'language': 'ru', 'segments': bounds},
         'diarization': {}, 'alignments': {}, 'alignment_windows': {}, 'transformed_audio': {}, 'outputs': {}}
    for key in a.TRANSFORMS:
        s['diarization'][key] = {'regular': timeline, 'exclusive': timeline, 'parameters': a.base.PARAMETERS,
            'filter_diagnostics': [{'min_active_ratio': .1, 'selected_embeddings': 1, 'chunk_speaker_pairs': 2}]}
        s['alignments'][key] = local
        s['alignment_windows'][key] = [{'window': win, 'aligned_local': local}]
        s['transformed_audio'][key] = {'frames': 32000, 'sample_rate': 16000,
            'n_steps': a.TRANSFORMS[key], 'pcm_sha256': 'a' * 64}
    for name, mode in a.MODES.items():
        s['outputs'][name] = a.base.attribute(s['asr'], local, timeline, 2.)
    return deepcopy(s)


class AcousticTests(unittest.TestCase):
    def test_variants_reproduce_and_do_not_change_text(self):
        s = sample(); original = deepcopy(s)
        a.validate_sample(s)
        self.assertEqual(s, original)
        self.assertEqual({o['rawAsrText'] for o in s['outputs'].values()}, {'word.'})
        self.assertEqual(len(a.MODES), 7)

    def test_time_offset_is_applied_once_and_original_bounds_retained(self):
        s = sample(); records = s['windows']; records[0]['window'] = {'start_frame': 16000, 'end_frame': 48000}
        before = deepcopy(records)
        def align(bounds, audio):
            self.assertEqual(len(audio), 32000)
            self.assertEqual(bounds, records[0]['bounds'])
            return records[0]['aligned_local']
        aligned, local = a.align_windows(records, [0] * 48000, align)
        self.assertAlmostEqual(aligned['segments'][0]['start'], 1.4)
        self.assertAlmostEqual(local[0]['aligned_local']['segments'][0]['start'], .4)
        self.assertEqual(records, before)

    def test_clock_changed_or_missing_variant_rejected(self):
        s = sample(); s['transformed_audio']['lower3']['frames'] = 31999
        with self.assertRaises(ValueError): a.validate_sample(s)
        s = sample(); del s['outputs']['lower6_both']
        with self.assertRaises(ValueError): a.validate_sample(s)

    def test_changed_alignment_or_speaker_export_rejected(self):
        s = sample(); s['alignment_windows']['lower3'][0]['window']['start_frame'] = 1
        with self.assertRaises(ValueError): a.validate_sample(s)
        s = sample(); s['outputs']['lower3_both']['segments'][0]['speaker'] = 'invented'
        with self.assertRaises(ValueError): a.validate_sample(s)

    def test_flags_are_exclusive_and_no_manual_hints_allowed(self):
        flags = {'FULL_PIPELINE_IMAGE': '1', 'FULL_PIPELINE_TEST': '1', 'ACOUSTIC_TURNS_TEST': '1'}
        with patch.dict(a.os.environ, flags, clear=True):
            a.require_test()
            with self.assertRaises(ValueError): a.validate_input({'profile': a.PROFILE, 'samples': [], 'num_speakers': 4})
        with patch.dict(a.os.environ, {**flags, 'TIMED_TURNS_TEST': '1'}, clear=True):
            with self.assertRaises(ValueError): a.require_test()
        with patch.dict(a.os.environ, {}, clear=True):
            with self.assertRaises(ValueError): a.require_test()

    def test_original_and_transformed_paths_have_independent_outputs(self):
        modes = a.MODES
        self.assertEqual(modes['lower3_align'], {'alignment': 'lower3', 'diarization': 'original'})
        self.assertEqual(modes['lower3_diar'], {'alignment': 'original', 'diarization': 'lower3'})
        self.assertEqual(modes['lower6_both'], {'alignment': 'lower6', 'diarization': 'lower6'})


if __name__ == '__main__': unittest.main()
