from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pause_context as p


def native():
    return [{'text': ' Go there! Go there!', 'words': [
        {'word': ' Go', 'start': .5, 'end': .8}, {'word': ' there!', 'start': .9, 'end': 1.2},
        {'word': ' Go', 'start': 2.4, 'end': 3.1}, {'word': ' there!', 'start': 3.2, 'end': 4.9}]}]


def sample_evidence():
    duration = 6.
    timeline = [{'start': 0., 'end': duration, 'speaker': 'A'}]
    plan = p.base.pause_windows(timeline, int(duration * p.base.RATE))
    bounds = p.alignment_bounds(native(), duration, 'control')
    local = {'segments': [{**b, 'words': []} for b in bounds]}
    records = [{'window': plan[0], 'native': native(), 'bounds': bounds, 'aligned_local': local}]
    sample = {'duration_seconds': duration, 'windows': records,
              'asr': {'language': 'ru', 'segments': bounds}, 'diarization': {},
              'alignments': {}, 'alignment_windows': {}, 'outputs': {}}
    for name, settings in p.DIARIZATION.items():
        sample['diarization'][name] = {'settings': {**settings, 'step_seconds': 10 * settings['step_ratio']},
            'parameters': deepcopy(p.base.PARAMETERS), 'regular': deepcopy(timeline), 'exclusive': deepcopy(timeline),
            'filter_diagnostics': [{'min_active_ratio': .1, 'selected_embeddings': 1, 'chunk_speaker_pairs': 2}]}
    for mode in p.ALIGNMENTS:
        sample['alignments'][mode], sample['alignment_windows'][mode] = p.realign(records, [0] * int(duration*p.base.RATE),
            lambda b, _: {'segments': [{**s, 'words': []} for s in b]}, mode)
    for mode, definition in p.MODES.items():
        sample['outputs'][mode] = p.base.attribute(sample['asr'], sample['alignments'][definition['alignment']], timeline, duration)
    return sample


class PauseContextTests(unittest.TestCase):
    def test_tail_can_reach_speech_after_decoder_end_without_changing_other_sentence(self):
        original = native(); before = deepcopy(original)
        control = p.alignment_bounds(original, 6.3, 'control')
        extended = p.alignment_bounds(original, 6.3, 'extend_tail')
        self.assertLess(control[-1]['end'], 5)
        self.assertEqual(extended[-1]['end'], 6.3)
        self.assertEqual(extended[-1]['start'], control[-1]['start'])
        self.assertEqual(extended[:-1], control[:-1])
        self.assertEqual(''.join(s['text'] for s in extended), original[0]['text'])
        self.assertEqual(original, before)

    def test_repeated_phrases_are_jointly_aligned_once_in_order(self):
        rows = p.alignment_bounds(native(), 6.3, 'joint_window')
        self.assertEqual(rows, [{'start': 0., 'end': 6.3, 'text': native()[0]['text']}])
        # No independently overlapping windows that could both find the first "Go".
        self.assertEqual(len(rows), 1)

    def test_multiple_decoder_segments_are_joined_without_losing_punctuation(self):
        rows = native() + [{'text': ' Done.', 'words': [{'word': ' Done.', 'start': 5., 'end': 5.3}]}]
        before = deepcopy(rows)
        joined = p.alignment_bounds(rows, 6., 'joint_window')
        self.assertEqual(joined[0]['text'], ' Go there! Go there! Done.')
        self.assertEqual(rows, before)

    def test_silence_stays_empty_and_invalid_context_or_tokens_fail(self):
        for mode in p.ALIGNMENTS:
            self.assertEqual(p.alignment_bounds([], 3, mode), [])
        for duration in (0, -1, 15.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError): p.alignment_bounds(native(), duration, 'joint_window')
        with self.assertRaises(ValueError): p.alignment_bounds(native(), 6., 'manual')
        rows = native(); rows[0]['words'][0]['word'] = 'oops'
        with self.assertRaises(ValueError): p.alignment_bounds(rows, 6., 'joint_window')

    def test_historical_control_unchanged_and_reused_without_extra_inference(self):
        bounds = p.base.sentence_bounds(native(), 6.)
        self.assertEqual(p.alignment_bounds(native(), 6., 'control'), bounds)
        win = {'start_frame': 2 * p.base.RATE, 'end_frame': 8 * p.base.RATE}
        local = {'segments': [{'start': .4, 'end': 5., 'text': native()[0]['text'],
                              'words': [{'word': 'Go', 'start': .4, 'end': .7}]}]}
        record = {'window': win, 'native': native(), 'bounds': bounds, 'aligned_local': local}
        with patch('builtins.print'):
            out, records = p.realign([record], [0] * 8 * p.base.RATE,
                lambda *_: self.fail('Control must reuse its saved alignment'), 'control')
        self.assertEqual(out['segments'][0]['start'], 2.4)
        self.assertEqual(out['segments'][0]['words'][0]['start'], 2.4)
        self.assertEqual(records[0]['aligned_local'], local)
        self.assertEqual(local['segments'][0]['start'], .4)

    def test_broader_audio_is_passed_and_offset_added_exactly_once(self):
        win = {'start_frame': 2 * p.base.RATE, 'end_frame': 8 * p.base.RATE}
        record = {'window': win, 'native': native()}
        calls = []
        def align(bounds, chunk):
            calls.append((deepcopy(bounds), len(chunk)))
            word = {'word': 'there!', 'start': 5., 'end': 5.8}
            return {'segments': [{'start': 5., 'end': 5.8, 'text': bounds[0]['text'], 'words': [word]}], 'word_segments': [word]}
        out, records = p.realign([record], [0] * 8 * p.base.RATE, align, 'joint_window')
        self.assertEqual(calls, [([{'start': 0., 'end': 6., 'text': native()[0]['text']}], 6 * p.base.RATE)])
        self.assertEqual(out['segments'][0]['words'][0]['end'], 7.8)
        self.assertEqual(records[0]['aligned_local']['word_segments'][0]['end'], 5.8)

    def test_isolation_and_reference_hints_rejected(self):
        with patch.dict(p.os.environ, {}, clear=True):
            with self.assertRaises(ValueError): p.require_test()
        env = {'FULL_PIPELINE_IMAGE': '1', 'FULL_PIPELINE_TEST': '1', 'PAUSE_CONTEXT_TEST': '1'}
        with patch.dict(p.os.environ, env, clear=True):
            p.require_test()
            for hint in ('words', 'num_speakers', 'start', 'reference'):
                with self.assertRaises(ValueError): p.validate_input({'profile': p.PROFILE, 'samples': [], hint: 1})
            with patch.dict(p.os.environ, {'ACOUSTIC_TURNS_TEST': '1'}):
                with self.assertRaises(ValueError): p.require_test()

    def test_saved_context_and_export_replay_detect_tampering(self):
        sample = sample_evidence(); before = deepcopy(sample)
        p.validate_sample(sample)
        self.assertEqual(sample, before)
        changed = deepcopy(sample)
        changed['alignment_windows']['joint_window'][0]['bounds'][0]['start'] = 2.
        with self.assertRaises(ValueError): p.validate_sample(changed)
        changed = deepcopy(sample)
        changed['outputs']['joint_window_short']['rawText'] = 'silently rewritten'
        with self.assertRaises(ValueError): p.validate_sample(changed)

    def test_changed_local_audio_clock_and_missing_variant_rejected(self):
        sample = sample_evidence()
        sample['alignment_windows']['joint_window'][0]['aligned_local']['segments'][0]['end'] = 7.
        with self.assertRaises(ValueError): p.validate_sample(sample)
        sample = sample_evidence(); del sample['alignments']['control']
        with self.assertRaises(ValueError): p.validate_sample(sample)


if __name__ == '__main__': unittest.main()
