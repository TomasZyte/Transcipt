from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import channel_asr as c


def evidence(channel, text, words):
    segment = {'text': text, 'start': 0., 'end': 5., 'words': [
        {'word': word, 'start': start, 'end': end} for word, start, end in words]}
    return {'channel': channel, 'raw': {'language': 'ru', 'segments': [{'text': text}]},
            'aligned': {'segments': [segment]}}


def sample():
    return [evidence('left', 'Да,  да! Потом.', [('Да,', .2, .4), ('да!', 1.5, 1.8), ('Потом.', 3., 3.4)]),
            evidence('right', 'Ответ.', [('Ответ.', 1., 1.6)])]


class ChannelMergeTests(unittest.TestCase):
    def test_interleaves_turns_preserves_overlap_repetitions_and_original_evidence(self):
        rows = sample()
        before = deepcopy(rows)
        result = c.merge_channels(rows, 5 * c.RATE)
        self.assertEqual(rows, before)
        self.assertEqual(result['evidence'], before)
        turns = result['candidate_segments']
        self.assertEqual([t['text'] for t in turns], ['Да,', 'Ответ.', 'да!', 'Потом.'])
        self.assertEqual([t['speaker'] for t in turns], ['Спикер 1', 'Спикер 2', 'Спикер 1', 'Спикер 1'])
        self.assertEqual(turns[1]['endSec'], 1.6)
        self.assertEqual(turns[2]['startSec'], 1.5)
        self.assertEqual(len(result['cross_channel_word_overlaps']), 1)
        self.assertFalse(result['release_ready'])
        self.assertFalse(result['quality_validated'])

    def test_numbering_is_channel_based_even_if_right_speaks_first(self):
        rows = sample()
        rows[1]['aligned']['segments'][0]['words'][0].update(start=.1, end=.3)
        turns = c.merge_channels(rows, 5 * c.RATE)['candidate_segments']
        self.assertEqual(turns[0]['speaker'], 'Спикер 2')
        self.assertEqual(turns[1]['speaker'], 'Спикер 1')

    def test_no_deduplication_of_possible_echo(self):
        rows = [evidence(channel, 'Да!', [('Да!', 1., 1.2)]) for channel in c.CHANNELS]
        result = c.merge_channels(rows, 5 * c.RATE)
        self.assertEqual([t['text'] for t in result['candidate_segments']], ['Да!', 'Да!'])
        self.assertEqual(len(result['cross_channel_word_overlaps']), 1)

    def test_preserves_punctuation_and_internal_spacing(self):
        rows = [evidence('left', '«Да,  да!»', [('Да,', .1, .3), ('да!', .4, .6)]), sample()[1]]
        self.assertEqual(c.merge_channels(rows, 5 * c.RATE)['candidate_segments'][0]['text'], '«Да,  да!»')

    def test_bad_alignment_retains_evidence_and_suppresses_candidate(self):
        for change in (
            lambda s: s['words'][0].pop('start'),
            lambda s: s['words'][0].update(start=float('nan')),
            lambda s: s['words'][0].update(start=True),
            lambda s: s['words'][0].update(start=-1),
            lambda s: s['words'][0].update(end=6),
            lambda s: s['words'][0].update(end=3),
            lambda s: s['words'][1].update(start=.1),
            lambda s: s['words'].pop(0),
            lambda s: s.update(text='Другая фраза.'),
            lambda s: s.update(words=[]),
        ):
            with self.subTest(change=change):
                rows = sample()
                change(rows[0]['aligned']['segments'][0])
                result = c.merge_channels(rows, 5 * c.RATE)
                self.assertEqual(result['status'], 'review_required')
                self.assertIsNone(result['candidate_segments'])
                self.assertEqual(result['evidence'][0]['raw'], rows[0]['raw'])

    def test_empty_channel_is_visible_not_a_successful_two_speaker_result(self):
        rows = sample()
        rows[1]['raw']['segments'] = []
        rows[1]['aligned']['segments'] = []
        result = c.merge_channels(rows, 5 * c.RATE)
        self.assertEqual(result['issues'][0]['reason'], 'channel_without_transcript')
        self.assertIsNone(result['candidate_segments'])

    def test_json_replay_produces_same_result(self):
        rows = sample()
        self.assertEqual(c.merge_channels(rows, 5 * c.RATE),
                         c.merge_channels(json.loads(json.dumps(rows)), 5 * c.RATE))

    def test_flag_lengths_and_confirmation_checked_before_inference(self):
        callback = Mock(side_effect=AssertionError('must not execute'))
        options = dict(transcribe=callback, align=callback, confirmation={
            'kind': 'owner_preview', 'start_seconds': 0, 'end_seconds': .5})
        with patch.dict(os.environ, {'CHANNEL_ASR_TEST': '0'}), self.assertRaises(ValueError):
            c.decode_channels(range(c.RATE), range(c.RATE), **options)
        with patch.dict(os.environ, {'CHANNEL_ASR_TEST': '1'}):
            for left, right in (([], []), (range(10), range(9)), (range(241*c.RATE), range(241*c.RATE))):
                with self.assertRaises(ValueError): c.decode_channels(left, right, **options)
            for confirmation in ({}, {'kind': 'stereo_detected', 'start_seconds': 0, 'end_seconds': .5}):
                with self.assertRaises(ValueError):
                    c.decode_channels(range(c.RATE), range(c.RATE), **{**options, 'confirmation': confirmation})
        callback.assert_not_called()

    def test_callbacks_use_full_channels_same_settings_and_no_reference_text(self):
        left, right = range(5*c.RATE), range(5*c.RATE, 10*c.RATE)
        rows, calls = sample(), []
        def transcribe(audio, **options):
            calls.append((audio, options))
            return deepcopy(rows[0 if audio is left else 1]['raw'])
        def align(segments, audio):
            segments[0]['text'] = 'mutated callback input'
            return deepcopy(rows[0 if audio is left else 1]['aligned'])
        with patch.dict(os.environ, {'CHANNEL_ASR_TEST': '1'}):
            result = c.decode_channels(left, right, transcribe=transcribe, align=align,
                confirmation={'kind': 'owner_preview', 'start_seconds': 0, 'end_seconds': 3})
        self.assertIs(calls[0][0], left)
        self.assertIs(calls[1][0], right)
        self.assertEqual(calls[0][1], {'language': 'ru', 'batch_size': 8, 'task': 'transcribe', 'chunk_size': 30})
        self.assertEqual(calls[0][1], calls[1][1])
        self.assertEqual(result['evidence'][0]['raw'], rows[0]['raw'])
        self.assertFalse(result['full_call_speaker_identity_verified'])


class ChannelInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / 'input.json'
        self.data = {'profile': c.PROFILE, 'source_sha256': 'f'*64,
                     'confirmation': {'kind': 'owner_preview', 'start_seconds': 0, 'end_seconds': 1},
                     'channels': []}
        for i, channel in enumerate(c.CHANNELS):
            path = self.root / f'{channel}.wav'
            with wave.open(str(path), 'wb') as wav:
                wav.setparams((1, 2, c.RATE, 0, 'NONE', 'not compressed'))
                wav.writeframes(bytes((i, 0)) * c.RATE)
            self.data['channels'].append({'channel': channel, 'file': path.name,
                'sha256': c.digest(path), 'source_start_seconds': 0})

    def prepare(self):
        self.manifest.write_text(json.dumps(self.data), encoding='utf-8')
        return c.prepare(self.manifest)

    def test_prepare_is_offline_and_counts_both_channels(self):
        result = self.prepare()
        self.assertEqual(result['status'], 'prepared_not_run')
        self.assertEqual(result['total_asr_audio_seconds'], 2.)
        self.assertEqual(result['paid_jobs'], 0)
        self.assertTrue(result['cloud_runner_connected'])

    def test_rejects_drift_truncation_unequal_lengths_and_format(self):
        path = self.root / 'right.wav'
        original = path.read_bytes()
        path.write_bytes(original[:-2])
        with self.assertRaises(ValueError): self.prepare()
        self.data['channels'][1]['sha256'] = c.digest(path)
        with self.assertRaises(ValueError): self.prepare()
        for rate, channels, count in ((c.RATE, 1, c.RATE-1), (8000, 1, 8000), (c.RATE, 2, c.RATE)):
            with wave.open(str(path), 'wb') as wav:
                wav.setparams((channels, 2, rate, 0, 'NONE', 'not compressed'))
                wav.writeframes(b'\0\0' * count * channels)
            self.data['channels'][1]['sha256'] = c.digest(path)
            with self.assertRaises(ValueError): self.prepare()

    def test_rejects_trimmed_swapped_duplicate_and_injected_inputs(self):
        original = deepcopy(self.data)
        for change in (
            lambda d: d['channels'][0].update(source_start_seconds=1),
            lambda d: d['channels'].reverse(),
            lambda d: d['channels'][1].update(file='left.wav', sha256=d['channels'][0]['sha256']),
            lambda d: d['channels'][0].update(file='../left.wav'),
            lambda d: d.update(prompt='expected words'),
            lambda d: d['confirmation'].update(end_seconds=32),
        ):
            self.data = deepcopy(original)
            change(self.data)
            with self.assertRaises(ValueError): self.prepare()


if __name__ == '__main__':
    unittest.main()
