from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from candidate_acceptance import evaluate, phrase_matches


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        output = {'attributed': {'segments': [{'words': [
            {'word': 'Hello,', 'start': 4., 'end': 4.2, 'speaker': 'C'},
            {'word': 'world!', 'start': 4.3, 'end': 4.8, 'speaker': 'C'}]}]},
            'rawAsrText': 'Hello, world!', 'quality': {'issues': {}}}
        self.result = {'status': 'completed', 'profile': 'release-candidate-v1',
            'modes': {'control30': {'diarization': 'short'}}, 'samples': [
                {'id': 'example', 'sha256': 'abc', 'duration_seconds': 10,
                 'outputs': {'control30': output}, 'diarization': {'short': {'regular': [
                     {'speaker': 'C'}, {'speaker': 'A'}]}}}]}
        self.refs = {'purpose': 'evaluation_only', 'samples': {'example': {
            'audio_sha256': 'abc', 'phrases': [{'id': 'child', 'text': 'hello world',
                'seconds': [4, 5], 'forbidden_speakers': ['A']}]}}}

    def check(self):
        return evaluate(self.result, self.refs)['rows'][0]['checks'][0]

    def test_passed_reference_does_not_claim_production_readiness(self):
        report = evaluate(self.result, self.refs)
        self.assertTrue(self.check()['pass'])
        self.assertFalse(report['release_ready'])
        self.assertTrue(report['rows'][0]['manual_review_required'])

    def test_early_alignment_and_wrong_adult_label_are_independent_failures(self):
        words = self.result['samples'][0]['outputs']['control30']['attributed']['segments'][0]['words']
        words[0]['start'] = 2
        self.assertTrue(self.check()['text_present'])
        self.assertFalse(self.check()['timing_in_reference_window'])
        self.assertTrue(self.check()['separated_from_forbidden_speakers'])
        self.assertFalse(self.check()['pass'])
        words[0]['start'] = 4
        words[0]['speaker'] = 'A'
        self.assertTrue(self.check()['timing_in_reference_window'])
        self.assertFalse(self.check()['separated_from_forbidden_speakers'])
        words[0]['speaker'] = 'SPEAKER_UNKNOWN'
        self.assertFalse(self.check()['pass'])

    def test_missing_text_and_changed_audio_block_acceptance(self):
        self.refs['samples']['example']['phrases'][0]['text'] = 'another phrase'
        self.assertFalse(self.check()['text_present'])
        self.refs['samples']['example']['audio_sha256'] = 'different'
        with self.assertRaises(ValueError): self.check()

    def test_repeated_phrase_uses_correct_occurrence_without_mutation(self):
        before = deepcopy(self.result)
        words = self.result['samples'][0]['outputs']['control30']['attributed']['segments'][0]['words']
        self.assertEqual(len(phrase_matches(words+words, 'hello world')), 2)
        evaluate(self.result, self.refs)
        self.assertEqual(self.result, before)

    def test_new_mode_uses_its_own_labels_and_has_no_implicit_control(self):
        sample = self.result['samples'][0]
        sample['outputs']['timed_dense'] = sample['outputs'].pop('control30')
        self.result['modes'] = {'timed_dense': {'diarization': 'short'}}
        self.result['profile'] = 'timed-turns-v1'
        phrase = self.refs['samples']['example']['phrases'][0]
        phrase['forbidden_speakers_by_mode'] = {'timed_dense': ['C']}
        report = evaluate(self.result, self.refs)
        self.assertFalse(report['rows'][0]['checks'][0]['pass'])
        self.assertIsNone(report['rows'][0]['text_changed_from_control'])
        phrase['forbidden_speakers_by_mode'] = {'other': ['A']}
        with self.assertRaises(ValueError): evaluate(self.result, self.refs)

    def test_different_occurrences_cannot_combine_into_false_pass(self):
        output = self.result['samples'][0]['outputs']['control30']
        early = deepcopy(output['attributed']['segments'][0]['words'])
        for word in early:
            word['start'] -= 3; word['end'] -= 3
        for word in output['attributed']['segments'][0]['words']: word['speaker'] = 'A'
        output['attributed']['segments'].insert(0, {'words': early})
        self.refs['samples']['example']['phrases'][0]['timing_precision'] = 'approximate'
        check = self.check()
        self.assertTrue(check['timing_in_reference_window'])
        self.assertTrue(check['separated_from_forbidden_speakers'])
        self.assertFalse(check['pass'])
        self.assertEqual(len(check['observed_matches']), 2)
        self.assertEqual(check['timing_precision'], 'approximate')


if __name__ == '__main__': unittest.main()
