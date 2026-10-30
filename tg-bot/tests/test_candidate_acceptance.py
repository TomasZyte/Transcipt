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


if __name__ == '__main__': unittest.main()
