from copy import deepcopy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import channel_repair as r
import test_channel_asr as fixture


class RecoveryTests(unittest.TestCase):
    def decode(self, native_text='Ответ.', no_speech=.05, logprob=-.2, empty=False, bad_alignment=False):
        # Two valid runs plus the two regression types. No reference is sent.
        audio = range(15*r.RATE)
        self.calls = []
        def transcribe(audio, rows):
            return {'language': 'ru', 'segments': [
                {'start': w['start_frame']/r.RATE, 'end': w['end_frame']/r.RATE,
                 'text': text} for w, text in zip(rows, ('Да.', 'Длинное.', 'Невыровнено.'))]}
        def recover(chunk):
            self.calls.append(chunk)
            return {'language': 'ru', 'segments': [] if empty else [
                {'start': .5, 'end': 1., 'text': native_text, 'no_speech_prob': no_speech,
                 'avg_logprob': logprob, 'words': [{'word': native_text, 'start': .5, 'end': 1., 'probability': .9}]}]}
        transcribe.recover = recover
        def align(segments, chunk):
            s = deepcopy(segments[0]); text = s['text']
            s['words'] = [] if text == 'Невыровнено.' or (bad_alignment and text == native_text) else [
                {'word': text, 'start': .1, 'end': 2.5 if text == 'Длинное.' else .4}]
            return {'segments': [s], 'word_segments': s['words']}
        with patch.dict(os.environ, {'CHANNEL_ASR_TEST': '1'}):
            return r.decode_channels(audio, audio,
                detect=lambda a: [{'start': 1, 'end': 2}, {'start': 5, 'end': 8}, {'start': 13, 'end': 13.4}],
                transcribe=transcribe, align=align,
                confirmation={'kind': 'owner_preview', 'start_seconds': 0, 'end_seconds': 15})

    def test_long_word_and_missing_alignment_recovered_without_changing_control(self):
        result = self.decode()
        self.assertEqual(len(self.calls), 4)
        self.assertTrue(result['original']['issues'])
        self.assertFalse(result['issues'])
        self.assertEqual(result['status'], 'candidate_unvalidated')
        self.assertFalse(result['release_ready'])
        self.assertTrue(all(c['text_changed'] and not c['human_verified'] for c in result['changes']))
        self.assertEqual(result['original']['run_evidence'][0]['runs'][1]['aligned_local']['segments'][0]['words'][0]['end'], 2.5)
        word = result['evidence'][0]['aligned']['segments'][1]['words'][0]
        # global region 2.88 + native crop .4 + CTC word .1, exactly once.
        self.assertAlmostEqual(word['start'], 3.38)
        self.assertAlmostEqual(word['end'], 3.68)

    def test_empty_uncertain_or_unaligned_recovery_keeps_review_gate(self):
        for options, reason in (({'empty': True}, 'recovery_without_transcript'),
                                ({'no_speech': .95}, 'uncertain_recognition'),
                                ({'logprob': -1.2}, 'uncertain_recognition'),
                                ({'bad_alignment': True}, 'missing_word_times')):
            with self.subTest(options=options):
                result = self.decode(**options)
                self.assertIsNone(result['candidate_segments'])
                self.assertTrue(any(i['reason'] == reason for i in result['issues']))
                self.assertTrue(result['original']['run_evidence'])

    def test_no_phrase_blacklist_or_manual_replacement(self):
        result = self.decode(native_text='Продолжение следует...')
        self.assertEqual(result['changes'][0]['after'], 'Продолжение следует...')
        self.assertIn('Продолжение следует...', [s['text'] for s in result['candidate_segments']])

    def test_intersecting_context_includes_neighbour_in_full_and_merges_regions(self):
        result = self.decode()
        original = result['original']
        # Larger generic context makes the two repairs intersect, and takes a
        # healthy neighbour whole. No duplication and no half-neighbour drop.
        with patch.dict(r.POLICY, {'context_seconds': 4.}):
            regions = r.region_plan(original, 15*r.RATE)
        self.assertEqual(len(regions), 2)
        self.assertEqual(regions[0]['replaced_runs'], [0, 1, 2])
        self.assertEqual(regions[0]['trigger_runs'], [1, 2])

    def test_compute_cap_checked_before_recovery(self):
        for limit in ({'max_regions': 1}, {'max_total_seconds': 1.}, {'max_region_seconds': 1.}):
            with patch.dict(r.POLICY, limit), self.assertRaises(ValueError): self.decode()
            self.assertEqual(self.calls, [])

    def test_rebuild_detects_missing_duplicate_or_changed_native_evidence(self):
        result = self.decode()
        plan = {'frames_per_channel': 15*r.RATE, 'confirmation': result['confirmation']}
        self.assertEqual(r.rebuild(result, plan), result)
        for change in (lambda a: a['recovery_evidence'].pop(),
                       lambda a: a['recovery_evidence'].append(a['recovery_evidence'][0]),
                       lambda a: a['recovery_evidence'][0]['native']['segments'][0].update(text='Подмена'),
                       lambda a: a['recovery_evidence'][0]['phrases'][0]['window'].update(start_frame=0),
                       lambda a: a['original']['issues'].clear()):
            bad = deepcopy(result); change(bad)
            with self.assertRaises(ValueError): r.rebuild(bad, plan)

    def test_bad_native_boundaries_rejected_not_clamped_into_validity(self):
        for a, b in ((-1, .2), (1, .1), (0, 1.03), (True, 1), (0, float('nan'))):
            with self.assertRaises(ValueError):
                r.alignment_window({'start': a, 'end': b, 'text': 'Да.'}, r.RATE)

    def test_guard_precedes_models(self):
        callback = Mock()
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            r.decode_channels([0], [0], detect=callback, transcribe=callback, align=callback, confirmation={})
        callback.assert_not_called()


class ManifestTests(unittest.TestCase):
    setUp = fixture.ChannelInputTests.setUp

    def test_separate_profile_accepts_only_original_audio_without_hints(self):
        import json
        self.data['profile'] = r.PROFILE
        self.manifest.write_text(json.dumps(self.data), encoding='utf-8')
        plan = r.prepare(self.manifest)
        self.assertEqual(plan['recovery_decoder']['initial_prompt'], None)
        self.assertFalse(plan['repair_policy']['reference_hints'])
        self.data['reference'] = 'Не передавать модели'
        self.manifest.write_text(json.dumps(self.data), encoding='utf-8')
        with self.assertRaises(ValueError): r.prepare(self.manifest)


if __name__ == '__main__': unittest.main()
