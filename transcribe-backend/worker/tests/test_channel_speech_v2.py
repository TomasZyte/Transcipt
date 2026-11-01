from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import channel_speech_v2 as v2
import channel_pipeline as worker
import full_pipeline_handler as router
import test_channel_asr as inputs
import test_channel_pipeline as payloads


class RecognitionRecoveryTests(unittest.TestCase):
    def decode(self, *, module=v2, empty=False, weak=True, count=2):
        self.recovery_calls, self.alignment_calls = [], []
        audio = range((4*count+1)*v2.base.RATE)
        def transcribe(audio, windows):
            return {'language': 'ru', 'segments': [
                {'start': w['start_frame']/v2.base.RATE, 'end': w['end_frame']/v2.base.RATE,
                 'text': 'Тихая реплика.'} for w in windows]}
        def recover(chunk):
            self.recovery_calls.append(chunk)
            # Fixture native times are relative to the complete context audio.
            segments = []
            for i in range(count):
                a = (1+4*i)-chunk.start/v2.base.RATE
                if 0 <= a < len(chunk)/v2.base.RATE:
                    segments.append({'text': 'Внятный ответ.', 'start': a, 'end': a+.5,
                        'avg_logprob': -.2, 'no_speech_prob': .1,
                        'words': [{'word': 'Внятный', 'start': a, 'end': a+.2},
                                  {'word': 'ответ.', 'start': a+.25, 'end': a+.5}]})
            return {'language': 'ru', 'segments': [] if empty else segments}
        transcribe.recover = recover
        def align(segments, chunk):
            self.alignment_calls.append((deepcopy(segments), chunk))
            segment = deepcopy(segments[0])
            score = .05 if weak and segment['text'] == 'Тихая реплика.' else .9
            segment['words'] = [{'word': word, 'start': .05+.2*i, 'end': .15+.2*i, 'score': score}
                                for i, word in enumerate(segment['text'].split())]
            return {'segments': [segment]}
        with patch.dict(os.environ, {'CHANNEL_ASR_TEST': '1'}):
            return module.decode_channels(audio, audio, detect=lambda a: [
                {'start': 1+4*i, 'end': 1.6+4*i} for i in range(count)],
                transcribe=transcribe, align=align,
                confirmation={'kind': 'owner_preview', 'start_seconds': 0, 'end_seconds': len(audio)/v2.base.RATE})

    def test_valid_times_no_longer_skip_weak_recognition(self):
        old = self.decode(module=v2.speech)
        self.assertEqual(self.recovery_calls, [])
        self.assertEqual(old['issues'], [])
        new = self.decode()
        self.assertTrue(self.recovery_calls)
        self.assertEqual(len(new['quality_trigger_evidence']), 4)
        self.assertIsNone(new['candidate_segments'])
        self.assertFalse(new['release_ready'])
        self.assertTrue(all(r['status'] == 'hypothesis_requires_review' for r in new['review_spans']))
        self.assertEqual(new['original'], old['original'])

    def test_context_never_reaches_alignment_and_offset_is_applied_once(self):
        result = self.decode()
        calls = [(s, c) for s, c in self.alignment_calls if s[0]['text'] == 'Внятный ответ.']
        self.assertEqual(len(calls), 4)
        windows = [r['window'] for c in result['original']['run_evidence'] for r in c['runs']]
        for (segments, chunk), window in zip(calls, windows):
            self.assertEqual((chunk.start, chunk.stop), (window['start_frame'], window['end_frame']))
            self.assertEqual(segments[0]['start'], 0.)
            self.assertLess(len(chunk)/v2.base.RATE, 1.)
        word = result['evidence'][0]['aligned']['segments'][1]['words'][0]
        self.assertAlmostEqual(word['start'], windows[1]['start_frame']/v2.base.RATE+.05)

    def test_empty_second_pass_keeps_evidence_and_explicit_unresolved_spans(self):
        result = self.decode(empty=True)
        self.assertEqual(result['evidence'], result['original']['evidence'])
        self.assertTrue(all(r['hypothesis'] is None and r['status'] == 'unresolved_speech'
                            for r in result['review_spans']))
        self.assertIsNone(result['candidate_segments'])

    def test_repeated_good_confidence_is_not_deleted_or_retried(self):
        result = self.decode(weak=False)
        self.assertEqual(self.recovery_calls, [])
        self.assertEqual(result['quality_trigger_evidence'], [])
        self.assertEqual(result['evidence'], result['original']['evidence'])
        self.assertFalse(result['release_ready'])

    def test_missing_scores_and_single_words_are_not_misread_as_weak_phrases(self):
        result = self.decode()
        for missing in ('score', 'word'):
            evidence = deepcopy(result['original'])
            for channel in evidence['run_evidence']:
                for row in channel['runs']:
                    if missing == 'score':
                        row['aligned_local']['segments'][0]['words'][0].pop('score')
                    else:
                        row['raw_local']['segments'][0]['text'] = 'Да.'
            self.assertEqual(v2.quality_triggers(evidence), [])

    def test_no_phrase_blacklist_or_owner_reference_is_needed(self):
        original = self.decode()['original']
        before = deepcopy(original)
        expected = v2.quality_triggers(original)
        for channel in original['run_evidence']:
            for row in channel['runs']:
                row['raw_local']['segments'][0]['text'] = 'Другие слова!'
        self.assertEqual(v2.quality_triggers(original), expected)
        self.assertEqual(v2.quality_triggers(before), expected)

    def test_compute_cap_fails_before_any_recovery_callback(self):
        with self.assertRaisesRegex(ValueError, 'bounded compute allowance'):
            self.decode(count=20)
        self.assertEqual(self.recovery_calls, [])

    def test_replay_rejects_changed_triggers_and_dropped_recovery(self):
        result = self.decode()
        plan = {'profile': v2.PROFILE, 'frames_per_channel': 9*v2.base.RATE,
                'confirmation': result['confirmation']}
        self.assertEqual(v2.rebuild(result, plan), result)
        state = {'profile': v2.PROFILE, 'status': 'completed', 'input_plan': plan,
                 'source_manifest_sha256': worker.digest(worker.SOURCES),
                 'human_quality_validated': False, 'result': result}
        worker.validate_result(state, plan)
        for change in (lambda r: r['quality_trigger_evidence'].clear(),
                       lambda r: r['recovery_evidence'].pop(),
                       lambda r: r.update(candidate_segments=[])):
            bad = deepcopy(state); change(bad['result'])
            with self.assertRaises(ValueError): worker.validate_result(bad, plan)

    def test_flags_are_checked_before_audio_and_unknown_seeds_rejected(self):
        cb = Mock()
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            v2.decode_channels([0], [0], detect=cb, transcribe=cb, align=cb, confirmation={})
        cb.assert_not_called()
        original = self.decode()['original']
        for seed in (('third', 0), ('left', -1), ('left', True)):
            with self.assertRaises(ValueError):
                v2.repair.region_plan(original, 9*v2.base.RATE, extra_seeds=[seed])

    def test_review_export_retains_all_windows_and_new_quality_reasons(self):
        import channel_review
        result = self.decode()
        plan = {'profile': v2.PROFILE, 'frames_per_channel': 9*v2.base.RATE,
                'duration_seconds': 9, 'confirmation': result['confirmation'],
                'channels': [], 'source_sha256': 'a'*64}
        state = {'profile': v2.PROFILE, 'status': 'completed', 'input_plan': plan,
                 'source_manifest_sha256': worker.digest(worker.SOURCES),
                 'human_quality_validated': False, 'result': result}
        control_result = v2.base.merge_channels(result['original']['evidence'], plan['frames_per_channel'])
        control_result.update(confirmation=plan['confirmation'], full_call_speaker_identity_verified=False)
        control = {**deepcopy(state), 'profile': v2.base.PROFILE, 'result': control_result,
                   'input_plan': {**plan, 'profile': v2.base.PROFILE}}
        report = channel_review.build(state, control)
        self.assertEqual(len(report['rows']), 4)
        self.assertTrue(all('weak_phrase_alignment' in row['reasons'] for row in report['rows']))
        self.assertFalse(report['production_export_allowed'])


class IntegrationTests(unittest.TestCase):
    setUp = inputs.ChannelInputTests.setUp

    def test_manifest_and_test_route_accept_v2_but_reject_hints_and_disabled_flags(self):
        self.data['profile'] = v2.PROFILE
        self.manifest.write_text(json.dumps(self.data), encoding='utf-8')
        plan = v2.prepare(self.manifest)
        self.assertEqual(plan['speech_alignment_policy']['recognition_selection'], v2.SELECTION)
        self.assertIs(worker.adapter_for(v2.PROFILE), v2)
        self.data['reference'] = 'Приватная правка'
        self.manifest.write_text(json.dumps(self.data), encoding='utf-8')
        with self.assertRaises(ValueError): v2.prepare(self.manifest)
        payload = payloads.ChannelWorkerTests().payload()
        payload['profile'] = v2.PROFILE
        worker.validate_input(payload, payloads.HOST)
        with patch.dict(os.environ, {}, clear=True), patch.object(worker, 'fetch_channel') as fetch:
            result = router.handler({'input': payload})
        self.assertEqual(result['phase'], 'validate')
        fetch.assert_not_called()


if __name__ == '__main__': unittest.main()
