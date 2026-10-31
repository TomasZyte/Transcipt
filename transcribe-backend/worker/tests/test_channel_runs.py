from copy import deepcopy
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import channel_runs as r
import channel_pipeline as worker
import test_channel_asr as fixture


class RunWindowTests(unittest.TestCase):
    def test_long_silence_stays_between_runs_and_small_gap_keeps_phrase(self):
        rows = r.windows([{'start': .1, 'end': 1}, {'start': 1.3, 'end': 2},
                          {'start': 10, 'end': 11}], 12*r.RATE)
        self.assertEqual([(w['start_frame'], w['end_frame']) for w in rows],
                         [(0, 33920), (158080, 177920)])
        self.assertEqual(len(rows), 2)

    def test_equal_threshold_is_split_and_padding_never_overlaps(self):
        rows = r.windows([{'start': 0, 'end': 1}, {'start': 1.4, 'end': 2}], 2*r.RATE)
        self.assertEqual(len(rows), 2)
        self.assertLess(rows[0]['end_frame'], rows[1]['start_frame'])
        self.assertEqual(rows[-1]['end_frame'], 2*r.RATE)

    def test_duration_cap_preserves_all_frames_without_overlapping_copies(self):
        rows = r.windows([{'start': 0, 'end': 65}], 65*r.RATE)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(w['duration_cap_split'] for w in rows))
        self.assertEqual(sum(w['end_frame']-w['start_frame'] for w in rows), 65*r.RATE)
        self.assertEqual([w['start_frame'] for w in rows], [0, 30*r.RATE, 60*r.RATE])

    def test_empty_invalid_reversed_overlapping_and_excessive_runs(self):
        self.assertEqual(r.windows([], 100), [])
        for rows in ([{'start': True, 'end': 1}], [{'start': 1, 'end': float('nan')}],
                     [{'start': 1, 'end': .5}], [{'start': 0, 'end': 2.2}],
                     [{'start': 1, 'end': 2}, {'start': .5, 'end': 1}],
                     [{'start': 0, 'end': 1.5}, {'start': 1, 'end': 2}]):
            with self.assertRaises(ValueError): r.windows(rows, 2*r.RATE)
        with self.assertRaises(ValueError):
            r.windows([{'start': i, 'end': i+.1} for i in range(161)], 162*r.RATE)


class RunDecodeTests(unittest.TestCase):
    def decode(self, alter=None):
        audio = range(12*r.RATE)
        self.align_calls = []
        def transcribe(a, rows):
            return {'language': 'ru', 'segments': [
                {'start': round(w['start_frame']/r.RATE, 3), 'end': round(w['end_frame']/r.RATE, 3),
                 'text': 'Да, да!', 'avg_logprob': -.2} for w in rows]}
        def align(segments, chunk):
            self.align_calls.append((deepcopy(segments), chunk))
            segment = deepcopy(segments[0])
            # Alias intentionally mimics WhisperX; it must not double-offset.
            segment['words'] = [{'word': 'Да,', 'start': .15, 'end': .3},
                                {'word': 'да!', 'start': .4, 'end': .6}]
            result = {'segments': [segment], 'word_segments': segment['words']}
            if alter: alter(result)
            segments[0]['text'] = 'callback mutation must not affect ASR evidence'
            return result
        with patch.dict(os.environ, {'CHANNEL_ASR_TEST': '1'}):
            return r.decode_channels(audio, audio, detect=lambda a: [{'start': 0, 'end': 1}, {'start': 10, 'end': 11}],
                transcribe=transcribe, align=align,
                confirmation={'kind': 'owner_preview', 'start_seconds': 0, 'end_seconds': 12})

    def test_alignment_uses_each_run_audio_and_offsets_once(self):
        result = self.decode()
        self.assertEqual(len(self.align_calls), 4)
        self.assertEqual(self.align_calls[1][1].start, 158080)
        self.assertEqual(self.align_calls[1][0][0]['start'], 0)
        self.assertAlmostEqual(result['evidence'][0]['aligned']['segments'][1]['words'][0]['start'], 10.03)
        self.assertEqual(result['status'], 'candidate_unvalidated')
        self.assertEqual([s['text'] for s in result['evidence'][0]['raw']['segments']], ['Да, да!', 'Да, да!'])
        self.assertFalse(result['release_ready'])

    def test_escape_from_local_run_suppresses_candidate_even_inside_full_file(self):
        result = self.decode(lambda a: a['segments'][0]['words'][0].update(end=1.5))
        self.assertEqual(result['status'], 'review_required')
        self.assertIsNone(result['candidate_segments'])
        self.assertTrue(any(i['reason'] == 'unreliable_word_time' for i in result['issues']))
        self.assertEqual(result['run_evidence'][0]['runs'][0]['aligned_local']['segments'][0]['words'][0]['end'], 1.5)

    def test_missing_word_text_change_and_long_words_keep_review_gate(self):
        for change in (lambda a: a['segments'][0]['words'][0].pop('start'),
                       lambda a: a['segments'][0].update(text='Другое'),
                       lambda a: a['segments'][0]['words'][0].update(end=4)):
            result = self.decode(change)
            self.assertIsNone(result['candidate_segments'])

    def test_replay_rejects_tampered_windows_text_or_global_export(self):
        result = self.decode()
        plan = {'profile': r.PROFILE, 'frames_per_channel': 12*r.RATE, 'confirmation': result['confirmation']}
        self.assertEqual(r.rebuild(result, plan), result)
        for change in (lambda x: x['run_evidence'][0]['runs'][0]['window'].update(start_frame=1),
                       lambda x: x['run_evidence'][0]['runs'][0]['raw_local']['segments'][0].update(text='Подмена')):
            bad = deepcopy(result); change(bad)
            with self.assertRaises(ValueError): r.rebuild(bad, plan)
        state = {'profile': r.PROFILE, 'status': 'completed', 'input_plan': plan,
                 'source_manifest_sha256': worker.digest(worker.SOURCES), 'human_quality_validated': False, 'result': result}
        worker.validate_result(state, plan)
        state['result']['evidence'][0]['aligned']['segments'][0]['words'][0]['start'] += .1
        with self.assertRaises(ValueError): worker.validate_result(state, plan)

    def test_known_millisecond_rounding_is_allowed_without_editing_times_or_relaxing_long_words(self):
        result = self.decode()
        evidence = deepcopy(result['run_evidence'])
        # One-sample-exact crop whose end rounds upwards at millisecond precision.
        evidence[0]['speech'][0]['end'] = .9995625
        planned = r.windows(evidence[0]['speech'], 12*r.RATE)
        run = evidence[0]['runs'][0]
        run['window'] = planned[0]
        duration = (planned[0]['end_frame']-planned[0]['start_frame'])/r.RATE
        run['raw_local']['segments'][0]['end'] = duration
        run['decoder_segment_global']['end'] = round(duration, 3)
        run['aligned_local']['segments'][0]['words'][-1]['end'] = round(duration, 3)
        before = deepcopy(evidence)
        modern = r.assemble(evidence, 12*r.RATE, result['confirmation'])
        legacy = r.assemble(evidence, 12*r.RATE, result['confirmation'], assembly_version=1)
        self.assertEqual(modern['status'], 'candidate_unvalidated')
        self.assertEqual(legacy['status'], 'review_required')
        self.assertEqual(evidence, before)
        plan = {'frames_per_channel': 12*r.RATE, 'confirmation':result['confirmation']}
        self.assertEqual(r.rebuild(legacy, plan), legacy)
        self.assertEqual(r.rebuild(modern, plan), modern)
        run['aligned_local']['segments'][0]['words'][-1]['end'] += .001
        self.assertIsNone(r.assemble(evidence, 12*r.RATE, result['confirmation'])['candidate_segments'])
        item = {'channel':'left','raw':{'language':'ru','segments':[{'text':'Слово.'}]},
                'aligned':{'segments':[{'text':'Слово.','words':[{'word':'Слово.','start':0.,'end':2.0001}]}]}}
        with self.assertRaisesRegex(ValueError, 'long_word_requires_review'):
            r.base._tokens(item, 3., time_tolerance=.0005)

    def test_missing_guard_stops_before_any_callback(self):
        callback = Mock()
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            r.decode_channels([0], [0], detect=callback, transcribe=callback, align=callback, confirmation={})
        callback.assert_not_called()

    def test_no_speech_is_not_a_successful_dialogue(self):
        with patch.dict(os.environ, {'CHANNEL_ASR_TEST': '1'}):
            align = Mock()
            result = r.decode_channels(range(r.RATE), range(r.RATE), detect=lambda a: [],
                transcribe=lambda a, w: {'language': 'ru', 'segments': []}, align=align,
                confirmation={'kind': 'owner_preview', 'start_seconds': 0, 'end_seconds': 1})
        self.assertIsNone(result['candidate_segments']); align.assert_not_called()

    def test_fixed_vad_returns_runs_without_merging_and_restores_model_on_error(self):
        class Vad:
            def __init__(self, onset): pass
        original = object()
        model = SimpleNamespace(vad_model=original)
        def transcribe(audio, **kw):
            fixed = model.vad_model
            self.assertEqual(fixed.preprocess_audio(audio), audio)
            segments = fixed({'waveform': audio})
            self.assertEqual(fixed.merge_chunks(segments, 30), segments)
            self.assertEqual(len(segments), 2)
            raise RuntimeError('injected failure')
        model.transcribe = transcribe
        modules = {'torch': SimpleNamespace(), 'whisperx.vads.vad': SimpleNamespace(Vad=Vad),
                   'whisperx.vads.pyannote': SimpleNamespace(Binarize=Mock())}
        with patch.dict(sys.modules, modules), patch.object(r, 'verify_api'):
            detect, run = r.callbacks(model, Mock())
            with self.assertRaises(RuntimeError):
                run([], r.windows([{'start': 0, 'end': 1}, {'start': 10, 'end': 11}], 12*r.RATE))
        self.assertIs(model.vad_model, original)


class RunManifestTests(unittest.TestCase):
    setUp = fixture.ChannelInputTests.setUp

    def test_profile_cannot_reuse_old_campaign_and_retains_audio_contract(self):
        import json
        self.data['profile'] = r.PROFILE
        self.manifest.write_text(json.dumps(self.data), encoding='utf-8')
        plan = r.prepare(self.manifest)
        self.assertEqual(plan['speech_run_policy'], r.POLICY)
        with self.assertRaises(ValueError): r.base.prepare(self.manifest)
        self.data['channels'][0]['source_start_seconds'] = 1
        self.manifest.write_text(json.dumps(self.data), encoding='utf-8')
        with self.assertRaises(ValueError): r.prepare(self.manifest)

    def test_new_profile_executes_through_isolated_worker_with_pinned_models(self):
        import json
        self.data['profile'] = r.PROFILE
        self.manifest.write_text(json.dumps(self.data), encoding='utf-8')
        model = Mock()
        def transcribe(audio, windows):
            return {'language': 'ru', 'segments': [{'start': 0., 'end': 1., 'text': 'Да.'}]}
        def align(segments, am, metadata, chunk, device, **options):
            return {'segments': [{**segments[0], 'words': [{'word': 'Да.', 'start': .1, 'end': .3}]}]}
        wx = SimpleNamespace(load_model=Mock(return_value=model), load_align_model=Mock(return_value=(None, {})),
            load_audio=Mock(side_effect=[range(r.RATE), range(r.RATE)]), align=Mock(side_effect=align))
        modules = {'torch': SimpleNamespace(manual_seed=Mock()), 'whisperx': wx,
                   'whisperx.vads.pyannote': SimpleNamespace(load_vad_model=Mock())}
        with patch.dict(os.environ, worker.FLAGS, clear=True), patch.dict(sys.modules, modules), \
             patch.object(worker, 'verify_assets', return_value={'root': '/fake', 'vad_path': '/fake/vad'}), \
             patch.object(worker, 'verify_bundle', return_value={}), patch.object(worker, 'verify_nltk', return_value='a'*64), \
             patch.object(worker, 'version', side_effect=lambda name: worker.config_for('wx_ru')['packages'][name]), \
             patch.object(r, 'callbacks', return_value=(lambda a: [{'start': 0, 'end': 1}], transcribe)):
            result = worker.execute(self.manifest, self.root/'result.json')
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['profile'], r.PROFILE)
        self.assertEqual(result['result']['status'], 'candidate_unvalidated')
        self.assertEqual(wx.align.call_count, 2)
        self.assertEqual(wx.load_model.call_count, 1)


if __name__ == '__main__': unittest.main()
