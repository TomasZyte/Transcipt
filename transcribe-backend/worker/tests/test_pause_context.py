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
    def test_local_voice_then_silence_is_not_a_speaker_change(self):
        turns = [{'start': 0., 'end': 1.3, 'speaker': 'A'}, {'start': 4., 'end': 5., 'speaker': 'B'}]
        def build(*, extra_regular=(), last_start=.8, tail_start=.8, single_word=False):
            sample = {'diarization': {}, 'alignments': {}, 'outputs': {}}
            for track in p.DIARIZATION:
                rows = deepcopy(turns)
                sample['diarization'][track] = {'regular': sorted(rows + list(extra_regular), key=lambda r:r['start'])
                                                         if track == 'short' else rows,
                                                 'exclusive': deepcopy(turns)}
            for mode, config in p.MODES.items():
                control = config['alignment'] == 'control'
                words = [] if single_word else [{'word': 'Уже', 'start': .2, 'end': .5}]
                words.append({'word': 'поздно.', 'start': last_start if control else tail_start,
                              'end': 1. if control else 3.5})
                seg = {'start': 0., 'end': words[-1]['end'], 'text': ' '.join(w['word'] for w in words),
                       'words': words}
                aligned = {'segments': [seg]}
                sample['alignments'][config['alignment']] = aligned
                sample['outputs'][mode] = p.base.attribute(aligned, aligned,
                    sample['diarization'][config['diarization']]['regular'], 5.)
            return sample
        sample = build()
        for v in (1, 2):
            self.assertEqual(p.phrase_speaker_review(sample, version=v)[0]['speakerNote'], p.UNRESOLVED_SPEAKER)
        self.assertEqual(p.phrase_speaker_review(sample)[0]['speakerSupport'], 'same_local_voice_before_silence')
        for params in [
            {'extra_regular': [{'start': 1.27, 'end': 1.31, 'speaker': 'B'}]},  # background voice only in regular
            {'extra_regular': [{'start': 1.4, 'end': 1.7, 'speaker': 'A'}]},  # voice returns after silence
            {'tail_start': 1.1},  # not the same supported onset
            {'single_word': True},  # no preceding words to establish a voice
        ]:
            with self.subTest(params=params):
                self.assertEqual(p.phrase_speaker_review(build(**params))[0]['speakerNote'], p.UNRESOLVED_SPEAKER)

    def test_single_voice_abstention_is_resolved_only_with_local_speech_support(self):
        words = [{'word': 'До', 'start': .2, 'end': .5, 'speaker': 'A'},
                 {'word': 'паузы', 'start': .6, 'end': 1.1, 'speaker': p.UNKNOWN},
                 {'word': 'после', 'start': 1.2, 'end': 1.4, 'speaker': 'A'}]
        seg = {'text': 'До паузы после', 'start': 0., 'end': 2., 'words': words}
        track = [{'start': 0., 'end': .8, 'speaker': 'A'}, {'start': 1., 'end': 2., 'speaker': 'A'}]
        sample = {'alignments': {'control': {'segments': [deepcopy(seg)]}},
                  'diarization': {d: {'regular': deepcopy(track), 'exclusive': deepcopy(track)} for d in p.DIARIZATION},
                  'outputs': {m: {'attributed': {'segments': [deepcopy(seg)]}} for m in p.MODES}}
        def decision():
            return p.phrase_speaker_review(sample)[0]
        self.assertEqual(p.phrase_speaker_review(sample, version=1)[0]['speakerNote'], p.UNRESOLVED_SPEAKER)
        self.assertEqual(decision()['speakerSupport'], 'single_voice_across_short_pause')
        for d in sample['diarization'].values():
            d['regular'][1]['start'] = 1.15
        self.assertEqual(decision()['speakerNote'], p.UNRESOLVED_SPEAKER)  # voice no longer covers both edges
        for d in p.DIARIZATION:
            w = sample['outputs']['control_'+d]['attributed']['segments'][0]['words'][1]
            w.update(speaker='A', end=.75)
        self.assertEqual(decision()['speakerSupport'], 'same_word_confirmed_on_both_tracks')
        before = deepcopy(sample)
        sample['outputs']['extend_tail_dense']['attributed']['segments'][0]['words'][0]['speaker'] = p.UNKNOWN
        self.assertEqual(decision()['speakerNote'], p.UNRESOLVED_SPEAKER)  # two unsupported words
        sample = deepcopy(before)
        sample['outputs']['joint_window_dense']['attributed']['segments'][0]['words'][1].update(start=.82,end=.98)
        self.assertEqual(decision()['speakerNote'], p.UNRESOLVED_SPEAKER)  # no speech at all
        sample = deepcopy(before)
        sample['diarization']['dense']['regular'].append({'start': .7,'end': .8,'speaker': 'B'})
        self.assertEqual(decision()['speakerNote'], p.UNRESOLVED_SPEAKER)  # another voice only on overlapping track
        sample = deepcopy(before)
        sample['outputs']['joint_window_dense']['attributed']['segments'][0]['words'][1]['word'] = 'иное'
        self.assertEqual(decision()['speakerNote'], p.UNRESOLVED_SPEAKER)  # incomplete word match

    def test_phrase_speakers_follow_occurrences_across_renamed_and_joined_turns(self):
        sample = {'alignments': {'control': {'segments': [{'text': ' Да!'}, {'text': 'Да!'}]}},
                  'diarization': {}, 'outputs': {}}
        for mode, ids in [('dense', ('A', 'B')), ('short', ('B', 'A'))]:
            sample['diarization'][mode] = {'exclusive': [
                {'start': i, 'end': i+1, 'speaker': label} for i, label in enumerate(ids)]}
        for name, definition in p.MODES.items():
            speaker = 'A' if definition['diarization'] == 'dense' else 'B'
            sample['outputs'][name] = {'attributed': {'segments': [
                {'text': 'Да! Да!', 'words': [{'word': 'Да!', 'speaker': speaker},
                                              {'word': 'Да!', 'speaker': speaker}]}]}}
        def notes():
            return [row['speakerNote'] for row in p.phrase_speaker_review(sample)]
        self.assertEqual(notes(), [None, None])
        joined = sample['outputs']['joint_window_short']['attributed']['segments'][0]
        for label in ('A', p.UNKNOWN, None):
            joined['words'][1]['speaker'] = label
            self.assertEqual(notes(), [None, p.UNRESOLVED_SPEAKER])
        joined['words'].pop()
        self.assertEqual(notes(), [p.UNRESOLVED_SPEAKER] * 2)

    def test_merged_half_and_half_voice_cannot_be_matched_by_name(self):
        tracks = {'dense': {'exclusive': [{'start': 0., 'end': 1., 'speaker': 'A'},
                                         {'start': 1., 'end': 2., 'speaker': 'B'}]},
                  'short': {'exclusive': [{'start': 0., 'end': 2., 'speaker': 'A'}]}}
        self.assertEqual(p._speaker_maps(tracks), {'dense': {}, 'short': {}})

    def test_missing_words_and_tampered_speaker_overlay(self):
        sample = sample_evidence()
        sample['phraseSpeakerReview'] = p.phrase_speaker_review(sample)
        self.assertTrue(all(row['speakerNote'] == p.UNRESOLVED_SPEAKER for row in sample['phraseSpeakerReview']))
        p.validate_sample(sample)
        sample['phraseSpeakerReview'][0]['speakerNote'] = None
        with self.assertRaisesRegex(ValueError, 'Phrase speaker review'):
            p.validate_sample(sample)

    def test_phrase_note_follows_the_correct_occurrence_across_sentence_splits(self):
        def word(start, end):
            return {'word': 'Да!', 'start': start, 'end': end}

        control = [{'text': ' Да!', 'start': a, 'end': b, 'words': [word(a, b)]}
                   for a, b in [(0., .2), (1., 1.3)]]
        joined = [{'text': ' Да! Да!', 'start': 0., 'end': 2.,
                   'words': [word(.02, .21), word(1.6, 1.9)]}]
        records = {mode: [{'window': {'start_frame': p.base.RATE * 4},
                   'aligned_local': {'segments': joined if mode == 'joint_window' else control}}]
                   for mode in p.ALIGNMENTS}
        sample = {'alignment_windows': records}
        original = deepcopy(sample)
        rows = p.phrase_timing_review(sample)
        self.assertEqual([(r['text'], r['startSec'], r['endSec'], r['timingNote']) for r in rows],
                         [(' Да!', 4., 4.2, None), (' Да!', 5., 5.3, p.APPROXIMATE_TIMING)])
        self.assertEqual(rows[1]['reasons'], ['timings_disagree'])
        self.assertEqual(sample, original)
        del joined[0]['words'][1]['end']
        self.assertEqual(p.phrase_timing_review(sample)[1]['reasons'], ['missing_or_invalid_time'])
        joined[0]['words'].pop(0)
        self.assertTrue(all(r['reasons'] == ['incomplete_word_comparison'] for r in p.phrase_timing_review(sample)))

    def test_long_and_missing_times_but_not_small_jitter_or_speaker_disagreement(self):
        self.assertEqual(p._timing_issue([(0., 2.1)] * 3), 'long_word')
        self.assertIsNone(p._timing_issue([(0., .04), (.08, .12), (.10, .14)]))
        words = [{'word': 'тест', 'start': 0., 'end': .5, 'speaker': 'A'}]
        other = [{**words[0], 'speaker': 'B'}]
        def times(group):
            return p._word_times([{'words': group, 'start': 0., 'end': 1.}])[1][0]
        self.assertIsNone(p._timing_issue([times(words), times(other), times(words)]))
        self.assertIsNone(times([{**words[0], 'end': 1.5}]))

    def test_legacy_sample_and_forged_timing_review(self):
        sample = sample_evidence()
        p.validate_sample(sample)
        sample['phraseTimingReview'] = p.phrase_timing_review(sample)
        self.assertTrue(all(r['timingNote'] == p.APPROXIMATE_TIMING for r in sample['phraseTimingReview']))
        p.validate_sample(sample)
        sample['phraseTimingReview'][0]['timingNote'] = None
        with self.assertRaisesRegex(ValueError, 'Phrase timing review'):
            p.validate_sample(sample)

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
