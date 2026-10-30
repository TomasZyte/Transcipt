from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import transcript_candidate as c
import candidate_pipeline as p
from transcript_segments import build_segments


def aligned(words):
    return {'segments': [{'start': 0., 'end': 5., 'text': ' '.join(w['word'] for w in words), 'words': words}]}


class CandidatePolicyTests(unittest.TestCase):
    def test_pause_plan_covers_every_frame_once_and_keeps_short_tail(self):
        for frames in (1, 15*c.RATE, 15*c.RATE+1, 31*c.RATE, 180*c.RATE):
            plan = c.pause_windows([], frames)
            self.assertEqual(plan[0]['start_frame'], 0)
            self.assertEqual(plan[-1]['end_frame'], frames)
            self.assertEqual(sum(w['end_frame']-w['start_frame'] for w in plan), frames)
            self.assertTrue(all(0 < w['end_frame']-w['start_frame'] <= 15*c.RATE for w in plan))
            for left, right in zip(plan, plan[1:]):
                self.assertEqual(left['end_frame'], right['start_frame'])
            if frames > 2*c.RATE:
                self.assertGreaterEqual(plan[-1]['end_frame']-plan[-1]['start_frame'], 2*c.RATE)

    def test_pauses_depend_on_speech_union_not_speaker_ids(self):
        timeline = [{'start': 0., 'end': 3., 'speaker': 'A'}, {'start': 4., 'end': 7., 'speaker': 'B'},
                    {'start': 5., 'end': 8., 'speaker': 'C'}]
        plan = c.pause_windows(timeline, 10*c.RATE)
        self.assertEqual(plan[0]['end_frame'], int(3.5*c.RATE))
        renamed = [{**r, 'speaker': 'same'} for r in timeline]
        self.assertEqual(plan, c.pause_windows(renamed, 10*c.RATE))

    def test_invalid_timeline_and_frame_counts_rejected(self):
        for frames in (0, -1, True, 1., 7200*c.RATE+1):
            with self.assertRaises(ValueError): c.pause_windows([], frames)
        for row in ({'start': float('nan'), 'end': 2., 'speaker': 'A'},
                    {'start': 0., 'end': 7., 'speaker': 'A'},
                    {'start': 0., 'end': 1., 'speaker': ''}):
            with self.assertRaises(ValueError): c.pause_windows([row], 5*c.RATE)

    def test_local_global_offset_applied_once_and_json_replay_identical(self):
        word = {'word': 'тест', 'start': .2, 'end': .8}
        source = {'segments': [{'start': .2, 'end': .8, 'text': 'тест', 'words': [word]}], 'word_segments': [word]}
        shifted = c.global_alignment(source, 12.)
        self.assertEqual(shifted['segments'][0]['words'][0]['start'], 12.2)
        self.assertEqual(shifted['word_segments'][0]['start'], 12.2)
        self.assertEqual(source['word_segments'][0]['start'], .2)
        self.assertEqual(shifted, c.global_alignment(json.loads(json.dumps(source)), 12.))

    def test_ambiguous_overlap_is_unknown_and_not_a_new_numbered_person(self):
        words = [{'word': 'Раз,', 'start': .1, 'end': .8}, {'word': 'два!', 'start': 1., 'end': 1.6}]
        source = aligned(words)
        timeline = [{'start': 0., 'end': 2., 'speaker': 'A'}, {'start': 0., 'end': .9, 'speaker': 'B'}]
        result = c.attribute(source, source, timeline, 5.)
        self.assertEqual([s['speaker'] for s in result['segments']], ['Спикер не определён', 'Спикер 1'])
        self.assertEqual(result['quality']['issues'], {'ambiguous_speaker_overlap': 1})
        self.assertNotIn('speaker', source['segments'][0]['words'][0])
        self.assertEqual(result, c.attribute(json.loads(json.dumps(source)), source, timeline, 5.))

    def test_long_word_or_uncovered_word_never_gets_a_confident_speaker(self):
        source = aligned([{'word': 'горка', 'start': .1, 'end': 4.5}])
        timeline = [{'start': 0., 'end': 5., 'speaker': 'A'}]
        out = c.attribute(source, source, timeline, 5.)
        self.assertEqual(out['segments'][0]['speaker'], 'Спикер не определён')
        self.assertEqual(out['quality']['issues'], {'long_word': 1})
        source = aligned([{'word': 'тест', 'start': .1, 'end': .7}])
        out = c.attribute(source, source, [], 5.)
        self.assertEqual(out['quality']['issues'], {'insufficient_speaker_coverage': 1})

    def test_duplicate_same_speaker_overlap_does_not_inflate_coverage(self):
        source = aligned([{'word': 'тест', 'start': 0., 'end': 1.}])
        timeline = [{'start': 0., 'end': .4, 'speaker': 'A'}] * 3
        out = c.attribute(source, source, timeline, 5.)
        self.assertEqual(out['segments'][0]['speaker'], 'Спикер не определён')

    def test_bad_alignment_text_and_out_of_range_segment_fail(self):
        source = aligned([{'word': 'тест', 'start': .1, 'end': .8}])
        bad = deepcopy(source); bad['segments'][0]['text'] = 'другая фраза'
        with self.assertRaises(ValueError): c.attribute(source, bad, [], 5.)
        bad = deepcopy(source); bad['segments'][0]['end'] = 10
        with self.assertRaises(ValueError): c.attribute(source, bad, [], 5.)

    def test_decode_pause_uses_exact_pcm_slices_and_keeps_empty_windows(self):
        audio = list(range(10*c.RATE))
        timeline = [{'start': 0., 'end': 3., 'speaker': 'A'}, {'start': 4., 'end': 9., 'speaker': 'A'}]
        seen = []
        def transcribe(chunk):
            seen.extend(chunk)
            return {'language': 'ru', 'segments': []}
        raw, align, evidence = p.decode_pause(audio, timeline, transcribe=transcribe,
                                               align=lambda segments, chunk: {'segments': []})
        self.assertEqual(seen, audio)
        self.assertEqual(len(evidence), 2)
        self.assertEqual(raw['segments'], [])
        self.assertEqual(align['segments'], [])

    def test_flags_and_reference_injection_block_before_download(self):
        inp = {'profile': p.PROFILE, 'samples': []}
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(p.handler({'input': inp})['phase'], 'validate')
        with patch.dict(os.environ, {'FULL_PIPELINE_IMAGE': '1', 'FULL_PIPELINE_TEST': '1',
                                     'RELEASE_CANDIDATE_TEST': '1'}, clear=True):
            with self.assertRaises(ValueError): p.validate_input({**inp, 'num_speakers': 4})

    def test_full_evidence_replay_rejects_export_config_and_offset_drift(self):
        raw = {'language': 'ru', 'segments': [{'start': 0., 'end': 5., 'text': 'тест'}]}
        ali = aligned([{'word': 'тест', 'start': .2, 'end': .8}])
        timeline = [{'start': 0., 'end': 4.9, 'speaker': 'A'}]
        sample = {'duration_seconds': 5., 'asr': {'control': raw, 'pause': deepcopy(raw)},
                  'alignment': {'control': ali, 'pause': deepcopy(ali)}, 'diarization': {}, 'outputs': {},
                  'windows': [{'window': c.pause_windows(timeline, 5*c.RATE)[0],
                               'raw_local': deepcopy(raw), 'aligned_local': deepcopy(ali)}]}
        for key, settings in p.DIARIZATION.items():
            params = deepcopy(p.PARAMETERS); params['clustering']['threshold'] = settings['threshold']
            sample['diarization'][key] = {'regular': timeline, 'exclusive': timeline, 'parameters': params,
                'filter_diagnostics': [{'min_active_ratio': settings['ratio'], 'selected_embeddings': 2, 'chunk_speaker_pairs': 3}]}
        for key in p.MODES:
            sample['outputs'][key] = c.attribute(raw, ali, timeline, 5.)
        p.validate_sample(sample)
        p.validate_sample(json.loads(json.dumps(sample)))
        for edit in (
            lambda x: x['outputs']['pause_short'].update(rawText='подмена'),
            lambda x: x['windows'][0]['aligned_local']['segments'][0].update(start=.3),
            lambda x: x['diarization']['merged']['parameters']['clustering'].update(threshold=.8)):
            bad = deepcopy(sample); edit(bad)
            with self.assertRaises(ValueError): p.validate_sample(bad)


if __name__ == '__main__': unittest.main()
