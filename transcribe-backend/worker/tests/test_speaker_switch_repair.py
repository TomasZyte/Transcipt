from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import speaker_switch_repair as repair
import pause_context as pause


def fixture():
    words = [{'word':'До','start':.2,'end':.5},
             {'word':'середины','start':.8,'end':1.4},
             {'word':'дальше','start':1.5,'end':1.8}]
    segment = {'text':'До середины дальше','start':.2,'end':1.8,'words':words}
    sample = {'duration_seconds':3.5,
              'asr':{'language':'ru','segments':[deepcopy(segment)]},
              'windows':[{'window':{'start_frame':0,'end_frame':56000},
                          'native':[deepcopy(segment)]}],
              'alignments':{m:{'segments':[deepcopy(segment)]} for m in pause.ALIGNMENTS},
              'diarization':{},'outputs':{}}
    for track,end in [('dense',1.2),('short',1.4)]:
        rows = [{'start':0.,'end':1.,'speaker':'A'},
                {'start':1.,'end':end,'speaker':'B'},
                {'start':end,'end':2.,'speaker':'A'},
                {'start':2.5,'end':3.5,'speaker':'B'}]
        sample['diarization'][track] = {k:deepcopy(rows) for k in ('regular','exclusive')}
    rebuild(sample)
    return sample


def rebuild(sample):
    for mode,definition in pause.MODES.items():
        sample['outputs'][mode] = pause.base.attribute(sample['asr'],sample['alignments'][definition['alignment']],
            sample['diarization'][definition['diarization']]['regular'],sample['duration_seconds'])


class SwitchRepairTests(unittest.TestCase):
    def test_in_word_island_is_repaired_without_rewriting_words_times_or_other_turns(self):
        sample = fixture(); before = deepcopy(sample)
        result,audit = repair.propose(sample)
        self.assertEqual(sample,before)
        self.assertEqual(audit['applied_pairs'],1)
        for key in ('windows','asr','alignments'):
            self.assertEqual(result[key],sample[key])
        for mode in pause.MODES:
            self.assertEqual(result['outputs'][mode]['rawAsrText'],sample['outputs'][mode]['rawAsrText'])
            self.assertEqual(result['outputs'][mode]['attributed']['segments'][0]['words'][1]['speaker'],'A')
        for track in ('dense','short'):
            self.assertEqual(result['diarization'][track]['regular'][-1],sample['diarization'][track]['regular'][-1])
        with self.assertRaisesRegex(ValueError,'original evidence'):
            repair.propose(result)

    def test_even_very_short_whole_word_reply_is_preserved(self):
        for duration in (.07,.12,.16):
            with self.subTest(duration=duration):
                sample = fixture()
                for target in [sample['windows'][0]['native'][0]] + [
                        sample['alignments'][m]['segments'][0] for m in pause.ALIGNMENTS]:
                    target['words'][1].update(start=1.02,end=1.02+duration)
                rebuild(sample)
                result,audit = repair.propose(sample)
                self.assertEqual(audit['applied_pairs'],0)
                self.assertEqual(audit['decisions'][0]['rejection'],'supported_short_reply')
                self.assertEqual(result['outputs'],sample['outputs'])

    def test_short_reply_supported_by_two_alignments_survives_bad_native_time(self):
        sample = fixture()
        for mode in ('control','extend_tail'):
            sample['alignments'][mode]['segments'][0]['words'][1].update(start=1.02,end=1.18)
        rebuild(sample)
        _,audit = repair.propose(sample)
        self.assertEqual(audit['decisions'][0]['rejection'],'supported_short_reply')

    def test_real_turn_change_without_return_to_previous_voice_is_not_smoothed(self):
        sample = fixture()
        for track in sample['diarization'].values():
            for kind in ('regular','exclusive'):
                track[kind][2]['speaker'] = 'C'
        rebuild(sample)
        result,audit = repair.propose(sample)
        self.assertEqual(audit['applied_pairs'],0)
        self.assertEqual(result['outputs'],sample['outputs'])

    def test_overlapping_background_voice_is_protected_even_if_exclusive_hides_it(self):
        sample = fixture()
        sample['diarization']['short']['regular'].append({'start':1.05,'end':1.15,'speaker':'C'})
        rebuild(sample)
        _,audit = repair.propose(sample)
        self.assertEqual(audit['applied_pairs'],0)
        self.assertEqual(audit['decisions'][0]['rejection'],'overlap_or_uncovered_speech')

    def test_pause_long_turn_short_flank_and_unmatched_onsets_block_repair(self):
        for change in ('pause','long','flank','offset'):
            with self.subTest(change=change):
                sample = fixture()
                for kind in ('regular','exclusive'):
                    rows = sample['diarization']['short'][kind]
                    if change == 'pause':
                        rows[0]['end'] = .98
                    elif change == 'long':
                        rows[1]['end'] = rows[2]['start'] = 1.6
                    elif change == 'flank':
                        rows[0]['start'] = .9
                    else:
                        rows[0]['end'] = rows[1]['start'] = 1.1
                rebuild(sample)
                _,audit = repair.propose(sample)
                self.assertEqual(audit['applied_pairs'],0)

    def test_a_phrase_boundary_inside_the_switch_is_preserved(self):
        sample = fixture()
        for alignment in sample['alignments'].values():
            original = alignment['segments'][0]
            first = {'text':'До середины','start':.2,'end':1.3,'words':deepcopy(original['words'][:2])}
            first['words'][1]['end'] = 1.3
            last = {'text':'дальше','start':1.31,'end':1.8,'words':[deepcopy(original['words'][2])]}
            last['words'][0]['start'] = 1.31
            alignment['segments'] = [first,last]
        rebuild(sample)
        _,audit = repair.propose(sample)
        self.assertEqual(audit['applied_pairs'],0)
        self.assertEqual(audit['decisions'][0]['rejection'],'phrase_boundary')

    def test_same_text_with_different_speaker_id_names_has_same_decision(self):
        sample = fixture()
        for kind in ('regular','exclusive'):
            for row in sample['diarization']['short'][kind]:
                row['speaker'] = {'A':'X','B':'Y'}[row['speaker']]
        rebuild(sample)
        result,audit = repair.propose(sample)
        self.assertEqual(audit['applied_pairs'],1)
        for mode in ('control_short','extend_tail_short','joint_window_short'):
            self.assertEqual(result['outputs'][mode]['attributed']['segments'][0]['words'][1]['speaker'],'X')

    def test_decoder_hyphen_split_is_matched_by_text_position(self):
        sample = fixture()
        for segment in [sample['asr']['segments'][0]] + [
                sample['alignments'][m]['segments'][0] for m in pause.ALIGNMENTS]:
            segment['text']='До Ну-ка дальше'; segment['words'][1]['word']='Ну-ка'
        native = sample['windows'][0]['native'][0]
        native['text']='До Ну-ка дальше'
        native['words'][1:2] = [{'word':'Ну','start':.8,'end':1.1},{'word':'-ка','start':1.1,'end':1.4}]
        rebuild(sample)
        _,audit = repair.propose(sample)
        self.assertEqual(audit['applied_pairs'],1)
        self.assertEqual(audit['decisions'][0]['word_support']['word'],'Ну-ка')

    def test_missing_or_changed_word_evidence_never_enables_repair(self):
        for edit in ('different_word','missing_time','onset_at_edge'):
            with self.subTest(edit=edit):
                sample = fixture()
                if edit == 'different_word':
                    sample['windows'][0]['native'][0]['words'][1]['word']='иное'
                elif edit == 'missing_time':
                    for mode in ('control','extend_tail'):
                        sample['alignments'][mode]['segments'][0]['words'][1].pop('start')
                else:
                    sample['windows'][0]['native'][0]['words'][1]['start']=1.
                _,audit = repair.propose(sample)
                self.assertEqual(audit['applied_pairs'],0)

    def test_policy_rejects_unknown_or_nonfinite_parameters(self):
        for policy in ({**repair.POLICY,'reference':'child'},
                       {**repair.POLICY,'max_island_seconds':float('nan')},
                       {**repair.POLICY,'min_crossing_alignment_modes':1}):
            with self.assertRaises(ValueError):
                repair.propose(fixture(),policy=policy)


if __name__ == '__main__':
    unittest.main()
