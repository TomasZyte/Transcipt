"""Offline regressions: no model downloads, credentials or GPU required."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipeline
from transcript_segments import build_segments, split_speaker_turns


def example():
    return {"start": 0, "end": 4, "text": "Да, согласен. Нет, подождите!",
            "speaker": "SPEAKER_00", "words": [
                {"word": "Да,", "start": 0, "end": .5, "speaker": "SPEAKER_00"},
                {"word": "согласен.", "start": .5, "end": 1.5, "speaker": "SPEAKER_00"},
                {"word": "Нет,", "start": 2, "end": 2.5, "speaker": "SPEAKER_01"},
                {"word": "подождите!", "start": 2.5, "end": 4, "speaker": "SPEAKER_01"}]}


class SpeakerTurns(unittest.TestCase):
    def test_speaker_change_inside_sentence_is_preserved(self):
        segments, stats = build_segments([example()])
        self.assertEqual([(s["speaker"], s["text"]) for s in segments], [
            ("Спикер 1", "Да, согласен."), ("Спикер 2", "Нет, подождите!")])
        self.assertEqual([(s["startSec"], s["endSec"]) for s in segments], [(0, 1.5), (2, 4)])
        self.assertEqual(stats["splitSegments"], 1)
        self.assertIn("[Спикер 2]: Нет, подождите!", pipeline._build_srt(segments))

    def test_returning_speaker_keeps_identity_across_segments(self):
        third = {"start": 5, "end": 6, "speaker": "SPEAKER_00", "text": "Хорошо."}
        out, _ = build_segments([example(), third])
        self.assertEqual([s["speaker"] for s in out], ["Спикер 1", "Спикер 2", "Спикер 1"])
        self.assertEqual([s["id"] for s in out], ["seg-1", "seg-2", "seg-3"])

    def test_repeated_words_and_punctuation_not_dropped(self):
        seg = {"start": 0, "end": 3, "text": '«Да», да — да!', "words": [
            {"word": "Да", "start": 0, "end": .5, "speaker": "a"},
            {"word": "да", "start": 1, "end": 1.5, "speaker": "b"},
            {"word": "да", "start": 2, "end": 3, "speaker": "a"}]}
        out, reason = split_speaker_turns(seg)
        self.assertIsNone(reason)
        self.assertEqual([s["text"] for s in out], ['«Да»,', 'да —', 'да!'])
        self.assertEqual([s["speaker"] for s in out], ['a', 'b', 'a'])

    def test_cjk_text_does_not_acquire_spaces(self):
        seg = {"start": 0, "end": 2, "text": "你好世界", "words": [
            {"word": "你好", "start": 0, "end": 1, "speaker": "a"},
            {"word": "世界", "start": 1, "end": 2, "speaker": "b"}]}
        out, _ = split_speaker_turns(seg)
        self.assertEqual("".join(s["text"] for s in out), seg["text"])

    def test_missing_aligned_word_keeps_whole_original(self):
        seg = example()
        del seg["words"][1]
        out, reason = split_speaker_turns(seg)
        self.assertEqual(reason, "text_mismatch")
        self.assertEqual([s["text"] for s in out], [seg["text"]])

    def test_untimed_number_inside_turn_is_retained(self):
        seg = example()
        seg["text"] = "Да, 2026 согласен. Нет, подождите!"
        seg["words"].insert(1, {"word": "2026"})
        out, stats = build_segments([seg])
        self.assertEqual(out[0]["text"], "Да, 2026 согласен.")
        self.assertEqual(len(out), 2)
        self.assertEqual(stats["wordsWithoutSpeaker"], 1)

    def test_unknown_leading_word_follows_first_known_speaker(self):
        seg = example()
        del seg["words"][0]["speaker"]
        out, _ = split_speaker_turns(seg)
        self.assertEqual([s["speaker"] for s in out], ["SPEAKER_00", "SPEAKER_01"])

    def test_speaker_change_without_time_does_not_invent_boundary(self):
        seg = example()
        del seg["words"][2]["start"]
        out, reason = split_speaker_turns(seg)
        self.assertEqual(reason, "untimed_speaker_change")
        self.assertEqual(len(out), 1)

    def test_invalid_word_time_falls_back_with_reason(self):
        for value in (float("nan"), float("inf"), -1, 9):
            with self.subTest(value=value):
                seg = example()
                seg["words"][2]["start"] = value
                out, reason = split_speaker_turns(seg)
                self.assertTrue(reason)
                self.assertEqual(len(out), 1)

    def test_one_speaker_preserves_original_whitespace(self):
        seg = example()
        seg["text"] = seg["text"].replace(" ", "  ")
        for word in seg["words"]:
            word["speaker"] = "b"
        out, reason = split_speaker_turns(seg)
        self.assertIsNone(reason)
        self.assertEqual(out[0]["text"], seg["text"])
        self.assertEqual(out[0]["speaker"], "b")

    def test_does_not_mutate_model_result(self):
        seg = example()
        before = deepcopy(seg)
        build_segments([seg])
        self.assertEqual(seg, before)


class PipelineOutput(unittest.TestCase):
    def run_pipeline(self, *, align_error=False, diar_error=False, token="test", enabled=True):
        source = example()
        asr = SimpleNamespace(transcribe=lambda *args, **kwargs: {"segments": [source], "language": "ru"})
        wx = SimpleNamespace(load_audio=lambda p: "audio", align=lambda *a, **k: {"segments": [source]},
                             assign_word_speakers=lambda *a: {"segments": [source]})
        def diarize(*args, **kwargs):
            if diar_error:
                raise RuntimeError("offline diarization failure")
            return "speakers"
        if align_error:
            def fail(*a, **k):
                raise RuntimeError("offline alignment failure")
            wx.align = fail
        with patch.dict(sys.modules, {"whisperx": wx}), patch.object(pipeline, "extract_audio", return_value="offline.wav"), \
                patch.object(pipeline, "_load_asr", return_value=asr), \
                patch.object(pipeline, "_load_align", return_value=(None, None)), \
                patch.object(pipeline, "_load_diarizer", return_value=diarize), \
                patch.object(pipeline, "HF_TOKEN", token), patch.object(pipeline.os, "remove"), \
                patch.object(pipeline, "summarize", return_value={}) as summarize:
            out = pipeline.process("offline.mp4", {"enableDiarization": enabled})
            self.assertEqual(summarize.call_args.args[0], out["rawText"])
            return out

    def test_process_exposes_two_turns_in_text_and_srt(self):
        out = self.run_pipeline()
        self.assertIn("Спикер 2: Нет, подождите!", out["rawText"])
        self.assertIn("[Спикер 2]: Нет, подождите!", out["srtContent"])
        self.assertEqual(out["processing"]["diarization"], "ok")
        self.assertEqual(out["wordCount"], 4)  # Speaker labels are not spoken words.

    def test_processing_failures_are_distinguishable(self):
        self.assertEqual(self.run_pipeline(align_error=True)["processing"]["alignment"], "failed")
        self.assertEqual(self.run_pipeline(diar_error=True)["processing"]["diarization"], "failed")
        self.assertEqual(self.run_pipeline(token=None)["processing"]["diarization"], "missing_token")
        self.assertEqual(self.run_pipeline(enabled=False)["processing"]["diarization"], "disabled")

    def test_benchmark_skips_llm_and_preserves_coarse_baseline(self):
        with patch.object(pipeline, "QUALITY_BENCHMARK", True):
            source = example()
            wx = SimpleNamespace(load_audio=lambda p: "audio", align=lambda *a, **k: {"segments": [source]},
                                 assign_word_speakers=lambda *a: {"segments": [source]})
            asr = SimpleNamespace(transcribe=lambda *a, **k: {"segments": [source], "language": "ru"})
            with patch.dict(sys.modules, {"whisperx": wx}), patch.object(pipeline, "extract_audio", return_value="offline.wav"), \
                    patch.object(pipeline, "_load_asr", return_value=asr), patch.object(pipeline, "_load_align", return_value=(None, None)), \
                    patch.object(pipeline, "_load_diarizer", return_value=lambda *a, **k: None), \
                    patch.object(pipeline, "HF_TOKEN", "test"), patch.object(pipeline.os, "remove"), \
                    patch.object(pipeline, "summarize", side_effect=AssertionError("LLM must stay unloaded")):
                out = pipeline.process("offline.mp4")
        self.assertEqual(len(out["benchmark"]["baselineSegments"]), 1)
        self.assertEqual(len(out["segments"]), 2)
        self.assertEqual(out["benchmark"]["alignedSegments"][0], source)
        self.assertEqual(out["processing"]["summary"], "disabled_for_benchmark")

    def test_production_does_not_emit_private_alignment_dump(self):
        with patch.object(pipeline, "QUALITY_BENCHMARK", False):
            self.assertNotIn("benchmark", self.run_pipeline())


if __name__ == "__main__":
    unittest.main()
