from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quality_diagnostics import audit_result, snapshot_asr, snapshot_diarization
from transcript_segments import build_segments


def result():
    aligned = [{"start": 0, "end": 5, "text": "Да. Подождите.", "words": [
        {"word": "Да.", "start": 0, "end": .4, "speaker": "a"},
        {"word": "Подождите.", "start": 1, "end": 5, "speaker": "b"}]}]
    return {"segments": build_segments(aligned)[0], "benchmark": {"alignedSegments": aligned}}


class QualityDiagnostics(unittest.TestCase):
    def test_asr_snapshot_is_independent_and_filters_metadata(self):
        source = [{"start": 0, "end": 1, "text": "Да.", "media_url": "private", "embedding": [1]}]
        captured = snapshot_asr(source)
        source[0]["text"] = "Изменено"
        self.assertEqual(captured, [{"start": 0, "end": 1, "text": "Да."}])

    def test_timeline_preserves_overlaps_and_omits_other_columns(self):
        rows = [{"start": 0, "end": 2, "speaker": "a", "segment": object()},
                {"start": 1, "end": 3, "speaker": "b", "embedding": [1]}]
        captured = snapshot_diarization(SimpleNamespace(to_dict=lambda orient: rows))
        rows[0]["speaker"] = "changed"
        self.assertEqual(captured, {"status": "ok", "invalidRows": 0, "segments": [
            {"start": 0, "end": 2, "speaker": "a"}, {"start": 1, "end": 3, "speaker": "b"}]})

    def test_missing_timeline_differs_from_successful_empty_timeline(self):
        self.assertEqual(snapshot_diarization(None)["status"], "unavailable")
        self.assertEqual(snapshot_diarization(SimpleNamespace(to_dict=lambda orient: []))["status"], "ok")

    def test_invalid_rows_are_not_silently_complete_evidence(self):
        rows = [{"start": 0, "end": 1, "speaker": "a"},
                {"start": 0, "end": float("nan"), "speaker": "b"},
                {"start": True, "end": 2, "speaker": "c"}]
        captured = snapshot_diarization(SimpleNamespace(to_dict=lambda orient: rows))
        self.assertEqual((captured["status"], captured["invalidRows"]), ("partial", 2))

    def test_capture_error_does_not_echo_private_message(self):
        def fail(orient):
            raise ValueError("private signed URL")
        captured = snapshot_diarization(SimpleNamespace(to_dict=fail))
        self.assertEqual(captured, {"status": "failed", "segments": None, "errorType": "ValueError"})

    def test_old_results_do_not_invent_missing_timeline_or_asr(self):
        audit = audit_result(result())
        self.assertTrue(audit["exportMatchesRebuild"])
        self.assertTrue(audit["exportTextMatchesAlignedIgnoringWhitespace"])
        self.assertEqual(audit["diarizationTimelineStatus"], "not_recorded")
        self.assertIsNone(audit["timelineSpeakerCount"])
        self.assertIsNone(audit["timelineSpeakersWithoutAssignedWords"])
        self.assertIsNone(audit["alignedTextMatchesAsrIgnoringWhitespace"])

    def test_export_loss_is_distinct_from_upstream_labels(self):
        data = result()
        data["segments"].pop()
        audit = audit_result(data)
        self.assertFalse(audit["exportMatchesRebuild"])
        self.assertFalse(audit["exportTextMatchesAlignedIgnoringWhitespace"])
        self.assertEqual(audit["alignedWordSpeakerCount"], 2)
        self.assertEqual(audit["exportedSpeakerCount"], 1)

    def test_review_flags_do_not_rewrite_long_words_or_short_real_turns(self):
        data = result()
        original = deepcopy(data)
        audit = audit_result(data)
        self.assertEqual(audit["longWordReview"][0]["durationSeconds"], 4)
        self.assertEqual(audit["singleWordSpeakers"], ["a", "b"])
        self.assertEqual(data, original)

    def test_raw_speaker_without_words_can_be_located(self):
        data = result()
        data["benchmark"]["diarizationTimeline"] = {"status": "ok", "segments": [
            {"start": 0, "end": .4, "speaker": "a"},
            {"start": 1, "end": 5, "speaker": "b"},
            {"start": 2, "end": 3, "speaker": "c"}]}
        audit = audit_result(data)
        self.assertEqual(audit["timelineSpeakerCount"], 3)
        self.assertEqual(audit["timelineSpeakersWithoutAssignedWords"], ["c"])
        data["benchmark"]["diarizationTimeline"]["status"] = "partial"
        self.assertIsNone(audit_result(data)["timelineSpeakersWithoutAssignedWords"])

    def test_asr_text_change_and_failed_job_are_visible(self):
        data = result()
        data["benchmark"]["asrSegments"] = [{"text": "Совсем другой текст"}]
        self.assertFalse(audit_result(data)["alignedTextMatchesAsrIgnoringWhitespace"])
        with self.assertRaises(ValueError):
            audit_result({"status": "FAILED", "output": data})
        self.assertTrue(audit_result({"status": "COMPLETED", "output": data})["exportMatchesRebuild"])

    def test_review_threshold_must_be_positive_finite(self):
        for threshold in (0, -1, float("nan"), float("inf")):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                audit_result(result(), threshold)


if __name__ == "__main__":
    unittest.main()
