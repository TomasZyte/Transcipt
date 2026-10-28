import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quality_compare import compare, word_error_rate


def output(text="Привет, мир!", speakers=("Спикер 1",)):
    return {"segments": [{"text": text if i == 0 else "", "speaker": s}
                         for i, s in enumerate(speakers)],
            "processing": {"alignment": "ok", "diarization": "ok"}}


class QualityComparison(unittest.TestCase):
    def test_known_word_errors(self):
        self.assertEqual(word_error_rate("один два три", "один четыре пять")['errors'], 2)
        self.assertEqual(word_error_rate("один два три", "один три")['errors'], 1)
        self.assertEqual(word_error_rate("один", "один два")['wer'], 1)

    def test_normalization_is_explicit(self):
        self.assertEqual(word_error_rate("Ёлка — ЗЕЛЁНАЯ!", "елка зеленая")['wer'], 0)
        self.assertGreater(word_error_rate("два", "2")['wer'], 0)

    def test_empty_reference_is_not_a_quality_score(self):
        with self.assertRaises(ValueError):
            word_error_rate("...", "что-то")

    def test_speaker_labels_not_counted_as_spoken_words(self):
        source = output()
        source["rawText"] = "Спикер 1: Привет, мир!"
        report = compare(source, source, {"human_verified": True, "text": "Привет мир", "speaker_count": 1})
        self.assertEqual(report["candidate"]["wer"]["referenceWords"], 2)
        self.assertEqual(report["werDelta"], 0)
        self.assertTrue(report["candidate"]["speakerCountMatchesReference"])

    def test_no_reference_means_no_quality_claim(self):
        report = compare(output(), output("Не тот текст"))
        self.assertIsNone(report["werDelta"])
        self.assertIsNone(report["candidate"]["der"])

    def test_machine_generated_reference_not_accepted(self):
        with self.assertRaises(ValueError):
            compare(output(), output(), {"text": "Привет мир", "human_verified": False})

    def test_diarization_failure_is_visible(self):
        failed = output()
        failed["processing"]["diarization"] = "missing_token"
        self.assertIn("candidate: diarization=missing_token", compare(output(), failed)["warnings"])

    def test_reject_failed_provider_response(self):
        with self.assertRaises(ValueError):
            compare(output(), {"status": "FAILED", "output": output()})

    def test_provider_wrapper_accepted(self):
        self.assertEqual(compare(output(), {"status": "COMPLETED", "output": output()})["warnings"], [])


if __name__ == "__main__":
    unittest.main()
