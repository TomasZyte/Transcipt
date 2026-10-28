"""Compare private result JSONs locally. No API calls, model loads or GPU jobs."""
import argparse
import json
from pathlib import Path
import re
import unicodedata


def tokens(text):
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold().replace("ё", "е"))


def word_error_rate(reference, hypothesis):
    ref, hyp = tokens(reference), tokens(hypothesis)
    if not ref:
        raise ValueError("A non-empty human reference is required for WER")
    if max(len(ref), len(hyp)) > 2000:
        raise ValueError("Use short evaluation samples (at most 2000 words)")
    row = list(range(len(hyp) + 1))
    for i, target in enumerate(ref, 1):
        next_row = [i]
        for j, actual in enumerate(hyp, 1):
            next_row.append(min(row[j] + 1, next_row[j - 1] + 1,
                                row[j - 1] + (target != actual)))
        row = next_row
    return {"errors": row[-1], "referenceWords": len(ref),
            "wer": round(row[-1] / len(ref), 6)}


def result_summary(result, reference=None):
    if "status" in result:
        if result["status"] != "COMPLETED":
            raise ValueError("RunPod result is not COMPLETED")
        result = result.get("output")
    if not isinstance(result, dict) or result.get("error") or not isinstance(result.get("segments"), list):
        raise ValueError("Expected a successful transcription with segments")
    segments = result["segments"]
    # rawText contains speaker labels; they are not spoken words and must not enter WER.
    text = " ".join(s.get("text", "") for s in segments)
    speakers = [s["speaker"] for s in segments if s.get("speaker") and s.get("text", "").strip()]
    processing = result.get("processing") or {}
    summary = {"words": len(tokens(text)), "segments": len(segments),
               "speakers": len(set(speakers)),
               "speakerChanges": sum(a != b for a, b in zip(speakers, speakers[1:])),
               "processing": processing, "wer": None,
               "speakerCountMatchesReference": None,
               "der": None, "derNote": "Not measured: requires human timed speaker annotations."}
    if reference is not None:
        if not reference.get("human_verified"):
            raise ValueError("Reference must be marked human_verified after listening")
        summary["wer"] = word_error_rate(reference.get("text", ""), text)
        count = reference.get("speaker_count")
        if count is not None:
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError("Reference speaker_count must be a positive integer")
            summary["speakerCountMatchesReference"] = summary["speakers"] == count
    return summary


def compare(baseline, candidate, reference=None):
    left, right = result_summary(baseline, reference), result_summary(candidate, reference)
    measured = left["wer"] is not None and right["wer"] is not None
    warnings = []
    for name, result in (("baseline", left), ("candidate", right)):
        for stage in ("alignment", "diarization"):
            status = result["processing"].get(stage, "not_reported")
            if status != "ok":
                warnings.append(f"{name}: {stage}={status}")
    return {"baseline": left, "candidate": right,
            "werDelta": round(right["wer"]["wer"] - left["wer"]["wer"], 6) if measured else None,
            "warnings": warnings,
            "note": "Speaker count alone does not measure diarization quality. No automatic winner."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    def read(path):
        return json.loads(path.read_text(encoding="utf-8-sig"))
    report = compare(read(args.baseline), read(args.candidate), read(args.reference) if args.reference else None)
    # Never overwrite prior evidence. Reports and references belong under .test-runtime.
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print("Comparison saved; no network requests or GPU jobs were made.")


if __name__ == "__main__":
    main()
