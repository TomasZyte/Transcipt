"""Private benchmark evidence and offline audit; no models, network or corrections."""
import argparse
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path

from transcript_segments import build_segments


def snapshot_asr(segments):
    # Keep an independent copy before alignment/assignment can mutate the result.
    # Only transcription fields: no transport metadata, URLs or model embeddings.
    return deepcopy([{key: row[key] for key in ("start", "end", "text") if key in row}
                     for row in segments])


def _time(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def snapshot_diarization(frame):
    """Copy the two supported WhisperX versions' DataFrame without importing pandas.

    Missing/failed capture must not masquerade as a successful empty timeline or
    interrupt transcription. Capture it before assign_word_speakers mutates it.
    """
    if frame is None:
        return {"status": "unavailable", "segments": None}
    try:
        rows = frame.to_dict("records")
        segments, invalid = [], 0
        for row in rows:
            start, end, speaker = _time(row.get("start")), _time(row.get("end")), row.get("speaker")
            if start is None or end is None or end <= start or not isinstance(speaker, str) or not speaker:
                invalid += 1
                continue
            segments.append({"start": start, "end": end, "speaker": speaker})
        return {"status": "partial" if invalid else "ok", "segments": segments, "invalidRows": invalid}
    except Exception as exc:
        # Do not echo exception messages, which may contain private metadata.
        return {"status": "failed", "segments": None, "errorType": type(exc).__name__}


def _text(segments):
    return "".join("".join(row.get("text", "").split()) for row in segments)


def audit_result(result, long_word_seconds=2.0):
    """Review flags are heuristics, not quality scores or corrected timestamps."""
    if not math.isfinite(long_word_seconds) or long_word_seconds <= 0:
        raise ValueError("Use a positive finite review threshold")
    if "status" in result:
        if result["status"] != "COMPLETED":
            raise ValueError("Expected a completed provider result")
        result = result.get("output")
    if not isinstance(result, dict) or result.get("error") or not isinstance(result.get("segments"), list):
        raise ValueError("Expected a successful transcription")
    benchmark = result.get("benchmark") or {}
    aligned = benchmark.get("alignedSegments")
    if not isinstance(aligned, list):
        raise ValueError("Private alignedSegments are required")
    rebuilt, _ = build_segments(aligned)
    exported = result["segments"]
    words = [word for row in aligned for word in row.get("words", [])]
    labelled = Counter(word["speaker"] for word in words if word.get("speaker"))
    long_words, invalid_times = [], 0
    for word in words:
        start, end = _time(word.get("start")), _time(word.get("end"))
        if start is None or end is None or end < start:
            invalid_times += 1
        elif end - start > long_word_seconds:
            long_words.append({"word": word.get("word", ""), "start": start, "end": end,
                               "durationSeconds": round(end - start, 3), "speaker": word.get("speaker")})
    timeline = benchmark.get("diarizationTimeline") or {"status": "not_recorded", "segments": None}
    timeline_labels = sorted({row["speaker"] for row in timeline.get("segments") or []})
    complete_timeline = timeline.get("status") == "ok"
    asr = benchmark.get("asrSegments")
    return {
        "exportMatchesRebuild": rebuilt == exported,
        "exportTextMatchesAlignedIgnoringWhitespace": _text(exported) == _text(aligned),
        "alignedTextMatchesAsrIgnoringWhitespace": _text(aligned) == _text(asr) if isinstance(asr, list) else None,
        "alignedWordSpeakerCount": len(labelled),
        "exportedSpeakerCount": len({row["speaker"] for row in exported if row.get("speaker")}),
        "wordsPerRawSpeaker": dict(sorted(labelled.items())),
        "singleWordSpeakers": sorted(s for s, count in labelled.items() if count == 1),
        "wordsWithoutSpeaker": sum(not word.get("speaker") for word in words),
        "wordsWithMissingOrInvalidTimes": invalid_times,
        "longWordReviewThresholdSeconds": long_word_seconds,
        "longWordReview": long_words,
        "diarizationTimelineStatus": timeline.get("status", "not_recorded"),
        "timelineSpeakerCount": len(timeline_labels) if complete_timeline else None,
        "timelineSpeakersWithoutAssignedWords": sorted(set(timeline_labels) - set(labelled)) if complete_timeline else None,
        "note": "Review flags only. Short real turns and sustained words may be valid. No rewriting, WER or DER."
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8-sig"))
    report = audit_result(result)
    # Never overwrite evidence. Store this private report under .test-runtime.
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
    print("Private audit saved; no network, model loads or GPU tasks.")


if __name__ == "__main__":
    main()
