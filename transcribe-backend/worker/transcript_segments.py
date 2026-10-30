"""Turn aligned word speakers into transcript turns without rewriting ASR text."""
import math


def _time(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def split_speaker_turns(segment):
    """Return (turns, fallback_reason). No fabricated word times or dropped text.

    WhisperX assigns speakers both to sentences and words. A sentence can contain
    several speakers; only split where its ordered word times support a boundary.
    Unlabelled words inherit a neighbour, never introduce a new speaker.
    """
    text = (segment.get("text") or "").strip()
    original = {"start": segment.get("start", 0.0),
                "end": segment.get("end", segment.get("start", 0.0)),
                "speaker": segment.get("speaker") or "SPEAKER_00", "text": text}
    words = segment.get("words") or []
    if not words:
        return [original], "no_words"

    # Match tokens against the original string. Joining words would alter Chinese,
    # punctuation, repeated whitespace, or a token omitted by the aligner.
    mapped, cursor = [], 0
    for word in words:
        token = (word.get("word") or "").strip()
        if not token:
            continue
        offset = text.find(token, cursor)
        if offset < 0 or any(c.isalnum() for c in text[cursor:offset]):
            return [original], "text_mismatch"
        mapped.append((word, offset))
        cursor = offset + len(token)
    if not mapped or any(c.isalnum() for c in text[cursor:]):
        return [original], "text_mismatch"

    start, end = _time(original["start"]), _time(original["end"])
    if start is None or end is None or start < 0 or end <= start:
        return [original], "invalid_segment_times"
    timed = []
    previous_start = start
    for word, _ in mapped:
        ws, we = _time(word.get("start")), _time(word.get("end"))
        if ws is None or we is None:
            timed.append(False)
            continue
        if ws < previous_start or we < ws or ws < start or we > end:
            return [original], "invalid_word_times"
        previous_start = ws
        timed.append(True)

    # Leading unknown words follow the first labelled word. Later unknown words
    # follow the previous labelled word. Keep this imputation visible in metrics.
    labelled = [word.get("speaker") for word, _ in mapped]
    current = next((s for s in labelled if s), original["speaker"])
    labels = []
    for speaker in labelled:
        current = speaker or current
        labels.append(current)

    boundaries = [i for i in range(1, len(mapped)) if labels[i] != labels[i - 1]]
    if not boundaries:
        original["speaker"] = labels[0]
        return [original], None
    # A label without a timestamp is insufficient to place a speaker change.
    if any(not timed[i] for i in boundaries):
        return [original], "untimed_speaker_change"

    cuts = [0] + boundaries + [len(mapped)]
    turns = []
    for left, right in zip(cuts, cuts[1:]):
        turn_start = start if left == 0 else _time(mapped[left][0]["start"])
        turn_end = end if right == len(mapped) else _time(mapped[right][0]["start"])
        if right < len(mapped) and timed[right - 1]:
            turn_end = min(turn_end, _time(mapped[right - 1][0]["end"]))
        if turn_end <= turn_start:
            return [original], "empty_turn_times"
        char_start = 0 if left == 0 else mapped[left][1]
        char_end = len(text) if right == len(mapped) else mapped[right][1]
        turns.append({"start": turn_start, "end": turn_end,
                      "speaker": labels[left], "text": text[char_start:char_end].strip()})
    return turns, None


def build_segments(source_segments):
    """Preserve public output schema and stable speaker numbering across a file."""
    speakers, output = {}, []
    stats = {"inputSegments": 0, "splitSegments": 0, "fallbackReasons": {},
             "wordsWithoutSpeaker": 0}
    for source in source_segments:
        stats["inputSegments"] += 1
        stats["wordsWithoutSpeaker"] += sum(
            1 for word in (source.get("words") or []) if not word.get("speaker"))
        turns, reason = split_speaker_turns(source)
        if reason:
            counts = stats["fallbackReasons"]
            counts[reason] = counts.get(reason, 0) + 1
        if len(turns) > 1:
            stats["splitSegments"] += 1
        for turn in turns:
            raw_speaker = turn["speaker"]
            if raw_speaker != 'SPEAKER_UNKNOWN':
                speakers.setdefault(raw_speaker, f"Спикер {len(speakers) + 1}")
            start, end = float(turn["start"]), float(turn["end"])
            output.append({"id": f"seg-{len(output) + 1}",
                           "startSec": round(start, 2), "endSec": round(end, 2),
                           "timestamp": f"{int(start // 60):02d}:{int(start % 60):02d}",
                           "speaker": 'Спикер не определён' if raw_speaker == 'SPEAKER_UNKNOWN' else speakers[raw_speaker],
                           "text": turn["text"]})
    return output, stats
