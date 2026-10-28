"""Prepare short private WAVs and reference templates locally; never uploads."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import wave


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def prepare(source, destination, ffmpeg):
    files = sorted(p for p in source.iterdir() if p.is_file()
                   and p.suffix.lower() in {".mp4", ".wav", ".mp3", ".ogg", ".m4a", ".flac"})
    if not 1 <= len(files) <= 5:
        raise ValueError("Choose a folder containing 1–5 recordings")
    if len({p.stem.casefold() for p in files}) != len(files):
        raise ValueError("Sample stems must be unique")
    probes = []
    for path in files:
        probe = subprocess.run([ffmpeg, "-hide_banner", "-i", str(path)],
                               capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
        match = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", probe.stderr)
        if not match or "Audio:" not in probe.stderr:
            raise ValueError(f"No readable audio duration for {path.name}")
        duration = int(match[1]) * 3600 + int(match[2]) * 60 + float(match[3])
        if not 0 < duration <= 60:
            raise ValueError(f"{path.name} exceeds 60 seconds; choose a representative excerpt explicitly")
        probes.append((path, duration))
    if sum(seconds for _, seconds in probes) > 180:
        raise ValueError("The evaluation set must fit within 180 seconds")
    # New destination only: originals and prior evidence must remain unchanged.
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {"source_directory": str(source.resolve()), "status": "local_only",
                "sample_rate": 16000, "channels": 1, "samples": []}
    for path, duration in probes:
        wav_path = destination / (path.stem + ".wav")
        subprocess.run([ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
                        "-i", str(path), "-map", "0:a:0", "-vn", "-ar", "16000", "-ac", "1",
                        "-c:a", "pcm_s16le", str(wav_path)], check=True, capture_output=True, timeout=120)
        with wave.open(str(wav_path), "rb") as wav:
            seconds = wav.getnframes() / wav.getframerate()
            if wav.getframerate() != 16000 or wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                raise ValueError("Unexpected WAV format")
        row = {"id": path.stem, "source_name": path.name, "source_sha256": digest(path),
               "source_duration_seconds": duration, "wav": wav_path.name,
               "duration_seconds": round(seconds, 3), "wav_sha256": digest(wav_path),
               "bytes": wav_path.stat().st_size}
        manifest["samples"].append(row)
        (destination / (path.stem + ".reference.json")).write_text(json.dumps({
            "human_verified": False, "text": "", "speaker_count": None,
            "notes": "Listen to the sample, correct every word and mark human_verified only afterwards."},
            ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["total_seconds"] = round(sum(s["duration_seconds"] for s in manifest["samples"]), 3)
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg")
    args = parser.parse_args()
    ffmpeg = args.ffmpeg or shutil.which("ffmpeg")
    if not ffmpeg:
        import imageio_ffmpeg  # Local preparation dependency only, not part of production.
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    result = prepare(args.source, args.output, ffmpeg)
    print(json.dumps({"status": result["status"], "count": len(result["samples"]),
                      "total_seconds": result["total_seconds"]}))


if __name__ == "__main__":
    main()
