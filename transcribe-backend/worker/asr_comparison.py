"""Isolated ASR comparison CLI. Default: validate/plan, no ML imports or network.

One clip and one allowlisted variant per process. Not imported by production.
It does not submit RunPod jobs, access R2, read secrets, or reconcile a budget.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import re
import socket
import time
from unittest.mock import patch
import wave

VARIANTS = ("wx_auto", "wx_ru", "wx_ru_vad", "fw_v3", "fw_v2", "gigaam_v3")
SHA = re.compile(r"[0-9a-f]{64}\Z")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def config_for(variant):
    if variant not in VARIANTS:
        raise ValueError("Unknown comparison variant")
    common = {"variant": variant, "device": "cuda", "alignment": False, "diarization": False, "summary": False}
    if variant.startswith("wx_"):
        lang = None if variant == "wx_auto" else "ru"
        return dict(common, engine="whisperx", model="large-v3", max_seconds=60,
            packages={"whisperx": "3.8.6", "faster-whisper": "1.2.1", "ctranslate2": "4.5.0"},
            load={"compute_type": "float16", "language": lang, "vad_method": "pyannote",
                  "asr_options": {"beam_size": 5, "initial_prompt": None, "hotwords": None},
                  "vad_options": {"chunk_size": 30, "vad_onset": .3 if variant == "wx_ru_vad" else .5,
                                  "vad_offset": .2 if variant == "wx_ru_vad" else .363}},
            transcribe={"language": lang, "batch_size": 8, "task": "transcribe", "chunk_size": 30})
    if variant.startswith("fw_"):
        return dict(common, engine="faster_whisper", model="large-v2" if variant == "fw_v2" else "large-v3", max_seconds=60,
            packages={"faster-whisper": "1.2.1", "ctranslate2": "4.5.0"},
            load={"compute_type": "float16"},
            transcribe={"language": "ru", "task": "transcribe", "beam_size": 5, "temperature": 0.0,
                "vad_filter": False, "condition_on_previous_text": False, "without_timestamps": False,
                "word_timestamps": False, "initial_prompt": None, "prefix": None, "hotwords": None,
                "compression_ratio_threshold": 2.4, "log_prob_threshold": -1.0, "no_speech_threshold": .6})
    return dict(common, engine="gigaam", model="v3_rnnt", max_seconds=25,
                packages={}, load={"fp16_encoder": True, "use_flash": False}, transcribe={"word_timestamps": False})


def prepare(manifest_path, clip_id, variant):
    """Read only the audio manifest; neither the plan nor human references are inputs."""
    config = config_for(variant)
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    matches = [c for c in manifest.get("clips", []) if c.get("id") == clip_id]
    if len(matches) != 1:
        raise ValueError("Clip must identify exactly one manifest entry")
    clip = matches[0]
    if not isinstance(clip.get("file"), str) or Path(clip["file"]).name != clip["file"] or ":" in clip["file"] or "\\" in clip["file"]:
        raise ValueError("Clip must be a file beside the manifest")
    path = manifest_path.parent / clip["file"]
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("Missing, linked or oversized audio")
    expected = clip.get("sha256", "")
    if not isinstance(expected, str) or not SHA.fullmatch(expected) or digest(path) != expected:
        raise ValueError("Audio checksum mismatch")
    start, end = clip.get("source_start_seconds"), clip.get("source_end_seconds")
    if not finite(start) or not finite(end) or not 0 <= start < end:
        raise ValueError("Invalid source offsets")
    with wave.open(str(path), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) != (1, 2, 16000, "NONE"):
            raise ValueError("Expected mono 16kHz PCM16 WAV")
        frames = wav.getnframes()
        if len(wav.readframes(frames)) != frames * 2:
            raise ValueError("Truncated WAV")
    duration = frames / 16000
    if not 0 < duration <= config["max_seconds"] or abs(end - start - duration) > 1 / 16000:
        raise ValueError("Audio duration/offset mismatch or engine limit exceeded")
    if "duration_seconds" in clip and (not finite(clip["duration_seconds"]) or abs(clip["duration_seconds"] - duration) > 1 / 16000):
        raise ValueError("Manifest duration mismatch")
    return {"schema": 1, "status": "not_run", "variant_id": variant, "clip_id": clip_id,
            "audio_file": str(path), "audio_sha256": expected, "duration_seconds": duration,
            "source_start_seconds": start, "source_end_seconds": end, "config": config,
            "raw_asr_text": None, "raw_asr_segments": None, "wer": None, "der": None,
            "billed_cost_usd": None, "network_submission": False}


def verify_assets(lock_path, config):
    """Verify an explicitly staged bundle before any model import/load."""
    lock_path = Path(lock_path).resolve()
    lock = read_json(lock_path)
    if lock.get("model") != config["model"] or not re.fullmatch(r"[0-9a-f]{40,64}", lock.get("revision", "")):
        raise ValueError("Model identity/revision is missing")
    root = (lock_path.parent / lock["root"]).resolve()
    files = lock.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Model file hashes are required")
    actual = {p.relative_to(root).as_posix(): p for p in root.rglob("*") if p.is_file()}
    if set(actual) != set(files) or any(p.is_symlink() for p in root.rglob("*")):
        raise ValueError("Bundle must contain exactly the listed regular files")
    for name, expected in files.items():
        if not isinstance(expected, str) or not SHA.fullmatch(expected) or digest(actual[name]) != expected:
            raise ValueError("Model file checksum mismatch")
    required = {"v3_rnnt.ckpt"} if config["engine"] == "gigaam" else {"model.bin", "config.json", "tokenizer.json"}
    if not required <= set(files):
        raise ValueError("Required model files are missing")
    packages = lock.get("packages", {})
    if not isinstance(packages, dict) or not all(isinstance(k, str) and k for k in packages):
        raise ValueError("Invalid package version lock")
    # Python distribution names are case-insensitive; dot/underscore/dash match.
    canonical = lambda name: re.sub(r"[-_.]+", "-", name).lower()
    locked_names = {canonical(name) for name in packages}
    required_packages = set(config["packages"]) | {"torch"}
    if config["engine"] == "gigaam":
        required_packages |= {"gigaam", "omegaconf", "hydra-core"}
    if config["engine"] == "whisperx":
        required_packages |= {"pyannote.audio", "transformers", "torchaudio"}
    if not {canonical(name) for name in required_packages} <= locked_names:
        raise ValueError("Incomplete runtime version lock")
    profile_versions = {canonical(name): v for name, v in config["packages"].items()}
    for package, expected in packages.items():
        if not isinstance(expected, str) or not expected or metadata.version(package) != expected:
            raise ValueError("Installed package differs from runtime lock")
        if canonical(package) in profile_versions and expected != profile_versions[canonical(package)]:
            raise ValueError("Runtime differs from comparison profile")
    if config["engine"] == "gigaam":
        direct = json.loads(metadata.distribution("gigaam").read_text("direct_url.json") or "{}")
        if direct.get("vcs_info", {}).get("commit_id") != lock["revision"] or direct.get("url", "").rstrip("/").removesuffix(".git") != "https://github.com/salute-developers/GigaAM":
            raise ValueError("GigaAM must be installed from the locked official Git commit")
    vad_path = None
    if config["engine"] == "whisperx":
        vad = lock.get("vad", {})
        vad_path = (lock_path.parent / vad.get("path", "")).resolve()
        if not vad_path.is_file() or vad_path.is_symlink() or digest(vad_path) != vad.get("sha256"):
            raise ValueError("Local VAD model checksum mismatch")
    return {"root": str(root), "vad_path": str(vad_path) if vad_path else None,
            "revision": lock["revision"], "packages": packages, "lock_sha256": digest(lock_path)}


@contextmanager
def offline_models():
    def denied(*args, **kwargs):
        raise RuntimeError("Network disabled in ASR comparison process")
    # This is an accidental-download guard, not an OS security sandbox.
    with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}), \
         patch.object(socket, "create_connection", denied), patch.object(socket.socket, "connect", denied), \
         patch.object(socket.socket, "connect_ex", denied):
        yield


def backend(config, path, assets):
    """Lazy entry points. Keep GigaAM in its own process/environment."""
    engine = importlib.import_module(config["engine"])
    started = time.perf_counter()
    resolved = {}
    if config["engine"] == "whisperx":
        vad_module = importlib.import_module("whisperx.vads.pyannote")
        vad_opts = config["load"]["vad_options"]
        vad = vad_module.load_vad_model(config["device"], model_fp=assets["vad_path"],
            vad_onset=vad_opts["vad_onset"], vad_offset=vad_opts["vad_offset"])
        model = engine.load_model(assets["root"], device=config["device"], local_files_only=True,
                                  vad_model=vad, **deepcopy(config["load"]))
        if hasattr(model.options, "_asdict"):
            resolved["decoder_options"] = model.options._asdict()
        loaded = time.perf_counter()
        result = model.transcribe(str(path), **config["transcribe"])
        segments = result["segments"]
        language = result.get("language")
        text = " ".join(s["text"].strip() for s in segments).strip()
    elif config["engine"] == "faster_whisper":
        model = engine.WhisperModel(assets["root"], device=config["device"], local_files_only=True, **config["load"])
        loaded = time.perf_counter()
        stream, info = model.transcribe(str(path), **config["transcribe"])
        # Consume the lazy generator inside the timed, network-disabled block.
        segments = [{"start": s.start, "end": s.end, "text": s.text,
                     "avg_logprob": s.avg_logprob, "no_speech_prob": s.no_speech_prob} for s in stream]
        language = info.language
        text = " ".join(s["text"].strip() for s in segments).strip()
        options = getattr(info, "transcription_options", None)
        if hasattr(options, "_asdict"):
            resolved["decoder_options"] = options._asdict()
    else:
        # Named model uses the already hashed cache file. A checkpoint path would
        # invoke GigaAM's *fine-tuned* checkpoint loader, which has a different schema.
        model = engine.load_model("v3_rnnt", device=config["device"], download_root=assets["root"], **config["load"])
        loaded = time.perf_counter()
        result = model.transcribe(str(path), **config["transcribe"])
        text = result if isinstance(result, str) else result.text
        segments = [{"start": None, "end": None, "text": text}]
        language = None  # Russian profile, not measured language detection.
        resolved["parameter_dtypes"] = sorted({str(p.dtype) for p in model.parameters()})
    ended = time.perf_counter()
    return {"text": text, "segments": segments, "language_detected": language, "resolved": resolved,
            "model_load_seconds": loaded - started, "inference_seconds": ended - loaded}


def normalize_result(plan, raw):
    if not isinstance(raw.get("text"), str) or not isinstance(raw.get("segments"), list):
        raise ValueError("Invalid ASR output")
    rows = []
    for segment in raw["segments"]:
        start, end = segment.get("start"), segment.get("end")
        if not isinstance(segment.get("text"), str):
            raise ValueError("Invalid ASR segment text")
        if start is None and end is None and plan["config"]["engine"] == "gigaam":
            rows.append({"text": segment["text"], "start": None, "end": None, "source_start": None, "source_end": None})
            continue
        if not finite(start) or not finite(end) or not 0 <= start <= end <= plan["duration_seconds"] + .05:
            raise ValueError("Invalid ASR segment timing")
        rows.append({"text": segment["text"], "start": start, "end": end,
                     "source_start": start + plan["source_start_seconds"], "source_end": end + plan["source_start_seconds"]})
    return dict(plan, status="completed", raw_asr_text=raw["text"], raw_asr_segments=rows,
                raw_engine_output=raw, language_detected=raw.get("language_detected"),
                vad_spans=None, timestamp_kind="merged_vad_chunks" if plan["config"]["engine"] == "whisperx" else
                ("asr_segments" if plan["config"]["engine"] == "faster_whisper" else "unavailable"))


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def run(plan, lock_path, output):
    if os.environ.get("ASR_COMPARISON_TEST") != "1":
        raise ValueError("Explicit ASR_COMPARISON_TEST=1 is required for execution")
    if plan["config"] != config_for(plan["variant_id"]):
        raise ValueError("Only the fixed comparison profiles may execute")
    if Path(output).exists() or Path(str(output) + ".started.json").exists():
        raise ValueError("Output or attempt record exists; no implicit retry/overwrite")
    if digest(plan["audio_file"]) != plan["audio_sha256"]:
        raise ValueError("Audio changed after preparation")
    assets = verify_assets(lock_path, plan["config"])
    attempt = Path(str(output) + ".started.json")
    write_new(attempt, {"variant": plan["variant_id"], "audio_sha256": plan["audio_sha256"],
                        "model_lock_sha256": assets["lock_sha256"], "status": "started"})
    try:
        with offline_models():
            raw = backend(plan["config"], plan["audio_file"], assets)
        result = normalize_result(plan, raw)
        result["runtime"] = assets
        write_new(output, result)
    except Exception as exc:
        write_new(str(output) + ".failed.json", {"status": "failed", "error_type": type(exc).__name__,
                  "variant": plan["variant_id"], "retry_automatic": False})
        raise
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--clip", default="context")
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--model-lock", type=Path)
    args = parser.parse_args()
    try:
        plan = prepare(args.manifest, args.clip, args.variant)
        if args.execute:
            if args.model_lock is None:
                raise ValueError("Execution requires a verified local model lock")
            result = run(plan, args.model_lock, args.output)
        else:
            write_new(args.output, plan)
            result = plan
        print(json.dumps({"status": result["status"], "variant": args.variant, "network_submission": False}))
    except Exception as exc:
        # Do not leak credentials, audio text or transport URLs in error messages.
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
