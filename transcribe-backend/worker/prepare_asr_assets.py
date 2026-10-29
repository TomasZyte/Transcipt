"""Prepare a pinned model bundle at image BUILD time, never during inference.

Default prints a download plan only. --download is explicit and makes large
public downloads. No torch/model imports, GPU inference, or RunPod requests.
"""
import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import shutil
import urllib.request

from asr_comparison import config_for, digest, read_json, verify_assets, write_new

SOURCES = Path(__file__).with_name("asr_model_sources.json")


def recipe(model):
    source = read_json(SOURCES)[model]
    if model == "v3_rnnt":
        return {"model": model, "revision": source["revision"], "files": [
            {"name": "v3_rnnt.ckpt", "url": source["url"], "md5": source["upstream_md5"],
             "max_bytes": source["max_download_bytes"]}]}
    return {"model": model, "revision": source["revision"], "files": [
        dict(info, name=name, max_bytes=info["size"],
             url=f"https://huggingface.co/{source['repository']}/resolve/{source['revision']}/{name}")
        for name, info in source["files"].items()]}


def fetch(row, target):
    total = 0
    with urllib.request.urlopen(row["url"], timeout=60) as source, target.open("xb") as output:
        while chunk := source.read(1024 * 1024):
            total += len(chunk)
            if total > row["max_bytes"]:
                raise ValueError("Model download exceeds pinned size limit")
            output.write(chunk)
    if "size" in row and total != row["size"]:
        raise ValueError("Model size mismatch")
    if "sha256" in row and digest(target) != row["sha256"]:
        raise ValueError("Model SHA256 mismatch")
    if "git_blob_sha1" in row:
        content = target.read_bytes()  # Only small JSON/vocabulary files use Git blob hashes.
        actual = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        if actual != row["git_blob_sha1"]:
            raise ValueError("Pinned Git blob mismatch")
    if "md5" in row:
        with target.open("rb") as source:
            if hashlib.file_digest(source, "md5").hexdigest() != row["md5"]:
                raise ValueError("GigaAM upstream checksum mismatch")


def assemble(model, destination):
    spec = recipe(model)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    root = destination / "model"
    root.mkdir()
    for row in spec["files"]:
        fetch(row, root / row["name"])
    lock = {"model": model, "revision": spec["revision"], "root": "model",
            "files": {p.name: digest(p) for p in root.iterdir()},
            "packages": {d.metadata["Name"]: d.version for d in metadata.distributions() if d.metadata.get("Name")},
            "source_manifest_sha256": digest(SOURCES)}
    if model != "v3_rnnt":
        vad = metadata.distribution("whisperx").locate_file("whisperx/assets/pytorch_model.bin")
        shutil.copyfile(vad, destination / "vad.bin")
        lock["vad"] = {"path": "vad.bin", "sha256": digest(destination / "vad.bin")}
    write_new(destination / "model-lock.json", lock)
    # Validate installed versions and VCS provenance without importing models.
    variants = ["gigaam_v3"] if model == "v3_rnnt" else (["fw_v2"] if model == "large-v2" else ["wx_auto", "wx_ru", "wx_ru_vad", "fw_v3"])
    for variant in variants:
        verify_assets(destination / "model-lock.json", config_for(variant))
    return lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("large-v3", "large-v2", "v3_rnnt"), required=True)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if not args.download:
        print(json.dumps({"status": "download_plan_only", **recipe(args.model)}, indent=2))
        return
    if args.destination is None:
        parser.error("--download requires a new --destination directory")
    lock = assemble(args.model, args.destination)
    print(json.dumps({"status": "bundle_verified", "model": lock["model"], "revision": lock["revision"]}))


if __name__ == "__main__":
    main()
