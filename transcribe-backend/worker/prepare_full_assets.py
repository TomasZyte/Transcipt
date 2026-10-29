"""Public alignment assets at build time; gated test diarization at runtime."""
import argparse
import hashlib
from pathlib import Path
import re
import tempfile

from asr_comparison import digest, read_json, write_new
from prepare_asr_assets import fetch

SOURCES = Path(__file__).with_name('full_model_sources.json')


def verify_file(path, row):
    if path.stat().st_size != row['size']:
        raise ValueError('Model size mismatch')
    if 'sha256' in row and digest(path) != row['sha256']:
        raise ValueError('Model checksum mismatch')
    if 'git_blob_sha1' in row:
        data = path.read_bytes()
        if hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() != row['git_blob_sha1']:
            raise ValueError('Model Git blob mismatch')


def verify_bundle(root, kind):
    source = read_json(SOURCES)[kind]
    for name, row in source['files'].items():
        path = Path(root) / name
        if not path.is_file():
            raise ValueError('Missing pinned model asset')
        verify_file(path, row)
    return {'repository':source['repository'], 'revision':source['revision'],
            'files':{name:digest(Path(root)/name) for name in source['files']}}


def prepare_alignment(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    source = read_json(SOURCES)['alignment']
    for name, row in source['files'].items():
        fetch(dict(row, name=name, max_bytes=row['size'],
              url=f"https://huggingface.co/{source['repository']}/resolve/{source['revision']}/{name}"), root/name)
    return verify_bundle(root, 'alignment')


def prepare_diarization(root, token):
    if not token:
        raise ValueError('Test HF token is required')
    from huggingface_hub import hf_hub_download
    import shutil
    source = read_json(SOURCES)['diarization']
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    # Only the previously authorized Community-1 repository, fixed revision/files.
    for name, row in source['files'].items():
        cached = Path(hf_hub_download(source['repository'], name,
            revision=source['revision'], token=token))
        verify_file(cached, row)
        target = root/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, target)
    return verify_bundle(root, 'diarization')


def verify_nltk(root):
    lock = read_json(Path(root)/'nltk-lock.json')
    base = Path(root)/'nltk'
    actual = {p.relative_to(base).as_posix():p for p in base.rglob('*') if p.is_file()}
    if set(actual) != set(lock) or any(digest(actual[k]) != v for k,v in lock.items()):
        raise ValueError('NLTK data differs from build lock')
    return digest(Path(root)/'nltk-lock.json')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--destination', type=Path, required=True)
    args = p.parse_args()
    prepare_alignment(args.destination/'alignment')
    import nltk
    base = args.destination/'nltk'
    if not nltk.download('punkt_tab', download_dir=str(base), quiet=True):
        raise RuntimeError('Could not stage sentence tokenizer')
    write_new(args.destination/'nltk-lock.json', {
        p.relative_to(base).as_posix():digest(p) for p in base.rglob('*') if p.is_file()})
    verify_nltk(args.destination)
    print('Pinned Russian alignment and sentence tokenizer staged; no inference.')


if __name__ == '__main__':
    main()
