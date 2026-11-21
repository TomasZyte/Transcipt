"""Offline handler -> duration probe -> pipeline.extract_audio integration tests.

Requires real ffmpeg/ffprobe. Only network retrieval, RunPod registration and
ASR after extract_audio are replaced. All media and privacy markers are synthetic.
"""
from contextlib import redirect_stdout, redirect_stderr
import importlib.util
import io
import json
import logging
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

WORKER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKER))
import pipeline
import ffmpeg_diagnostics as diagnostics

CANARY = 'SYNTHETIC_PRIVATE_79991234567'
SECRET = 'private@example.invalid https://example.invalid/?signature=FAKE_SECRET'


def load_handler():
    spec = importlib.util.spec_from_file_location('privacy_chain_handler', WORKER/'handler.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'runpod': SimpleNamespace(serverless=SimpleNamespace(start=lambda _: None))}):
        spec.loader.exec_module(module)
    return module


class MediaPrivacyChainTests(unittest.TestCase):
    def setUp(self):
        self.handler = load_handler()

    def assert_private_absent(self, text):
        for value in (CANARY, '79991234567', 'private@example', 'example.invalid', 'FAKE_SECRET', 'signature='):
            self.assertNotIn(value, text)

    def test_probe_exception_types_never_print_exception_data(self):
        cases = [
            (subprocess.CalledProcessError(1, [CANARY], output=SECRET, stderr=SECRET), 'probe_failed'),
            (subprocess.TimeoutExpired([CANARY], 60, output=SECRET, stderr=SECRET), 'timeout'),
            (FileNotFoundError(SECRET), 'executable_or_file_missing'),
            (PermissionError(SECRET), 'permission_denied'),
            (ValueError(SECRET), 'invalid_probe_output'),
            (RuntimeError(SECRET), 'duration_unavailable'),
        ]
        for error, kind in cases:
            with self.subTest(kind=kind), patch.object(self.handler.subprocess, 'run', side_effect=error):
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertEqual(self.handler._media_duration(CANARY), 0.0)
                self.assert_private_absent(output.getvalue())
                payload = json.loads(output.getvalue().removeprefix('FFPROBE_DIAGNOSTIC_V1 '))
                self.assertEqual(payload, {'schema':'ffprobe-safe-v1', 'stage':'duration_probe', 'kind':kind})

    def test_malformed_probe_json_is_private(self):
        with patch.object(self.handler.subprocess, 'run', return_value=SimpleNamespace(stdout=SECRET.encode())):
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(self.handler._media_duration(CANARY), 0.0)
            self.assert_private_absent(output.getvalue())
            self.assertIn('invalid_probe_output', output.getvalue())

    def fixture_command(self, args):
        result = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', *args], capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, 'Synthetic fixture generation failed; raw stderr withheld')

    def exercise(self, kind):
        self.assertTrue(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'Install ffmpeg/ffprobe to run integration tests')
        with tempfile.TemporaryDirectory(prefix=CANARY) as directory:
            root = Path(directory)
            original = root/'tone.wav'
            with wave.open(str(original), 'wb') as out:
                out.setparams((2, 2, 44100, 0, 'NONE', 'not compressed'))
                out.writeframes(struct.pack('<hh', 1000, -500)*44100)
            source = original
            if kind == 'corrupt_m4a':
                source = root/'broken.m4a'
                self.fixture_command(['-i', str(original), '-metadata', 'comment='+SECRET, '-c:a', 'aac', str(source)])
                data = source.read_bytes()
                offset = 0
                while offset+8 <= len(data):
                    size, atom = struct.unpack('>I4s', data[offset:offset+8])
                    self.assertGreaterEqual(size, 8)
                    if atom == b'moov':
                        source.write_bytes(data[:offset])
                        break
                    offset += size
                else:
                    self.fail('Synthetic moov atom not found')
            elif kind == 'video_only':
                source = root/'video.mp4'
                self.fixture_command(['-f', 'lavfi', '-i', 'color=c=black:s=32x32:r=1:d=1', '-an', '-c:v', 'mpeg4', '-metadata', 'title='+SECRET, str(source)])
            work = root/'work'
            work.mkdir()
            stdout, stderr, logs = io.StringIO(), io.StringIO(), io.StringIO()
            sink = logging.StreamHandler(logs)
            logger = logging.getLogger('ffmpeg_diagnostics')
            def retrieve(url, destination):
                shutil.copyfile(source, destination)
                return destination, None
            def convert(path, **kwargs):
                output = Path(pipeline.extract_audio(path))
                try:
                    with wave.open(str(output), 'rb') as audio:
                        self.assertEqual((audio.getframerate(), audio.getnchannels(), audio.getsampwidth(), audio.getnframes()), (16000, 1, 2, 16000))
                    return {'conversion_ok': True}
                finally:
                    output.unlink()
            with patch.object(tempfile, 'tempdir', str(work)), patch.object(self.handler.urllib.request, 'urlretrieve', side_effect=retrieve), patch.object(self.handler, 'process', side_effect=convert), patch.object(logger, 'handlers', [sink]), patch.object(logger, 'propagate', False), redirect_stdout(stdout), redirect_stderr(stderr):
                response = self.handler.handler({'input': {'media_url':'https://example.invalid/'+CANARY+source.suffix+'?signature=FAKE_SECRET', 'max_duration_sec':1800}})
            self.assert_private_absent(stdout.getvalue()+stderr.getvalue()+logs.getvalue()+json.dumps(response))
            self.assertEqual(stderr.getvalue(), '')
            self.assertEqual(list(work.iterdir()), [], 'Temporary conversion/download file remains')
            if kind == 'valid_wav':
                self.assertEqual(response, {'conversion_ok':True})
                self.assertEqual(stdout.getvalue()+logs.getvalue(), '')
            else:
                self.assertTrue(response['error'].startswith(diagnostics.MARKER))
                payload = json.loads(response['error'][len(diagnostics.MARKER):])
                self.assertIn('missing_moov_atom' if kind == 'corrupt_m4a' else 'missing_audio_stream', payload['categories'])
                self.assertEqual(len(logs.getvalue().splitlines()), 1)
                if kind == 'corrupt_m4a':
                    self.assertIn('FFPROBE_DIAGNOSTIC_V1 ', stdout.getvalue())
                    self.assertIn('probe_failed', stdout.getvalue())
                else:
                    self.assertEqual(stdout.getvalue(), '')

    def test_chain_valid_wav(self):
        self.exercise('valid_wav')

    def test_chain_corrupt_m4a(self):
        self.exercise('corrupt_m4a')

    def test_chain_video_without_audio(self):
        self.exercise('video_only')


if __name__ == '__main__':
    unittest.main()
