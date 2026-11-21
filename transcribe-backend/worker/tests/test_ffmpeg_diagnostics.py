from contextlib import redirect_stderr
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import ffmpeg_diagnostics as f

PRIVATE = 'PRIVATE_TOKEN email@example.invalid +79991234567 /private/customer.m4a https://private.invalid/?signature=SECRET'


class SafeDiagnosticsTests(unittest.TestCase):
    def launch(self, code):
        real_popen = subprocess.Popen
        self.command = None
        def replacement(command, **kwargs):
            self.command = command
            return real_popen([sys.executable, '-I', '-B', '-c', code, command[-1]], **kwargs)
        return patch.object(f.subprocess, 'Popen', side_effect=replacement)

    def assert_private_absent(self, text):
        for value in ('PRIVATE_TOKEN', 'email@example', '+7999', '/private/', 'private.invalid', 'signature=', 'SECRET'):
            self.assertNotIn(value, text)

    def test_all_categories_and_unknown_stderr(self):
        for category, phrases in f.RULES.items():
            for phrase in phrases:
                with self.subTest(category=category, phrase=phrase):
                    result = f.diagnostic(PRIVATE.encode()+b'\n'+phrase+b'\n', 1)
                    self.assertIn(category, result['categories'])
                    self.assert_private_absent(json.dumps(result))
        result = f.diagnostic(PRIVATE.encode()+b'\xff\x00', 1)
        self.assertEqual(result['categories'], ['unclassified'])

    def test_full_pipe_failure_has_no_stderr_or_command_in_traceback(self):
        code = 'import sys; sys.stderr.buffer.write(' + repr((PRIVATE+'\nmoov atom not found\n').encode()) + '); sys.exit(1)'
        private_path = '/private/customer.m4a'
        with self.launch(code), self.assertLogs('ffmpeg_diagnostics', level='ERROR') as logs:
            try:
                f.extract_audio(private_path)
            except f.SafeMediaConversionError as exc:
                rendered = traceback.format_exc()
                self.assertIsNone(exc.__context__)
                self.assertFalse(hasattr(exc, 'stderr'))
                payload = json.loads(str(exc).removeprefix(f.MARKER))
            else:
                self.fail('Expected failure')
        self.assertEqual(payload['exit_code'], 1)
        self.assertEqual(payload['categories'], ['missing_moov_atom'])
        self.assertEqual(len(logs.output), 1)
        self.assert_private_absent(rendered + ''.join(logs.output))
        self.assertFalse(Path(self.command[-1]).exists())
        self.assertIn('-nostdin', self.command)

    def test_large_stderr_is_drained_and_tail_is_bounded(self):
        code = 'import sys; sys.stderr.buffer.write(b"X"*300000+b"\\nPermission denied\\n"); sys.exit(1)'
        with self.launch(code), self.assertLogs('ffmpeg_diagnostics', level='ERROR'), self.assertRaises(f.SafeMediaConversionError) as ctx:
            f.extract_audio('/private/input.m4a')
        payload = json.loads(str(ctx.exception).removeprefix(f.MARKER))
        self.assertTrue(payload['stderr_tail_truncated'])
        self.assertIn('permission_denied', payload['categories'])
        self.assertLess(len(str(ctx.exception)), 600)

    def test_success_returns_pcm_file_without_diagnostics(self):
        code = 'import sys,wave; w=wave.open(sys.argv[1],"wb"); w.setparams((1,2,16000,0,"NONE","not compressed")); w.writeframes(b"\\0\\0"*160); w.close()'
        with self.launch(code), patch.object(f.logging.getLogger('ffmpeg_diagnostics'), 'error') as log:
            path = f.extract_audio('/private/input.m4a')
        try:
            import wave
            with wave.open(path) as wav:
                self.assertEqual((wav.getframerate(), wav.getnchannels(), wav.getsampwidth()), (16000, 1, 2))
            self.assertEqual(self.command[-7:-1], ['-ar','16000','-ac','1','-c:a','pcm_s16le'])
            log.assert_not_called()
        finally:
            os.remove(path)

    def test_launch_exceptions_are_safe_and_output_removed(self):
        for error in (FileNotFoundError(PRIVATE), PermissionError(PRIVATE), OSError(PRIVATE), ValueError(PRIVATE)):
            with self.subTest(error=type(error).__name__):
                with patch.object(f.subprocess, 'Popen', side_effect=error) as launch, self.assertLogs('ffmpeg_diagnostics', level='ERROR') as logs:
                    with self.assertRaises(f.SafeMediaConversionError) as ctx:
                        f.extract_audio('/private/input.m4a')
                self.assert_private_absent(str(ctx.exception)+''.join(logs.output))
                self.assertIsNone(ctx.exception.__context__)
                self.assertFalse(Path(launch.call_args.args[0][-1]).exists())

    def test_empty_stderr_and_untrusted_exit_or_kind(self):
        result = f.diagnostic(b'', 1)
        self.assertFalse(result['stderr_present'])
        self.assertEqual(result['categories'], ['unclassified'])
        result = f.diagnostic(b'', PRIVATE, kind=PRIVATE)
        self.assertIsNone(result['exit_code'])
        self.assertEqual(result['kind'], 'local_execution_failure')
        self.assert_private_absent(json.dumps(result))

    def test_partial_output_cleanup_failure_does_not_leak(self):
        code = 'import sys; sys.exit(1)'
        with self.launch(code), patch.object(f.os, 'remove', side_effect=PermissionError(PRIVATE)), self.assertLogs('ffmpeg_diagnostics', level='ERROR'):
            with self.assertRaises(f.SafeMediaConversionError) as ctx:
                f.extract_audio('/private/input.m4a')
        try:
            payload = json.loads(str(ctx.exception).removeprefix(f.MARKER))
            self.assertTrue(payload['partial_output_cleanup_failed'])
            self.assert_private_absent(str(ctx.exception))
        finally:
            os.remove(self.command[-1])

    def test_pipe_read_error_kills_child_and_cleans_output(self):
        child = MagicMock()
        child.__enter__.return_value = child
        child.stderr.read.side_effect = OSError(PRIVATE)
        with patch.object(f.subprocess, 'Popen', return_value=child) as launch, self.assertLogs('ffmpeg_diagnostics', level='ERROR'):
            with self.assertRaises(f.SafeMediaConversionError) as ctx:
                f.extract_audio(PRIVATE)
        child.kill.assert_called_once()
        child.wait.assert_called_once()
        self.assertFalse(Path(launch.call_args.args[0][-1]).exists())
        self.assert_private_absent(str(ctx.exception))

    def test_negative_process_returncode_and_truncated_old_signal(self):
        result = f.diagnostic(b'no space left on device'+b'X'*(f.TAIL_LIMIT+10), -9)
        self.assertEqual(result['exit_code'], -9)
        self.assertEqual(result['categories'], ['unclassified'])
        self.assertTrue(result['stderr_tail_truncated'])

    def test_handler_returns_safe_json_without_traceback(self):
        fake_runpod = SimpleNamespace(serverless=SimpleNamespace(start=lambda config: None))
        fake_pipeline = SimpleNamespace(process=lambda *a, **k: None)
        spec = importlib.util.spec_from_file_location('candidate_handler', HERE/'handler.py')
        handler = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'runpod': fake_runpod, 'pipeline': fake_pipeline}):
            spec.loader.exec_module(handler)
        message = f.MARKER + json.dumps(f.diagnostic(b'moov atom not found', 1))
        console = io.StringIO()
        with patch.object(handler, '_handle_transcribe', side_effect=f.SafeMediaConversionError(message)), redirect_stderr(console):
            response = handler.handler({'input': {}})
        self.assertEqual(response, {'error': message})
        self.assertEqual(console.getvalue(), '')
        self.assert_private_absent(json.dumps(response))

    def test_pipeline_calls_new_converter_and_docker_includes_module(self):
        spec = importlib.util.spec_from_file_location('candidate_pipeline_test', HERE/'pipeline.py')
        pipeline = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'transcript_segments': SimpleNamespace(build_segments=lambda *a: None)}):
            spec.loader.exec_module(pipeline)
        with patch.object(f, 'extract_audio', return_value='test.wav') as convert:
            self.assertEqual(pipeline.extract_audio('test.m4a'), 'test.wav')
            convert.assert_called_once_with('test.m4a')
        self.assertIn('local_server.py ffmpeg_diagnostics.py ./', (HERE/'Dockerfile').read_text())


if __name__ == '__main__':
    unittest.main()
