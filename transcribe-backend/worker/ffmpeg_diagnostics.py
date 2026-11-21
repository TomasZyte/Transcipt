"""Bounded, allowlisted diagnostics. Never persist command, filenames or raw stderr."""
import json
import logging
import os
import subprocess
import tempfile

TAIL_LIMIT = 65536
MARKER = 'FFMPEG_DIAGNOSTIC_V1 '
RULES = {
    'missing_moov_atom': (b'moov atom not found',),
    'invalid_media': (b'invalid data found when processing input', b'error reading header'),
    'missing_audio_stream': (b'does not contain any stream', b'matches no streams', b'no audio stream'),
    'decoder_unavailable': (b'unknown decoder', b'decoder not found', b'decoding requested, but no decoder found'),
    'decode_failure': (b'error while decoding', b'error submitting packet to decoder'),
    'disk_full': (b'no space left on device',),
    'permission_denied': (b'permission denied',),
    'input_missing': (b'no such file or directory',),
    'resource_exhausted': (b'cannot allocate memory', b'resource temporarily unavailable'),
}


class SafeMediaConversionError(RuntimeError):
    """Contains only the already classified JSON, no subprocess exception or stderr."""


def diagnostic(stderr, exit_code, *, kind='conversion_failed', truncated=False, cleanup_failed=False):
    lowered = stderr[-TAIL_LIMIT:].lower()
    categories = [key for key, values in RULES.items() if any(v in lowered for v in values)]
    if kind not in ('conversion_failed', 'local_file_or_executable_missing', 'local_permission_denied',
                    'local_io_failure', 'local_execution_failure'):
        kind = 'local_execution_failure'
    return {'schema': 'ffmpeg-safe-v1', 'stage': 'audio_conversion', 'kind': kind,
            'exit_code': exit_code if type(exit_code) is int and -255 <= exit_code <= 255 else None,
            'categories': categories or ['unclassified'], 'stderr_present': bool(stderr),
            'stderr_tail_truncated': bool(truncated or len(stderr) > TAIL_LIMIT),
            'partial_output_cleanup_failed': bool(cleanup_failed)}


def extract_audio(media_path):
    """Same PCM conversion as before; bounded in-memory stderr and safe failures."""
    output = None
    failure = None
    tail = bytearray()
    truncated = False
    returncode = None
    try:
        fd, output = tempfile.mkstemp(suffix='.wav')
        os.close(fd)
        command = ['ffmpeg', '-nostdin', '-hide_banner', '-nostats', '-loglevel', 'error',
                   '-y', '-i', media_path, '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', output]
        with subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                              stderr=subprocess.PIPE) as child:
            try:
                while True:
                    chunk = child.stderr.read(8192)
                    if not chunk:
                        break
                    tail.extend(chunk)
                    if len(tail) > TAIL_LIMIT:
                        truncated = True
                        del tail[:-TAIL_LIMIT]
                returncode = child.wait()
            except BaseException:
                child.kill()
                child.wait()
                raise
        if returncode == 0:
            return output
        failure = diagnostic(bytes(tail), returncode, truncated=truncated)
    except FileNotFoundError:
        failure = diagnostic(b'', None, kind='local_file_or_executable_missing')
    except PermissionError:
        failure = diagnostic(b'', None, kind='local_permission_denied')
    except OSError:
        failure = diagnostic(bytes(tail), returncode, kind='local_io_failure', truncated=truncated)
    except Exception:
        failure = diagnostic(bytes(tail), returncode, kind='local_execution_failure', truncated=truncated)
    finally:
        # Success transfers ownership to the pipeline. On all failures/cancellation remove partial WAV.
        if output is not None and (failure is not None or returncode != 0):
            try:
                os.remove(output)
            except FileNotFoundError:
                pass
            except OSError:
                if failure is not None:
                    failure['partial_output_cleanup_failed'] = True
    message = MARKER + json.dumps(failure, sort_keys=True, separators=(',', ':'))
    logging.getLogger(__name__).error('%s', message)
    # Outside the except block, so the original exception/command is not chained.
    raise SafeMediaConversionError(message) from None
