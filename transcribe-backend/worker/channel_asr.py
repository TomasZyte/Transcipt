"""Prepared, test-only WhisperX adapter for explicitly confirmed split channels.

No model loading, downloads, cloud submission or production imports. The CLI
only validates local inputs; inference requires caller-supplied callbacks.
Channel identity is not proof of speaker identity throughout an entire call.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import wave

from asr_comparison import SHA, config_for, digest, finite, read_json
from contextual_asr import model_evidence

PROFILE = 'channel-asr-v1'
RATE = 16000
MAX_SECONDS = 240
CHANNELS = ('left', 'right')


def _confirmation(value, duration):
    if not isinstance(value, dict) or set(value) != {'kind', 'start_seconds', 'end_seconds'}:
        raise ValueError('Explicit owner preview confirmation required')
    start, end = value['start_seconds'], value['end_seconds']
    if value['kind'] != 'owner_preview' or not all(finite(x) for x in (start, end)):
        raise ValueError('Invalid confirmation')
    if not 0 <= start < end <= duration:
        raise ValueError('Confirmation outside audio')


def prepare(manifest_path, *, expected_profile=PROFILE):
    """Validate two complete, synchronised PCM channels without running models."""
    manifest_path = Path(manifest_path).resolve()
    data = read_json(manifest_path)
    if expected_profile not in (PROFILE, 'channel-runs-v1'):
        raise ValueError('Unknown channel profile')
    if set(data) != {'profile', 'source_sha256', 'confirmation', 'channels'} or data['profile'] != expected_profile:
        raise ValueError('Unexpected channel manifest')
    if not isinstance(data['source_sha256'], str) or not SHA.fullmatch(data['source_sha256']):
        raise ValueError('Original source fingerprint required')
    rows = data['channels']
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValueError('Exactly left and right channels required')
    frames, paths = [], []
    for name, row in zip(CHANNELS, rows):
        if not isinstance(row, dict) or set(row) != {'channel', 'file', 'sha256', 'source_start_seconds'}:
            raise ValueError('Unexpected channel fields')
        filename = row['file']
        if row['channel'] != name or not isinstance(filename, str) or not filename.endswith('.wav'):
            raise ValueError('Channel identity or WAV filename invalid')
        if any(c in filename for c in ('/', '\\', ':')) or filename in ('.', '..'):
            raise ValueError('WAV must be beside the manifest')
        if type(row['source_start_seconds']) not in (int, float) or row['source_start_seconds'] != 0:
            raise ValueError('Full channels must retain the original time origin')
        path = manifest_path.parent / filename
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_SECONDS * RATE * 2 + 4096:
            raise ValueError('Missing, linked or oversized WAV')
        if not isinstance(row['sha256'], str) or not SHA.fullmatch(row['sha256']) or digest(path) != row['sha256']:
            raise ValueError('Channel checksum mismatch')
        with wave.open(str(path), 'rb') as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) != (1, 2, RATE, 'NONE'):
                raise ValueError('Expected mono PCM16 at 16 kHz')
            count = wav.getnframes()
            if not 0 < count <= MAX_SECONDS * RATE or len(wav.readframes(count)) != count * 2:
                raise ValueError('Truncated or out-of-limit channel')
        frames.append(count)
        paths.append(path.resolve())
    if frames[0] != frames[1] or paths[0] == paths[1] or rows[0]['sha256'] == rows[1]['sha256']:
        raise ValueError('Distinct channels must have exactly equal lengths')
    duration = frames[0] / RATE
    _confirmation(data['confirmation'], duration)
    return {'profile': expected_profile, 'status': 'prepared_not_run', 'manifest_sha256': digest(manifest_path),
            'source_sha256': data['source_sha256'], 'confirmation': deepcopy(data['confirmation']),
            'channels': deepcopy(rows), 'frames_per_channel': frames[0], 'duration_seconds': duration,
            'total_asr_audio_seconds': 2 * duration, 'asr_model': 'large-v3',
            'transcribe_options': config_for('wx_ru')['transcribe'],
            'full_call_speaker_identity_verified': False, 'quality_validated': False,
            'cloud_runner_connected': True, 'paid_jobs': 0}


def _plain(segments):
    return ''.join(''.join(s['text'].split()) for s in segments)


def _tokens(item, duration, *, time_tolerance=0.):
    raw, aligned = item['raw']['segments'], item['aligned']['segments']
    if item['raw'].get('language') != 'ru':
        raise ValueError('unexpected_language')
    if _plain(raw) != _plain(aligned):
        raise ValueError('alignment_changed_text')
    result, previous = [], -1.0
    for si, segment in enumerate(aligned):
        text, words = segment['text'], segment.get('words', [])
        if not words and text.strip():
            raise ValueError('missing_word_times')
        mapped, cursor = [], 0
        for word in words:
            token = word.get('word', '').strip()
            offset = text.find(token, cursor)
            if not token or offset < 0 or any(c.isalnum() for c in text[cursor:offset]):
                raise ValueError('unmapped_text')
            start, end = word.get('start'), word.get('end')
            if not all(finite(x) for x in (start, end)) or not 0 <= start < end <= duration + time_tolerance:
                raise ValueError('unreliable_word_time')
            if start < previous:
                raise ValueError('unordered_words')
            if end - start > 2:
                raise ValueError('long_word_requires_review')
            previous = start
            mapped.append((offset, start, end))
            cursor = offset + len(token)
        if any(c.isalnum() for c in text[cursor:]):
            raise ValueError('unmapped_text')
        for wi, (offset, start, end) in enumerate(mapped):
            left = 0 if wi == 0 else offset
            right = mapped[wi + 1][0] if wi + 1 < len(mapped) else len(text)
            result.append({'channel': item['channel'], 'segment_index': si, 'word_index': wi,
                           'startSec': start, 'endSec': end, 'text': text[left:right]})
    if not result:
        raise ValueError('channel_without_transcript')
    return result


def merge_channels(evidence, frame_count, *, time_tolerance=0.):
    """Interleave words without clipping overlaps, assigning roles or removing echo."""
    if type(frame_count) is not int or not 0 < frame_count <= RATE * MAX_SECONDS:
        raise ValueError('Unsupported channel duration')
    if type(time_tolerance) not in (int, float) or time_tolerance not in (0., .0005):
        raise ValueError('Only the pinned millisecond rounding tolerance is supported')
    evidence = model_evidence(deepcopy(evidence))
    if not isinstance(evidence, list) or [r.get('channel') for r in evidence] != list(CHANNELS):
        raise ValueError('Exactly two ordered channel results required')
    tokens, issues = [], []
    for item in evidence:
        try:
            tokens.extend(_tokens(item, frame_count / RATE, time_tolerance=time_tolerance))
        except (ValueError, KeyError, TypeError) as exc:
            issues.append({'channel': item['channel'], 'reason': str(exc) if isinstance(exc, ValueError)
                           else 'invalid_model_evidence'})
    output = {'profile': PROFILE, 'evidence': evidence, 'issues': issues,
              'quality_validated': False, 'release_ready': False,
              'speaker_source': 'audio_channel', 'diarization': 'not_run', 'summary': 'not_run',
              'status': 'review_required' if issues else 'candidate_unvalidated',
              'candidate_segments': None, 'cross_channel_word_overlaps': []}
    if issues:
        return output
    tokens.sort(key=lambda t: (t['startSec'], CHANNELS.index(t['channel']), t['segment_index'], t['word_index']))
    for i, a in enumerate(tokens):
        for b in tokens[i + 1:]:
            if b['startSec'] >= a['endSec']:
                break
            if a['channel'] != b['channel']:
                output['cross_channel_word_overlaps'].append({
                    'startSec': b['startSec'], 'endSec': min(a['endSec'], b['endSec']),
                    'left_channel': a['channel'], 'right_channel': b['channel']})
    turns = []
    for token in tokens:
        # A pause, sentence boundary or other-channel word starts a fresh turn.
        if (turns and turns[-1]['channel'] == token['channel']
                and turns[-1]['source_segment'] == token['segment_index']
                and token['startSec'] - turns[-1]['endSec'] <= .8):
            turns[-1]['text'] += token['text']
            turns[-1]['endSec'] = max(turns[-1]['endSec'], token['endSec'])
        else:
            turns.append({'id': f'seg-{len(turns) + 1}', 'channel': token['channel'],
                          'speaker': f'Спикер {CHANNELS.index(token["channel"]) + 1}',
                          'source_segment': token['segment_index'], 'startSec': token['startSec'],
                          'endSec': token['endSec'], 'text': token['text']})
    for turn in turns:
        turn['text'] = turn['text'].strip()
    for item in evidence:
        exported = [t for t in turns if t['channel'] == item['channel']]
        if _plain(exported) != _plain(item['raw']['segments']):
            raise ValueError('Export changed channel text')
    output['candidate_segments'] = turns
    return output


def decode_channels(left, right, *, transcribe, align, confirmation):
    """Callbacks use one preloaded WhisperX model, then Russian word alignment.

    Caller owns audio loading, runtime/model verification and budget enforcement.
    No diarizer or summary callback exists. Do not treat this as a cloud runner.
    """
    if os.getenv('CHANNEL_ASR_TEST') != '1':
        raise ValueError('Explicit isolated channel test flag required')
    frames = len(left)
    if frames != len(right) or not 0 < frames <= RATE * MAX_SECONDS:
        raise ValueError('Expected equal complete channels up to 240 seconds')
    _confirmation(confirmation, frames / RATE)
    evidence = []
    for channel, audio in zip(CHANNELS, (left, right)):
        raw = model_evidence(deepcopy(transcribe(audio, **config_for('wx_ru')['transcribe'])))
        item = {'channel': channel, 'raw': raw, 'aligned': None}
        evidence.append(item)
        item['aligned'] = model_evidence(align(deepcopy(raw['segments']), audio))
    result = merge_channels(evidence, frames)
    result['confirmation'] = deepcopy(confirmation)
    result['full_call_speaker_identity_verified'] = False
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate local channel inputs; never run inference')
    parser.add_argument('--manifest', type=Path, required=True)
    print(json.dumps(prepare(parser.parse_args().manifest), ensure_ascii=False, indent=2))
