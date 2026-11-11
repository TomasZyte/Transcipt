"""Test-only alignment ablation: decoder bounds versus the full acoustic window.

Keep WhisperX text and original audio common to every mode. No reference words,
speaker roles, expected counts or manual timestamps are accepted by this worker.
"""
import argparse
from collections import Counter
from copy import deepcopy
import gc
from itertools import combinations
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import timed_turns as base
from transcript_candidate import UNKNOWN, merge_intervals

PROFILE = 'pause-context-v1'
ALIGNMENTS = ('control', 'extend_tail', 'joint_window')
APPROXIMATE_TIMING = 'тайминг приблизительный'
UNRESOLVED_SPEAKER = 'спикер не определён'
DIARIZATION = {key: deepcopy(base.DIARIZATION[key]) for key in ('short', 'dense')}
MODES = {a + '_' + d: {'alignment': a, 'diarization': d}
         for a in ALIGNMENTS for d in DIARIZATION}


def require_test():
    expected = {'FULL_PIPELINE_IMAGE': '1', 'FULL_PIPELINE_TEST': '1', 'PAUSE_CONTEXT_TEST': '1',
                'ACOUSTIC_TURNS_TEST': '0', 'TIMED_TURNS_TEST': '0', 'RELEASE_CANDIDATE_TEST': '0',
                'SHORT_TURN_COMPARISON_TEST': '0', 'SPEAKER_COMPARISON_TEST': '0',
                'CONTEXT_ASR_TEST': '0', 'ASR_COMPARISON_TEST': '0'}
    if any(os.getenv(key, '0') != value for key, value in expected.items()):
        raise ValueError('Dedicated pause-context test flags required')


def validate_input(inp):
    require_test()
    if (not isinstance(inp, dict) or set(inp) != {'profile', 'samples'} or inp['profile'] != PROFILE
            or not isinstance(inp['samples'], list) or len(inp['samples']) != 4
            or any(not isinstance(s, dict) or set(s) != {'id', 'sha256', 'media_url'} for s in inp['samples'])
            or [s['id'] for s in inp['samples']] != base.IDS):
        raise ValueError('Only four complete audio files without inference hints')
    base.validate_request({'profile': 'community', 'samples': inp['samples']}, os.getenv('QUALITY_R2_HOST', ''))
    return inp['samples']


def alignment_bounds(native, duration, mode):
    """Never copy a human boundary or re-decode text to fit an expected answer.

    extend_tail isolates the last sentence's right-bound restriction; joint_window
    gives CTC all ordered text in a single, at most 15-second acoustic context.
    A single joined segment avoids aligning repeated phrases independently into
    the same audio occurrence. The historical control remains byte-for-byte.
    """
    if mode not in ALIGNMENTS or not base.finite(duration) or not 0 < duration <= base.POLICY['max_window_seconds']:
        raise ValueError('Invalid bounded alignment context')
    bounds = base.sentence_bounds(native, duration)
    if mode == 'control' or not bounds:
        return bounds
    if mode == 'extend_tail':
        bounds[-1]['end'] = duration
        return bounds
    text = ''.join(s['text'] for s in bounds)
    return [{'start': 0., 'end': duration, 'text': text}]


def realign(records, audio, align, mode):
    segments, local = [], []
    for record in records:
        win = record['window']; a, b = win['start_frame'], win['end_frame']
        bounds = alignment_bounds(record['native'], (b - a) / base.RATE, mode)
        aligned = (deepcopy(record['aligned_local']) if mode == 'control' else
                   base.model_evidence(align(deepcopy(bounds), audio[a:b])))
        local.append({'window': deepcopy(win), 'bounds': bounds, 'aligned_local': aligned})
        segments.extend(base.global_alignment(aligned, a / base.RATE)['segments'])
    return {'segments': segments}, local


def _word_times(segments):
    tokens, times = [], []
    for seg in segments:
        previous = seg['start']
        for word in seg.get('words', []):
            a, b = word.get('start'), word.get('end')
            valid = base.finite(a) and base.finite(b) and previous <= a < b <= seg['end']
            tokens.append(word['word'].strip())
            times.append((a, b) if valid else None)
            if valid:
                previous = a
    return tokens, times


def _timing_issue(intervals):
    if None in intervals:
        return 'missing_or_invalid_time'
    if any(b - a > base.POLICY['max_word_seconds'] for a, b in intervals):
        return 'long_word'
    for (a, b), (c, d) in combinations(intervals, 2):
        overlap = max(0., min(b, d) - max(a, c))
        union = (b - a) + (d - c) - overlap
        if max(abs(a - c), abs(b - d)) >= .25 and overlap < .5 * union:
            return 'timings_disagree'
    return None


def phrase_timing_review(sample):
    """Compare words by occurrence within the same audio window. Preserve the control phrases."""
    phrases = []
    for records in zip(*(sample['alignment_windows'][mode] for mode in ALIGNMENTS), strict=True):
        window = records[0]['window']
        if any(record['window'] != window for record in records):
            raise ValueError('Cannot compare timings from different windows')
        variants = [record['aligned_local']['segments'] for record in records]
        tokens, times = zip(*(_word_times(segments) for segments in variants))
        comparable = all(seq == tokens[0] for seq in tokens[1:])
        offset = window['start_frame'] / base.RATE
        index = 0
        for seg in variants[0]:
            count = len(seg.get('words', []))
            phrase_tokens = tokens[0][index:index + count]
            if not comparable or not count or base.plain(seg['text']) != base.plain(''.join(phrase_tokens)):
                reasons = {'incomplete_word_comparison'}
            else:
                reasons = {_timing_issue([group[i] for group in times]) for i in range(index, index + count)}
                reasons.discard(None)
            phrases.append({'text': seg['text'], 'startSec': offset + seg['start'], 'endSec': offset + seg['end'],
                            'timingNote': APPROXIMATE_TIMING if reasons else None, 'reasons': sorted(reasons)})
            index += count
    return phrases


def _speaker_maps(tracks):
    """Match anonymous IDs only when each voice overlaps the other for a majority of its time."""
    durations, timelines = {}, {}
    for mode in ('dense', 'short'):
        totals = Counter()
        intervals = []
        for row in tracks[mode]['exclusive']:
            start, end = (round(row[k] * 1_000_000) for k in ('start', 'end'))
            if intervals and start < intervals[-1][1]:
                raise ValueError('Cannot match speakers on overlapping exclusive turns')
            label = row['speaker']
            intervals.append((start, end, label))
            totals[label] += end - start
        durations[mode], timelines[mode] = totals, intervals
    shared = Counter()
    dense, short = timelines['dense'], timelines['short']
    i = j = 0
    while i < len(dense) and j < len(short):
        a, b, first = dense[i]
        c, d, second = short[j]
        overlap = min(b, d) - max(a, c)
        if overlap > 0:
            shared[first, second] += overlap
        if b <= d:
            i += 1
        if d <= b:
            j += 1
    matched = {second: first for (first, second), time in shared.items()
               if UNKNOWN not in (first, second) and 2 * time > max(durations['dense'][first], durations['short'][second])}
    return {'short': matched, 'dense': {label: label for label in matched.values()}}


def _single_voice_phrases(sample, maps):
    """Resolve one coverage-related abstention per phrase, without changing any word label."""
    speech = {}
    for track, mapping in maps.items():
        evidence = sample['diarization'][track]
        if len(mapping) != 1 or not evidence.get('regular'):
            return {}
        labels = {row['speaker'] for kind in ('regular', 'exclusive') for row in evidence[kind]}
        if labels != set(mapping):
            return {}
        speech[track] = merge_intervals((row['start'], row['end']) for row in evidence['regular'])

    resolved = {}
    for index, variants in _matched_phrase_words(sample):
        size = len(next(iter(variants.values())))
        uncertain = {i for mode, words in variants.items() for i, (speaker, span) in enumerate(words)
                     if speaker not in maps[MODES[mode]['diarization']] or span is None}
        if size > 1 and len(uncertain) == 1:
            i = uncertain.pop()
            reason = _coverage_support({mode: words[i] for mode, words in variants.items()},
                                       maps, speech, interior=0 < i < size - 1)
            if reason:
                resolved[index] = reason
    return resolved


def _matched_phrase_words(sample):
    """Yield comparable word occurrences only if all six outputs reproduce the full transcript."""
    control = sample['alignments']['control']['segments']
    groups = [[base.plain(w['word']) for w in s.get('words', [])] for s in control]
    tokens = [token for group in groups for token in group]
    if any(''.join(g) != base.plain(s['text']) for g, s in zip(groups, control)):
        return
    variants = {}
    for mode in MODES:
        segments = sample['outputs'][mode]['attributed']['segments']
        sequence, times = _word_times(segments)
        if sequence != tokens or any(base.plain(seg['text']) != ''.join(base.plain(w['word']) for w in seg.get('words', []))
                                     for seg in segments):
            return
        variants[mode] = [(w.get('speaker'), span) for w, span in
                          zip((w for seg in segments for w in seg.get('words', [])), times, strict=True)]

    pos = 0
    for index, group in enumerate(groups):
        end = pos + len(group)
        yield index, {mode: words[pos:end] for mode, words in variants.items()}
        pos = end


def _coverage_support(observations, maps, speech, *, interior):
    anchors, missing = [], []
    for alignment in ALIGNMENTS:
        pair = []
        for track in DIARIZATION:
            speaker, span = observations[alignment + '_' + track]
            if span is None or span[1] - span[0] > base.POLICY['max_word_seconds']:
                return None
            if speaker in maps[track]:
                pair.append(span)
                continue
            if speaker != UNKNOWN:
                return None
            a, b = span
            pieces = [(max(a, x), min(b, y)) for x, y in speech[track] if a < y and x < b]
            covered = sum(y-x for x, y in pieces)
            if not 0 < covered < base.POLICY['speaker_min_coverage'] * (b-a):
                return None
            missing.append((span, pieces, covered))
        if len(pair) == 2:
            anchors.append((max(s[0] for s in pair), min(s[1] for s in pair)))
    if not missing:
        return None
    if anchors and all(any(max(x, a) < min(y, b) for x, y in pieces for a, b in anchors)
                       for _, pieces, _ in missing):
        return 'same_word_confirmed_on_both_tracks'
    if interior and all(len(pieces) == 2 and pieces[0][0] == a and pieces[-1][1] == b
                        and pieces[1][0] - pieces[0][1] <= base.POLICY['min_pause_seconds']
                        and covered > .5 * (b-a) for (a,b), pieces, covered in missing):
        return 'single_voice_across_short_pause'
    return None


def _ends_in_silence(variants, tracks, maps):
    """One stable local turn and word onset; only the final offset escapes into silence."""
    labels, anchors, missing_tracks, starts = set(), [], set(), []
    phrase_start, latest_end = float('inf'), 0.
    for alignment in ALIGNMENTS:
        pair = []
        for track in DIARIZATION:
            words = variants[alignment + '_' + track]
            if len(words) < 2 or any(span is None for _, span in words):
                return False
            for speaker, span in words[:-1]:
                if speaker not in maps[track] or span[1] - span[0] > base.POLICY['max_word_seconds']:
                    return False
                labels.add(maps[track][speaker])
            speaker, (a, b) = words[-1]
            if max(span[1] for _, span in words[:-1]) > a:
                return False
            if speaker in maps[track] and b - a <= base.POLICY['max_word_seconds']:
                labels.add(maps[track][speaker])
                pair.append((a, b))
            elif speaker == UNKNOWN:
                missing_tracks.add((track, b))
            else:
                return False
            starts.append(a)
            phrase_start = min(phrase_start, words[0][1][0])
            latest_end = max(latest_end, b)
        if len(pair) == 2:
            anchors.extend(pair)
    if len(labels) != 1 or not anchors or not missing_tracks:
        return False
    # The spread in onsets must be small relative to the shortest confirmed reading.
    if max(starts) - min(starts) > .5 * min(b-a for a,b in anchors):
        return False
    voice = labels.pop()
    for track, evidence in tracks.items():
        for kind in ('regular', 'exclusive'):
            rows = [r for r in evidence[kind] if r['start'] < latest_end and r['end'] > phrase_start]
            if not rows or any(maps[track].get(r['speaker']) != voice for r in rows):
                return False
            runs = merge_intervals((r['start'], r['end']) for r in rows)
            if len(runs) != 1:
                return False
            start, end = runs[0]
            if start > phrase_start or max(starts) >= end or max(b for _, b in anchors) > end:
                return False
            if any(b <= end for t, b in missing_tracks if t == track):
                return False
    return True


def phrase_speaker_review(sample, *, version=3):
    """Use the original text positions, even when an alignment joins sentences or splits a word."""
    phrases = [base.plain(s['text']) for s in sample['alignments']['control']['segments']]
    text = ''.join(phrases)
    maps = _speaker_maps(sample['diarization'])
    if version not in (1, 2, 3):
        raise ValueError('Unsupported phrase speaker review version')
    supported = _single_voice_phrases(sample, maps) if version >= 2 else {}
    tracks = sample['diarization']
    if version >= 3 and all(len(maps[t]) >= 2 and tracks[t].get('regular') for t in DIARIZATION):
        for index, variants in _matched_phrase_words(sample):
            if _ends_in_silence(variants, tracks, maps):
                supported[index] = 'same_local_voice_before_silence'
    voices = []
    for mode, definition in MODES.items():
        segments = sample['outputs'][mode]['attributed']['segments']
        if ''.join(base.plain(s['text']) for s in segments) != text:
            raise ValueError('Cannot compare speakers on different transcripts')
        mapping = maps[definition['diarization']]
        labels = []
        for seg in segments:
            words = seg.get('words') or []
            tokens = [base.plain(w['word']) for w in words]
            sentence = base.plain(seg['text'])
            if ''.join(tokens) != sentence:
                # No positional match is reliable in this segment (e.g. omitted repeated word).
                labels.extend([None] * len(sentence))
                continue
            for token, word in zip(tokens, words):
                labels.extend([mapping.get(word.get('speaker'))] * len(token))
        voices.append(labels)
    result = []
    start = 0
    for index, phrase in enumerate(phrases):
        end = start + len(phrase)
        labels = {label for group in voices for label in group[start:end]}
        reasons = []
        if not labels or None in labels:
            reasons.append('unassigned_or_unmatched_speaker')
        if len(labels - {None}) > 1:
            reasons.append('multiple_speaker_labels')
        row = {'speakerNote': UNRESOLVED_SPEAKER if reasons else None, 'speakerReasons': reasons}
        if reasons == ['unassigned_or_unmatched_speaker'] and index in supported:
            row = {'speakerNote': None, 'speakerReasons': [], 'speakerSupport': supported[index]}
        result.append(row)
        start = end
    return result


def validate_sample(sample):
    duration = sample['duration_seconds']
    if (set(sample['diarization']) != set(DIARIZATION) or set(sample['alignments']) != set(ALIGNMENTS)
            or set(sample['alignment_windows']) != set(ALIGNMENTS) or set(sample['outputs']) != set(MODES)):
        raise ValueError('Incomplete context comparison')
    for name, settings in DIARIZATION.items():
        evidence = sample['diarization'][name]
        if evidence['settings'] != {**settings, 'step_seconds': 10 * settings['step_ratio']} or evidence['parameters'] != base.PARAMETERS:
            raise ValueError('Diarization changed')
        diag = evidence['filter_diagnostics']
        if (len(diag) != 1 or diag[0]['min_active_ratio'] != .1
                or not 0 < diag[0]['selected_embeddings'] <= diag[0]['chunk_speaker_pairs']):
            raise ValueError('Missing embedding selection evidence')
        for kind in ('regular', 'exclusive'):
            base.validate_timeline(evidence[kind], duration)
    plan = base.pause_windows(sample['diarization']['short']['exclusive'], round(duration * base.RATE))
    if [w['window'] for w in sample['windows']] != plan:
        raise ValueError('Acoustic windows changed')
    raw = []
    for record in sample['windows']:
        win = record['window']; offset = win['start_frame'] / base.RATE
        bounds = alignment_bounds(record['native'], (win['end_frame'] - win['start_frame']) / base.RATE, 'control')
        if record['bounds'] != bounds:
            raise ValueError('Native control changed')
        raw.extend(base.global_alignment({'segments': bounds}, offset)['segments'])
    if sample['asr'] != {'language': 'ru', 'segments': raw}:
        raise ValueError('Common WhisperX text changed')
    for mode in ALIGNMENTS:
        records = sample['alignment_windows'][mode]
        if len(records) != len(plan):
            raise ValueError('Missing alignment window')
        global_segments = []
        for original, record in zip(sample['windows'], records):
            win = original['window']; a, b = win['start_frame'], win['end_frame']
            expected = alignment_bounds(original['native'], (b - a) / base.RATE, mode)
            if record['window'] != win or record['bounds'] != expected:
                raise ValueError('Context bounds cannot be reproduced')
            if mode == 'control' and record['aligned_local'] != original['aligned_local']:
                raise ValueError('Control alignment changed')
            aligned = record['aligned_local']
            if base.plain(''.join(s['text'] for s in aligned['segments'])) != base.plain(''.join(s['text'] for s in expected)):
                raise ValueError('Alignment changed ordered text')
            for segment in aligned['segments']:
                start, end = segment.get('start'), segment.get('end')
                if (not base.finite(start) or not base.finite(end) or not 0 <= start < end <= (b-a)/base.RATE + .001):
                    raise ValueError('Alignment escaped the acoustic window')
            global_segments.extend(base.global_alignment(aligned, a / base.RATE)['segments'])
        if sample['alignments'][mode] != {'segments': global_segments}:
            raise ValueError('Audio offset changed')
    for name, mode in MODES.items():
        rebuilt = base.attribute(sample['asr'], sample['alignments'][mode['alignment']],
                                 sample['diarization'][mode['diarization']]['regular'], duration)
        if sample['outputs'][name] != rebuilt:
            raise ValueError('Export cannot be reproduced')
    if 'phraseTimingReview' in sample and sample['phraseTimingReview'] != phrase_timing_review(sample):
        raise ValueError('Phrase timing review cannot be reproduced')
    if 'phraseSpeakerReview' in sample and sample['phraseSpeakerReview'] != phrase_speaker_review(
            sample, version=sample.get('phraseSpeakerReviewVersion', 1)):
        raise ValueError('Phrase speaker review cannot be reproduced')


def execute(manifest, output_path):
    require_test()
    rows = base.read_json(manifest)['samples']; base.validate_audio_rows(rows)
    state = {'profile': PROFILE, 'status': 'started', 'phase': 'assets', 'samples': [],
             'source_manifest_sha256': base.digest(base.SOURCES), 'modes': MODES, 'policy': base.POLICY,
             'diarization_configs': DIARIZATION, 'alignment_configs': list(ALIGNMENTS),
             'decoder': base.DECODER, 'human_quality_validated': False}
    begun = time.perf_counter()
    try:
        config = base.config_for('wx_auto'); assets = base.verify_assets('/opt/asr/model-lock.json', config)
        alignment = base.verify_bundle('/opt/full/alignment', 'alignment'); nltk = base.verify_nltk('/opt/full')
        diar_root = Path(manifest).parent / 'diarization'
        diar_assets = base.prepare_diarization(diar_root, os.getenv('HF_TOKEN', ''))
        state['runtime'] = {'asr': assets, 'alignment': alignment, 'diarization': diar_assets, 'nltk_lock_sha256': nltk}
        with base.offline_models():
            import torch, whisperx
            from importlib.metadata import version
            from pyannote.audio import Pipeline
            from whisperx.vads.pyannote import load_vad_model
            base.verify_decoder_api()
            state['runtime']['packages'] = {p: version(p) for p in ('torch', 'whisperx', 'pyannote.audio', 'faster-whisper')}
            vad = load_vad_model('cuda', model_fp=assets['vad_path'], vad_onset=.5, vad_offset=.363)
            model = whisperx.load_model(assets['root'], device='cuda', local_files_only=True, vad_model=vad, **deepcopy(config['load']))
            am, metadata = whisperx.load_align_model('ru', 'cuda', model_name='/opt/full/alignment', model_cache_only=True)
            pipeline = Pipeline.from_pretrained(str(diar_root), token=False).to(torch.device('cuda'))
            base.validate_runtime_pipeline(pipeline, version('pyannote.audio'))
            if pipeline._segmentation.model.specifications.duration != 10:
                raise ValueError('Unexpected segmentation duration')
            for row in rows:
                audio = whisperx.load_audio(row['path'])
                sample = {'id': row['id'], 'sha256': row['sha256'], 'duration_seconds': row['duration_seconds'],
                          'diarization': {}, 'alignments': {}, 'alignment_windows': {}, 'outputs': {}}
                state['samples'].append(sample)
                for name, settings in DIARIZATION.items():
                    state['phase'] = row['id'] + ':' + name
                    diagnostics = []; torch.manual_seed(0)
                    with base.acoustic_settings(pipeline, settings) as actual, base.short_turn_filter(pipeline.clustering, .1, diagnostics):
                        result = pipeline({'waveform': torch.from_numpy(audio[None, :]), 'sample_rate': base.RATE})
                    sample['diarization'][name] = {'settings': actual, 'parameters': base.model_evidence(pipeline.parameters(instantiated=True)),
                        'filter_diagnostics': diagnostics,
                        'regular': base.annotation_rows(result.speaker_diarization, row['duration_seconds'], False),
                        'exclusive': base.annotation_rows(result.exclusive_speaker_diarization, row['duration_seconds'], True)}
                windows = base.pause_windows(sample['diarization']['short']['exclusive'], len(audio))
                align = lambda segments, chunk: whisperx.align(segments, am, metadata, chunk, 'cuda', return_char_alignments=False)
                state['phase'] = row['id'] + ':decode'
                sample['asr'], _, sample['windows'] = base.decode(audio, windows, model, align)
                for name in ALIGNMENTS:
                    state['phase'] = row['id'] + ':' + name
                    sample['alignments'][name], sample['alignment_windows'][name] = realign(sample['windows'], audio, align, name)
                for name, mode in MODES.items():
                    sample['outputs'][name] = base.attribute(sample['asr'], sample['alignments'][mode['alignment']],
                        sample['diarization'][mode['diarization']]['regular'], row['duration_seconds'])
                sample['phraseTimingReview'] = phrase_timing_review(sample)
                sample['phraseSpeakerReview'] = phrase_speaker_review(sample)
                sample['phraseSpeakerReviewVersion'] = 3
                validate_sample(sample)
                gc.collect(); torch.cuda.empty_cache()
        state.update(status='completed', phase='complete')
    except Exception as exc:
        state.update(status='failed', error_type=type(exc).__name__)
    state['elapsed_seconds'] = round(time.perf_counter() - begun, 3)
    base.write_new(output_path, state)
    return state


def handler(job):
    stage = 'validate'
    try:
        samples = validate_input(job.get('input'))
        with tempfile.TemporaryDirectory(prefix='pause-context-') as directory:
            root = Path(directory); rows = []; stage = 'download_audio'
            for sample in samples:
                path = root / (sample['id'] + '.wav'); base.fetch_sample(sample, path)
                rows.append({'id': sample['id'], 'sha256': sample['sha256'], 'path': str(path)})
            base.validate_audio_rows(rows)
            manifest, result_path = root / 'manifest.json', root / 'result.json'
            base.write_new(manifest, {'samples': rows}); stage = 'child'
            child = subprocess.run([sys.executable, str(Path(__file__)), '--manifest', str(manifest),
                '--output', str(result_path)], timeout=270, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            if child.returncode or not result_path.is_file():
                raise RuntimeError('No pause-context evidence')
            return base.read_json(result_path)
    except Exception as exc:
        return {'profile': PROFILE, 'status': 'failed', 'phase': stage, 'error_type': type(exc).__name__}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--manifest', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); execute(args.manifest, args.output)
