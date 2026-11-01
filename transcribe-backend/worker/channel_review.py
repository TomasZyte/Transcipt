"""Offline review export of every saved speech run. Never a production transcript.

No inference, secrets or API clients. Evidence is validated, then copied into a
separate review document. Missing hypotheses remain explicit placeholders.
"""
from collections import Counter
from copy import deepcopy
from difflib import SequenceMatcher
import re

import channel_asr as base
import channel_pipeline
import channel_runs

VERSION = 'channel-review-v1'
REASONS = {
    'unresolved_speech': 'Речь не разобрана',
    'recognition_disagreement': 'Два прохода распознали разные слова',
    'native_timing_unreliable': 'Исходные времена распознавания ненадёжны',
    'uncertain_recognition': 'Модель сомневается в наличии или распознавании речи',
    'low_alignment_score': 'Низкая оценка выравнивания отдельных слов',
    'repeated_phrase': 'Фраза повторяется в нескольких отдельных участках',
    'control_text_disagreement': 'Распознавание полного канала дало другой текст',
    'long_word_requires_review': 'Слово растянуто более чем на две секунды',
    'missing_word_times': 'Нет выравнивания слов',
    'unreliable_word_time': 'Некорректное время слова',
    'weak_phrase_alignment': 'У значительной части фразы слабое выравнивание; слова требуют проверки',
    'repeated_low_confidence_phrase': 'Повторяющаяся фраза со слабым выравниванием',
}


def tokens(text):
    return re.findall(r'\w+', text.casefold().replace('ё', 'е'))


def disagreements(rows, control_text):
    """Compare wording only: bad control timestamps cannot locate new words."""
    expanded = [(token, r['run']) for r in rows for token in tokens(r['original_text'])]
    changed = set()
    for tag, _, _, a, b in SequenceMatcher(None, tokens(control_text),
            [p[0] for p in expanded], autojunk=False).get_opcodes():
        if tag == 'equal':
            continue
        changed.update(run for _, run in expanded[a:b])
        if a == b and expanded:  # Control-only words are not silently ignored.
            changed.add(expanded[min(a, len(expanded)-1)][1])
    return changed


def build(state, control):
    if state.get('profile') not in ('channel-speech-v1', 'channel-speech-v2') or control.get('profile') != 'channel-asr-v1':
        raise ValueError('Expected saved speech-bounded and full-channel results')
    for item in (state, control):
        channel_pipeline.validate_result(item, item['input_plan'])
    plan = state['input_plan']
    for key in ('source_sha256', 'channels', 'frames_per_channel', 'duration_seconds'):
        if plan[key] != control['input_plan'][key]:
            raise ValueError('Control belongs to different audio')
    result = state['result']
    flags = {(r['channel'], r['run']): r for r in result['review_spans']}
    recovered = {(r['region']['channel'], a['run']): a
                 for r in result['recovery_evidence'] for a in r['alignments']}
    rows = []
    for channel in result['original']['run_evidence']:
        for index, run in enumerate(channel['runs']):
            name = channel['channel']; key = name, index
            flag, replacement = flags.get(key), recovered.get(key)
            raw = replacement['raw_local'] if replacement else run['raw_local']
            aligned = replacement['aligned_local'] if replacement else run['aligned_local']
            text = None if raw is None else ''.join(s['text'] for s in raw['segments']).strip()
            win = run['window']; start, end = win['start_frame']/base.RATE, win['end_frame']/base.RATE
            reasons = set(flag['reasons'] if flag else [])
            if text is not None:
                try:
                    base._tokens({'raw': raw, 'aligned': aligned, 'channel': name},
                                 end-start, time_tolerance=.0005)
                except (ValueError, KeyError, TypeError) as exc:
                    reasons.add(str(exc) if isinstance(exc, ValueError) else 'invalid_alignment')
                if any(base.finite(w.get('score')) and w['score'] < .1
                       for s in aligned['segments'] for w in s.get('words', [])):
                    reasons.add('low_alignment_score')
            else:
                reasons.add('unresolved_speech')
            rows.append({'id': f'{name}-{index}', 'channel': name, 'run': index,
                'startSec': start, 'endSec': end, 'time_basis': 'original_vad_window',
                'original_text': run['raw_local']['segments'][0]['text'].strip(),
                'text': text, 'words': [] if aligned is None else [deepcopy(w)
                    for s in channel_runs.global_alignment(aligned, start)['segments']
                    for w in s.get('words', [])], 'reasons': sorted(reasons),
                'human_verified': False})
    counts = Counter(tuple(tokens(r['text'] or '')) for r in rows)
    control_text = {e['channel']: ''.join(s['text'] for s in e['raw']['segments'])
                    for e in control['result']['evidence']}
    changed = {name: disagreements([r for r in rows if r['channel'] == name], text)
               for name, text in control_text.items()}
    for row in rows:
        phrase = tuple(tokens(row['text'] or ''))
        if len(phrase) >= 2 and counts[phrase] >= 3:
            row['reasons'].append('repeated_phrase')
        if row['run'] in changed[row['channel']]:
            row['reasons'].append('control_text_disagreement')
        row['reasons'] = sorted(set(row['reasons']))
        row['status'] = ('unresolved_speech' if row['text'] is None else
                         'requires_review' if row['reasons'] else 'unverified')
        row['overlaps'] = [r['id'] for r in rows if r['channel'] != row['channel']
                           and r['startSec'] < row['endSec'] and r['endSec'] > row['startSec']]
    rows.sort(key=lambda r: (r['startSec'], r['channel'], r['run']))
    return {'profile': VERSION, 'purpose': 'human_review_only', 'release_ready': False,
            'production_export_allowed': False, 'quality_validated': False,
            'full_call_speaker_identity_verified': False, 'recognition_performed': False,
            'duration_seconds': plan['duration_seconds'], 'rows': rows,
            'control_text': control_text, 'counts': {
                'speech_runs': len(rows), 'requires_review': sum(bool(r['reasons']) for r in rows),
                'unresolved': sum(r['text'] is None for r in rows),
                'by_channel': dict(Counter(r['channel'] for r in rows))},
            'limitations': ['All VAD runs are included; speech outside detected runs may be missing.',
                'No independent listening or complete reference transcript.',
                'Agreement, CTC score and a channel label do not prove correctness or speaker identity.']}


def timestamp(value):
    minutes, seconds = divmod(value, 60)
    return f'{int(minutes):02}:{seconds:06.3f}'


def text_export(review):
    if review.get('profile') != VERSION or review.get('production_export_allowed') is not False:
        raise ValueError('Only marked review exports are supported')
    lines = ['ЧЕРНОВИК ДЛЯ ПРОВЕРКИ — WhisperX, без нового распознавания',
        'Все сохранённые речевые участки. Полнота речи и правильность слов не подтверждены.',
        'Канал 1 — левый, канал 2 — правый. Это каналы записи; роли не установлены.',
        'Время строки — окно речи, не точные границы каждого слова. Перекрытия сохранены.', '']
    for row in review['rows']:
        label = 'Канал 1' if row['channel'] == 'left' else 'Канал 2'
        mark = ' [ПРОВЕРИТЬ]' if row['reasons'] else ''
        content = row['text'] if row['text'] is not None else '[НЕРАЗБОРЧИВАЯ РЕЧЬ — текст не восстановлен]'
        lines.append(f"[{timestamp(row['startSec'])}–{timestamp(row['endSec'])}] {label}{mark}: {content}")
        if row['reasons']:
            lines.append('  Причины: ' + '; '.join(REASONS.get(r, r) for r in row['reasons']))
        if row['overlaps']:
            lines.append('  Одновременно есть участок другого канала.')
        lines.append('')
    return '\n'.join(lines)
