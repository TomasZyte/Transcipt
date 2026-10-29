"""Test-only contextual ASR adapter. No models, network, secrets or production imports.

Callers supply decoder/aligner callbacks and must explicitly enable execution.
All window evidence is retained. Ambiguous seams produce no stitched candidate.
This is a prepared hypothesis, not a validated quality improvement.
"""
from copy import deepcopy
import math
import os
import re

RATE = 16000
CORE = 12 * RATE
PADDING = 2 * RATE
MIN_TAIL = 4 * RATE
MODE = 'context12_pad2'


def plan_windows(frame_count):
    if type(frame_count) is not int or not 0 < frame_count <= 60 * RATE:
        raise ValueError('Expected a complete mono 16 kHz sample up to 60 seconds')
    boundaries = list(range(0,frame_count,CORE)) + [frame_count]
    # Avoid a tiny last inference (sound2 has only ~0.095 s after 36 s).
    # Redistribute the last two cores; keep every source frame and the 12 s cap.
    if len(boundaries) > 2 and boundaries[-1]-boundaries[-2] < MIN_TAIL:
        boundaries[-2] = (boundaries[-3]+boundaries[-1])//2
    return [{'index':i, 'core_start_frame':left, 'core_end_frame':right,
             'read_start_frame':max(0,left-PADDING), 'read_end_frame':min(frame_count,right+PADDING)}
            for i,(left,right) in enumerate(zip(boundaries,boundaries[1:]))]


def _finite(value):
    return type(value) in (float,int) and math.isfinite(value)


def _plain(segments):
    return ''.join(''.join(s['text'].split()) for s in segments)


def _words(evidence):
    """Map aligned tokens to original text; reject guessed/dropped word evidence."""
    window = evidence['window']
    duration = (window['read_end_frame']-window['read_start_frame'])/RATE
    offset = window['read_start_frame']/RATE
    raw, aligned = evidence['raw']['segments'], evidence['aligned']['segments']
    if _plain(raw) != _plain(aligned):
        raise ValueError('alignment_text_mismatch')
    result, previous, previous_midpoint = [], -1.0, -1.0
    for si,segment in enumerate(aligned):
        text, cursor = segment['text'], 0
        words = segment.get('words',[])
        if not words and text.strip():
            raise ValueError('missing_words')
        for wi,word in enumerate(words):
            token = word.get('word','').strip()
            if not token:
                raise ValueError('empty_word')
            char_start = text.find(token,cursor)
            if char_start < 0 or any(c.isalnum() for c in text[cursor:char_start]):
                raise ValueError('word_text_mismatch')
            start,end = word.get('start'),word.get('end')
            if not all(_finite(x) for x in (start,end)) or not 0 <= start < end <= duration:
                raise ValueError('unreliable_word_time')
            midpoint = (start+end)*RATE/2+window['read_start_frame']
            if start < previous or midpoint < previous_midpoint:
                raise ValueError('unordered_words')
            previous, previous_midpoint = start, midpoint
            result.append({'segment_index':si,'word_index':wi,'char_start':char_start,
                'char_end':char_start+len(token),'word':token,'start':start+offset,'end':end+offset,
                'midpoint_frame':midpoint,
                'score':word.get('score')})
            cursor = char_start+len(token)
        if any(c.isalnum() for c in text[cursor:]):
            raise ValueError('unmapped_text')
    return result


def _token(word):
    # Comparison only. Never substitute this normalization into the transcript.
    return re.sub(r'[^\w]+','',word.casefold())


def reconcile(evidence, frame_count):
    windows = plan_windows(frame_count)
    if [e.get('window') for e in evidence] != windows:
        raise ValueError('Evidence must cover the exact ordered window plan')
    reviewed, issues, seams = [], [], []
    for item in evidence:
        try:
            words = _words(item)
        except ValueError as exc:
            issues.append({'window':item['window']['index'],'reason':str(exc)})
            words = []
        reviewed.append(words)
        for word in words:
            # Review threshold only, never a confidence probability or a correction.
            if word['end']-word['start'] > 2:
                issues.append({'window':item['window']['index'],'reason':'long_word',
                               'word_index':word['word_index'],'segment_index':word['segment_index']})
    for i in range(len(windows)-1):
        left,right = windows[i:i+2]
        low, high, boundary = right['read_start_frame'],left['read_end_frame'],left['core_end_frame']
        a = [w for w in reviewed[i] if low <= w['midpoint_frame'] < high]
        b = [w for w in reviewed[i+1] if low <= w['midpoint_frame'] < high]
        tokens_equal = [_token(w['word']) for w in a] == [_token(w['word']) for w in b]
        ownership_equal = tokens_equal and [w['midpoint_frame'] < boundary for w in a] == [w['midpoint_frame'] < boundary for w in b]
        seams.append({'left_window':i,'right_window':i+1,'boundary_seconds':boundary/RATE,
                      'tokens_equal':tokens_equal,'ownership_equal':ownership_equal,
                      'left_words':[w['word'] for w in a],'right_words':[w['word'] for w in b]})
        if not ownership_equal:
            issues.append({'window':i,'reason':'ambiguous_seam'})
    result = {'mode':MODE,'status':'review_required' if issues else 'candidate_unvalidated',
              'issues':issues,'seams':seams,'candidate_segments':None,
              'note':'Seam agreement does not establish ASR or speaker accuracy. No diarization labels are inferred.'}
    if issues:
        return result
    candidate = []
    for item,words in zip(evidence,reviewed):
        win = item['window']
        owned = [w for w in words if win['core_start_frame'] <= w['midpoint_frame'] < win['core_end_frame']]
        for si,segment in enumerate(item['aligned']['segments']):
            selected = [w for w in owned if w['segment_index']==si]
            if not selected:
                continue
            all_words = [w for w in words if w['segment_index']==si]
            first,last = selected[0],selected[-1]
            # Preserve punctuation, internal spacing and real repeated words.
            char_start = 0 if first==all_words[0] else first['char_start']
            char_end = len(segment['text']) if last==all_words[-1] else all_words[last['word_index']+1]['char_start']
            candidate.append({'start':first['start'],'end':last['end'],
                              'text':segment['text'][char_start:char_end].strip(),
                              'words':[{k:w[k] for k in ('word','start','end','score')} for w in selected],
                              'source_window':win['index']})
    result['candidate_segments'] = candidate
    return result


def decode_with_context(audio, *, transcribe, align, language, evidence=None):
    """Callbacks use the same already-verified models as the full-file control.

    language must come from that control's detection, not human reference text.
    No GPU/model imports happen here. A separate paid runner is not connected yet.
    """
    if os.getenv('CONTEXT_ASR_TEST') != '1':
        raise ValueError('Explicit CONTEXT_ASR_TEST=1 required')
    if language != 'ru':
        raise ValueError('This experiment requires the control to detect Russian')
    if len(getattr(audio,'shape',(len(audio),))) != 1:
        raise ValueError('Expected mono audio')
    windows = plan_windows(len(audio))
    if evidence is None:
        evidence = []
    elif not isinstance(evidence,list) or evidence:
        raise ValueError('Evidence sink must start empty')
    for window in windows:
        chunk = audio[window['read_start_frame']:window['read_end_frame']]
        raw = deepcopy(transcribe(chunk,language=language,batch_size=8,task='transcribe',chunk_size=30))
        if raw.get('language') != language or not isinstance(raw.get('segments'),list):
            raise ValueError('Unexpected ASR result')
        item = {'window':deepcopy(window),'raw':raw}
        evidence.append(item)
        aligned = deepcopy(align(deepcopy(raw['segments']),chunk))
        item['aligned'] = aligned
    return {'mode':MODE,'language':language,'source_frames':len(audio),'sample_rate':RATE,
            'read_frames':sum(w['read_end_frame']-w['read_start_frame'] for w in windows),
            'window_evidence':evidence,'reconciliation':reconcile(evidence,len(audio)),
            'quality_validated':False,'diarization':'not_run','network_submission':False}
