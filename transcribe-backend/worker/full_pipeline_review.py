"""Offline diagnostics of saved full-pipeline outputs; never rewrites evidence."""
from collections import defaultdict
from difflib import SequenceMatcher
import math
import re


def _time(value):
    return type(value) in (float,int) and math.isfinite(value)


def word_support(word, timeline):
    start,end = word.get('start'),word.get('end')
    if not all(_time(x) for x in (start,end)) or not 0 <= start < end:
        return {'status':'invalid_word_time','word':word.get('word'),'assigned':word.get('speaker')}
    spans = timeline.get('segments')
    if timeline.get('status') != 'ok' or not isinstance(spans,list):
        return {'status':'timeline_unavailable','word':word.get('word'),'assigned':word.get('speaker')}
    sums = defaultdict(float)
    for span in spans:
        a,b = span.get('start'),span.get('end')
        if not all(_time(x) for x in (a,b)) or not 0 <= a < b or not isinstance(span.get('speaker'),str):
            raise ValueError('Invalid timeline segment')
        overlap = min(end,b)-max(start,a)
        if overlap > 0:
            sums[span['speaker']] += overlap
    leaders = sorted(k for k,v in sums.items() if math.isclose(v,max(sums.values()),abs_tol=1e-9)) if sums else []
    flags = []
    if end-start > 2: flags.append('long_word')
    score = word.get('score')
    if _time(score) and score < .1: flags.append('low_alignment_score')
    if len(sums)>1: flags.append('multiple_overlapping_labels')
    if not sums: flags.append('no_timeline_overlap')
    return {'status':'review','word':word.get('word'),'start':start,'end':end,
            'alignment_score':score,'assigned':word.get('speaker'),
            'overlap_seconds':{k:round(v,6) for k,v in sorted(sums.items())},
            'largest_overlap_labels':leaders,'assignment_matches_overlap':word.get('speaker') in leaders if leaders else None,
            'flags':flags,
            'note':'Overlap and alignment scores are not speaker-identity confidence; labels may already merge people.'}


def text_changes(control, candidate):
    def tokens(text): return re.findall(r'\w+',text.casefold(),flags=re.UNICODE)
    a,b = tokens(control),tokens(candidate)
    rows = []
    for tag,i,j,k,l in SequenceMatcher(None,a,b,autojunk=False).get_opcodes():
        if tag != 'equal':
            rows.append({'operation':tag,'control_tokens':a[i:j],'candidate_tokens':b[k:l],
                         'control_range':[i,j],'candidate_range':[k,l]})
    return {'control_token_count':len(a),'candidate_token_count':len(b),
            'changes':rows,'note':'Lexical disagreement only; neither output is a verified reference. Not WER.'}


def review_sample(sample):
    outputs = sample['outputs']
    if set(outputs) != {'standard30','short12'}:
        raise ValueError('Expected the two saved full-file modes')
    reports = {}
    for mode,output in outputs.items():
        benchmark = output['benchmark']
        timeline = benchmark['diarizationTimeline']
        reports[mode] = {
            'asr_segments':benchmark['asrSegments'],
            'timeline':timeline,
            'words':[word_support(w,timeline) for s in benchmark['alignedSegments'] for w in s.get('words',[])],
            'text_preserved':all(output['audit'][k] for k in ('exportMatchesRebuild',
                'exportTextMatchesAlignedIgnoringWhitespace','alignedTextMatchesAsrIgnoringWhitespace'))}
    return {'id':sample['id'],'audio_sha256':sample['sha256'],'modes':reports,
            'same_diarization':reports['standard30']['timeline']==reports['short12']['timeline'],
            'asr_disagreements':text_changes(outputs['standard30']['rawAsrText'],outputs['short12']['rawAsrText']),
            'audio_listened_by_agent':False,'recognition_performed':False,'source_modified':False}
