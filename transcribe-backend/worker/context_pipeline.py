"""Fixed test comparison on the full audio, retaining every overlapping window."""
from copy import deepcopy
import os

from contextual_asr import decode_with_context, reconcile, plan_windows, RATE, MODE
from full_pipeline import format_output, finite

PROFILE = 'context-pipeline-v1'
CONTROL = 'standard30_ru'
MODES = {CONTROL:30, MODE:{'core_seconds':12,'padding_seconds':2,'short_tail_seconds':4}}


def require_context():
    if any(os.getenv(k) != '1' for k in ('FULL_PIPELINE_IMAGE','FULL_PIPELINE_TEST','CONTEXT_ASR_TEST')):
        raise ValueError('Isolated image and both explicit test flags required')


def global_segments(segments, offset):
    result = deepcopy(segments)
    for segment in result:
        for item in [segment]+segment.get('words',[]):
            for key in ('start','end'):
                if key in item and finite(item[key]):
                    item[key] += offset
    return result


def export(raw, aligned, timeline, mode, duration):
    result = format_output(raw,aligned,timeline,'standard30',duration)
    result['processing'].update(mode=mode,languagePolicy='ru_from_full_file_detection')
    if mode != CONTROL:
        result['processing']['windowing'] = deepcopy(MODES[MODE])
        result['processing']['asrOrigin'] = 'window_reconciliation' if mode==MODE else 'context_window'
    return result


def assigned_segments(segments, assign):
    return assign({'segments':deepcopy(segments)})


def execute_sample(audio, *, model, align, assign, timeline, duration, output_sink=None):
    require_context()
    outputs = {} if output_sink is None else output_sink
    if not isinstance(outputs,dict) or outputs:
        raise ValueError('Output sink must start empty')
    # Determine language once from the full source; use it in BOTH arms.
    language = model.detect_language(audio)
    if language != 'ru':
        raise ValueError('This experiment requires detected Russian')
    raw = deepcopy(model.transcribe(audio,language=language,batch_size=8,task='transcribe',chunk_size=30))
    if raw.get('language') != language:
        raise ValueError('Unexpected control language')
    aligned = align(deepcopy(raw['segments']),audio)
    control = export(raw,assigned_segments(aligned['segments'],assign),timeline,CONTROL,duration)
    outputs[CONTROL] = control
    context = outputs[MODE] = {'status':'started','window_evidence':[]}
    context.update(decode_with_context(audio,transcribe=model.transcribe,align=align,language=language,
                                       evidence=context['window_evidence']))
    context['status'] = context['reconciliation']['status']
    context['diarization'] = 'shared_full_file'
    context['diarizationTimeline'] = deepcopy(timeline)
    context['window_outputs'] = []
    for item in context['window_evidence']:
        offset = item['window']['read_start_frame']/RATE
        raw_global = {'language':language,'segments':global_segments(item['raw']['segments'],offset)}
        aligned_global = global_segments(item['aligned']['segments'],offset)
        # Empty speech is retained as evidence, not turned into a fabricated transcript.
        output = export(raw_global,assigned_segments(aligned_global,assign),timeline,
                        'context_window',duration) if raw_global['segments'] else None
        context['window_outputs'].append({'index':item['window']['index'],'output':output})
    context['candidate'] = None
    candidate = context['reconciliation']['candidate_segments']
    if candidate:
        context['candidate'] = export({'language':language,'segments':candidate},
            assigned_segments(candidate,assign),timeline,MODE,duration)
    return outputs


def verify_export(output, timeline, mode, duration, expected_raw=None, expected_aligned=None):
    benchmark = output['benchmark']
    if benchmark['diarizationTimeline'] != timeline:
        raise ValueError('Speaker timeline changed between outputs')
    # Assignments may add speakers, but must never rewrite the aligned words/times.
    def without_speakers(segments):
        result = deepcopy(segments)
        for seg in result:
            seg.pop('speaker',None)
            for word in seg.get('words',[]):
                word.pop('speaker',None)
        return result
    if expected_raw is not None:
        from quality_diagnostics import snapshot_asr
        if benchmark['asrSegments'] != snapshot_asr(expected_raw):
            raise ValueError('Window ASR evidence changed')
    if expected_aligned is not None and without_speakers(benchmark['alignedSegments']) != without_speakers(expected_aligned):
        raise ValueError('Aligned window evidence changed')
    rebuilt = export({'language':'ru','segments':benchmark['asrSegments']},
        {'segments':benchmark['alignedSegments']},timeline,mode,duration)
    for key in ('segments','rawText','rawAsrText','srtContent','audit','wordCount','languageDetected'):
        if output[key] != rebuilt[key]:
            raise ValueError('Export does not match retained evidence')
    if output['processing'] != rebuilt['processing']:
        raise ValueError('Processing settings mismatch')


def validate_sample(outputs, frames, duration):
    if set(outputs) != set(MODES) or frames/RATE != duration:
        raise ValueError('Context sample identity mismatch')
    control, ctx = outputs[CONTROL], outputs[MODE]
    timeline = control['benchmark']['diarizationTimeline']
    if timeline.get('status') != 'ok' or not timeline.get('segments'):
        raise ValueError('Complete full-file timeline required')
    verify_export(control,timeline,CONTROL,duration)
    if (ctx.get('mode') != MODE or ctx.get('language') != 'ru' or ctx.get('source_frames') != frames
            or ctx.get('sample_rate') != RATE or ctx.get('quality_validated') is not False
            or ctx.get('network_submission') is not False or ctx.get('diarization') != 'shared_full_file'
            or ctx.get('diarizationTimeline') != timeline):
        raise ValueError('Unexpected contextual evidence')
    plan = plan_windows(frames)
    if ctx.get('read_frames') != sum(w['read_end_frame']-w['read_start_frame'] for w in plan):
        raise ValueError('Context coverage mismatch')
    evidence = ctx['window_evidence']
    rebuilt = reconcile(evidence,frames)
    if ctx['reconciliation'] != rebuilt or ctx.get('status') != rebuilt['status'] or len(ctx['window_outputs']) != len(plan):
        raise ValueError('Context reconciliation changed')
    for item,wrapped in zip(evidence,ctx['window_outputs']):
        if wrapped['index'] != item['window']['index'] or item['raw'].get('language') != 'ru':
            raise ValueError('Window identity mismatch')
        offset = item['window']['read_start_frame']/RATE
        raw = global_segments(item['raw']['segments'],offset)
        aligned = global_segments(item['aligned']['segments'],offset)
        if raw:
            verify_export(wrapped['output'],timeline,'context_window',duration,raw,aligned)
        elif wrapped['output'] is not None or aligned:
            raise ValueError('Unexpected output for empty window')
    candidate = rebuilt['candidate_segments']
    if candidate:
        verify_export(ctx['candidate'],timeline,MODE,duration,candidate,candidate)
    elif ctx['candidate'] is not None:
        raise ValueError('Ambiguous or empty context must not become a candidate')
