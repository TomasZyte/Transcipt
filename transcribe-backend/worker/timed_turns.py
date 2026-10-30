"""Test-only timestamp-guided WhisperX alignment and acoustic diarization variants."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import gc
import inspect
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

import candidate_pipeline as base
from asr_comparison import config_for, digest, offline_models, read_json, verify_assets, write_new
from contextual_asr import model_evidence
from full_pipeline import IDS, validate_audio_rows
from prepare_full_assets import SOURCES, prepare_diarization, verify_bundle, verify_nltk
from quality_batch_handler import validate_request, fetch_sample
from short_turn_comparison import short_turn_filter, validate_runtime_pipeline, PARAMETERS
from speaker_comparison import annotation_rows
from transcript_candidate import RATE, POLICY, finite, pause_windows, global_alignment, attribute, validate_timeline

PROFILE = 'timed-turns-v1'
DIARIZATION = {
    'short': {'step_ratio': .1, 'exclude_overlap': True},
    'dense': {'step_ratio': .05, 'exclude_overlap': True},
    'overlap': {'step_ratio': .05, 'exclude_overlap': False}}
MODES = {f'timed_{key}': {'asr': 'timed', 'diarization': key} for key in DIARIZATION}
DECODER = {'language': 'ru', 'task': 'transcribe', 'beam_size': 5, 'temperature': 0.,
           'condition_on_previous_text': False, 'word_timestamps': True, 'without_timestamps': False,
           'vad_filter': False, 'initial_prompt': None, 'prefix': None, 'hotwords': None}


def require_test():
    expected = {'FULL_PIPELINE_IMAGE': '1', 'FULL_PIPELINE_TEST': '1', 'TIMED_TURNS_TEST': '1',
                'RELEASE_CANDIDATE_TEST': '0', 'SHORT_TURN_COMPARISON_TEST': '0',
                'SPEAKER_COMPARISON_TEST': '0', 'CONTEXT_ASR_TEST': '0', 'ASR_COMPARISON_TEST': '0'}
    if any(os.getenv(k, '0') != v for k, v in expected.items()):
        raise ValueError('Dedicated timed-turn test flags required')


def validate_input(inp):
    require_test()
    if (not isinstance(inp, dict) or set(inp) != {'profile', 'samples'} or inp['profile'] != PROFILE
            or not isinstance(inp['samples'], list) or len(inp['samples']) != 4
            or any(not isinstance(s, dict) or set(s) != {'id', 'sha256', 'media_url'} for s in inp['samples'])
            or [s['id'] for s in inp['samples']] != IDS):
        raise ValueError('Expected four audio files without text, timing or speaker hints')
    validate_request({'profile': 'community', 'samples': inp['samples']}, os.getenv('QUALITY_R2_HOST', ''))
    return inp['samples']


@contextmanager
def acoustic_settings(pipeline, settings):
    original = (pipeline.segmentation_step, pipeline._segmentation.step, pipeline.embedding_exclude_overlap)
    duration = float(pipeline._segmentation.model.specifications.duration)
    if original != (.1, duration * .1, True) or settings not in DIARIZATION.values():
        raise ValueError('Unexpected acoustic implementation or setting')
    try:
        pipeline.segmentation_step = settings['step_ratio']
        pipeline._segmentation.step = duration * settings['step_ratio']
        pipeline.embedding_exclude_overlap = settings['exclude_overlap']
        yield {'step_seconds': pipeline._segmentation.step,
               'step_ratio': pipeline.segmentation_step,
               'exclude_overlap': pipeline.embedding_exclude_overlap}
    finally:
        pipeline.segmentation_step, pipeline._segmentation.step, pipeline.embedding_exclude_overlap = original


def plain(text):
    return ''.join(text.split())


def sentence_bounds(native, duration):
    """Bound CTC sentences by this same decoder's word times, not human labels.

    Keep the original text including punctuation. No fixed sound2 intervals,
    guessed speaker identities, reference phrases or corrected transcript.
    """
    output = []
    for segment in native:
        text = segment['text']
        words = segment.get('words') or []
        if not words or plain(''.join(w['word'] for w in words)) != plain(text):
            raise ValueError('Timestamp decoder did not retain every text token')
        cursor, mapped, previous = 0, [], 0.
        for word in words:
            token = word['word'].strip()
            pos = text.find(token, cursor)
            a, b = word['start'], word['end']
            if (not token or pos < 0 or text[cursor:pos].strip()
                    or not finite(a) or not finite(b) or not previous <= a <= b <= duration + .05):
                raise ValueError('Invalid decoder text or local timing')
            mapped.append((word, pos)); cursor = pos + len(token); previous = a
        if text[cursor:].strip():
            raise ValueError('Decoder omitted trailing text')
        cuts = [0] + [i+1 for i,(w,_) in enumerate(mapped[:-1]) if re.search(r'[.!?…]["»\)]?$',w['word'].strip())] + [len(mapped)]
        for left, right in zip(cuts, cuts[1:]):
            a = max(0., mapped[left][0]['start'] - .08)
            b = min(duration, mapped[right-1][0]['end'] + .08)
            # Adjacent speech is not available as extra alignment context.
            if left: a = max(a, mapped[left-1][0]['end'])
            if right < len(mapped): b = min(b, mapped[right][0]['start'])
            if not a < b:
                raise ValueError('Decoder supplied no usable sentence interval')
            char_a = 0 if left == 0 else mapped[left][1]
            char_b = len(text) if right == len(mapped) else mapped[right][1]
            output.append({'start': a, 'end': b, 'text': text[char_a:char_b]})
    if plain(''.join(s['text'] for s in output)) != plain(''.join(s['text'] for s in native)):
        raise ValueError('Sentence splitting lost text')
    return output


def decode(audio, windows, model, align):
    records, raw_all, aligned_all = [], [], []
    for window in windows:
        a,b = window['start_frame'],window['end_frame']
        chunk = audio[a:b]
        segments, info = model.model.transcribe(chunk, **DECODER)
        if info.language != 'ru': raise ValueError('Unexpected decoder language')
        native = model_evidence([asdict(s) for s in segments])
        bounds = sentence_bounds(native, len(chunk)/RATE)
        aligned = model_evidence(align(deepcopy(bounds), chunk))
        raw = {'language': 'ru', 'segments': bounds}
        records.append({'window': deepcopy(window), 'native': native, 'bounds': bounds, 'aligned_local': aligned})
        raw_all.extend(global_alignment(raw, a/RATE)['segments'])
        aligned_all.extend(global_alignment(aligned,a/RATE)['segments'])
    return {'language': 'ru', 'segments': raw_all}, {'segments': aligned_all}, records


def validate_sample(sample):
    duration=sample['duration_seconds']
    if set(sample['diarization']) != set(DIARIZATION) or set(sample['outputs']) != set(MODES):
        raise ValueError('Missing comparison variants')
    for name, settings in DIARIZATION.items():
        observed=sample['diarization'][name]
        expected={**settings,'step_seconds':10*settings['step_ratio']}
        if observed['settings'] != expected or observed['parameters'] != PARAMETERS:
            raise ValueError('Unexpected diarization settings')
        diag=observed['filter_diagnostics']
        if (len(diag)!=1 or diag[0]['min_active_ratio']!=.1
                or not 0 < diag[0]['selected_embeddings'] <= diag[0]['chunk_speaker_pairs']):
            raise ValueError('Missing embedding filter evidence')
        for kind in ('regular','exclusive'): validate_timeline(observed[kind],duration)
    plan=pause_windows(sample['diarization']['short']['exclusive'],round(duration*RATE))
    if [w['window'] for w in sample['windows']]!=plan: raise ValueError('Window plan changed')
    raw, aligned=[],[]
    for record in sample['windows']:
        win=record['window']; offset=win['start_frame']/RATE
        bounds=sentence_bounds(record['native'],(win['end_frame']-win['start_frame'])/RATE)
        if record['bounds']!=bounds: raise ValueError('Sentence bounds changed')
        raw.extend(global_alignment({'segments':bounds},offset)['segments'])
        aligned.extend(global_alignment(record['aligned_local'],offset)['segments'])
    if sample['asr']!={'language':'ru','segments':raw} or sample['alignment']!={'segments':aligned}:
        raise ValueError('Global evidence differs from local windows')
    for name,mode in MODES.items():
        if sample['outputs'][name]!=attribute(sample['asr'],sample['alignment'],
                sample['diarization'][mode['diarization']]['regular'],duration):
            raise ValueError('Export cannot be reproduced')


def verify_decoder_api():
    from whisperx.asr import WhisperModel
    from faster_whisper import WhisperModel as Parent
    if WhisperModel.transcribe is not Parent.transcribe:
        raise ValueError('The reviewed timestamp decoder changed')
    if not set(DECODER) <= set(inspect.signature(WhisperModel.transcribe).parameters):
        raise ValueError('Timestamp decoder options unavailable')


def execute(manifest, output_path):
    require_test()
    rows=read_json(manifest)['samples']; validate_audio_rows(rows)
    state={'profile':PROFILE,'status':'started','phase':'assets','samples':[],
           'source_manifest_sha256':digest(SOURCES),'modes':MODES,'policy':POLICY,
           'diarization_configs':DIARIZATION,'decoder':DECODER,'human_quality_validated':False}
    begun=time.perf_counter()
    try:
        config=config_for('wx_auto'); assets=verify_assets('/opt/asr/model-lock.json',config)
        alignment=verify_bundle('/opt/full/alignment','alignment'); nltk=verify_nltk('/opt/full')
        diar_root=Path(manifest).parent/'diarization'; diar_assets=prepare_diarization(diar_root,os.getenv('HF_TOKEN',''))
        state['runtime']={'asr':assets,'alignment':alignment,'diarization':diar_assets,'nltk_lock_sha256':nltk}
        with offline_models():
            import torch, whisperx
            from importlib.metadata import version
            from pyannote.audio import Pipeline
            from whisperx.vads.pyannote import load_vad_model
            verify_decoder_api()
            state['runtime']['packages']={p:version(p) for p in ('torch','whisperx','pyannote.audio','faster-whisper')}
            vad=load_vad_model('cuda',model_fp=assets['vad_path'],vad_onset=.5,vad_offset=.363)
            model=whisperx.load_model(assets['root'],device='cuda',local_files_only=True,vad_model=vad,**deepcopy(config['load']))
            am,metadata=whisperx.load_align_model('ru','cuda',model_name='/opt/full/alignment',model_cache_only=True)
            pipeline=Pipeline.from_pretrained(str(diar_root),token=False).to(torch.device('cuda'))
            validate_runtime_pipeline(pipeline,version('pyannote.audio'))
            if pipeline._segmentation.model.specifications.duration != 10: raise ValueError('Unexpected segmentation duration')
            for row in rows:
                audio=whisperx.load_audio(row['path'])
                sample={'id':row['id'],'sha256':row['sha256'],'duration_seconds':row['duration_seconds'],
                        'diarization':{},'outputs':{}}
                state['samples'].append(sample)
                for name, settings in DIARIZATION.items():
                    state['phase']=row['id']+':'+name
                    diagnostics=[]; torch.manual_seed(0)
                    with acoustic_settings(pipeline,settings) as actual, short_turn_filter(pipeline.clustering,.1,diagnostics):
                        result=pipeline({'waveform':torch.from_numpy(audio[None,:]),'sample_rate':RATE})
                    sample['diarization'][name]={'settings':actual,'parameters':model_evidence(pipeline.parameters(instantiated=True)),
                        'filter_diagnostics':diagnostics,
                        'regular':annotation_rows(result.speaker_diarization,row['duration_seconds'],False),
                        'exclusive':annotation_rows(result.exclusive_speaker_diarization,row['duration_seconds'],True)}
                state['phase']=row['id']+':timed_asr'
                windows=pause_windows(sample['diarization']['short']['exclusive'],len(audio))
                align=lambda segments,chunk: whisperx.align(segments,am,metadata,chunk,'cuda',return_char_alignments=False)
                sample['asr'],sample['alignment'],sample['windows']=decode(audio,windows,model,align)
                for name, mode in MODES.items():
                    sample['outputs'][name]=attribute(sample['asr'],sample['alignment'],
                        sample['diarization'][mode['diarization']]['regular'],row['duration_seconds'])
                validate_sample(sample)
                gc.collect(); torch.cuda.empty_cache()
        state.update(status='completed',phase='complete')
    except Exception as exc: state.update(status='failed',error_type=type(exc).__name__)
    state['elapsed_seconds']=round(time.perf_counter()-begun,3)
    write_new(output_path,state)
    return state


def handler(job):
    stage='validate'
    try:
        samples=validate_input(job.get('input'))
        with tempfile.TemporaryDirectory(prefix='timed-turns-') as directory:
            root=Path(directory); rows=[]; stage='download_audio'
            for sample in samples:
                path=root/(sample['id']+'.wav'); fetch_sample(sample,path)
                rows.append({'id':sample['id'],'sha256':sample['sha256'],'path':str(path)})
            validate_audio_rows(rows)
            manifest,result_path=root/'manifest.json',root/'result.json'
            write_new(manifest,{'samples':rows}); stage='child'
            child=subprocess.run([sys.executable,str(Path(__file__)),'--manifest',str(manifest),
                '--output',str(result_path)],timeout=270,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=False)
            if child.returncode or not result_path.is_file(): raise RuntimeError('No timed-turn evidence')
            return read_json(result_path)
    except Exception as exc: return {'profile':PROFILE,'status':'failed','phase':stage,'error_type':type(exc).__name__}


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--manifest',required=True); parser.add_argument('--output',required=True)
    args=parser.parse_args(); execute(args.manifest,args.output)
