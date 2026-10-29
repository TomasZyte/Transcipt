"""Test-only full-file ASR comparison, alignment, global diarization and export.

Two fixed chunk sizes, automatic speaker count, no reference words or LLM.
The normal production pipeline never imports this module.
"""
import argparse
from copy import deepcopy
import gc
import os
from pathlib import Path
import time

from asr_comparison import config_for, digest, finite, offline_models, read_json, verify_assets, write_new
from prepare_full_assets import prepare_diarization, verify_bundle, verify_nltk, SOURCES
from quality_diagnostics import snapshot_asr, snapshot_diarization, audit_result
from transcript_segments import build_segments

PROFILE = 'full-pipeline-v1'
MODES = {'standard30':30, 'short12':12}
IDS = ['sound1','sound2','sound3','sound4']


def validate_audio_rows(rows):
    import wave
    if not isinstance(rows, list) or [r.get('id') for r in rows] != IDS:
        raise ValueError('Exactly sound1-4 in fixed order are required')
    total = 0
    for row in rows:
        path = Path(row['path'])
        if path.is_symlink() or path.stat().st_size > 2*1024*1024 or digest(path) != row['sha256']:
            raise ValueError('Unexpected audio bytes')
        with wave.open(str(path), 'rb') as wav:
            if (wav.getnchannels(),wav.getsampwidth(),wav.getframerate(),wav.getcomptype()) != (1,2,16000,'NONE'):
                raise ValueError('Expected PCM16 mono 16 kHz')
            n = wav.getnframes()
            if len(wav.readframes(n)) != n*2:
                raise ValueError('Truncated WAV')
        duration = n/16000
        if not 0 < duration <= 60:
            raise ValueError('Sample exceeds duration limit')
        row['duration_seconds'] = duration
        total += duration
    if total > 180:
        raise ValueError('Batch exceeds total duration limit')


def _plain(rows):
    return ''.join(''.join(row.get('text','').split()) for row in rows)


def srt(segments):
    def timestamp(value):
        ms = round(value*1000)
        h, ms = divmod(ms,3600000)
        m, ms = divmod(ms,60000)
        sec, ms = divmod(ms,1000)
        return f'{h:02}:{m:02}:{sec:02},{ms:03}'
    return '\n\n'.join(f"{i}\n{timestamp(s['startSec'])} --> {timestamp(s['endSec'])}\n{s['speaker']}: {s['text']}"
                         for i,s in enumerate(segments,1)) + ('\n' if segments else '')


def format_output(raw, aligned, timeline, mode, duration):
    if mode not in MODES or not raw.get('segments'):
        raise ValueError('Expected non-empty fixed-mode ASR output')
    for row in raw['segments']:
        if not isinstance(row.get('text'),str) or not all(finite(row.get(k)) for k in ('start','end')):
            raise ValueError('Invalid ASR segment')
        if not 0 <= row['start'] < row['end'] <= duration+.05:
            raise ValueError('ASR times outside audio')
    # Preserve ASR evidence. Do not rewrite a model's output to fit a reference.
    if _plain(raw['segments']) != _plain(aligned['segments']):
        raise ValueError('Alignment changed ASR text')
    segments, stats = build_segments(aligned['segments'])
    if any(not 0 <= s['startSec'] <= s['endSec'] <= duration+.06 for s in segments):
        raise ValueError('Exported times outside audio')
    output = {'languageDetected':raw['language'], 'durationEstimateSec':duration,
        'rawText':'\n\n'.join(f"{s['speaker']}: {s['text']}" for s in segments),
        'rawAsrText':' '.join(s['text'].strip() for s in raw['segments']),
        'segments':segments, 'srtContent':srt(segments), 'summary':{},
        'wordCount':sum(len(s['text'].split()) for s in segments),
        'processing':{'asrModel':'large-v3','diarizationModel':'pyannote/speaker-diarization-community-1',
            'alignment':'ok','diarization':'ok','summary':'disabled_for_benchmark',
            'speakerTurns':stats,'mode':mode,'chunkSizeSeconds':MODES[mode]},
        'benchmark':{'asrSegments':snapshot_asr(raw['segments']),
            'alignedSegments':deepcopy(aligned['segments']), 'diarizationTimeline':deepcopy(timeline)}}
    output['audit'] = audit_result(output)
    if not all(output['audit'][k] for k in ('exportMatchesRebuild',
            'exportTextMatchesAlignedIgnoringWhitespace','alignedTextMatchesAsrIgnoringWhitespace')):
        raise ValueError('Export lost transcription evidence')
    return output


def execute(manifest, output_path):
    if os.getenv('FULL_PIPELINE_TEST') != '1' or os.getenv('FULL_PIPELINE_IMAGE') != '1':
        raise ValueError('Explicit isolated full-pipeline image and flag required')
    output_path = Path(output_path)
    if output_path.exists():
        raise ValueError('Do not overwrite an earlier attempt')
    rows = read_json(manifest)['samples']
    validate_audio_rows(rows)
    state = {'profile':PROFILE,'status':'started','samples':[], 'phase':'verify_assets',
             'modes':MODES, 'source_manifest_sha256':digest(SOURCES)}
    started = time.perf_counter()
    try:
        assets = verify_assets('/opt/asr/model-lock.json',config_for('wx_auto'))
        alignment = verify_bundle('/opt/full/alignment','alignment')
        nltk_lock = verify_nltk('/opt/full')
        state['phase'] = 'download_test_diarization'
        diar_root = Path(manifest).parent/'diarization'
        diar_assets = prepare_diarization(diar_root, os.getenv('HF_TOKEN',''))
        state['runtime'] = {'asr':assets,'alignment':alignment,'diarization':diar_assets,
                            'nltk_lock_sha256':nltk_lock}
        # After the fixed gated assets have been staged, inference is offline.
        with offline_models():
            import torch
            import whisperx
            from whisperx.vads.pyannote import load_vad_model
            from whisperx.diarize import DiarizationPipeline
            state['phase'] = 'load_models'
            torch.manual_seed(0)
            config = config_for('wx_auto')
            vad = load_vad_model('cuda', model_fp=assets['vad_path'], vad_onset=.5,vad_offset=.363)
            model = whisperx.load_model(assets['root'],device='cuda',local_files_only=True,
                vad_model=vad,**deepcopy(config['load']))
            align_model, metadata = whisperx.load_align_model('ru','cuda',
                model_name='/opt/full/alignment',model_cache_only=True)
            diarizer = DiarizationPipeline(model_name=str(diar_root),device='cuda',token=False)
            state['modelLoadSeconds'] = round(time.perf_counter()-started,3)
            for row in rows:
                state['phase'] = row['id'] + ':load_audio'
                audio = whisperx.load_audio(row['path'])
                state['phase'] = row['id'] + ':diarization'
                torch.manual_seed(0)
                diar_frame = diarizer(audio)  # Once on the full file; no forced count.
                timeline = snapshot_diarization(diar_frame)
                if timeline['status'] != 'ok' or not timeline['segments']:
                    raise ValueError('Full-file speaker timeline unavailable')
                sample = {'id':row['id'],'sha256':row['sha256'],
                    'durationSeconds':row['duration_seconds'],'outputs':{}}
                state['samples'].append(sample)
                for mode, chunk_size in MODES.items():
                    state['phase'] = row['id'] + ':' + mode + ':asr'
                    phase_start = time.perf_counter()
                    options = {**config['transcribe'],'chunk_size':chunk_size}
                    raw = model.transcribe(audio,**options)
                    raw = deepcopy(raw)
                    if raw.get('language') != 'ru':
                        raise ValueError('This fixed Russian alignment test requires detected Russian')
                    state['phase'] = row['id'] + ':' + mode + ':alignment'
                    aligned = whisperx.align(deepcopy(raw['segments']),align_model,metadata,audio,
                        'cuda',return_char_alignments=False)
                    state['phase'] = row['id'] + ':' + mode + ':speaker_assignment'
                    aligned = whisperx.assign_word_speakers(diar_frame.copy(deep=True),aligned)
                    result = format_output(raw,aligned,timeline,mode,row['duration_seconds'])
                    result['processing']['pipelineSeconds'] = round(time.perf_counter()-phase_start,3)
                    sample['outputs'][mode] = result
                gc.collect()
                torch.cuda.empty_cache()
        state.update(status='completed',phase='complete',elapsedSeconds=round(time.perf_counter()-started,3))
    except Exception as exc:
        # Keep partial evidence, not traceback/messages containing URLs or credentials.
        state.update(status='failed',error_type=type(exc).__name__,elapsedSeconds=round(time.perf_counter()-started,3))
    write_new(output_path,state)
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    execute(args.manifest,args.output)


if __name__ == '__main__':
    main()
