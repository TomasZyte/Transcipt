"""Test-only speaker experiment: frozen WhisperX text, fresh alignment/diarization."""
import argparse
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import wave

from asr_comparison import digest, finite, offline_models, read_json, SHA, write_new
from contextual_asr import model_evidence
from full_pipeline import srt
from prepare_full_assets import prepare_diarization, verify_bundle, verify_nltk, SOURCES
from quality_batch_handler import fetch_sample, validate_request
from transcript_segments import build_segments

PROFILE = 'speaker-children-v1'
MODES = ('auto_regular', 'auto_exclusive', 'four_regular', 'four_exclusive')
RATE = 16000


def evidence_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def require_test():
    if (os.getenv('FULL_PIPELINE_IMAGE') != '1' or os.getenv('FULL_PIPELINE_TEST') != '1'
            or os.getenv('SPEAKER_COMPARISON_TEST') != '1'
            or os.getenv('ASR_COMPARISON_TEST', '0') != '0'
            or os.getenv('CONTEXT_ASR_TEST', '0') != '0'):
        raise ValueError('Dedicated speaker comparison flags required')


def validate_evidence(e):
    keys = {'variant', 'model', 'language', 'source_result_sha256', 'clip_sha256',
            'source_start_seconds', 'source_end_seconds', 'segments'}
    if (not isinstance(e, dict) or set(e) != keys or e['variant'] != 'wx_auto'
            or e['model'] != 'large-v3' or e['language'] != 'ru'
            or any(not isinstance(e[k], str) or not SHA.fullmatch(e[k])
                   for k in ('source_result_sha256', 'clip_sha256'))
            or e['source_start_seconds'] != 8.5 or e['source_end_seconds'] != 19.5):
        raise ValueError('Expected saved WhisperX evidence for the fixed clip')
    rows = e['segments']
    if not isinstance(rows, list) or not 0 < len(rows) <= 20:
        raise ValueError('Invalid frozen segments')
    previous = -1
    for row in rows:
        if (not isinstance(row, dict) or set(row) != {'start', 'end', 'text'}
                or not isinstance(row['text'], str) or not row['text'].strip()
                or len(row['text']) > 8000 or not all(finite(row[k]) for k in ('start', 'end'))
                or not 0 <= row['start'] < row['end'] <= 11.05 or row['start'] < previous):
            raise ValueError('Invalid saved ASR segment')
        previous = row['start']
    return e


def validate_request_input(inp, host):
    if (not isinstance(inp, dict) or set(inp) != {'profile', 'samples', 'asr_evidence'}
            or inp['profile'] != PROFILE or not isinstance(inp['samples'], list)
            or len(inp['samples']) != 1 or inp['samples'][0].get('id') != 'sound2'):
        raise ValueError('Only one sound2 speaker comparison is supported')
    validate_request({'profile': 'community', 'samples': inp['samples']}, host)
    validate_evidence(inp['asr_evidence'])
    return inp['samples'][0]


def validate_audio(path, expected_sha, evidence):
    validate_evidence(evidence)
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > 2*1024*1024 or digest(path) != expected_sha:
        raise ValueError('Audio identity mismatch')
    with wave.open(str(path), 'rb') as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) != (1,2,RATE,'NONE'):
            raise ValueError('Expected mono PCM16 16 kHz')
        frames = wav.getnframes()
        if not 19.5*RATE <= frames <= 60*RATE:
            raise ValueError('Audio duration outside test bounds')
        pcm = wav.readframes(frames)
        if len(pcm) != frames*2:
            raise ValueError('Truncated WAV')
    clip_pcm = pcm[int(8.5*RATE)*2:int(19.5*RATE)*2]
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setparams((1,2,RATE,0,'NONE','not compressed'))
        wav.writeframes(clip_pcm)
    if hashlib.sha256(buffer.getvalue()).hexdigest() != evidence['clip_sha256']:
        raise ValueError('Saved ASR clip does not match full sound2')
    return frames/RATE


def text_key(segments):
    return ''.join(''.join(s['text'].split()) for s in segments)


def global_alignment(aligned, evidence):
    result = model_evidence(deepcopy(aligned))
    if text_key(result['segments']) != text_key(evidence['segments']):
        raise ValueError('Alignment changed frozen text')
    # Normalize before timing decisions; numpy scalars must behave like saved JSON.
    def shift(item):
        if isinstance(item, list):
            for child in item: shift(child)
        elif isinstance(item, dict):
            for key, value in item.items():
                if key in ('start', 'end'):
                    if not finite(value) or not 0 <= value <= 11.05:
                        raise ValueError('Alignment time outside clip')
                    item[key] = value + 8.5
                elif isinstance(value, (dict, list)):
                    shift(value)
    shift(result)
    return result


def annotation_rows(annotation, duration, exclusive=False):
    rows = [{'start':float(turn.start), 'end':float(turn.end), 'speaker':str(label)}
            for turn, _, label in annotation.itertracks(yield_label=True)]
    validate_timeline(rows, duration, exclusive)
    return rows


def validate_timeline(rows, duration, exclusive=False):
    if not isinstance(rows, list) or not rows:
        raise ValueError('Missing diarization timeline')
    previous = -1
    for row in sorted(rows, key=lambda r:r['start']):
        if (set(row) != {'start','end','speaker'} or not isinstance(row['speaker'],str)
                or not row['speaker'] or not all(finite(row[k]) for k in ('start','end'))
                or not 0 <= row['start'] < row['end'] <= duration+.05):
            raise ValueError('Invalid diarization timeline')
        if exclusive and row['start'] < previous-1e-6:
            raise ValueError('Exclusive timeline contains overlapping turns')
        previous = max(previous, row['end'])


def export_result(aligned):
    segments, stats = build_segments(aligned['segments'])
    return {'segments':segments, 'speaker_turns':stats, 'srt':srt(segments),
            'text':'\n\n'.join(s['speaker']+': '+s['text'] for s in segments)}


def execute(manifest, output_path):
    require_test()
    manifest = read_json(manifest)
    row, evidence = manifest['sample'], validate_evidence(manifest['asr_evidence'])
    duration = validate_audio(row['path'], row['sha256'], evidence)
    state = {'profile':PROFILE, 'status':'started', 'phase':'assets', 'outputs':{},
             'sample_id':'sound2', 'audio_sha256':row['sha256'], 'duration_seconds':duration,
             'asr_evidence':evidence, 'asr_evidence_sha256':evidence_sha(evidence),
             'asr_rerun':False, 'source_manifest_sha256':digest(SOURCES)}
    started = time.perf_counter()
    try:
        alignment = verify_bundle('/opt/full/alignment','alignment')
        nltk_lock = verify_nltk('/opt/full')
        diar_root = Path(row['path']).parent/'diarization'
        diar_assets = prepare_diarization(diar_root, os.getenv('HF_TOKEN',''))
        with offline_models():
            import torch
            import whisperx
            import pandas as pd
            from importlib.metadata import version
            from pyannote.audio import Pipeline
            state['runtime'] = {'alignment':alignment,'diarization':diar_assets,
                'nltk_lock_sha256':nltk_lock, 'packages':{p:version(p) for p in
                ('torch','torchaudio','whisperx','pyannote.audio')}}
            torch.manual_seed(0)
            audio = whisperx.load_audio(row['path'])
            clip = audio[int(8.5*RATE):int(19.5*RATE)]
            state['phase'] = 'align_saved_whisperx'
            model, metadata = whisperx.load_align_model('ru','cuda',
                model_name='/opt/full/alignment',model_cache_only=True)
            aligned = whisperx.align(deepcopy(evidence['segments']),model,metadata,clip,
                'cuda',return_char_alignments=False)
            state['alignment_local'] = model_evidence(aligned)
            state['alignment_global'] = global_alignment(aligned, evidence)
            del model
            torch.cuda.empty_cache()
            diarizer = Pipeline.from_pretrained(str(diar_root),token=False).to(torch.device('cuda'))
            sample = {'waveform':torch.from_numpy(audio[None,:]),'sample_rate':RATE}
            for name, count in (('auto',None),('four',4)):
                state['phase'] = name+':diarization'
                torch.manual_seed(0)
                output = diarizer(sample, **({'num_speakers':count} if count else {}))
                for kind, annotation in (('regular',output.speaker_diarization),
                                         ('exclusive',output.exclusive_speaker_diarization)):
                    timeline = annotation_rows(annotation,duration,kind=='exclusive')
                    assigned = whisperx.assign_word_speakers(pd.DataFrame(timeline),
                        deepcopy(state['alignment_global']),fill_nearest=False)
                    assigned = model_evidence(assigned)
                    if text_key(assigned['segments']) != text_key(evidence['segments']):
                        raise ValueError('Speaker assignment changed frozen text')
                    state['outputs'][name+'_'+kind] = {'num_speakers':count, 'timeline':timeline,
                        'aligned':assigned, 'export':export_result(assigned)}
        state.update(status='completed',phase='complete')
    except Exception as exc:
        state.update(status='failed',error_type=type(exc).__name__)
    state['elapsed_seconds'] = round(time.perf_counter()-started,3)
    write_new(output_path,state)
    return state


def handler(job):
    stage = 'validate'
    try:
        require_test()
        inp = job.get('input')
        sample = validate_request_input(inp,os.getenv('QUALITY_R2_HOST',''))
        with tempfile.TemporaryDirectory(prefix='speaker-children-') as directory:
            root = Path(directory)
            stage = 'download'
            audio = root/'sound2.wav'
            fetch_sample(sample,audio)
            validate_audio(audio,sample['sha256'],inp['asr_evidence'])
            manifest, output = root/'manifest.json', root/'result.json'
            write_new(manifest, {'sample':{'path':str(audio),'sha256':sample['sha256']},
                                 'asr_evidence':inp['asr_evidence']})
            stage = 'child'
            child = subprocess.run([sys.executable,str(Path(__file__)), '--manifest',str(manifest),
                '--output',str(output)],timeout=270,stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,check=False)
            if child.returncode or not output.is_file():
                raise RuntimeError('Speaker comparison did not produce evidence')
            return read_json(output)
    except Exception as exc:
        return {'profile':PROFILE,'status':'failed','phase':stage,'error_type':type(exc).__name__}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    execute(args.manifest,args.output)
