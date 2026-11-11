"""Dedicated, disabled-by-default model adapter for the frozen independent call check.

No production imports, reference words, speaker counts, manual times or channel
labels. A check-only command validates files without loading models or networking.
"""
import argparse
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import wave

import pause_context as pause
import speaker_switch_repair as repair

base = pause.base
PROFILE = 'independent-speaker-switch-v1'
IDS = [f'call-check-{i:02d}' for i in range(1,5)]
FROZEN_CODES = {
    'speaker_switch_repair.py':'19276da84dc51d1c8b258fa42f1e109aa27cfa2806e76537ed66bbbb4e75004e',
    'pause_context.py':'d106a447acb4469dc056862fde3c7e38e77294beab186434986283e8e15fa94a',
    'transcript_candidate.py':'b247bfb402f8c0bb45a04baa8c07e6dc87ffcd3228167cfbca4db4d0f3ca464c',
}
OTHER_FLAGS = ('ASR_COMPARISON_TEST','CONTEXT_ASR_TEST','SPEAKER_COMPARISON_TEST',
    'SHORT_TURN_COMPARISON_TEST','RELEASE_CANDIDATE_TEST','TIMED_TURNS_TEST',
    'ACOUSTIC_TURNS_TEST','PAUSE_CONTEXT_TEST','CHANNEL_ASR_TEST','CHANNEL_CONTEXT_TEST',
    'CHANNEL_CONTEXT_GUARD_TEST','QUALITY_BENCHMARK')


def require_test():
    if (any(os.getenv(k,'0') != '1' for k in ('FULL_PIPELINE_IMAGE','FULL_PIPELINE_TEST','INDEPENDENT_SWITCH_TEST'))
            or any(os.getenv(k,'0') != '0' for k in OTHER_FLAGS)):
        raise ValueError('Dedicated independent test flags required')


def check_lock(lock):
    if (not isinstance(lock,dict) or set(lock) != {'schema','profile','algorithm','model_sources_sha256',
                                                 'asr_sources_sha256','decoder','samples'}
            or lock['schema'] != 1 or lock['profile'] != PROFILE
            or lock['model_sources_sha256'] != base.digest(base.SOURCES)
            or lock['asr_sources_sha256'] != base.digest(Path(__file__).with_name('asr_model_sources.json'))
            or lock['decoder'] != base.DECODER
            or lock['algorithm'] != {'version':repair.VERSION,'policy':repair.POLICY,'code_sha256':FROZEN_CODES}):
        raise ValueError('Unexpected frozen lock')
    for name,expected in FROZEN_CODES.items():
        if base.digest(Path(__file__).with_name(name)) != expected:
            raise ValueError('Frozen algorithm changed')
    rows = lock['samples']
    if (not isinstance(rows,list) or len(rows)!=4 or any(not isinstance(r,dict) for r in rows)
            or [r.get('id') for r in rows]!=IDS):
        raise ValueError('Expected four independent sample IDs')
    for row in rows:
        if (set(row) != {'id','sha256','frames'} or not isinstance(row['sha256'],str)
                or not re.fullmatch('[a-f0-9]{64}',row['sha256'])
                or type(row['frames']) is not int or not 0 < row['frames'] <= 30*base.RATE):
            raise ValueError('Invalid locked audio')
    if sum(r['frames'] for r in rows)>60*base.RATE:
        raise ValueError('Independent batch exceeds one minute')
    return lock


def validate_input(inp, lock):
    require_test()
    check_lock(lock)
    if (not isinstance(inp,dict) or set(inp)!={'profile','samples'} or inp['profile']!=PROFILE
            or not isinstance(inp['samples'],list) or len(inp['samples'])!=4):
        raise ValueError('Expected audio-only independent request')
    for row,expected in zip(inp['samples'],lock['samples'],strict=True):
        if (not isinstance(row,dict) or set(row)!={'id','sha256','media_url'}
                or row['id']!=expected['id'] or row['sha256']!=expected['sha256']):
            raise ValueError('Unapproved audio or inference hints')
    return base.validate_request({'profile':'community','samples':inp['samples']},
                                 os.getenv('QUALITY_R2_HOST',''))


def validate_local(manifest, lock):
    check_lock(lock)
    if (not isinstance(manifest,dict) or set(manifest)!={'profile','samples'} or manifest['profile']!=PROFILE
            or not isinstance(manifest['samples'],list) or len(manifest['samples'])!=4):
        raise ValueError('Expected local audio-only manifest')
    rows = []
    for row,expected in zip(manifest['samples'],lock['samples'],strict=True):
        if (not isinstance(row,dict) or set(row)!={'id','sha256','path'}
                or row['id']!=expected['id'] or row['sha256']!=expected['sha256']):
            raise ValueError('Unapproved audio or local hints')
        path = Path(row['path'])
        if path.is_symlink() or path.stat().st_size>2*1024*1024 or base.digest(path)!=row['sha256']:
            raise ValueError('Audio checksum or size mismatch')
        with wave.open(str(path),'rb') as wav:
            if ((wav.getnchannels(),wav.getsampwidth(),wav.getframerate(),wav.getcomptype())!=(1,2,base.RATE,'NONE')
                    or wav.getnframes()!=expected['frames']
                    or len(wav.readframes(wav.getnframes()))!=2*expected['frames']):
                raise ValueError('Expected complete mono PCM16 audio')
        rows.append({**row,'duration_seconds':expected['frames']/base.RATE})
    return rows


def collect_sample(row, audio, *, diarize, decode, align):
    """Callback boundary permits CPU tests, without substituting fake evidence in execution."""
    if len(audio)!=round(row['duration_seconds']*base.RATE):
        raise ValueError('Audio length changed')
    sample={'id':row['id'],'sha256':row['sha256'],'duration_seconds':row['duration_seconds'],
            'diarization':{},'alignments':{},'alignment_windows':{},'outputs':{}}
    for name,settings in pause.DIARIZATION.items():
        # These are two real pipeline calls, never duplicates of a channel label.
        sample['diarization'][name]=deepcopy(diarize(audio,deepcopy(settings)))
    windows=base.pause_windows(sample['diarization']['short']['exclusive'],len(audio))
    sample['asr'],_,sample['windows']=decode(audio,windows,align)
    for name in pause.ALIGNMENTS:
        sample['alignments'][name],sample['alignment_windows'][name]=pause.realign(sample['windows'],audio,align,name)
    for name,mode in pause.MODES.items():
        sample['outputs'][name]=base.attribute(sample['asr'],sample['alignments'][mode['alignment']],
            sample['diarization'][mode['diarization']]['regular'],row['duration_seconds'])
    sample['phraseTimingReview']=pause.phrase_timing_review(sample)
    sample['phraseSpeakerReview']=pause.phrase_speaker_review(sample)
    sample['phraseSpeakerReviewVersion']=3
    pause.validate_sample(sample)
    preserved=deepcopy(sample)
    candidate,audit=repair.propose(sample)
    if sample!=preserved:
        raise ValueError('Repair mutated the baseline')
    return {'baseline':sample,'candidate':candidate,'repair_audit':audit}


def validate_result(state, lock):
    check_lock(lock)
    if (state.get('profile')!=PROFILE or state.get('status')!='completed'
            or state.get('modes')!=pause.MODES or state.get('decoder')!=base.DECODER
            or state.get('diarization_configs')!=pause.DIARIZATION
            or state.get('alignment_configs')!=list(pause.ALIGNMENTS)
            or state.get('algorithm')!=lock['algorithm']
            or state.get('source_manifest_sha256')!=lock['model_sources_sha256']
            or state.get('asr_sources_sha256')!=lock['asr_sources_sha256']
            or state.get('input_samples')!=lock['samples']
            or len(state.get('samples',[]))!=4):
        raise ValueError('Incomplete or changed independent probe')
    sources=base.read_json(base.SOURCES)
    asr_source=base.read_json(Path(__file__).with_name('asr_model_sources.json'))['large-v3']
    runtime=state.get('runtime',{})
    assets=runtime.get('asr',{})
    if assets.get('revision')!=asr_source['revision']:
        raise ValueError('Unexpected WhisperX model revision')
    for name,version in base.config_for('wx_auto')['packages'].items():
        if assets.get('packages',{}).get(name)!=version:
            raise ValueError('Unexpected ASR packages')
    for kind in ('alignment','diarization'):
        if any(runtime.get(kind,{}).get(k)!=sources[kind][k] for k in ('repository','revision')):
            raise ValueError('Unexpected alignment or diarization revision')
    for entry,row in zip(state['samples'],lock['samples'],strict=True):
        original=entry['baseline']
        if (original['id']!=row['id'] or original['sha256']!=row['sha256']
                or original['duration_seconds']!=row['frames']/base.RATE):
            raise ValueError('Result belongs to other audio')
        pause.validate_sample(original)
        candidate,audit=repair.propose(original)
        if entry['candidate']!=candidate or entry['repair_audit']!=audit:
            raise ValueError('Frozen correction does not replay')


def execute(manifest_path, lock_path, output_path):
    require_test()
    if Path(output_path).exists():
        raise ValueError('Never overwrite a probe result')
    lock=base.read_json(lock_path)
    rows=validate_local(base.read_json(manifest_path),lock)
    state={'profile':PROFILE,'status':'started','phase':'assets','samples':[],
           'source_manifest_sha256':base.digest(base.SOURCES),'modes':pause.MODES,
           'asr_sources_sha256':lock['asr_sources_sha256'],
           'decoder':deepcopy(base.DECODER),'diarization_configs':deepcopy(pause.DIARIZATION),
           'alignment_configs':list(pause.ALIGNMENTS),'algorithm':deepcopy(lock['algorithm']),
           'input_samples':deepcopy(lock['samples']),
           'human_quality_validated':False,'production_ready':False}
    begun=time.perf_counter()
    try:
        config=base.config_for('wx_auto')
        assets=base.verify_assets('/opt/asr/model-lock.json',config)
        alignment=base.verify_bundle('/opt/full/alignment','alignment'); nltk=base.verify_nltk('/opt/full')
        with tempfile.TemporaryDirectory(prefix='independent-model-') as temp:
            diar_root=Path(temp)/'diarization'
            diar_assets=base.prepare_diarization(diar_root,os.getenv('HF_TOKEN',''))
            state['runtime']={'asr':assets,'alignment':alignment,'diarization':diar_assets,'nltk_lock_sha256':nltk}
            with base.offline_models():
                import torch, whisperx
                from importlib.metadata import version
                from pyannote.audio import Pipeline
                from whisperx.vads.pyannote import load_vad_model
                base.verify_decoder_api()
                state['runtime']['packages']={p:version(p) for p in ('torch','whisperx','pyannote.audio','faster-whisper')}
                vad=load_vad_model('cuda',model_fp=assets['vad_path'],vad_onset=.5,vad_offset=.363)
                model=whisperx.load_model(assets['root'],device='cuda',local_files_only=True,vad_model=vad,**deepcopy(config['load']))
                am,metadata=whisperx.load_align_model('ru','cuda',model_name='/opt/full/alignment',model_cache_only=True)
                pipeline=Pipeline.from_pretrained(str(diar_root),token=False).to(torch.device('cuda'))
                base.validate_runtime_pipeline(pipeline,version('pyannote.audio'))
                if pipeline._segmentation.model.specifications.duration!=10:
                    raise ValueError('Unexpected segmentation duration')
                def diarize(audio, settings):
                    diagnostics=[]; torch.manual_seed(0)
                    with base.acoustic_settings(pipeline,settings) as actual, base.short_turn_filter(pipeline.clustering,.1,diagnostics):
                        result=pipeline({'waveform':torch.from_numpy(audio[None,:]),'sample_rate':base.RATE})
                    return {'settings':actual,'parameters':base.model_evidence(pipeline.parameters(instantiated=True)),
                        'filter_diagnostics':diagnostics,
                        'regular':base.annotation_rows(result.speaker_diarization,len(audio)/base.RATE,False),
                        'exclusive':base.annotation_rows(result.exclusive_speaker_diarization,len(audio)/base.RATE,True)}
                def align(segments, chunk):
                    return whisperx.align(segments,am,metadata,chunk,'cuda',return_char_alignments=False)
                def decode(audio, windows, alignment_callback):
                    return base.decode(audio,windows,model,alignment_callback)
                for row in rows:
                    state['phase']=row['id']
                    audio=whisperx.load_audio(row['path'])
                    state['samples'].append(collect_sample(row,audio,diarize=diarize,decode=decode,align=align))
                    gc.collect(); torch.cuda.empty_cache()
        state.update(status='completed',phase='complete')
        validate_result(state,lock)
    except Exception as exc:
        state.update(status='failed',error='Independent probe failed',error_type=type(exc).__name__)
    state['elapsed_seconds']=round(time.perf_counter()-begun,3)
    base.write_new(output_path,state)
    return state


def handler(job):
    stage='validate'
    try:
        lock_path=os.getenv('INDEPENDENT_SWITCH_LOCK','/opt/probe/lock.json')
        require_test()
        lock=base.read_json(lock_path)
        samples=validate_input(job.get('input'),lock)
        with tempfile.TemporaryDirectory(prefix='independent-switch-') as temp:
            root=Path(temp); rows=[]; stage='download_audio'
            for sample in samples:
                path=root/(sample['id']+'.wav')
                base.fetch_sample(sample,path)
                rows.append({'id':sample['id'],'sha256':sample['sha256'],'path':str(path)})
            manifest={'profile':PROFILE,'samples':rows}
            validate_local(manifest,lock)
            mpath,opath=root/'input.json',root/'result.json'
            base.write_new(mpath,manifest); stage='child'
            child=subprocess.run([sys.executable,str(Path(__file__)),'--execute','--manifest',str(mpath),
                '--lock',str(lock_path),'--output',str(opath)],timeout=270,
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=False)
            if child.returncode or not opath.is_file():
                raise RuntimeError('Missing independent evidence')
            result=base.read_json(opath)
            validate_result(result,lock)
            return result
    except Exception as exc:
        return {'profile':PROFILE,'status':'failed','phase':stage,
                'error':'Independent switch probe failed','error_type':type(exc).__name__}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group(required=True)
    for name in ('check','execute','serve'):
        action.add_argument('--'+name,action='store_true')
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--lock',type=Path,default=Path('/opt/probe/lock.json'))
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.serve:
        require_test(); check_lock(base.read_json(args.lock))
        os.environ['INDEPENDENT_SWITCH_LOCK']=str(args.lock.resolve())
        print(json.dumps({'event':'INDEPENDENT_PROBE_HANDLER_READY','profile':PROFILE,
                          'lock_sha256':base.digest(args.lock)}),flush=True)
        import runpod
        runpod.serverless.start({'handler':handler})
    else:
        if args.manifest is None:
            parser.error('--manifest is required')
        if args.check:
            rows=validate_local(base.read_json(args.manifest),base.read_json(args.lock))
            print(json.dumps({'status':'preflight_only','samples':len(rows),
                   'audio_seconds':sum(r['duration_seconds'] for r in rows),'models_loaded':False}))
        elif args.output is None:
            parser.error('--output is required')
        else:
            execute(args.manifest,args.lock,args.output)


if __name__=='__main__':
    main()
