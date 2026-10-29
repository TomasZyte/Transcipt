"""One sound2 speaker experiment, with frozen verified WhisperX ASR. Offline by default."""
import argparse
import asyncio
import base64
from dataclasses import replace
from decimal import Decimal
import hashlib
import json
import logging
from pathlib import Path
import sys

from asr_test import ENDPOINT, BUCKET, money
from full_pipeline_test import FullRunner, IMAGE_PREFIX, EXECUTION_MS, TTL_MS, CLIENT_SECONDS
from gigaam_test import history_files as previous_history
from integration_config import TestSettings, RESERVE_USD
from integration_guard import OneJobRunner, process_lock, save_json, SecretFormatter, TERMINAL
import speaker_comparison as comparison
from asr_comparison import digest, read_json, SHA
from prepare_full_assets import SOURCES

CAMPAIGN = comparison.PROFILE


def history_files(runtime):
    found = previous_history(runtime)
    folder = runtime/'gigaam-children-v1/jobs'
    if any(folder.glob('*.tmp')):
        raise ValueError('Unfinished previous journal write')
    for path in folder.glob('*.json'):
        if path.name.startswith(('message-','retry-','failed-')):
            found[path.relative_to(runtime).as_posix()] = path
    return found


def frozen_evidence(path):
    raw = read_json(path)
    output = raw['output']
    if (raw.get('profile') != 'asr-comparison' or output.get('status') != 'completed'
            or output.get('variant_id') != 'wx_auto' or output.get('duration_seconds') != 11
            or output['config']['model'] != 'large-v3' or output['config']['engine'] != 'whisperx'):
        raise ValueError('Only saved successful WhisperX evidence is supported')
    evidence = {'variant':'wx_auto','model':'large-v3','language':output['language_detected'],
        'source_result_sha256':digest(path),'clip_sha256':output['audio_sha256'],
        'source_start_seconds':output['source_start_seconds'],'source_end_seconds':output['source_end_seconds'],
        'segments':[{k:row[k] for k in ('start','end','text')} for row in output['raw_engine_output']['segments']]}
    comparison.validate_evidence(evidence)
    return evidence


class SpeakerRunner(FullRunner):
    campaign_name = output_profile = CAMPAIGN
    history_files = staticmethod(history_files)
    additional_preflight = {'context_asr_enabled':False,'speaker_comparison_enabled':True}

    def __init__(self, settings, **kwargs):
        if settings.endpoint != ENDPOINT or settings.bucket != BUCKET:
            raise ValueError('Dedicated test resources required')
        self.base_settings = settings
        root = settings.runtime/CAMPAIGN
        if root.resolve() != root or (root/'jobs').resolve() != root/'jobs':
            raise ValueError('No redirected runtime paths')
        self.campaign_path = root/'campaign.json'
        c = self.campaign = read_json(self.campaign_path)
        self.campaign_sha = digest(self.campaign_path)
        commit = c.get('commit','')
        if (c.get('schema') != 1 or c.get('campaign') != CAMPAIGN or c.get('endpoint') != ENDPOINT
                or c.get('bucket') != BUCKET or len(commit) != 40
                or any(ch not in '0123456789abcdef' for ch in commit)
                or c.get('image') != IMAGE_PREFIX+commit[:9]
                or c.get('source_manifest_sha256') != digest(SOURCES)):
            raise ValueError('Unexpected speaker campaign identity')
        previous = read_json(settings.runtime/'gigaam-children-v1/jobs/message-gigaam-children-v1.json')
        budget, prior = money(c['budget_usd']), money(c['baseline_spend_usd'])
        if (previous.get('status') not in TERMINAL
                or prior < max(settings.prior_spend,money(previous['planning_total_usd']))
                or not 0 < budget <= min(settings.budget,Decimal('2')) or prior+RESERVE_USD > budget):
            raise ValueError('All previous reserves and the new attempt must fit the cumulative budget')
        isolated = replace(settings,runtime=root,max_jobs=1,budget=budget,prior_spend=prior)
        OneJobRunner.__init__(self,isolated,CAMPAIGN,timeout_s=CLIENT_SECONDS,**kwargs)
        self.evidence = frozen_evidence(c['source_result'])
        if comparison.evidence_sha(self.evidence) != c['asr_evidence_sha256']:
            raise ValueError('Frozen ASR evidence changed')
        self.duration = comparison.validate_audio(c['audio_file'],c['audio_sha256'],self.evidence)
        self.state.update(campaign_sha256=self.campaign_sha,audio_sha256=c['audio_sha256'],
            asr_evidence_sha256=c['asr_evidence_sha256'],speaker_comparison_validated=False)
        self.validate_history()

    def check_available(self):
        self.validate_history()
        if frozen_evidence(self.campaign['source_result']) != self.evidence:
            raise ValueError('Saved ASR source changed')
        comparison.validate_audio(self.campaign['audio_file'],self.campaign['audio_sha256'],self.evidence)
        OneJobRunner.check_available(self)
        if any(self.path.parent.glob('*.json')) or any(self.path.parent.glob('*.tmp')):
            raise RuntimeError('One attempt only; no reset or automatic retry')

    def payload(self, url):
        return {'profile':CAMPAIGN,'samples':[{'id':'sound2','sha256':self.campaign['audio_sha256'],
                                             'media_url':url}], 'asr_evidence':self.evidence}

    def request_body(self,payload):
        url = payload.get('samples',[{}])[0].get('media_url')
        if payload != self.payload(url):
            raise ValueError('Unexpected audio, text, options or model hints')
        comparison.validate_request_input(payload,self.settings.storage_host)
        return {'input':payload,'policy':{'executionTimeout':EXECUTION_MS,'ttl':TTL_MS}}

    def validate_output(self, result):
        if (result.get('profile') != CAMPAIGN or result.get('status') != 'completed'
                or result.get('sample_id') != 'sound2' or result.get('asr_rerun') is not False
                or result.get('audio_sha256') != self.campaign['audio_sha256']
                or result.get('duration_seconds') != self.duration
                or result.get('asr_evidence') != self.evidence
                or result.get('asr_evidence_sha256') != comparison.evidence_sha(self.evidence)
                or result.get('source_manifest_sha256') != digest(SOURCES)
                or set(result.get('outputs',{})) != set(comparison.MODES)):
            raise ValueError('Incomplete or changed speaker evidence')
        sources = read_json(SOURCES)
        runtime = result['runtime']
        for kind in ('alignment','diarization'):
            if any(runtime[kind].get(k) != sources[kind][k] for k in ('repository','revision')):
                raise ValueError('Unexpected model revision')
            if set(runtime[kind]['files']) != set(sources[kind]['files']):
                raise ValueError('Missing model files')
            for name, info in sources[kind]['files'].items():
                checksum=runtime[kind]['files'][name]
                if not SHA.fullmatch(checksum) or ('sha256' in info and checksum != info['sha256']):
                    raise ValueError('Model file hash mismatch')
        if runtime['packages'].get('whisperx') != '3.8.6' or runtime['packages'].get('pyannote.audio') != '4.0.3':
            raise ValueError('Unexpected inference packages')
        rebuilt = comparison.global_alignment(result['alignment_local'],self.evidence)
        if result['alignment_global'] != rebuilt:
            raise ValueError('Incorrect alignment offset')
        for mode, output in result['outputs'].items():
            comparison.validate_timeline(output['timeline'],self.duration,mode.endswith('_exclusive'))
            if output.get('num_speakers') != (4 if mode.startswith('four') else None):
                raise ValueError('Unexpected speaker count constraint')
            if comparison.text_key(output['aligned']['segments']) != comparison.text_key(self.evidence['segments']):
                raise ValueError('Assigned text changed')
            def strip_speaker(value):
                if isinstance(value,list): return [strip_speaker(v) for v in value]
                if isinstance(value,dict): return {k:strip_speaker(v) for k,v in value.items() if k!='speaker'}
                return value
            if strip_speaker(output['aligned']) != strip_speaker(rebuilt):
                raise ValueError('Speaker assignment changed word times or alignment evidence')
            if output['export'] != comparison.export_result(output['aligned']):
                raise ValueError('Export changed raw evidence')
        self.state.update(speaker_comparison_validated=True,quality_validated=False)
        save_json(self.path,self.state)

    async def execute(self):
        with process_lock(self.base_settings.runtime):
            self.check_available(); self.preflight(); await self.health()
            url = await asyncio.to_thread(upload_sound2,self)
            self.check_available(); self.preflight(); await self.health()
            result = await OneJobRunner.__call__(self,self.payload(url))
            self.validate_output(result)
            return result


def upload_sound2(runner):
    import boto3
    from botocore.config import Config
    s, c = runner.settings, runner.campaign
    comparison.validate_audio(c['audio_file'],c['audio_sha256'],runner.evidence)
    data = Path(c['audio_file']).read_bytes()
    if hashlib.sha256(data).hexdigest() != c['audio_sha256']:
        raise ValueError('Audio changed before upload')
    client = boto3.client('s3',endpoint_url='https://'+s.storage_host,
        aws_access_key_id=s.secrets['S3_ACCESS_KEY'],aws_secret_access_key=s.secrets['S3_SECRET_KEY'],
        region_name='auto',config=Config(signature_version='s3v4',connect_timeout=5,read_timeout=20,
                                       retries={'total_max_attempts':1}))
    key='quality/'+CAMPAIGN+'/'+c['audio_sha256']+'/sound2.wav'
    client.put_object(Bucket=BUCKET,Key=key,Body=data,ContentType='audio/wav',
        ContentMD5=base64.b64encode(hashlib.md5(data).digest()).decode('ascii'))
    stream = client.get_object(Bucket=BUCKET,Key=key)['Body']
    try: received=stream.read(2*1024*1024+1)
    finally: stream.close()
    if hashlib.sha256(received).hexdigest() != c['audio_sha256']:
        raise ValueError('R2 roundtrip mismatch')
    return client.generate_presigned_url('get_object',Params={'Bucket':BUCKET,'Key':key},ExpiresIn=1200)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file',type=Path,required=True)
    parser.add_argument('--submit',action='store_true')
    args=parser.parse_args()
    try:
        settings=TestSettings.load(args.env_file)
        runner=SpeakerRunner(settings)
        runner.check_available()
        if not args.submit:
            print(json.dumps({'status':'prepared_not_run','network_calls':0,'max_attempts':1,
                'planning_total_usd':str(runner.settings.prior_spend+RESERVE_USD),'budget_usd':'2'}))
            return 0
        logging.basicConfig(level=logging.INFO)
        for handler in logging.getLogger().handlers:
            handler.setFormatter(SecretFormatter(settings.secrets.values()))
        logging.getLogger('httpx').setLevel(logging.WARNING)
        asyncio.run(runner.execute())
        print('Speaker evidence saved and verified; disable flags and Max workers.')
        return 0
    except Exception as exc:
        print(json.dumps({'status':'stopped','error_type':type(exc).__name__,'automatic_retry':False}))
        return 1


if __name__=='__main__':sys.exit(main())
