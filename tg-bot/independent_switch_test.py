"""One authorized four-clip probe, with fresh settled billing and durable no-retry journal."""
import argparse
import asyncio
import csv
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
import json
import logging
import re
from pathlib import Path

import full_pipeline_test as full
import channel_context_guard_test as previous
import independent_switch_probe as probe
from integration_guard import OneJobRunner, SecretFormatter, TERMINAL, process_lock, save_json

PROFILE=probe.PROFILE
EXPECTED_JOBS={'completed':24,'failed':1,'inProgress':0,'inQueue':0,'retried':2}
COMMIT='10ff864ce84ab7e06f7cb55144ba851f4e2b8326'
IMAGE_MARKER='dockerfile-independent-switch'
ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/'.test-runtime/quality-2026-10-28/independent-switch-run-2026-11-11'
PREP=ROOT/'.test-runtime/quality-2026-10-28/independent-switch-adapter-2026-11-11-rc2'


def stamp(value):
    result=datetime.fromisoformat(value.replace('Z','+00:00'))
    if result.tzinfo is None: raise ValueError('Timezone required')
    return result


def validate_build_publication(log):
    exported=re.findall(r'exporting config sha256:([a-f0-9]{64})\b',log)
    published=re.findall(r'\b([a-f0-9]{12})\s+config\s+',log)
    if len(exported)!=1 or len(published)!=1 or exported[0][:12]!=published[0]:
        raise ValueError('Published config differs from the image just built')
    return exported[0]


def history_files(runtime):
    found=previous.history_files(runtime)
    folder=runtime/previous.PROFILE/'jobs'
    if any(folder.glob('*.tmp')): raise ValueError('Unfinished historical write')
    for path in folder.glob('*.json'):
        if path.name.startswith(('message-','failed-','retry-')):
            found[path.relative_to(runtime).as_posix()]=path
    return found


def validate_health(health):
    if health.get('jobs')!=EXPECTED_JOBS: raise ValueError('Test job history changed')
    workers=health.get('workers',{})
    keys={'idle','initializing','ready','running','throttled','unhealthy'}
    if set(workers)!=keys or any(type(v) is not int or v<0 for v in workers.values()):
        raise ValueError('Unknown worker state')
    zero={k:0 for k in keys}
    allowed=[zero,{**zero,'idle':1,'ready':1},{**zero,'initializing':1},{**zero,'throttled':1}]
    if workers not in allowed: raise ValueError('Unexpected test workers')


def validate_budget(billing, rows, journals, health, now=None):
    now=now or datetime.now(timezone.utc)
    if (billing.get('endpoint')!=full.ENDPOINT or billing.get('period_timezone')!='UTC'
            or billing.get('period_start','9999')>'2026-10-28'
            or billing.get('period_end')!=now.date().isoformat()
            or not 0<=(now-stamp(billing['observed_at_utc'])).total_seconds()<=900
            or billing.get('billing_lag_hours')!=1):
        raise ValueError('Need fresh full-period test billing')
    validate_health(health)
    if len(journals)!=24 or any(j.get('status') not in TERMINAL for j in journals):
        raise ValueError('All previous attempts must be resolved')
    for journal in journals:
        if journal.get('endpoint_id') is None and journal.get('bucket') is None:
            # One historical raw failure snapshot is also retained alongside its journal.
            # Bind it by provider job ID AND status; never assume missing scope means test.
            peers=[j for j in journals if j.get('job_id')==journal.get('id') and j.get('job_id')
                   and j.get('status')==journal.get('status') and j.get('endpoint_id')==full.ENDPOINT
                   and j.get('bucket')==full.BUCKET]
            if len(peers)!=1: raise ValueError('Unbound historical snapshot')
            journal=peers[0]
        if journal.get('endpoint_id')!=full.ENDPOINT or journal.get('bucket')!=full.BUCKET:
            raise ValueError('Wrong historical resources')
        ended=float(journal.get('submitted_at') or 0)+float(journal.get('elapsed_sec') or 0)
        if ended>now.timestamp()-3600: raise ValueError('Previous billing may still lag')
    latest=max(journals,key=lambda j:float(j.get('submitted_at') or 0))
    if latest.get('job_id')!='13a77045-e4c8-46ab-8972-ad73336a43c0-e2':
        raise ValueError('Unexpected latest historical job')
    selected=[r for r in rows if r['Endpoint']==full.ENDPOINT]
    if not selected: raise ValueError('No test endpoint billing column')
    total=Decimal('0')
    seen=set()
    for row in selected:
        day=stamp(row['Date']).date().isoformat()
        if not billing['period_start']<=day<=billing['period_end'] or day in seen:
            raise ValueError('Invalid or duplicate billing day')
        seen.add(day)
        amount=full.money(row['Amount'])
        if amount<0: raise ValueError('Negative bill')
        total+=amount
    if total!=full.money(billing['observed_billed_usd']): raise ValueError('CSV total differs')
    # Prior planning reserves are replaced only by settled billing and unchanged history.
    # Retain rounding and an extra ten cents; a fresh half-dollar reserve covers this job.
    baseline=(total+Decimal('.02')+Decimal('.10')).quantize(Decimal('.01'),rounding=ROUND_CEILING)
    if baseline+full.RESERVE_USD>Decimal('2'): raise ValueError('Cumulative budget exhausted')
    return baseline


class IndependentRunner(OneJobRunner):
    def __init__(self, settings):
        if settings.endpoint!=full.ENDPOINT or settings.bucket!=full.BUCKET:
            raise ValueError('Test resources only')
        self.base_settings=settings
        self.lock=probe.check_lock(full.read_json(PREP/'independent-probe-lock.json'))
        self.manifest=full.read_json(PREP/'local-input.json')
        self.rows=probe.validate_local(self.manifest,self.lock)
        self.history_record=full.read_json(EVIDENCE/'reconciliation-initial.json')
        self.history_hashes=self.history_record['history_sha256']
        self.validate_history()
        billing,rows=self.read_billing()
        self.initial_health=full.read_json(EVIDENCE/'preflight.json')['health']
        baseline=validate_budget(billing,rows,self.journals,self.initial_health)
        root=settings.runtime/PROFILE
        if root.resolve()!=root: raise ValueError('Redirected runtime')
        super().__init__(replace(settings,runtime=root,max_jobs=1,budget=Decimal('2'),prior_spend=baseline),
                         PROFILE,timeout_s=450,poll_s=3)
        self.state.update(lock_sha256=full.digest(PREP/'independent-probe-lock.json'),
            manifest_sha256=full.digest(PREP/'local-input.json'),history_sha256=self.history_hashes,
            client_sha256=full.digest(__file__),
            prior_settled_billed_usd=billing['observed_billed_usd'],baseline_spend_usd=str(baseline),
            planning_total_usd=str(baseline+full.RESERVE_USD),independent_probe_validated=False)

    def read_billing(self):
        billing=full.read_json(EVIDENCE/'billing-submit.json')
        path=EVIDENCE/'billing-submit.csv'
        if full.digest(path)!=billing['csv_sha256']: raise ValueError('Billing file changed')
        return billing,list(csv.DictReader(path.read_text(encoding='utf-8-sig').splitlines()))

    def validate_history(self):
        paths=history_files(self.base_settings.runtime)
        if {k:full.digest(p) for k,p in paths.items()}!=self.history_hashes:
            raise ValueError('Prior journals changed')
        self.journals=[full.read_json(p) for p in paths.values()]

    def check_available(self):
        self.validate_history()
        probe.validate_local(self.manifest,self.lock)
        super().check_available()
        if any(self.path.parent.glob('*.json')) or any(self.path.parent.glob('*.tmp')):
            raise ValueError('One attempt only; never reset this campaign')

    def preflight(self):
        self.validate_history()
        p=full.read_json(EVIDENCE/'preflight.json')
        now=datetime.now(timezone.utc)
        expected={'endpoint':full.ENDPOINT,'branch':'codex/full-pipeline-test','commit':COMMIT,
            'paid_run_authorized':True,'active_workers':0,'max_workers':1,'gpu_count':1,
            'execution_timeout_seconds':300,'idle_timeout_seconds':5,'flashboot':False,
            'full_pipeline_enabled':True,'independent_switch_enabled':True,
            'other_test_modes_disabled':True,'build_status':'completed'}
        if any(type(p.get(k)) is not type(v) or p[k]!=v for k,v in expected.items()):
            raise ValueError('Unexpected test setup')
        # The provider built this exact image and ran its CPU checks. GitHub's
        # additional copy of the build may still be queued; never call it passed.
        build=full.read_json(EVIDENCE/'build.json')
        log=EVIDENCE/'runpod-build.log'
        if (p.get('ci_status') not in ('queued','success')
                or build.get('endpoint')!=full.ENDPOINT or build.get('commit')!=COMMIT
                or build.get('build_status')!='completed'
                or build.get('cpu_model_checks_verified') is not True
                or build.get('frozen_code_verified') is not True
                or full.digest(log)!=build.get('log_sha256')
                or p.get('image')!=build.get('image')
                or p.get('image_digest')!=build.get('build_manifest_digest')):
            raise ValueError('Unverified provider build')
        build_text=log.read_text(encoding='utf-8')
        validate_build_publication(build_text)
        if any(marker not in build_text for marker in (
                'FULL_PIPELINE_CPU_IMPORT_AND_ALIGNMENT_OK','INDEPENDENT_PROBE_FROZEN_CODE_OK')):
            raise ValueError('Missing provider CPU checks')
        if not 0<=(now-stamp(p['observed_at_utc'])).total_seconds()<=900:
            raise ValueError('Stale endpoint observation')
        if (not p['image'].startswith('registry.runpod.net/tomaszyte-transcipt-codex-full-pipeline-test-')
                or IMAGE_MARKER not in p['image'] or not p['image'].endswith(':'+COMMIT[:9])
                or not p['image_digest'].startswith('sha256:')
                or not 0<full.money(p['gpu_rate_usd_per_hour'])<=Decimal('.75')
                or p['lock_sha256']!=self.state['lock_sha256']):
            raise ValueError('Unverified image, lock or GPU rate')
        billed,rows=self.read_billing()
        if validate_budget(billed,rows,self.journals,p['health'],now)!=self.settings.prior_spend:
            raise ValueError('Budget changed')
        self.state.update(preflight_sha256=full.digest(EVIDENCE/'preflight.json'),image=p['image'],image_digest=p['image_digest'])

    async def health(self):
        async with asyncio.timeout(20), self.client_factory(timeout=15) as client:
            r=await client.get(self.url+'/health',headers=self.headers)
            r.raise_for_status(); health=r.json()
        validate_health(health)
        self.state['pre_submit_health']=health

    def request_body(self,payload):
        if not isinstance(payload,dict) or set(payload)!={'profile','samples'} or payload['profile']!=PROFILE:
            raise ValueError('Unexpected input')
        rows=payload['samples']
        if not isinstance(rows,list) or len(rows)!=4: raise ValueError('Exactly four clips required')
        for row,expected in zip(rows,self.lock['samples'],strict=True):
            if set(row)!={'id','sha256','media_url'} or any(row[k]!=expected[k] for k in ('id','sha256')):
                raise ValueError('Changed audio or forbidden hint')
        full.validate_request({'profile':'community','samples':rows},self.settings.storage_host)
        return {'input':payload,'policy':{'executionTimeout':300000,'ttl':420000}}

    async def execute(self):
        with process_lock(self.base_settings.runtime):
            self.check_available(); self.preflight(); await self.health()
            # The four locked clips use the same private R2 round-trip checks as earlier tests.
            urls=await asyncio.to_thread(upload,self.settings,self.rows,self.lock,self.manifest)
            self.check_available(); self.preflight(); await self.health()
            payload={'profile':PROFILE,'samples':[{'id':r['id'],'sha256':r['sha256'],'media_url':url}
                    for r,url in zip(self.rows,urls,strict=True)]}
            result=await super().__call__(payload)
            self.accept_result(result)
            save_json(EVIDENCE/'raw-result.json',result)
            return result

    def accept_result(self,result):
        # Provider COMPLETED only means the HTTP handler returned. Preserve the
        # distinction when a different handler or an invalid result is returned.
        try:
            probe.validate_result(result,self.lock)
        except (ValueError,KeyError,TypeError) as exc:
            self.state.update(independent_probe_validated=False,probe_status='rejected_output',
                              probe_validation_error_type=type(exc).__name__)
            save_json(self.path,self.state)
            raise
        self.state.update(independent_probe_validated=True,probe_status='validated')
        save_json(self.path,self.state)

    async def __call__(self,*args,**kwargs):
        raise RuntimeError('Use execute() to enforce preflight and the shared process lock')


def upload(settings,rows,lock,manifest):
    import base64,hashlib,boto3
    from botocore.config import Config
    probe.validate_local(manifest,lock)
    s3=boto3.client('s3',endpoint_url='https://'+settings.storage_host,
        aws_access_key_id=settings.secrets['S3_ACCESS_KEY'],aws_secret_access_key=settings.secrets['S3_SECRET_KEY'],
        region_name='auto',config=Config(signature_version='s3v4',connect_timeout=5,read_timeout=20,retries={'total_max_attempts':1}))
    urls=[]
    for row in rows:
        data=Path(row['path']).read_bytes()
        if hashlib.sha256(data).hexdigest()!=row['sha256']: raise ValueError('Audio changed')
        key='quality/'+PROFILE+'/'+row['sha256']+'/'+row['id']+'.wav'
        s3.put_object(Bucket=full.BUCKET,Key=key,Body=data,ContentType='audio/wav',ContentMD5=base64.b64encode(hashlib.md5(data).digest()).decode())
        stream=s3.get_object(Bucket=full.BUCKET,Key=key)['Body']
        try: received=stream.read(2*1024*1024+1)
        finally: stream.close()
        if hashlib.sha256(received).hexdigest()!=row['sha256']: raise ValueError('R2 checksum mismatch')
        urls.append(s3.generate_presigned_url('get_object',Params={'Bucket':full.BUCKET,'Key':key},ExpiresIn=1200))
    return urls


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submit',action='store_true'); args=parser.parse_args()
    try:
        settings=full.TestSettings.load(ROOT/'.env.integration-test')
        logging.basicConfig(level=logging.INFO)
        for handler in logging.getLogger().handlers: handler.setFormatter(SecretFormatter(settings.secrets.values()))
        logging.getLogger('httpx').setLevel(logging.WARNING)
        runner=IndependentRunner(settings); runner.check_available(); runner.preflight()
        if args.submit:
            asyncio.run(runner.execute())
            print(json.dumps({'status':'completed','validated':True,'job_id':runner.state['job_id']}))
        else: print(json.dumps({'status':'prepared_not_submitted','planning_total_usd':runner.state['planning_total_usd']}))
    except Exception as exc:
        print(json.dumps({'status':'stopped','error_type':type(exc).__name__,'automatic_retry':False}))
        raise SystemExit(1)
