"""Build local inference-only manifests from the previously frozen preparation.

Does not upload audio, obtain URLs, load models, build an image or submit a job.
Human reference data and stereo channel identity never enter the inference lock.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import offline_release as release
import independent_switch_probe as probe


def prepare(preparation, output):
    preparation,output=Path(preparation).resolve(),Path(output).resolve()
    if output.exists():
        raise ValueError('Never overwrite independent probe preparation')
    source=preparation/'manifest.json'
    data=json.loads(source.read_bytes())
    if (data.get('schema')!='speaker-switch-independent-preparation-v1'
            or data.get('status')!='prepared_not_executed'
            or data.get('model_inference_performed') is not False
            or data.get('paid_jobs_submitted')!=0
            or [c['id'] for c in data['clips']]!=probe.IDS):
        raise ValueError('Expected the unexecuted independent preparation')
    # Compare against the already frozen snapshot, not newly selected parameters.
    algorithm=data['algorithm']
    names={Path(p).name:h for p,h in algorithm['code_sha256'].items()}
    for name,expected in probe.FROZEN_CODES.items():
        if names.get(name)!=expected:
            raise ValueError('Preparation uses a different correction')
    lock={'schema':1,'profile':probe.PROFILE,
          'algorithm':{'version':algorithm['version'],'policy':algorithm['policy'],'code_sha256':probe.FROZEN_CODES},
          'model_sources_sha256':probe.base.digest(probe.base.SOURCES),
          'asr_sources_sha256':probe.base.digest(Path(probe.__file__).with_name('asr_model_sources.json')),
          'decoder':probe.base.DECODER,'samples':[]}
    local={'profile':probe.PROFILE,'samples':[]}
    for clip in data['clips']:
        file=clip['files']['mono']
        relative=Path(file['file'].replace('\\','/'))
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Audio must stay inside the preparation')
        audio=(preparation/relative).resolve()
        if not audio.is_relative_to(preparation):
            raise ValueError('Audio escaped the preparation')
        lock['samples'].append({'id':clip['id'],'sha256':file['sha256'],'frames':file['frames']})
        local['samples'].append({'id':clip['id'],'sha256':file['sha256'],'path':str(audio)})
    rows=probe.validate_local(local,lock)
    output.mkdir()
    release.save(output/'independent-probe-lock.json',lock)
    release.save(output/'local-input.json',local)
    # Explicit COPY inputs only; never copy audio, references, directories or env files.
    context=output/'image-context'
    context.mkdir()
    release.save(context/'independent-probe-lock.json',lock)
    dockerfile=Path(probe.__file__).with_name('Dockerfile.independent-switch')
    names={'Dockerfile.independent-switch'}
    for line in dockerfile.read_text(encoding='utf-8').splitlines():
        if line.startswith('COPY '):
            for name in line.split()[1:-1]:
                if name=='independent-probe-lock.json': continue
                if Path(name).name!=name or Path(name).suffix not in ('.py','.json','.txt') or name.startswith('.'):
                    raise ValueError('Unexpected build context source')
                names.add(name)
    code_paths=[dockerfile.with_name(name) for name in sorted(names)]
    hashes={}
    for path in code_paths:
        content=path.read_bytes()
        (context/path.name).write_bytes(content)
        hashes[path.name]=sha256(content).hexdigest()
    summary={'status':'prepared_for_build_not_executed','samples':len(rows),
        'audio_seconds':sum(r['duration_seconds'] for r in rows),
        'preparation_manifest_sha256':sha256(source.read_bytes()).hexdigest(),
        'lock_sha256':sha256((output/'independent-probe-lock.json').read_bytes()).hexdigest(),
        'build_context_sha256':hashes,'build_strategy':'rebuild_established_full_test_stack',
        'base_image':'pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime',
        'image_built':False,'audio_uploaded':False,'model_inference_performed':False,
        'paid_jobs_submitted':0,'production_changed':False,
        'next_requirements':['verify the completed image and CPU model checks',
                             'explicit authorization for a paid test and fresh cumulative USD 2 budget check']}
    release.save(output/'preflight.json',summary)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    sys.addaudithook(release.deny_network)
    print(json.dumps(prepare(args.preparation,args.output),ensure_ascii=False))
