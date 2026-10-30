"""One bounded timestamp/diarization comparison; no network without --submit."""
import argparse
import asyncio
import json
import logging
from pathlib import Path

import candidate_test as previous
import full_pipeline_test as full
import timed_turns as worker
from integration_guard import save_json, SecretFormatter


def history_files(runtime):
    found=previous.history_files(runtime)
    folder=runtime/previous.candidate.PROFILE/'jobs'
    if any(folder.glob('*.tmp')): raise ValueError('Unfinished historical write')
    for path in folder.glob('*.json'):
        if path.name.startswith(('message-','failed-','retry-')):
            found[path.relative_to(runtime).as_posix()]=path
    return found


class TimedRunner(full.FullRunner):
    campaign_name=output_profile=worker.PROFILE
    modes=worker.MODES
    validated_key='timed_turns_validated'
    history_files=staticmethod(history_files)
    health=previous.CandidateRunner.health
    additional_preflight={'timed_turns_enabled':True,'release_candidate_enabled':False,
                          'short_turn_comparison_enabled':False,'speaker_comparison_enabled':False,
                          'context_asr_enabled':False}

    def __init__(self,settings,**kwargs):
        super().__init__(settings,**kwargs)
        prior=previous.CandidateRunner(settings); journal=full.read_json(prior.path)
        if (journal.get('status')!='COMPLETED' or journal.get('candidate_technical_validated') is not True
                or journal.get('campaign_sha256')!=prior.campaign_sha
                or self.campaign.get('previous_campaign_sha256')!=prior.campaign_sha
                or full.money(self.campaign['baseline_spend_usd'])<full.money(journal['planning_total_usd'])):
            raise ValueError('Previous candidate allowance must be carried in full')
        shutdown=Path(self.campaign['previous_shutdown_file'])
        if full.digest(shutdown)!=self.campaign['previous_shutdown_sha256']: raise ValueError('Shutdown evidence changed')
        observed=full.read_json(shutdown); self.expected_health=observed['health']
        if (observed.get('endpoint')!=full.ENDPOINT or observed.get('max_workers')!=0
                or self.expected_health['jobs']['inQueue'] or self.expected_health['jobs']['inProgress']
                or any(self.expected_health['workers'].values())): raise ValueError('Previous workers not stopped')

    def validate_output(self,result):
        assets=self.validate_provenance(result); sources=full.read_json(full.SOURCES)
        if (result.get('policy')!=worker.POLICY or result.get('diarization_configs')!=worker.DIARIZATION
                or result.get('decoder')!=worker.DECODER or result.get('human_quality_validated') is not False
                or len(result.get('samples',[]))!=4): raise ValueError('Unexpected experiment identity')
        for kind in ('alignment','diarization'):
            files=result['runtime'][kind].get('files',{})
            if set(files)!=set(sources[kind]['files']): raise ValueError('Incomplete asset provenance')
            for name,checksum in files.items():
                expected=sources[kind]['files'][name].get('sha256')
                if not full.SHA.fullmatch(checksum) or (expected and checksum!=expected): raise ValueError('Model hash changed')
        for name,version in {'whisperx':'3.8.6','pyannote.audio':'4.0.3','faster-whisper':'1.2.1'}.items():
            if result['runtime']['packages'].get(name)!=version: raise ValueError('Package changed')
        for sample,row in zip(result['samples'],self.rows):
            if any(sample.get(k)!=row[k] for k in ('id','sha256','duration_seconds')): raise ValueError('Audio changed')
            worker.validate_sample(sample)
        self.state.update(timed_turns_validated=True,quality_validated=False,model_lock_sha256=assets['lock_sha256'])
        save_json(self.path,self.state)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file',required=True,type=Path); parser.add_argument('--submit',action='store_true')
    args=parser.parse_args()
    try:
        settings=full.TestSettings.load(args.env_file); runner=TimedRunner(settings); runner.check_available()
        if not args.submit:
            print(json.dumps({'status':'prepared_not_run','network_calls':0,'max_attempts':1,
                              'planning_total_usd':str(runner.settings.prior_spend+full.RESERVE_USD)})); return 0
        logging.basicConfig(level=logging.INFO)
        for h in logging.getLogger().handlers: h.setFormatter(SecretFormatter(settings.secrets.values()))
        logging.getLogger('httpx').setLevel(logging.WARNING)
        asyncio.run(runner.execute())
        print('Timed-turn evidence verified; disable test flags and Max workers. Quality review still required.')
        return 0
    except Exception as exc:
        print(json.dumps({'status':'stopped','error_type':type(exc).__name__,'automatic_retry':False})); return 1


if __name__=='__main__': raise SystemExit(main())
