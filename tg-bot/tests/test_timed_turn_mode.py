from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import timed_turn_test as t
from test_candidate_mode import fixture as old_fixture
from test_asr_mode import response


def fixture(root):
    settings=old_fixture(root); old=t.previous.CandidateRunner(settings)
    old.path.parent.mkdir()
    t.save_json(old.path,{**old.state,'status':'COMPLETED','candidate_technical_validated':True,
        'planning_total_usd':'1.25','campaign_sha256':old.campaign_sha})
    shutdown={'endpoint':t.full.ENDPOINT,'max_workers':0,'health':deepcopy(old.expected_health)}
    shutdown['health']['jobs']['completed']+=1
    t.save_json(root/'timed-shutdown.json',shutdown)
    campaign={**old.campaign,'campaign':t.worker.PROFILE,'modes':t.worker.MODES,
        'baseline_spend_usd':'1.25','history_sha256':{n:t.full.digest(p) for n,p in t.history_files(root).items()},
        'previous_campaign_sha256':old.campaign_sha,'previous_shutdown_file':str(root/'timed-shutdown.json'),
        'previous_shutdown_sha256':t.full.digest(root/'timed-shutdown.json')}
    folder=root/t.worker.PROFILE; folder.mkdir()
    t.save_json(folder/'campaign.json',campaign)
    preflight=t.full.read_json(old.settings.runtime/'preflight.json')
    preflight.update(release_candidate_enabled=False,timed_turns_enabled=True,campaign_sha256=t.full.digest(folder/'campaign.json'))
    t.save_json(folder/'preflight.json',preflight)
    return settings


class TimedClientTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve(); self.settings=fixture(self.root)
        self.client=AsyncMock(); self.client.__aenter__.return_value=self.client
        self.runner=t.TimedRunner(self.settings,client_factory=Mock(return_value=self.client))

    async def test_budget_history_production_and_single_attempt(self):
        self.assertEqual(self.runner.settings.prior_spend,Decimal('1.25'))
        self.runner.check_available(); self.runner.preflight()
        self.assertEqual(self.runner.state['planning_total_usd'],'1.75')
        with self.assertRaises(ValueError): t.TimedRunner(replace(self.settings,endpoint='l20o867nueti1d'))
        c=t.full.read_json(self.runner.campaign_path); c['baseline_spend_usd']='.75'
        t.save_json(self.runner.campaign_path,c)
        with self.assertRaises(ValueError): t.TimedRunner(self.settings)

    async def test_changed_health_stops_before_audio_upload(self):
        health=deepcopy(self.runner.expected_health); health['jobs']['completed']+=1
        self.client.get.return_value=response(health)
        with patch.object(t.full,'upload_samples',side_effect=AssertionError('upload')):
            with self.assertRaises(RuntimeError): await self.runner.execute()
        self.client.post.assert_not_called()

    async def test_old_flag_or_second_attempt_is_blocked(self):
        p=self.runner.settings.runtime/'preflight.json'; d=t.full.read_json(p)
        d['release_candidate_enabled']=True; t.save_json(p,d)
        with self.assertRaises(ValueError): self.runner.preflight()
        self.runner.path.parent.mkdir(); t.save_json(self.runner.path,{'status':'COMPLETED'})
        with self.assertRaises(RuntimeError): self.runner.check_available()


if __name__=='__main__': unittest.main()
