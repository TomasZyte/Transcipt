from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import candidate_test as c
from test_short_turn_mode import fixture as previous_fixture
from test_asr_mode import response


def fixture(root):
    settings = previous_fixture(root)
    old = c.previous.ShortTurnRunner(settings)
    old.path.parent.mkdir()
    c.save_json(old.path, {**old.state, 'status': 'COMPLETED', 'short_turn_validated': True,
                          'planning_total_usd': '.75', 'campaign_sha256': old.campaign_sha})
    shutdown = {'endpoint': c.full.ENDPOINT, 'max_workers': 0, 'health': deepcopy(old.expected_health)}
    shutdown['health']['jobs']['completed'] += 1
    c.save_json(root/'shutdown-candidate.json', shutdown)
    corpus = root/'candidate-corpus'; corpus.mkdir()
    rows = []
    for name in c.full.IDS:
        path = corpus/(name+'.wav'); shutil.copyfile(root/'sound2.wav', path)
        rows.append({'id': name, 'wav': path.name, 'wav_sha256': c.full.digest(path)})
    manifest = corpus/'manifest.json'; c.save_json(manifest, {'samples': rows})
    root_new = root/c.candidate.PROFILE; root_new.mkdir()
    campaign = {'schema': 1, 'campaign': c.candidate.PROFILE, 'endpoint': c.full.ENDPOINT, 'bucket': c.full.BUCKET,
        'commit': 'd'*40, 'image': c.full.IMAGE_PREFIX+'d'*9, 'modes': c.candidate.MODES,
        'budget_usd': '2', 'baseline_spend_usd': '.75', 'manifest': str(manifest),
        'manifest_sha256': c.full.digest(manifest), 'source_manifest_sha256': c.full.digest(c.full.SOURCES),
        'audio_sha256': {r['id']: r['wav_sha256'] for r in rows},
        'history_sha256': {n: c.full.digest(p) for n, p in c.history_files(root).items()},
        'previous_campaign_sha256': old.campaign_sha, 'previous_shutdown_file': str(root/'shutdown-candidate.json'),
        'previous_shutdown_sha256': c.full.digest(root/'shutdown-candidate.json')}
    c.save_json(root_new/'campaign.json', campaign)
    preflight = c.full.read_json(old.settings.runtime/'preflight.json')
    preflight.update(image=campaign['image'], campaign_sha256=c.full.digest(root_new/'campaign.json'),
                     short_turn_comparison_enabled=False, release_candidate_enabled=True)
    c.save_json(root_new/'preflight.json', preflight)
    return settings


class CandidateClientTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve(); self.settings = fixture(self.root)
        self.client = AsyncMock(); self.client.__aenter__.return_value = self.client
        self.runner = c.CandidateRunner(self.settings, client_factory=Mock(return_value=self.client))

    async def test_keeps_previous_full_reservation_and_blocks_production(self):
        self.assertEqual(self.runner.settings.prior_spend, Decimal('.75'))
        self.runner.check_available(); self.runner.preflight()
        self.assertFalse(self.runner.path.exists())
        for field, value in (('endpoint', 'l20o867nueti1d'), ('bucket', 'transcribe')):
            with self.assertRaises(ValueError): c.CandidateRunner(replace(self.settings, **{field: value}))
        campaign = c.full.read_json(self.runner.campaign_path); campaign['baseline_spend_usd'] = '.25'
        c.save_json(self.runner.campaign_path, campaign)
        with self.assertRaises(ValueError): c.CandidateRunner(self.settings)

    async def test_changed_history_and_live_counts_stop_before_upload(self):
        health = deepcopy(self.runner.expected_health); health['jobs']['completed'] += 1
        self.client.get.return_value = response(health)
        with patch.object(c.full, 'upload_samples', side_effect=AssertionError('upload')):
            with self.assertRaises(RuntimeError): await self.runner.execute()
        self.client.post.assert_not_called()
        old = self.root/c.previous.CAMPAIGN/'jobs/message-speaker-short-turns-v1.json'
        c.save_json(old, {'status': 'COMPLETED', 'changed': True})
        with self.assertRaises(ValueError): self.runner.check_available()

    async def test_flags_budget_and_single_attempt_guard(self):
        path = self.runner.settings.runtime/'preflight.json'; original = c.full.read_json(path)
        for change in ({'release_candidate_enabled': False}, {'short_turn_comparison_enabled': True},
                       {'test_billed_usd': '1.51'}):
            c.save_json(path, {**original, **change})
            with self.assertRaises((ValueError, RuntimeError)): self.runner.preflight()
        self.runner.path.parent.mkdir()
        c.save_json(self.runner.path, {'status': 'COMPLETED'})
        with self.assertRaises(RuntimeError): self.runner.check_available()


if __name__ == '__main__': unittest.main()
