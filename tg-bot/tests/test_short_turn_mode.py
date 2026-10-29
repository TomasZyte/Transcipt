from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import short_turn_test as s
from test_asr_mode import response
from test_speaker_mode import fixture as old_fixture, output as old_output


def fixture(root):
    settings = old_fixture(root)
    old_runner = s.previous.SpeakerRunner(settings)
    raw = old_output(old_runner)
    raw['alignment_local']['word_segments'] = deepcopy(raw['alignment_local']['segments'][0]['words'])
    raw['alignment_global'] = s.previous.comparison.global_alignment(raw['alignment_local'], old_runner.evidence)
    source = root / s.previous.CAMPAIGN / 'jobs/result-speaker-children-v1.json'
    source.parent.mkdir()
    s.save_json(source, raw)
    old_history = s.preparation.history_files(root)
    now = datetime.now(timezone.utc)
    for n in range(12 - len(old_history)):
        s.save_json(root / 'jobs' / ('message-reconciliation-' + str(n) + '.json'), {'status': 'COMPLETED'})
    history = s.preparation.history_files(root)
    for path in history.values():
        d = s.previous.read_json(path)
        d.update(status='COMPLETED', submitted_at=(now - timedelta(hours=3)).timestamp(),
                 elapsed_sec=25, planning_total_usd='1.616')
        s.save_json(path, d)
    health = {'endpoint': s.previous.ENDPOINT, 'observed_at_utc': now.isoformat(),
        'health': {'jobs': {'completed': 12, 'failed': 1, 'retried': 2, 'inProgress': 0, 'inQueue': 0},
                   'workers': {'idle': 0, 'running': 0, 'initializing': 0}}}
    shutdown = {'observed_at_utc': (now - timedelta(hours=2)).isoformat(), 'health': deepcopy(health['health'])}
    bill = {'endpoint': s.previous.ENDPOINT, 'observed_at_utc': now.isoformat(),
        'period_ui': 'окт. 1 30', 'period_timezone': 'UTC', 'group': 'Serverless / Endpoint',
        'active_workers': 0, 'max_workers': 0, 'endpoint_idle_observed_this_turn': True,
        'billing_lag_hours': 1, 'displayed_total_test_usd': '0.121',
        'table': 'Start date\tTotal\ttt17em69to6yua\n29 окт.\t$0,046\t$0,046\n28 окт.\t$0,075\t$0,075'}
    observations = {'billing': bill, 'health': health, 'shutdown': shutdown}
    for name, data in observations.items(): s.save_json(root / (name + '.json'), data)
    record = s.preparation.reconcile(bill, health, shutdown, [s.previous.read_json(p) for p in history.values()], now)
    record['history_sha256'] = {name: s.previous.digest(path) for name, path in history.items()}
    record['evidence_sha256'] = {name + '.json': s.previous.digest(root / (name + '.json')) for name in observations}
    s.save_json(root / 'reconciliation.json', record)
    frozen = {'source_result_sha256': s.previous.digest(source), 'asr_evidence': raw['asr_evidence'],
              'alignment_local': raw['alignment_local']}
    manifest = {'sample': {'path': str(root / 'sound2.wav'), 'sha256': s.previous.digest(root / 'sound2.wav')}, 'frozen': frozen}
    s.save_json(root / 'manifest.json', manifest)
    target = root / s.CAMPAIGN; target.mkdir()
    c = {'schema': 1, 'campaign': s.CAMPAIGN, 'endpoint': s.previous.ENDPOINT, 'bucket': s.previous.BUCKET,
        'commit': 'c' * 40, 'image': s.IMAGE_PREFIX + 'c' * 9,
        'source_manifest_sha256': s.previous.digest(s.previous.SOURCES),
        'reconciliation_file': str(root / 'reconciliation.json'),
        'reconciliation_sha256': s.previous.digest(root / 'reconciliation.json'),
        'history_sha256': record['history_sha256'], 'baseline_spend_usd': '0.25', 'budget_usd': '2',
        'frozen_manifest': str(root / 'manifest.json'), 'frozen_manifest_sha256': s.previous.digest(root / 'manifest.json'),
        'frozen_sha256': s.previous.comparison.evidence_sha(frozen),
        'audio_file': str(root / 'sound2.wav'), 'audio_sha256': manifest['sample']['sha256'],
        **{name + '_file': str(root / (name + '.json')) for name in observations}}
    s.save_json(target / 'campaign.json', c)
    p = s.previous.read_json(root / s.previous.CAMPAIGN / 'preflight.json')
    p.update(campaign_sha256=s.previous.digest(target / 'campaign.json'), short_turn_comparison_enabled=True,
             speaker_comparison_enabled=False, test_billed_usd='0.121')
    s.save_json(target / 'preflight.json', p)
    return settings


def output(runner):
    sources = s.previous.read_json(s.previous.SOURCES)['diarization']
    runtime = {'diarization': {'repository': sources['repository'], 'revision': sources['revision'],
        'files': {name: info.get('sha256', 'a' * 64) for name, info in sources['files'].items()}},
        'parameters': deepcopy(s.comparison.PARAMETERS), 'embedding_exclude_overlap': True,
        'packages': {'pyannote.audio': '4.0.3', 'whisperx': '3.8.6'}}
    outputs = {}
    for mode in s.comparison.MODES:
        ratio = .1 if mode.startswith('short_turns') else .2
        outputs[mode] = {'min_active_ratio': ratio, 'num_speakers': None,
            'filter_diagnostics': [{'min_active_ratio': ratio, 'selected_embeddings': 3, 'chunk_speaker_pairs': 9}],
            'timeline': [{'start': 0., 'end': 36., 'speaker': 'A'}],
            'aligned': deepcopy(runner.aligned), 'export': s.previous.comparison.export_result(runner.aligned)}
    return {'profile': s.CAMPAIGN, 'status': 'completed', 'sample_id': 'sound2', 'asr_rerun': False,
        'alignment_rerun': False, 'quality_validated': False, 'audio_sha256': runner.campaign['audio_sha256'],
        'duration_seconds': 36., 'frozen': runner.frozen, 'frozen_sha256': runner.campaign['frozen_sha256'],
        'source_manifest_sha256': s.previous.digest(s.previous.SOURCES), 'runtime': runtime, 'outputs': outputs}


class ShortTurnClientTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve(); self.settings = fixture(self.root)
        self.runner = s.ShortTurnRunner(self.settings)

    def test_offline_check_has_no_reservation_or_network(self):
        with patch.object(sys, 'argv', ['short_turn_test.py', '--env-file', 'fixture']), \
             patch.object(s.TestSettings, 'load', return_value=self.settings), \
             patch.object(s.asyncio, 'run', side_effect=AssertionError('network')), \
             patch('sys.stdout', new_callable=io.StringIO) as out:
            self.assertEqual(s.main(), 0)
            self.assertEqual(json.loads(out.getvalue())['planning_total_usd'], '0.75')
        self.assertFalse(self.runner.path.parent.exists())

    def test_reconciliation_cannot_be_forged_or_lowered(self):
        with self.assertRaises(ValueError): s.ShortTurnRunner(replace(self.settings, endpoint='l20o867nueti1d'))
        c = s.previous.read_json(self.runner.campaign_path)
        s.save_json(self.runner.campaign_path, {**c, 'baseline_spend_usd': '0.121'})
        with self.assertRaises(ValueError): s.ShortTurnRunner(self.settings)
        s.save_json(self.runner.campaign_path, c)
        record = s.previous.read_json(c['reconciliation_file']); record['baseline_spend_usd'] = '0.01'
        s.save_json(Path(c['reconciliation_file']), record)
        c['reconciliation_sha256'] = s.previous.digest(c['reconciliation_file'])
        s.save_json(self.runner.campaign_path, c)
        with self.assertRaises(ValueError): s.ShortTurnRunner(self.settings)

    def test_preflight_and_output_reject_drift(self):
        path = self.runner.settings.runtime / 'preflight.json'; original = s.previous.read_json(path)
        for change in ({'short_turn_comparison_enabled': False}, {'speaker_comparison_enabled': True},
                       {'test_billed_usd': '1.501'}):
            s.save_json(path, {**original, **change})
            with self.assertRaises((ValueError, RuntimeError)): self.runner.preflight()
        self.runner.path.parent.mkdir()
        result = output(self.runner); self.runner.validate_output(result)
        for change in ('ratio', 'time', 'package', 'rerun'):
            bad = deepcopy(result)
            if change == 'ratio': bad['outputs']['short_turns_regular']['min_active_ratio'] = .05
            if change == 'time': bad['outputs']['standard_regular']['aligned']['segments'][0]['start'] = 0
            if change == 'package': bad['runtime']['packages']['whisperx'] = '4.0'
            if change == 'rerun': bad['asr_rerun'] = True
            with self.assertRaises(ValueError): self.runner.validate_output(bad)


class ShortTurnSubmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_exactly_one_attempt_is_journaled_before_post_and_cannot_repeat(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); settings = fixture(root)
            client = AsyncMock(); client.__aenter__.return_value = client
            runner = s.ShortTurnRunner(settings, poll_s=0, client_factory=Mock(return_value=client))
            async def post(url, **kwargs):
                self.assertTrue((root / 'runner.lock').exists())
                self.assertEqual(s.previous.read_json(runner.path)['status'], 'reserved')
                self.assertEqual(kwargs['json']['policy'], {'executionTimeout': 300000, 'ttl': 420000})
                self.assertNotIn('evaluation', kwargs['json']['input'])
                return response({'id': 'short-turn-fixture'})
            async def get(url, **kwargs):
                if url.endswith('/health'): return response(runner.expected_health)
                return response({'status': 'COMPLETED', 'output': output(runner)})
            client.post.side_effect = post; client.get.side_effect = get
            url = 'https://' + settings.storage_host + '/poslovnica-test/quality/sound2.wav?sig=x'
            with patch.object(s.previous, 'upload_sound2', return_value=url) as upload:
                await runner.execute()
                upload.assert_called_once_with(runner, campaign=s.CAMPAIGN)
                with self.assertRaises(RuntimeError): await runner.execute()
            self.assertEqual(client.post.await_count, 1)
            self.assertEqual(s.previous.read_json(runner.path)['planning_total_usd'], '0.75')
            self.assertFalse((root / 'runner.lock').exists())

    async def test_changed_cloud_counts_stop_before_upload(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = fixture(Path(folder).resolve())
            client = AsyncMock(); client.__aenter__.return_value = client
            runner = s.ShortTurnRunner(settings, client_factory=Mock(return_value=client))
            changed = deepcopy(runner.expected_health); changed['jobs']['completed'] += 1
            client.get.return_value = response(changed)
            with patch.object(s.previous, 'upload_sound2', side_effect=AssertionError('upload')):
                with self.assertRaises(RuntimeError): await runner.execute()
            client.post.assert_not_called()


if __name__ == '__main__': unittest.main()
