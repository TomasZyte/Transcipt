"""Prepare a local short-turn experiment and reconcile closed test allowances.

No network client, uploads, Telegram polling, or paid submission are implemented.
"""
import argparse
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'transcribe-backend/worker'))
import short_turn_comparison as comparison
import speaker_test as previous
from integration_guard import TERMINAL


def stamp(text):
    value = datetime.fromisoformat(text.replace('Z', '+00:00'))
    if value.tzinfo is None:
        raise ValueError('Observation requires an explicit timezone')
    return value


def dollars(text):
    value = Decimal(str(text))
    if not value.is_finite() or value < 0:
        raise ValueError('Invalid amount')
    return value


def history_files(runtime):
    found = previous.history_files(runtime)
    folder = runtime / previous.CAMPAIGN / 'jobs'
    if any(folder.glob('*.tmp')):
        raise ValueError('Unfinished earlier journal')
    for path in folder.glob('*.json'):
        if path.name.startswith(('message-', 'retry-', 'failed-')):
            found[path.relative_to(runtime).as_posix()] = path
    return found


def reconcile(billing, health, shutdown, journals, now=None):
    """Append-only reconciliation, only after the billing lag and an idle check."""
    now = now or datetime.now(timezone.utc)
    observed = stamp(billing['observed_at_utc'])
    health_at = stamp(health['observed_at_utc'])
    if any(not 0 <= (now - t).total_seconds() <= 900 for t in (observed, health_at)):
        raise ValueError('Billing and health observations must be under 15 minutes old')
    if (billing.get('endpoint') != previous.ENDPOINT or health.get('endpoint') != previous.ENDPOINT
            or billing.get('period_ui') != 'окт. 1 30' or billing.get('period_timezone') != 'UTC'
            or billing.get('group') != 'Serverless / Endpoint'
            or billing.get('active_workers') != 0 or billing.get('max_workers') != 0
            or billing.get('endpoint_idle_observed_this_turn') is not True):
        raise ValueError('This reviewed observation must cover the entire October test campaign')
    current = health['health']
    if (current != shutdown['health'] or current['jobs'].get('inProgress') != 0
            or current['jobs'].get('inQueue') != 0 or not current.get('workers')
            or any(value != 0 for value in current['workers'].values())):
        raise ValueError('Cloud jobs/workers changed since the recorded shutdown')
    if len(journals) != 12 or any(j.get('status') not in TERMINAL for j in journals):
        raise ValueError('Expected the twelve unchanged closed journal records')
    last_end = max(float(j.get('submitted_at') or 0) + float(j.get('elapsed_sec') or 0) for j in journals)
    cutoff = max(last_end, stamp(shutdown['observed_at_utc']).timestamp())
    lag = billing.get('billing_lag_hours')
    if not isinstance(lag, (int, float)) or isinstance(lag, bool) or not 1 <= lag <= 24:
        raise ValueError('Billing delay must be known')
    if observed.timestamp() - cutoff < lag * 3600 + 900:
        raise ValueError('Wait for billing delay plus a fifteen-minute margin after shutdown')
    # Bind the human-readable observation to its endpoint column, not account totals.
    lines = billing['table'].splitlines()
    headings = lines[0].split('\t')
    index = headings.index(previous.ENDPOINT)
    amounts = [dollars(line.split('\t')[index].replace('$', '').replace(',', '.'))
               for line in lines[1:] if line.strip()]
    billed = sum(amounts, Decimal(0))
    if billed != dollars(billing['displayed_total_test_usd']):
        raise ValueError('Endpoint column and reported total disagree')
    # Two nonzero day amounts are displayed with three decimals. Add both a
    # rounding allowance and $0.10 contingency; never reduce old journal values.
    rounding = Decimal('0.001') * sum(amount != 0 for amount in amounts)
    baseline = max(Decimal('0.25'), (billed + rounding + Decimal('0.10')).quantize(Decimal('0.01'), rounding=ROUND_CEILING))
    reserve = Decimal('0.50')
    if baseline + reserve > Decimal('2'):
        raise ValueError('Cumulative budget cannot cover the next attempt')
    return {'schema': 1, 'kind': 'closed_allowance_reconciliation', 'endpoint': previous.ENDPOINT,
        'observed_billed_usd': str(billed), 'billing_rounding_allowance_usd': str(rounding),
        'previous_planning_total_usd': str(max(dollars(j.get('planning_total_usd', 0)) for j in journals)),
        'baseline_spend_usd': str(baseline), 'next_attempt_reserve_usd': str(reserve),
        'planning_total_usd': str(baseline + reserve), 'budget_usd': '2',
        'basis': 'All journal entries terminal, cloud unchanged and idle, billing lag plus margin elapsed',
        'old_journals_modified': False, 'paid_run_ready': False}


def prepare(*, audio, source, annotations, runtime, billing_path, health_path, shutdown_path, destination):
    raw, owner = previous.read_json(source), previous.read_json(annotations)
    if (raw.get('profile') != previous.CAMPAIGN or raw.get('status') != 'completed'
            or raw.get('sample_id') != 'sound2' or raw.get('asr_rerun') is not False):
        raise ValueError('Only the completed sound2 speaker comparison may supply frozen evidence')
    frozen = {'source_result_sha256': previous.digest(source), 'asr_evidence': raw['asr_evidence'],
              'alignment_local': raw['alignment_local']}
    aligned = comparison.validate_frozen(frozen)
    if aligned != raw['alignment_global']:
        raise ValueError('Saved local and global timing disagree')
    duration = previous.comparison.validate_audio(audio, raw['audio_sha256'], frozen['asr_evidence'])
    intervals = owner['intervals']
    if [[i['source_start_seconds'], i['source_end_seconds']] for i in intervals] != [[11.5, 12.5], [14.5, 17.0]]:
        raise ValueError('Owner reference changed; update the evaluation plan explicitly')
    history = history_files(runtime)
    checksums = {name: previous.digest(path) for name, path in history.items()}
    prior_campaign = previous.read_json(runtime / previous.CAMPAIGN / 'campaign.json')
    for name, expected in prior_campaign['history_sha256'].items():
        if checksums.get(name) != expected:
            raise ValueError('Earlier journal changed')
    if previous.digest(source) != previous.digest(runtime / previous.CAMPAIGN / 'jobs/result-speaker-children-v1.json'):
        raise ValueError('Source differs from the original cloud result')
    reconciliation = reconcile(previous.read_json(billing_path), previous.read_json(health_path),
                               previous.read_json(shutdown_path), [previous.read_json(p) for p in history.values()])
    reconciliation['history_sha256'] = checksums
    reconciliation['evidence_sha256'] = {p.name: previous.digest(p) for p in (billing_path, health_path, shutdown_path)}
    plan = {'profile': comparison.PROFILE, 'status': 'prepared_not_run', 'network_calls': 0,
        'endpoint': previous.ENDPOINT, 'bucket': previous.BUCKET, 'max_attempts': 1,
        'audio_sha256': raw['audio_sha256'], 'duration_seconds': duration,
        'frozen_sha256': previous.comparison.evidence_sha(frozen), 'modes': comparison.MODES,
        'min_active_ratio': comparison.RATIOS, 'unchanged_parameters': comparison.PARAMETERS,
        'asr_rerun': False, 'alignment_rerun': False, 'num_speakers': None,
        'evaluation_only_annotation_file': str(annotations.resolve()),
        'annotation_sha256': previous.digest(annotations),
        'evaluation': {'children': [[11.5, 12.5], [14.5, 17.0]],
            'father_reference_seconds': [[0.1, 2.0], [17.429, 18.0]],
            'wife_reference': 'unchanged saved roof-sentence anchor; approximate',
            'criteria': ['child intervals separate from adult reference labels',
                         'father labels consistent at both references',
                         'wife anchor not fragmented', 'inspect regular overlap; do not score as DER'],
            'same_child_across_intervals': 'unknown'},
        'budget': reconciliation, 'paid_run_ready': False, 'quality_validated': False,
        'before_submit': ['Build and verify the test image; the deployed image has no new adapter',
            'Create a one-attempt campaign bound to the built commit and these immutable inputs',
            'Refresh billing/idle preflight and revalidate the reconciled budget',
            'Enable only full pipeline and short-turn test flags on the test endpoint',
            'After terminal status disable flags/Max workers and confirm idle']}
    destination.mkdir(parents=True, exist_ok=False)
    previous.comparison.write_new(destination / 'worker-manifest.json',
        {'sample': {'path': str(audio.resolve()), 'sha256': raw['audio_sha256']}, 'frozen': frozen})
    previous.comparison.write_new(destination / 'reconciliation.json', reconciliation)
    previous.comparison.write_new(destination / 'plan.json', plan)
    if checksums != {name: previous.digest(path) for name, path in history.items()}:
        raise ValueError('Historical files changed during preparation')
    return {'status': 'prepared_not_run', 'network_calls': 0, 'paid_run_ready': False,
            'plan': str((destination / 'plan.json').resolve()),
            'billed_usd': reconciliation['observed_billed_usd'],
            'planning_total_usd': reconciliation['planning_total_usd'], 'budget_usd': '2'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('audio', 'source', 'annotations', 'runtime', 'billing', 'health', 'shutdown', 'destination'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = prepare(audio=args.audio, source=args.source, annotations=args.annotations, runtime=args.runtime,
        billing_path=args.billing, health_path=args.health, shutdown_path=args.shutdown, destination=args.destination)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
