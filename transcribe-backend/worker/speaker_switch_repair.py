"""Offline-only candidate for brief A-B-A label islands within a continuing word.

Never imported by the production handler. Input is validated pause-context
evidence, with no reference text, age, roles or human boundaries. Changes are
reversible proposals on a copy, not a claim of acoustically verified identity.
"""
from copy import deepcopy

import pause_context as pause
from transcript_candidate import attribute, merge_intervals

VERSION = 'speaker-switch-repair-v1'
POLICY = {
    'max_island_seconds': .5,
    'min_flank_seconds': .3,
    'max_onset_difference_seconds': .05,
    'min_shared_island_ratio': .8,
    'word_edge_margin_seconds': .02,
    'min_crossing_alignment_modes': 2,
}
EPS = 1e-6


def _overlap(a, b, c, d):
    return max(0., min(b, d) - max(a, c))


def _regular_agrees(rows, triple, scope):
    """Exclusive decoding alone can hide another simultaneous speaker."""
    for turn in triple:
        a, b = max(scope[0], turn['start']), min(scope[1], turn['end'])
        pieces = []
        for row in rows:
            if _overlap(a, b, row['start'], row['end']) > EPS:
                if row['speaker'] != turn['speaker']:
                    return False
                pieces.append((max(a, row['start']), min(b, row['end'])))
        if abs(sum(y-x for x,y in merge_intervals(pieces)) - (b-a)) > EPS:
            return False
    return True


def _islands(track, mapping, policy):
    found = []
    rows = track['exclusive']
    for i in range(1, len(rows)-1):
        left, middle, right = rows[i-1:i+2]
        outer, inner = mapping.get(left['speaker']), mapping.get(middle['speaker'])
        if (outer is None or inner is None or outer == inner
                or outer != mapping.get(right['speaker'])
                or middle['end']-middle['start'] > policy['max_island_seconds'] + EPS
                or min(left['end']-left['start'], right['end']-right['start']) < policy['min_flank_seconds']-EPS
                or abs(left['end']-middle['start']) > EPS
                or abs(middle['end']-right['start']) > EPS):
            continue
        scope = (middle['start']-policy['min_flank_seconds'], middle['end']+policy['min_flank_seconds'])
        found.append({'index':i, 'start':middle['start'], 'end':middle['end'],
                      'from_label':middle['speaker'], 'to_label':left['speaker'],
                      'outer':outer, 'inner':inner, 'scope':scope,
                      'regular_agrees':_regular_agrees(track['regular'], (left,middle,right), scope)})
    return found


def _word_evidence(sample):
    """Match tokens by occurrence, not by their spelling alone."""
    native_tokens, native = [], []
    for record in sample['windows']:
        offset = record['window']['start_frame'] / pause.base.RATE
        for segment in record['native']:
            for word in segment['words']:
                native_tokens.append(word['word'].strip())
                native.append((word['start']+offset, word['end']+offset))
    aligned = {}; tokens = None
    for mode in pause.ALIGNMENTS:
        seq, spans = pause._word_times(sample['alignments'][mode]['segments'])
        if tokens is None:
            tokens = seq
        if seq != tokens or ''.join(seq) != ''.join(native_tokens):
            return None
        aligned[mode] = spans
    # Decoder may split a hyphenated word that CTC joins. Keep text positions exact;
    # a split native word has no reliable independent span for either fragment.
    positions = []; pos = 0
    for token,span in zip(native_tokens,native,strict=True):
        positions.append((pos,pos+len(token),span)); pos += len(token)
    matched = []; pos = 0
    for token in tokens:
        end = pos+len(token)
        parts = [(a,b,span) for a,b,span in positions if a < end and b > pos]
        matched.append((parts[0][2][0],parts[-1][2][1])
                       if parts and parts[0][0] == pos and parts[-1][1] == end else None)
        pos = end
    return tokens, matched, aligned


def _reply_supported(sample, pair, native, aligned):
    """Protect even a one-word interjection supported by both diarization tracks."""
    def is_inner(track, a, b):
        if b <= a:
            return False
        switch = pair[track]
        pieces = merge_intervals((max(a,r['start'],switch['start']), min(b,r['end'],switch['end']))
            for r in sample['diarization'][track]['regular'] if r['speaker'] == pair[track]['from_label'])
        return sum(y-x for x,y in pieces) >= pause.base.POLICY['speaker_min_coverage'] * (b-a)
    if any(span is not None and all(is_inner(track,*span) for track in pair) for span in native):
        return True
    words = {mode:[w for seg in output['attributed']['segments'] for w in seg.get('words', [])]
             for mode,output in sample['outputs'].items()}
    for i in range(len(native)):
        support = sum(aligned[alignment][i] is not None and
                      all(_overlap(*aligned[alignment][i],pair[track]['start'],pair[track]['end']) > 0
                          and words[alignment+'_'+track][i].get('speaker') == pair[track]['from_label']
                          for track in pair) for alignment in pause.ALIGNMENTS)
        if support >= 2:
            return True
    return False


def _lexical_guard(sample, pair, evidence, policy):
    if evidence is None:
        return 'incomparable_words', None
    tokens, native, aligned = evidence
    low, high = min(c['start'] for c in pair.values()), max(c['end'] for c in pair.values())
    margin = policy['word_edge_margin_seconds']
    # Whole lexical units protect short replies, even without punctuation.
    if _reply_supported(sample, pair, native, aligned):
        return 'supported_short_reply', None
    # Both switches must cut the same word, confirmed by native and aligned clocks.
    crossing = []
    for i,span in enumerate(native):
        if span is None:
            continue
        a,b = span
        if b-a > pause.base.POLICY['max_word_seconds'] or not all(
                a+margin <= c['start'] <= b-margin for c in pair.values()):
            continue
        modes = [mode for mode,spans in aligned.items() if spans[i] is not None
                 and spans[i][1]-spans[i][0] <= pause.base.POLICY['max_word_seconds']
                 and all(spans[i][0]+margin <= c['start'] <= spans[i][1]-margin for c in pair.values())]
        if len(modes) >= policy['min_crossing_alignment_modes']:
            crossing.append({'word_index':i,'word':tokens[i],'alignment_modes':modes})
    if len(crossing) != 1:
        return 'no_shared_in_word_onset', None
    # Reject a sentence boundary anywhere inside the proposed correction.
    for segment in sample['alignments']['control']['segments']:
        if any(low+EPS < segment[k] < high-EPS for k in ('start','end')):
            return 'phrase_boundary', None
    return None, crossing[0]


def _replace_interval(rows, change):
    output = []
    for row in rows:
        a,b = max(row['start'],change['start']), min(row['end'],change['end'])
        if row['speaker'] != change['from_label'] or a >= b:
            output.append(deepcopy(row))
            continue
        if row['start'] < a:
            output.append({**row,'end':a})
        output.append({**row,'start':a,'end':b,'speaker':change['to_label']})
        if b < row['end']:
            output.append({**row,'start':b})
    return output


def propose(sample, *, policy=None):
    """Return candidate and audit; never mutate or reuse an already repaired sample."""
    if 'speaker_switch_repair' in sample:
        raise ValueError('Always start from original evidence, not an already repaired candidate')
    policy = dict(POLICY if policy is None else policy)
    if (set(policy) != set(POLICY)
            or any(not pause.base.finite(v) or v <= 0 for v in policy.values())
            or policy['min_shared_island_ratio'] > 1
            or policy['min_crossing_alignment_modes'] not in (2,3)):
        raise ValueError('Invalid switch repair policy')
    maps = pause._speaker_maps(sample['diarization'])
    islands = {t:_islands(sample['diarization'][t],maps[t],policy) for t in ('dense','short')}
    compatible = []
    for dense in islands['dense']:
        for short in islands['short']:
            if (dense['outer'] == short['outer'] and dense['inner'] == short['inner']
                    and abs(dense['start']-short['start']) <= policy['max_onset_difference_seconds']+EPS
                    and _overlap(dense['start'],dense['end'],short['start'],short['end'])+EPS
                    >= policy['min_shared_island_ratio'] * min(dense['end']-dense['start'], short['end']-short['start'])):
                compatible.append({'dense':dense,'short':short})
    evidence = _word_evidence(sample)
    accepted, decisions = [], []
    for pair in compatible:
        # No cascading smoothing or overlapping decisions in this pass.
        conflict = any(other is not pair and any(_overlap(*pair[t]['scope'],*other[t]['scope'])>0 for t in pair)
                       for other in compatible)
        if conflict:
            reason, word = 'overlapping_candidates', None
        elif not all(c['regular_agrees'] for c in pair.values()):
            reason, word = 'overlap_or_uncovered_speech', None
        else:
            reason, word = _lexical_guard(sample,pair,evidence,policy)
        decisions.append({'tracks':deepcopy(pair),'applied':reason is None,'rejection':reason,'word_support':word})
        if reason is None:
            accepted.append(pair)
    candidate = deepcopy(sample)
    for pair in accepted:
        for track,change in pair.items():
            for kind in ('regular','exclusive'):
                candidate['diarization'][track][kind] = _replace_interval(candidate['diarization'][track][kind],change)
    if pause._speaker_maps(candidate['diarization']) != maps:
        candidate = deepcopy(sample)
        for decision in decisions:
            if decision['applied']:
                decision.update(applied=False,rejection='global_matching_changed')
        accepted = []
    if accepted:
        for mode,definition in pause.MODES.items():
            candidate['outputs'][mode] = attribute(sample['asr'],sample['alignments'][definition['alignment']],
                candidate['diarization'][definition['diarization']]['regular'],sample['duration_seconds'])
        # These optional overlays must describe this candidate, not stale original labels.
        if 'phraseSpeakerReview' in candidate:
            candidate['phraseSpeakerReview'] = pause.phrase_speaker_review(candidate)
            candidate['phraseSpeakerReviewVersion'] = 3
    audit = {'version':VERSION, 'policy':policy, 'island_counts':{t:len(v) for t,v in islands.items()},
             'decisions':decisions, 'applied_pairs':len(accepted), 'production_enabled':False}
    candidate['speaker_switch_repair'] = deepcopy(audit)
    return candidate,audit
