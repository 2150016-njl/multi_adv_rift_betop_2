"""Read-only evidence audit of V6 ZIPs, excluding all preflight fixtures."""
import argparse
import collections
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import yaml


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--skip_clean_cache', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    pipe = load('audit_v6_pipeline', root / 'scripts/joint_rift_v6.py')
    comparator = load('audit_v6_comparator', root / 'scripts/compare_joint_v5_outcomes.py')
    old_values = comparator.values
    def extended(record):
        values = old_values(record)
        for label in ('vehicle', 'pedestrian', 'layout'):
            values['ego_' + label + '_collision_count'] = float(len(record.get('infractions', {}).get('collisions_' + label, [])))
        values['ego_blocked'] = float('blocked' in record['status'].lower())
        values['ego_timed_out'] = float('timed out' in record['status'].lower())
        return values
    comparator.values = extended
    with zipfile.ZipFile(args.results) as archive:
        config = yaml.safe_load(archive.read('eval.yaml'))
        original_stats = {n: json.loads(archive.read(n)) for n in archive.namelist() if n.startswith('stats') and n.endswith('/summary.json')}
        hashes = {n: hashlib.sha256(archive.read(n)).hexdigest() for n in archive.namelist() if n.startswith('source/')}
        differences = [n for n, h in hashes.items() if (root / n[7:]).is_file() and hashlib.sha256((root / n[7:]).read_bytes()).hexdigest() != h]
        inventory = json.loads(archive.read('PACKAGE_MANIFEST.json'))['sha256']
        invalid = [n for n, h in inventory.items() if hashlib.sha256(archive.read(n)).hexdigest() != h]
        modes, logs = {}, {}
        for n in archive.namelist():
            if not (n.startswith('teachers/') or n.startswith('capture_seed0/')) or not n.endswith('simulation_results.json'):
                continue
            mode = n.split('/')[1][:-6] if n.startswith('teachers/') else 'capture_independent'
            payload = json.loads(archive.read(n))['_checkpoint']
            modes[mode] = payload
            metric_path = n.rsplit('/', 1)[0] + '/joint_rift_eval_metrics.jsonl'
            logs[mode] = [json.loads(x) for x in archive.read(metric_path).decode().splitlines() if x.strip()]
    with zipfile.ZipFile(args.cache) as archive:
        states = []
        names = [n for n in archive.namelist() if n.startswith('cache/') and n.endswith('.npz')]
        excluded = [n for n in archive.namelist() if n.endswith('.npz') and not n.startswith('cache/')]
        for n in sorted(names):
            with np.load(io.BytesIO(archive.read(n)), allow_pickle=False) as d:
                data = {k: d[k] for k in d.files}
            states.append(dict(path=n, data=data, metadata=json.loads(archive.read(n[:-4] + '.json'))))
        # A clean input ZIP permits independent offline reproductions without fixtures.
        if not args.skip_clean_cache:
            with zipfile.ZipFile(args.output / 'real_cache_only.zip', 'x', zipfile.ZIP_DEFLATED) as clean:
                for n in sorted(names):
                    clean.writestr(n, archive.read(n))
                    clean.writestr(n[:-4] + '.json', archive.read(n[:-4] + '.json'))
    (args.output / 'eval.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    replay_error = pipe.relabel_semantics(states, 'forward_braid')
    pipe.prepare(states, config['decision_teacher'])
    teacher = pipe.teacher_module.decision_teacher
    rows = []
    for s in states:
        d, m, t, b = s['data'], s['metadata'], s['teacher'], s['baseline']
        f, score = t['feasible_mask'], t['score']
        p1, p2 = d['p1'][:, None], d['p2'][None, :]
        bilateral = f & (p1 >= .01) & (p2 >= .01) & (d['coalition_gain'] >= .01)
        best = pipe.teacher_module.stable_best(score, f, t['quality'], d['base_logits']) or b
        one_q = f.copy(); ii, jj = np.indices(f.shape)
        one_q &= (ii == t['quality'][0]) | (jj == t['quality'][1])
        one_b = f & ((ii == b[0]) | (jj == b[1]))
        weights = .98 ** np.arange(d['interaction_trace_1'].shape[-1])
        overlap = np.einsum('it,jt,t->ij', d['interaction_trace_1'], d['interaction_trace_2'], weights)
        qbest = np.unravel_index(np.where(d['valid_mask'], d['joint_q'], -np.inf).argmax(), f.shape)
        positive_score = []
        for axis in (1, 2):
            positive_score.append((d['interaction_trace_' + str(axis)] * (d['interaction_sign_' + str(axis)] > 0) * weights).sum(-1))
        row = dict(path=s['path'], route=m['route_id'], repetition=m['repetition_index'], step=m['time_step'], pair_ids=m['pair_ids'],
            baseline=b, quality=t['quality'], factorized=t['factorized'], best=best,
            feasible_pairs=int(f.sum()), baseline_feasible=bool(f[b]), quality_changed=t['quality'] != b,
            interaction=bool(t['metrics']['decision_interaction_support']), joint=bool(t['metrics']['decision_joint_support']),
            bilateral_pairs=int(bilateral.sum()), valid_bilateral_pairs=int((d['valid_mask'] & (p1 >= .01) & (p2 >= .01) & (d['coalition_gain'] >= .01)).sum()),
            axis1_pressure_max=float(p1.max()), axis2_pressure_max=float(p2.max()),
            best_gain_quality=float(score[best] - score[t['quality']]) if f.any() else None,
            best_gain_factorized=float(score[best] - score[t['factorized']]) if f.any() else None,
            best_gain_baseline=float(score[best] - score[b]) if f.any() else None,
            best_bilateral_gain_factorized=float(score[bilateral].max() - score[t['factorized']]) if bilateral.any() else None,
            best_bilateral_gain_quality=float(score[bilateral].max() - score[t['quality']]) if bilateral.any() else None,
            max_coalition_feasible=float(d['coalition_gain'][f].max()) if f.any() else None,
            best_vs_one_quality=float(score[best] - score[one_q].max()) if one_q.any() else None,
            best_vs_one_baseline=float(score[best] - score[one_b].max()) if one_b.any() else None,
            chosen_best_positive_p1=float(positive_score[0][best[0]]), chosen_best_positive_p2=float(positive_score[1][best[1]]),
            chosen_best_p1=float(p1[best[0], 0]), chosen_best_p2=float(p2[0, best[1]]),
            overlap_max=float(overlap.max()), union_identity_error=float(abs(d['p12'] - (p1 + p2 - overlap)).max()),
            full_q_gain=float(d['joint_q'][qbest] - d['joint_q'][b]),
            full_q_unary_gain=float(d['unary_return_1'][qbest[0]] + d['unary_return_2'][qbest[1]] - d['unary_return_1'][b[0]] - d['unary_return_2'][b[1]]),
            full_q_collision_gain=float(-20. * (int(d['pair_collision'][qbest]) - int(d['pair_collision'][b]))),
            full_q_interaction_gain=float(score[qbest] - score[b]),
            positive_events=sum(int((d['interaction_sign_' + str(a)] > 0).sum()) for a in (1, 2)),
            negative_events=sum(int((d['interaction_sign_' + str(a)] < 0).sum()) for a in (1, 2)))
        rows.append(row)
    variants = {}
    settings = [('main', {}, {}), ('label_only', {}, {'nominal_conflict_policy': 'label_only'}),
                ('no_budget', {'unary_budget': 1e9}, {}), ('no_collision_gate', {'require_unary_collision_free': False}, {}),
                ('no_offroad_gate', {'require_offroad_free': False}, {}),
                ('valid_pair_only', {'unary_budget': 1e9, 'require_unary_collision_free': False, 'require_offroad_free': False}, {}),
                ('zero_joint_margin', {}, {'min_joint_gain': 0.}),
                ('no_bilateral_thresholds', {}, {'min_member_pressure': 0., 'min_coalition': 0.})]
    settings += [('budget_' + str(v), {'unary_budget': v}, {}) for v in (1., 2., 5., 10.)]
    for name, quality_override, decision_override in settings:
        counts = collections.Counter(); route_sets = collections.defaultdict(set)
        for s in states:
            q = dict(s['quality_config'], **quality_override)
            cfg = dict(config['decision_teacher'], **decision_override)
            t = teacher(s['data'], s['baseline'], s['data']['base_logits'], q, cfg)
            counts['feasible_states'] += int(t['feasible_mask'].any())
            for label in ('interaction', 'joint', 'bilateral'):
                yes = bool(t['metrics']['decision_' + label + '_support'])
                counts[label] += yes
                if yes: route_sets[label].add(s['metadata']['route_id'])
        variants[name] = dict(counts=counts, routes={k: sorted(v) for k, v in route_sets.items()})
    # Isolate AABB hard-gate effects while holding exact-SAT unary returns fixed.
    counts = collections.Counter()
    for s in states:
        d = dict(s['data'])
        for a in (1, 2):
            d['unary_collision_' + str(a)] = (d['collision_aabb_ego_trace_' + str(a)] | d['collision_aabb_background_trace_' + str(a)]).any(-1)
        t = teacher(d, s['baseline'], d['base_logits'], s['quality_config'], config['decision_teacher'])
        counts['feasible_states'] += int(t['feasible_mask'].any())
        counts['interaction'] += int(t['metrics']['decision_interaction_support'])
        counts['joint'] += int(t['metrics']['decision_joint_support'])
    variants['aabb_gate_only_exact_returns_fixed'] = dict(counts)
    mode_reports, records = {}, {}
    for mode, payload in modes.items():
        rr = payload['records']; ll = logs[mode]
        records[mode] = {(0, r['route_id']): r for r in rr}
        numeric = ('decision_has_feasible', 'decision_quality_repair', 'decision_interaction_support', 'decision_joint_support', 'decision_bilateral_support', 'decision_executed_changed', 'decision_executed_feasible')
        intervention = [r for r in ll if r['decision_executed_changed'] and r['decision_executed_feasible'] and r['decision_executed_gain_over_quality'] >= .05 - 1e-8]
        joint_rows = [r for r in ll if r['decision_joint_support']]
        mode_reports[mode] = dict(progress=payload['progress'], episodes=len(rr), statuses=dict(collections.Counter(r['status'] for r in rr)),
            scores={k: float(np.mean([r['scores'][k] for r in rr])) for k in ('score_composed', 'score_route', 'score_penalty')},
            collisions=sum(r['meta']['cbv_collision_count'] for r in rr),
            joint_states=len(ll), diagnostic_counts={k: int(sum(r[k] for r in ll)) for k in numeric},
            interaction_interventions=len(intervention),
            interaction_interventions_by_route=dict(collections.Counter(r['route_id'] for r in intervention)),
            joint_examples=[{k: r[k] for k in ('route_id', 'repetition_index', 'time_step', 'pair_ids', 'decision_joint_increment', 'decision_quality_score', 'decision_factorized_score', 'executed_indices')} for r in joint_rows])
    comparisons = {}
    for a, b in [('quality_oracle', 'independent'), ('factorized_oracle', 'independent'), ('interaction_oracle', 'independent'), ('joint_oracle', 'independent'),
                 ('interaction_oracle', 'quality_oracle'), ('joint_oracle', 'factorized_oracle'), ('joint_oracle', 'quality_oracle'), ('capture_independent', 'independent')]:
        comparisons[a + '_vs_' + b] = comparator.compare(records[a], records[b])
    counts = dict(states=len(rows), feasible=sum(r['feasible_pairs'] > 0 for r in rows), baseline_feasible=sum(r['baseline_feasible'] for r in rows),
                  bilateral_feasible=sum(r['bilateral_pairs'] > 0 for r in rows), bilateral_valid=sum(r['valid_bilateral_pairs'] > 0 for r in rows),
                  interaction=sum(r['interaction'] for r in rows), joint=sum(r['joint'] for r in rows),
                  score_beats_factorized_001=sum(r['best_gain_factorized'] is not None and r['best_gain_factorized'] >= .01 for r in rows),
                  both_axes_have_pressure=sum(r['axis1_pressure_max'] >= .01 and r['axis2_pressure_max'] >= .01 for r in rows),
                  no_overlap=sum(r['overlap_max'] < 1e-8 for r in rows),
                  interaction_single_change_quality_sufficient=sum(r['interaction'] and r['best_vs_one_quality'] is not None and r['best_vs_one_quality'] < 1e-6 for r in rows),
                  positive_events=sum(r['positive_events'] for r in rows), negative_events=sum(r['negative_events'] for r in rows))
    report = dict(provenance=dict(results_sha256=hashlib.sha256(args.results.read_bytes()).hexdigest(), cache_sha256=hashlib.sha256(args.cache.read_bytes()).hexdigest(),
        source_files=len(hashes), source_differences=differences, invalid_package_hashes=invalid, excluded_fixture_npz=len(excluded), replay_error=replay_error),
        counts=counts, variants=variants, modes=mode_reports, comparisons=comparisons, original_stats=original_stats,
        full_q_decomposition_mean={k: float(np.mean([r[k] for r in rows])) for k in ('full_q_gain', 'full_q_unary_gain', 'full_q_collision_gain', 'full_q_interaction_gain')})
    (args.output / 'audit.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    (args.output / 'per_state.json').write_text(json.dumps(rows, indent=2, allow_nan=False), encoding='utf-8')
    with (args.output / 'per_state.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({k: report[k] for k in ('provenance', 'counts', 'variants', 'full_q_decomposition_mean')}, indent=2))


if __name__ == '__main__':
    main()
