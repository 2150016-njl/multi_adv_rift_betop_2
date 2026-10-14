"""Separate V6 action ranking from predicted quality/pressure gate failures.

Oracle-mask selections are diagnostic only, never deployable learned results.
"""
import argparse
import collections
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--fit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    path = Path(__file__).with_name('joint_rift_v6.py')
    spec = importlib.util.spec_from_file_location('probe_v6_pipeline', path)
    pipe = importlib.util.module_from_spec(spec); sys.modules[spec.name] = pipe; spec.loader.exec_module(pipe)
    torch.set_num_threads(1)
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))['decision_teacher']
    states = pipe.read_states(args.cache); pipe.prepare(states, cfg)
    fit = json.loads((args.fit / 'summary.json').read_text())
    saved = torch.load(args.fit / 'joint_head.ckpt', map_location='cpu', weights_only=False)
    model_cfg = dict(saved['scorer_config']); model_cfg.pop('head_type')
    model = pipe.head_module.NominalDirectScorer(**model_cfg)
    model.load_state_dict({k[len('joint_head.'):]: v for k, v in saved['state_dict'].items() if k.startswith('joint_head.')})
    model.eval()
    reports = {}
    for label in ('train', 'validation'):
        if fit['split']['mode'] == 'route':
            subset = [s for s in states if (str(s['metadata']['route_id']) in fit['split']['validation_routes']) == (label == 'validation')]
        else:
            subset = [s for s in states if s['teacher']['metrics']['decision_interaction_support']]
        counters = collections.defaultdict(collections.Counter)
        quality = collections.Counter()
        missed = []
        for s in subset:
            with torch.no_grad(): out = pipe.forward(model, s)
            f, score, b, t = s['teacher']['feasible_mask'], s['teacher']['score'], s['baseline'], s['teacher']
            predicted = (out.quality_logits[0].sigmoid().numpy() >= out.quality_threshold) & s['data']['valid_mask']
            logits = out.joint_logits[0].numpy()
            def best(mask):
                return tuple(np.unravel_index(np.where(mask, logits, -np.inf).argmax(), f.shape)) if mask.any() else b
            oracle_selected = best(f)
            true_guard = oracle_selected
            if f[b] and score[oracle_selected] - score[b] < .05:
                true_guard = b
            selections = dict(deployed=pipe.head_module.select_direct(out),
                raw_top1=best(s['data']['valid_mask']), pred_quality_without_pressure_guard=best(predicted),
                oracle_quality_without_pressure_guard=oracle_selected, oracle_quality_and_true_pressure_guard=true_guard)
            for name, selected in selections.items():
                c = counters[name]; c['states'] += 1
                c['feasible'] += bool(f[selected]); c['changed'] += selected != b
                c['quality_recovery'] += not f[b] and f[selected]
                c['quality_regression'] += f[b] and not f[selected]
                c['interaction_support'] += bool(t['metrics']['decision_interaction_support'])
                c['interaction_capture'] += bool(f[selected] and t['metrics']['decision_interaction_support'] and score[selected] - score[t['quality']] >= .05 - 1e-8)
            valid = s['data']['valid_mask']
            quality['tp'] += int((predicted & f).sum()); quality['fp'] += int((predicted & ~f & valid).sum())
            quality['fn'] += int((~predicted & f).sum()); quality['tn'] += int((~predicted & ~f & valid).sum())
            quality['F_nonempty_no_predicted_eligible'] += bool(f.any() and not predicted.any())
            if t['metrics']['decision_interaction_support']:
                quality['opportunity_teacher_rejected_by_quality_prediction'] += not predicted[t['interaction']]
                if selections['deployed'] != t['interaction']:
                    missed.append(dict(path=s['path'], baseline=b, teacher=t['interaction'], deployed=selections['deployed'],
                        teacher_quality_probability=float(out.quality_logits[0][t['interaction']].sigmoid()),
                        predicted_pressure_gain=float(out.interaction[0][t['interaction']] - out.interaction[0][b]),
                        true_pressure_gain=float(score[t['interaction']] - score[b])))
        reports[label] = dict(selections=dict(counters), candidate_quality_confusion=dict(quality), missed_teacher=missed)
    args.output.write_text(json.dumps(reports, indent=2, default=lambda v: v.item()), encoding='utf-8')
    print(json.dumps({label: {k: v for k, v in value.items() if k != 'missed_teacher'} for label, value in reports.items()}, indent=2, default=lambda v: v.item()))


if __name__ == '__main__':
    main()
