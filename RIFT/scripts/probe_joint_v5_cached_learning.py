"""Exploratory local overfit probes on uploaded cached inputs (not CARLA RL).

Starts a fresh real scorer with zero residual; no trained checkpoint required.
Checks the initial scores against the cache before any optimization. Results
describe this local Torch runtime, not a reproduction of server checkpoint weights.
"""
import argparse
import importlib.util
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache_zip", type=Path, required=True)
    p.add_argument("--output_json", type=Path, required=True)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    spec = importlib.util.spec_from_file_location("v5_probe_offline", Path(__file__).with_name("joint_rift_offline.py"))
    offline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = offline
    spec.loader.exec_module(offline)
    torch.set_num_threads(1)
    states = []
    with zipfile.ZipFile(args.cache_zip) as archive:
        for name in sorted(n for n in archive.namelist() if n.endswith(".npz")):
            meta = json.loads(archive.read(name[:-4] + ".json"))
            with np.load(io.BytesIO(archive.read(name)), allow_pickle=False) as data:
                d = {k: data[k] for k in data.files}
            d["valid_mask"] = d["valid_pair_mask"]
            states.append({"path": name, "metadata": meta, "data": d,
                           "objective": offline.objective_fn(d, tuple(meta["independent_indices"]), meta["cooperation_config"])})
    opportunities = [s for s in states if s["objective"]["metrics"]["coop_opportunity"]]
    cases = [("single_opportunity_lambda1", opportunities[:1], "distill", {}),
             ("single_opportunity_lambda2", opportunities[:1], "distill", {"lambda_delta": 2.}),
             ("six_opportunities_lambda1", opportunities, "distill", {}),
             ("six_opportunities_lambda2", opportunities, "distill", {"lambda_delta": 2.}),
             ("six_opportunities_fixed", opportunities, "distill", {"base_temperature": 3., "lambda_delta": 2., "residual_scale_mode": "fixed", "fixed_residual_scale": 10.}),
             ("six_opportunities_rift_lambda2", opportunities, "rift", {"lambda_delta": 2.}),
             ("six_opportunities_rift_aux_lambda2", opportunities, "rift_aux", {"lambda_delta": 2.})]
    report = {"scope": __doc__, "torch": torch.__version__, "steps": args.steps, "seed": args.seed, "cases": {}}
    for label, subset, loss_name, overrides in cases:
        torch.manual_seed(args.seed)
        scorer = offline.Scorer(**states[0]["metadata"]["scorer_config"])
        error = offline.verify(scorer, subset, 5e-4)
        cfg = dict(states[0]["metadata"]["scorer_config"], **overrides)
        calibrated = offline.Scorer(**cfg)
        calibrated.load_state_dict(scorer.state_dict())
        scorer = calibrated
        optimizer = torch.optim.AdamW(scorer.parameters(), lr=.001, weight_decay=1e-5)
        references = [offline.forward(scorer, s).joint_logits.detach() for s in subset]
        history = []
        for step in range(args.steps + 1):
            if step % 50 == 0 or step == args.steps:
                m = offline.metrics(scorer, subset)
                history.append({"step": step, **m})
            if step == args.steps:
                break
            optimizer.zero_grad()
            for state, reference in zip(subset, references):
                out = offline.forward(scorer, state)
                target = torch.as_tensor(state["objective"]["target_score"], dtype=torch.float32)[None]
                advantage = torch.as_tensor(state["objective"]["advantage"], dtype=torch.float32)[None]
                ce = offline.losses.soft_target_loss(out.joint_logits, target, out.valid_pair_mask, .25)
                if loss_name == "distill":
                    loss = ce
                else:
                    loss, _ = offline.losses.joint_dual_clip_loss(out.joint_logits, reference, advantage, out.valid_pair_mask)
                    if loss_name == "rift_aux":
                        loss = loss + .2 * ce
                (loss / len(subset)).backward()
            torch.nn.utils.clip_grad_norm_(scorer.parameters(), 5.)
            optimizer.step()
        per_state = []
        with torch.no_grad():
            for s in subset:
                out = offline.forward(scorer, s)
                chosen = np.unravel_index(int(out.joint_logits.argmax()), out.joint_logits.shape[1:])
                baseline = tuple(s["metadata"]["independent_indices"])
                per_state.append({"file": s["path"], "selected": [int(v) for v in chosen],
                                  "feasible": bool(s["objective"]["feasible_mask"][chosen]),
                                  "h_gain": float(s["objective"]["interaction"][chosen] - s["objective"]["interaction"][baseline]),
                                  "saturation_fraction": float((out.raw_delta.abs() > 3).float().mean()),
                                  "raw_delta_min": float(out.raw_delta.min()), "raw_delta_max": float(out.raw_delta.max())})
        report["cases"][label] = {"initial_verification_error": error, "overrides": overrides,
                                  "history": history, "per_state": per_state}
        print(label, history[-1], flush=True)
    with args.output_json.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
