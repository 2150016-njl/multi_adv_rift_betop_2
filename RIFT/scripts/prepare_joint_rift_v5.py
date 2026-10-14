"""Create isolated capture/train/eval configs; does not start CARLA or delete files."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess

import yaml


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_config", type=Path, default=root / "rift/cbv/planning/config/joint_rift_pluto_stage1.yaml")
    parser.add_argument("--base_ckpt", type=Path, required=True, help="Trusted trained RIFT or Joint-RIFT checkpoint")
    parser.add_argument("--fit_ckpt", type=Path, help="Optional output of joint_rift_offline.py fit")
    parser.add_argument("--eval_ckpt", type=Path, help="V5 Lightning checkpoint with scorer metadata")
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--variant", choices=("legacy", "coop", "coop_aux", "coop_calibrated"), default="coop")
    parser.add_argument("--max_states", type=int, default=1000)
    parser.add_argument("--stride_steps", type=int, default=10)
    parser.add_argument("--states_per_episode", type=int, default=30)
    parser.add_argument("--unary_budget", type=float, default=0.)
    parser.add_argument("--require_both", action="store_true", help="Require both nominal coverage marginals > epsilon; not a causal test")
    parser.add_argument("--require_comfortable", action="store_true")
    parser.add_argument("--opportunity_weight", type=float, default=1.)
    parser.add_argument("--pair_weighting", choices=("uniform", "old_policy"), default="uniform")
    parser.add_argument("--interaction_semantics", choices=("spatial_temporal", "forward_braid"), default="spatial_temporal")
    parser.add_argument("--base_temperature", type=float)
    parser.add_argument("--lambda_delta", type=float)
    parser.add_argument("--residual_scale_mode", choices=("base_std", "fixed"))
    parser.add_argument("--fixed_residual_scale", type=float)
    args = parser.parse_args()
    if args.fit_ckpt and args.eval_ckpt:
        raise ValueError("choose fit_ckpt or eval_ckpt")
    if not args.base_ckpt.is_file():
        raise FileNotFoundError(args.base_ckpt)
    run = args.run_dir.resolve()
    if run.exists():
        raise ValueError("run_dir already exists; use a new experiment directory")
    cfg = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    cfg["policy_name"] = "joint_rift_audited"
    cfg["model_path"] = str(run / "models")
    cfg["base_rift_ckpt_path"] = str(args.base_ckpt.resolve())
    cfg["joint_ckpt_path"] = str(args.base_ckpt.resolve())
    cfg["joint"]["stage"] = 1
    cfg["joint"].update(base_temperature=3. if args.variant == "coop_calibrated" else 1.,
                         residual_scale_floor=1., geometry_horizon=40,
                         lambda_delta=2. if args.variant == "coop_calibrated" else 1.)
    cfg["joint"].update(residual_scale_mode="fixed" if args.variant == "coop_calibrated" else "base_std",
                         fixed_residual_scale=10.)
    for key in ("base_temperature", "lambda_delta", "residual_scale_mode", "fixed_residual_scale"):
        if getattr(args, key) is not None:
            cfg["joint"][key] = getattr(args, key)
    cfg["cooperation"] = {"enabled": args.variant != "legacy", "diagnose": True,
                          "unary_budget": args.unary_budget, "coalition_weight": 1.,
                          "require_unary_collision_free": True, "require_offroad_free": True,
                          "min_gain": .1, "both_epsilon": .1, "interaction_scale_floor": .1,
                          "opportunity_weight": args.opportunity_weight}
    cfg["cooperation"]["require_both"] = args.require_both
    cfg["cooperation"]["require_comfortable"] = args.require_comfortable
    cfg["reward"]["interaction_semantics"] = args.interaction_semantics
    cfg["high_value_reweight"]["enabled"] = False
    cfg["training"].update(train_pi_head=False, pair_weighting=args.pair_weighting,
                            auxiliary_weight=.2 if args.variant in ("coop_aux", "coop_calibrated") else 0.,
                            teacher_temperature=.25, split_by="route")
    cfg["eval_joint_diagnostics"] = True
    cfg["oracle_eval_mode"] = "diagnose"
    cfg["joint_eval_action_mode"] = "policy"
    cfg["candidate_dump"] = {"enabled": False}
    configs = {"train": cfg, "eval": copy.deepcopy(cfg), "capture": copy.deepcopy(cfg)}
    configs["capture"]["joint_eval_action_mode"] = "independent"
    configs["capture"]["candidate_dump"] = {
        "enabled": True, "capture_all": True, "include_training_inputs": True,
        "output_dir": str(run / "cache"), "max_states": args.max_states,
        "stride_steps": args.stride_steps, "max_states_per_episode": args.states_per_episode}
    selected_checkpoint = args.fit_ckpt or args.eval_ckpt
    if selected_checkpoint:
        import torch
        fitted = torch.load(selected_checkpoint, map_location="cpu", weights_only=False)
        if args.fit_ckpt and not fitted.get("offline_diagnostic"):
            raise ValueError("fit_ckpt must be an offline pipeline export")
        if "scorer_config" not in fitted:
            raise ValueError("eval checkpoint lacks V5 scorer metadata; use its original config explicitly")
        if args.fit_ckpt and fitted["backbone_sha256"] != hashlib.sha256(args.base_ckpt.read_bytes()).hexdigest():
            raise ValueError("fitted head belongs to a different collection checkpoint")
        mapping = {"pair_dim": "latent_dim", "trajectory_stride": "traj_sample_interval"}
        for label in ("train", "eval"):
            configs[label]["joint_ckpt_path"] = str(selected_checkpoint.resolve())
            if fitted.get("reward_config"):
                configs[label]["reward"] = fitted["reward_config"]
            configs[label]["cooperation"].update(fitted.get("cooperation_config", {}))
            configs[label]["cooperation"].update(enabled=args.variant != "legacy", diagnose=True,
                                                  opportunity_weight=args.opportunity_weight)
            for key in ("representation_dim", "pair_dim", "geometry_hidden_dim", "lambda_delta", "trajectory_stride",
                        "base_temperature", "residual_scale_floor", "geometry_horizon", "residual_scale_mode", "fixed_residual_scale"):
                if key in fitted["scorer_config"]:
                    configs[label]["joint"][mapping.get(key, key)] = fitted["scorer_config"][key]
    run.mkdir(parents=True)
    hashes = {}
    for label, config in configs.items():
        path = run / (label + ".yaml")
        path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        hashes[label] = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"
    manifest = {"git_head": revision, "variant": args.variant,
                "base_checkpoint": str(args.base_ckpt.resolve()),
                "base_checkpoint_sha256": hashlib.sha256(args.base_ckpt.read_bytes()).hexdigest(),
                "selected_checkpoint": str(selected_checkpoint.resolve()) if selected_checkpoint else None,
                "selected_checkpoint_sha256": hashlib.sha256(selected_checkpoint.read_bytes()).hexdigest() if selected_checkpoint else None,
                "config_sha256": hashes, "semantics": args.interaction_semantics + "; fixed nominal coverage; no causal response or BeTopNet implementation",
                "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (root / "rift/cbv/planning/fine_tuner/rlft/joint_rift").glob("*.py")},
                "note": "Generated configs share geometry_horizon=40 and residual_scale_floor=1. legacy refers to the legacy Q target, not byte-identical historical model settings."}
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(str(run))


if __name__ == "__main__":
    main()
