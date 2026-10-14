"""Create V6 teacher/capture/student configs in a fresh directory."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base_ckpt", type=Path, required=True)
    p.add_argument("--student_ckpt", type=Path)
    p.add_argument("--run_dir", type=Path, required=True)
    p.add_argument("--interaction_semantics", choices=("spatial_temporal", "forward_braid"), default="forward_braid")
    p.add_argument("--nominal_conflict_policy", choices=("reject", "label_only"), default="reject")
    p.add_argument("--collision_geometry", choices=("exact_sat", "legacy_aabb"), default="exact_sat")
    p.add_argument("--pair_selection", choices=("nearest", "shared_conflict"), default="shared_conflict")
    p.add_argument("--max_states", type=int, default=4000)
    p.add_argument("--stride_steps", type=int, default=5)
    p.add_argument("--states_per_episode", type=int, default=100)
    args = p.parse_args()
    if args.run_dir.exists() or not args.base_ckpt.is_file():
        raise ValueError("fresh run directory and existing trained RIFT backbone required")
    base_hash = hashlib.sha256(args.base_ckpt.read_bytes()).hexdigest()
    cfg = yaml.safe_load((root / "rift/cbv/planning/config/joint_rift_pluto_stage1.yaml").read_text(encoding="utf-8"))
    cfg.update(policy_name="joint_rift_audited", base_rift_ckpt_path=str(args.base_ckpt.resolve()),
               joint_ckpt_path=str(args.base_ckpt.resolve()), model_path=str((args.run_dir / "models").resolve()),
               eval_joint_diagnostics=True, oracle_eval_mode="diagnose", joint_eval_action_mode="policy")
    cfg["joint"].update(stage=1, geometry_horizon=40, residual_scale_floor=1.)
    cfg["cooperation"] = dict(enabled=True, diagnose=True, unary_budget=0., require_unary_collision_free=True,
        require_offroad_free=True, require_comfortable=False, min_gain=.1, coalition_weight=1.)
    cfg["decision_teacher"] = dict(enabled=True, min_gain=.05, min_joint_gain=.01,
        min_member_pressure=.01, min_coalition=.01, pressure_weight=1., coalition_weight=1.,
        nominal_conflict_policy=args.nominal_conflict_policy)
    cfg["reward"]["interaction_semantics"] = args.interaction_semantics
    cfg["reward"]["collision_geometry"] = args.collision_geometry
    cfg["high_value_reweight"]["enabled"] = False
    cfg["candidate_dump"] = {"enabled": False}
    checkpoint = None
    if args.student_ckpt:
        import torch
        checkpoint = torch.load(args.student_ckpt, map_location="cpu", weights_only=False)
        if not checkpoint.get("v6_distillation") or checkpoint.get("backbone_sha256") != base_hash:
            raise ValueError("requires a V6 student export belonging to this frozen backbone")
        saved = checkpoint["scorer_config"]
        cfg["joint"].update(saved)
        cfg["joint"]["traj_sample_interval"] = saved["trajectory_stride"]
        cfg["joint_horizon"] = saved["trajectory_frames"]
        cfg["joint_ckpt_path"] = str(args.student_ckpt.resolve())
        cfg["reward"] = checkpoint["reward_config"]
        cfg["cooperation"] = checkpoint["cooperation_config"]
        cfg["decision_teacher"] = checkpoint["decision_teacher_config"]
    configs = {"eval": cfg, "capture": copy.deepcopy(cfg), "recollect": copy.deepcopy(cfg)}
    # Initial capture always uses the original prior, independent of a student.
    configs["capture"]["joint"] = copy.deepcopy(cfg["joint"])
    configs["capture"]["joint"]["head_type"] = "residual"
    configs["capture"]["joint_ckpt_path"] = str(args.base_ckpt.resolve())
    configs["capture"]["joint_eval_action_mode"] = "independent"
    for label in ("capture", "recollect"):
        configs[label]["candidate_dump"] = dict(enabled=True, capture_all=True, include_training_inputs=True,
            output_dir=str((args.run_dir / ("cache" if label == "capture" else "recollect_cache")).resolve()),
            max_states=args.max_states, stride_steps=args.stride_steps, max_states_per_episode=args.states_per_episode)
    args.run_dir.mkdir(parents=True)
    if checkpoint is not None and checkpoint.get("head_only"):
        source = torch.load(args.base_ckpt, map_location="cpu", weights_only=False)["state_dict"]
        merged = {k: v for k, v in source.items() if not k.startswith("joint_head.")}
        merged.update(checkpoint["state_dict"])
        checkpoint["state_dict"] = merged
        checkpoint["head_only"] = False
        merged_path = (args.run_dir / "student_merged.ckpt").resolve()
        torch.save(checkpoint, merged_path)
        for label in ("eval", "recollect"):
            configs[label]["joint_ckpt_path"] = str(merged_path)
    recognition = yaml.safe_load((root / "rift/cbv/recognition/config/rule.yaml").read_text(encoding="utf-8"))
    for mode in ("train", "eval"):
        recognition[mode]["pair_selection"] = args.pair_selection
    (args.run_dir / "recognition.yaml").write_text(yaml.safe_dump(recognition, sort_keys=False), encoding="utf-8")
    config_hashes = {}
    for label, config in configs.items():
        path = args.run_dir / (label + ".yaml")
        path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
        config_hashes[label] = hashlib.sha256(path.read_bytes()).hexdigest()
    config_hashes["recognition"] = hashlib.sha256((args.run_dir / "recognition.yaml").read_bytes()).hexdigest()
    manifest = {"base_checkpoint": str(args.base_ckpt.resolve()), "base_checkpoint_sha256": base_hash,
                "student_checkpoint": str(args.student_ckpt.resolve()) if args.student_ckpt else None,
                "config_sha256": config_hashes, "actual_reward_config": cfg["reward"],
                "decision_teacher_config": cfg["decision_teacher"],
                "source_sha256": {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest()
                    for f in (root / "rift/cbv/planning/fine_tuner/rlft/joint_rift").glob("*.py")},
                "scope": "Teacher experiments and frozen-generator distillation. Oracle modes are evaluator-assisted. No performance guarantee."}
    (args.run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(args.run_dir.resolve())


if __name__ == "__main__":
    main()
