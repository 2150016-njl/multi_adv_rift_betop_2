#!/usr/bin/env python
"""Apply the audited Joint-RIFT overlay to a RIFT checkout.

Run from the repository's RIFT root, i.e. the directory containing ``rift/``
and ``scripts/``:

    python /path/to/apply_joint_rift_audit.py

The script is idempotent and aborts if expected upstream snippets are missing,
so it will not silently patch an incompatible tree.
"""

from pathlib import Path
import shutil
import sys

ROOT = Path.cwd()
PKG = Path(__file__).resolve().parent
OVERLAY = PKG / "overlay"


def require(path: str) -> Path:
    target = ROOT / path
    if not target.exists():
        raise SystemExit(f"Expected RIFT repository file is missing: {target}")
    return target


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise SystemExit(
            f"Refusing to patch {path}: expected upstream snippet not found. "
            "Your checkout may differ from the audited main branch."
        )
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def copy_overlay() -> None:
    for source in OVERLAY.rglob("*"):
        if source.is_dir():
            continue
        relative = source.relative_to(OVERLAY)
        destination = ROOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        print(f"overlay: {relative}")


def patch_policy_registry() -> None:
    path = require("rift/cbv/planning/__init__.py")
    import_old = (
        "from rift.cbv.planning.fine_tuner.rlft.joint_rift.joint_rift_pluto "
        "import JointRIFTPluto\n"
    )
    import_new = import_old + (
        "from rift.cbv.planning.fine_tuner.rlft.joint_rift.joint_rift_audited "
        "import AuditedJointRIFTPluto\n"
    )
    replace_once(path, import_old, import_new)
    mapping_old = "    'joint_rift_pluto': JointRIFTPluto,\n"
    mapping_new = mapping_old + "    'joint_rift_audited': AuditedJointRIFTPluto,\n"
    replace_once(path, mapping_old, mapping_new)


def patch_runner() -> None:
    path = require("rift/carla_runner.py")
    train_old = """                CBVs_actions_dict = self.cbv_policy.get_action(\n                    CBVs_obs_list,\n                    info_list,\n                    deterministic=False,\n                    ego_nominal_trajectories=ego_actions_dict.get('ego_nominal_trajectories'),\n                )\n"""
    train_new = """                if self.cbv_policy_name in ('joint_rift_pluto', 'joint_rift_audited'):\n                    CBVs_actions_dict = self.cbv_policy.get_action(\n                        CBVs_obs_list,\n                        info_list,\n                        deterministic=False,\n                        ego_nominal_trajectories=ego_actions_dict.get('ego_nominal_trajectories'),\n                    )\n                else:\n                    CBVs_actions_dict = self.cbv_policy.get_action(\n                        CBVs_obs_list, info_list, deterministic=False\n                    )\n"""
    replace_once(path, train_old, train_new)

    eval_old = """                ego_actions_dict = self.ego_policy.get_action(ego_obs_list, info_list, deterministic=True)\n                CBVs_actions_dict = self.cbv_policy.get_action(CBVs_obs_list, info_list, deterministic=True)\n"""
    eval_new = """                ego_actions_dict = self.ego_policy.get_action(ego_obs_list, info_list, deterministic=True)\n                if self.cbv_policy_name in ('joint_rift_pluto', 'joint_rift_audited'):\n                    CBVs_actions_dict = self.cbv_policy.get_action(\n                        CBVs_obs_list,\n                        info_list,\n                        deterministic=True,\n                        ego_nominal_trajectories=ego_actions_dict.get('ego_nominal_trajectories'),\n                    )\n                else:\n                    CBVs_actions_dict = self.cbv_policy.get_action(\n                        CBVs_obs_list, info_list, deterministic=True\n                    )\n"""
    replace_once(path, eval_old, eval_new)

    buffer_old = """            buffer_class = (\n                JointRolloutBuffer\n                if self.cbv_policy_name == 'joint_rift_pluto'\n                else CBVRolloutBuffer\n            )\n"""
    buffer_new = """            buffer_class = (\n                JointRolloutBuffer\n                if self.cbv_policy_name in ('joint_rift_pluto', 'joint_rift_audited')\n                else CBVRolloutBuffer\n            )\n"""
    replace_once(path, buffer_old, buffer_new)


def main() -> None:
    require("rift/cbv/planning/__init__.py")
    require("rift/carla_runner.py")
    copy_overlay()
    patch_policy_registry()
    patch_runner()
    print("Audited Joint-RIFT patch applied successfully.")
    print("Next: python -m compileall -q rift scripts && python scripts/validate_joint_rift_core.py")


if __name__ == "__main__":
    main()
