#!/usr/bin/env bash
# Linux server entry point. May be invoked directly from the root terminal.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$(id -u)" == "0" ]]; then
  exec runuser -u riftuser -- env V6_BASE_CKPT="${V6_BASE_CKPT:-}" V6_STUDENT_CKPT="${V6_STUDENT_CKPT:-}" \
    V6_SEEDS="${V6_SEEDS:-0}" V6_ROUTES="${V6_ROUTES:-}" V6_SUBSET="${V6_SUBSET:-}" \
    V6_SEMANTICS="${V6_SEMANTICS:-forward_braid}" V6_CONFLICT_POLICY="${V6_CONFLICT_POLICY:-reject}" \
    V6_COLLISION_GEOMETRY="${V6_COLLISION_GEOMETRY:-exact_sat}" V6_PAIR_SELECTION="${V6_PAIR_SELECTION:-shared_conflict}" \
    V6_TEACHER="${V6_TEACHER:-joint}" V6_VALIDATION_ROUTES="${V6_VALIDATION_ROUTES:-}" \
    V6_STEPS="${V6_STEPS:-1500}" V6_HEAD_KIND="${V6_HEAD_KIND:-joint}" V6_GPU="${V6_GPU:-0}" \
    bash "$SCRIPT_DIR/run_joint_v6.sh" "$@"
fi
STAGE="${1:?Usage: bash scripts/run_joint_v6.sh STAGE /absolute/fresh/run_dir}"
RUN="${2:?Specify the experiment directory}"
PROJECT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RIFT="$PROJECT/RIFT"
PY=/zeron-vepfs/tjqc/njl/envs/rift/bin/python
export CARLA_ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop/carla
SCENARIO_ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop/scenario-dreamer
CARLA_EGG="$(find "$CARLA_ROOT/PythonAPI/carla/dist" -maxdepth 1 -name "carla-*.egg" -print -quit)"
export PYTHONPATH="$RIFT:$SCENARIO_ROOT:$CARLA_ROOT/PythonAPI/carla:$CARLA_EGG"
export WANDB_MODE=offline PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
BASE="${V6_BASE_CKPT:-/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop/RIFT/rift/cbv/planning/model_ckpt/rift_pluto/pdm_lite-rule-seed0/carla_episode=427-epoch=12-val_loss=0.183.ckpt}"
ROUTES="${V6_ROUTES:-rift/scenario/route/drivetransformer_bench2drive_dev10.xml}"
cd "$RIFT"
simulate() {
  local cfg="$1" out="$2" mode="$3" seed="$4" offset="$5"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="${V6_GPU:-0}" "$PY" -u scripts/run.py --mode eval --ego_cfg pdm_lite.yaml \
    --cbv_cfg "$cfg" --cbv_recog_cfg "$RUN/recognition.yaml" --routes "$ROUTES" \
    --routes_subset "${V6_SUBSET:-}" --seed "$seed" --repetitions 2 \
    --base_port "$((38000 + seed * 100 + offset))" --base_tm_port "$((58000 + seed * 100 + offset))" \
    --joint_eval_action_mode "$mode" --output_dir "$out" 2>&1 | tee "$out/console.log"
}
case "$STAGE" in
  pilot)
    bash "$SCRIPT_DIR/run_joint_v6.sh" prepare "$RUN"
    bash "$SCRIPT_DIR/run_joint_v6.sh" continue_pilot "$RUN"
    ;;
  continue_pilot)
    test -r "$RUN/capture.yaml"
    test -r "$RUN/eval.yaml"
    test -r "$RUN/recognition.yaml"
    for PART in preflight capture stats teachers summarize package; do
      bash "$SCRIPT_DIR/run_joint_v6.sh" "$PART" "$RUN"
    done
    ;;
  preflight)
    export JOINT_RIFT_TEST_OUTPUT_DIR="$RUN/preflight_tests"
    mkdir -p "$JOINT_RIFT_TEST_OUTPUT_DIR"
    test -w "$JOINT_RIFT_TEST_OUTPUT_DIR"
    "$PY" -c 'import torch, carla; from agents.navigation.local_planner import RoadOption; print(torch.__version__, torch.cuda.is_available(), RoadOption)'
    test -r data/map_data/Town03_HD_map.npz
    test -r data/speed_limits/Town03_speed_limits.npy
    "$PY" -c 'from rift.cbv.planning.fine_tuner.rlft.joint_rift.joint_rift_audited import AuditedJointRIFTPluto; from rift.cbv.recognition.rule_based.rule_cbv import RuleCBVRecog; print("V6 production imports: PASS")'
    "$PY" -c 'import carla; from rift.cbv.recognition.rule_based.rule_cbv import RuleCBVRecog; speed = RuleCBVRecog._velocity_speed_mps; assert abs(speed(carla.Vector3D(x=3., y=4., z=12.)) - 13.) < 1e-6; assert speed(carla.Vector3D()) == 0.; assert speed(None) == 0.; print("V6 CARLA Vector3D speed contract: PASS")'
    "$PY" scripts/validate_joint_rift_core.py
    "$PY" tools/test/test_joint_v5_pipeline.py
    "$PY" tools/test/test_joint_v6.py
    ;;
  prepare)
    EXTRA=()
    if [[ -n "${V6_STUDENT_CKPT:-}" ]]; then EXTRA+=(--student_ckpt "$V6_STUDENT_CKPT"); fi
    "$PY" scripts/prepare_joint_rift_v6.py --base_ckpt "$BASE" --run_dir "$RUN" \
      --interaction_semantics "${V6_SEMANTICS:-forward_braid}" \
      --nominal_conflict_policy "${V6_CONFLICT_POLICY:-reject}" \
      --collision_geometry "${V6_COLLISION_GEOMETRY:-exact_sat}" \
      --pair_selection "${V6_PAIR_SELECTION:-shared_conflict}" "${EXTRA[@]}"
    ;;
  capture) simulate "$RUN/capture.yaml" "$RUN/capture_seed0" independent 0 0 ;;
  teachers)
    for SEED in ${V6_SEEDS:-0}; do
      OFFSET=10
      for MODE in independent quality_oracle factorized_oracle interaction_oracle joint_oracle; do
        simulate "$RUN/eval.yaml" "$RUN/teachers/${MODE}_seed${SEED}" "$MODE" "$SEED" "$OFFSET"
        OFFSET=$((OFFSET + 10))
      done
    done
    ;;
  stats)
    "$PY" scripts/joint_rift_v6.py stats --cache "$RUN/cache" --config "$RUN/eval.yaml" --verify_semantics --output_dir "$RUN/stats_$(date +%Y%m%d_%H%M%S)"
    "$PY" scripts/joint_rift_v6.py stats --cache "$RUN/cache" --config "$RUN/eval.yaml" \
      --nominal_conflict_policy label_only --output_dir "$RUN/stats_nominal_conflict_ablation_$(date +%Y%m%d_%H%M%S)"
    "$PY" scripts/joint_rift_v6.py stats --cache "$RUN/cache" --config "$RUN/eval.yaml" \
      --semantics_override spatial_temporal --output_dir "$RUN/stats_spatial_replay_$(date +%Y%m%d_%H%M%S)"
    ;;
  overfit|fit)
    EXTRA=()
    if [[ "$STAGE" == "overfit" ]]; then EXTRA+=(--overfit_opportunities); fi
    if [[ -n "${V6_VALIDATION_ROUTES:-}" ]]; then EXTRA+=(--validation_routes "$V6_VALIDATION_ROUTES"); fi
    "$PY" -u scripts/joint_rift_v6.py fit --cache "$RUN/cache" --config "$RUN/eval.yaml" --head_only \
      --teacher "${V6_TEACHER:-joint}" --head_kind "${V6_HEAD_KIND:-joint}" \
      --steps "${V6_STEPS:-1500}" \
      --output_dir "$RUN/${STAGE}_$(date +%Y%m%d_%H%M%S)" "${EXTRA[@]}"
    ;;
  eval)
    for SEED in ${V6_SEEDS:-0}; do
      simulate "$RUN/eval.yaml" "$RUN/eval_student/policy_seed${SEED}" policy "$SEED" 60
      simulate "$RUN/eval.yaml" "$RUN/eval_student/independent_seed${SEED}" independent "$SEED" 70
      simulate "$RUN/eval.yaml" "$RUN/eval_student/independent_aa_seed${SEED}" independent "$SEED" 80
      simulate "$RUN/eval.yaml" "$RUN/eval_student/verified_policy_seed${SEED}" verified_policy "$SEED" 90
    done
    ;;
  recollect) simulate "$RUN/recollect.yaml" "$RUN/recollect_seed0" policy 0 95 ;;
  summarize)
    for GROUP in teachers eval_student; do
      if [[ -d "$RUN/$GROUP" ]]; then
        "$PY" scripts/summarize_joint_v6.py --run_dir "$RUN/$GROUP" --output_json "$RUN/${GROUP}_summary_$(date +%Y%m%d_%H%M%S).json"
      fi
    done
    ;;
  package) "$PY" -u scripts/package_joint_v6.py --run_dir "$RUN" ;;
  *) echo "Unknown stage: $STAGE" >&2; exit 2 ;;
esac
