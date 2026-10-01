# Audited Joint-RIFT fix package

This package implements the fixes identified in the code audit without overwriting the original `joint_rift_pluto` policy. It adds a new policy name:

```text
joint_rift_audited
```

The original implementation therefore remains available as a control/debug reference.

## 1. Apply the patch

From the repository's `RIFT/` root (the directory containing `rift/` and `scripts/`):

```bash
python /PATH/TO/joint_rift_audit_fix/apply_joint_rift_audit.py
```

Then inspect the changes:

```bash
git status --short
git diff -- rift/carla_runner.py rift/cbv/planning/__init__.py
```

The patch is intentionally idempotent. It aborts if the audited upstream snippets are no longer present rather than silently patching the wrong code.

## 2. Static and math validation

```bash
python -m compileall -q rift scripts
python scripts/validate_joint_rift_core.py
```

Expected final line:

```text
Joint-RIFT core validation: PASS
```

This CPU-only validator checks:

- zero-initialized joint residual exactly reduces to the product/Cartesian independent policy;
- pair-swap symmetry;
- temporally aligned crossing receives much larger interaction weight than the same spatial crossing with a large arrival-time gap;
- additive reward matrices have near-zero non-factorizable residual, while XOR-like pair rewards do not.

## 3. Verify / train the single-CBV RIFT prior

Stage 1 now refuses to bootstrap from IL Pluto. It auto-discovers the newest RIFT checkpoint under:

```text
rift/cbv/planning/model_ckpt/rift_pluto/<ego-policy>-<recognition>-seed<seed>/
```

Check what exists:

```bash
find rift/cbv/planning/model_ckpt/rift_pluto -name '*.ckpt' -print
```

If no matching RIFT checkpoint exists for PDM-Lite + rule recognition + seed 0, train it first:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run.py \
  --ego_cfg pdm_lite.yaml \
  --cbv_cfg rift_pluto.yaml \
  --mode train_cbv \
  --seed 0 \
  --repetitions 2 \
  --no_resume
```

For crash-prone long runs, the repository's resume wrapper can still be used:

```bash
bash scripts/run_multi.sh \
  -t 3 \
  -e pdm_lite.yaml \
  -c rift_pluto.yaml \
  -m train_cbv \
  -r 2 \
  -s 0 \
  -g 0
```

## 4. Joint-RIFT Stage 1: residual head only

This is a **fresh** rollout/training run in its own checkpoint directory. Pluto/RIFT generator and single-agent `pi_head` are frozen; only the joint residual scorer is trained.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run.py \
  --ego_cfg pdm_lite.yaml \
  --cbv_cfg joint_rift_pluto_stage1.yaml \
  --mode train_cbv \
  --seed 0 \
  --repetitions 2 \
  --no_resume
```

Stage-1 checkpoints are written under:

```text
rift/cbv/planning/model_ckpt/joint_rift_pluto_stage1_audited/
```

The run logs include the new feasibility diagnostics:

```text
feasibility/oracle_joint_gain
feasibility/oracle_rank_1
feasibility/oracle_rank_2
feasibility/oracle_nll_1
feasibility/oracle_nll_2
feasibility/oracle_both_necessary
feasibility/both_necessary_rate
feasibility/nonfactorizable_reward_ratio
selection/oracle_policy_gap
selection/independent_same_as_oracle_rate
selection/joint_same_as_oracle_rate
reward_scale/P12_std
reward_scale/coalition_std
reward_scale/unary_pressure_std
reward_scale/unary_return_std
```

Interpretation before accepting Stage 1 as scientifically meaningful:

- `oracle_joint_gain` should be materially above zero on a nontrivial fraction of states;
- `nonfactorizable_reward_ratio` should not collapse near zero everywhere;
- `both_necessary_rate` / `oracle_both_necessary` should be nontrivial if the claimed behavior is genuinely two-CBV cooperation;
- oracle candidate ranks/NLL should not be so extreme that the RIFT prior places essentially no mass on useful coordinated modes;
- pair collision should not dominate oracle solutions.

## 5. Joint-RIFT Stage 2: joint head + shared RIFT pi-head

Stage 2 uses a **separate checkpoint directory** and automatically initializes from the latest Stage-1 audited checkpoint. It recollects a fresh on-policy buffer; it does not reuse Stage-1 rollout states.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run.py \
  --ego_cfg pdm_lite.yaml \
  --cbv_cfg joint_rift_pluto_stage2.yaml \
  --mode train_cbv \
  --seed 0 \
  --repetitions 2 \
  --no_resume
```

Stage-2 checkpoints are written under:

```text
rift/cbv/planning/model_ckpt/joint_rift_pluto_stage2_audited/
```

Stage 2 keeps the trajectory generator frozen and only enables:

```text
joint residual head
planning_decoder.pi_head
```

This preserves candidate geometry/identity for the RIFT/PPO old-vs-new ratio.

## 6. Evaluation baselines

Use the same seed, PDM-Lite ego, rule recognizer, routes, and repetition count for all compared policies.

### IL Pluto x2

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run.py \
  --ego_cfg pdm_lite.yaml \
  --cbv_cfg pluto.yaml \
  --mode eval \
  --seed 0 \
  --pretrain_seed 0 \
  --repetitions 1
```

### Independent RIFT x2

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run.py \
  --ego_cfg pdm_lite.yaml \
  --cbv_cfg rift_pluto.yaml \
  --mode eval \
  --seed 0 \
  --pretrain_seed 0 \
  --repetitions 1
```

### Joint-RIFT Stage 1

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run.py \
  --ego_cfg pdm_lite.yaml \
  --cbv_cfg joint_rift_pluto_stage1.yaml \
  --mode eval \
  --seed 0 \
  --pretrain_seed 0 \
  --repetitions 1
```

### Joint-RIFT Stage 2

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run.py \
  --ego_cfg pdm_lite.yaml \
  --cbv_cfg joint_rift_pluto_stage2.yaml \
  --mode eval \
  --seed 0 \
  --pretrain_seed 0 \
  --repetitions 1
```

The patched runner passes the PDM nominal trajectory only to joint policies, so GRPO/RIFT/other baselines keep their original `get_action` API. Joint evaluation writes per-state diagnostics to:

```text
<run-output-dir>/joint_rift_eval_metrics.jsonl
```

Locate and summarize them with:

```bash
find log -name joint_rift_eval_metrics.jsonl -print
python scripts/summarize_joint_rift_metrics.py /PATH/TO/joint_rift_eval_metrics.jsonl
```

## 7. Recommended multi-seed experiment

At minimum run seeds 0, 1, 2 independently. For each seed, train its own RIFT prior, Stage 1, Stage 2, then evaluate all baselines with the matching `--pretrain_seed`.

Example shell skeleton:

```bash
for SEED in 0 1 2; do
  CUDA_VISIBLE_DEVICES=0 python scripts/run.py \
    --ego_cfg pdm_lite.yaml --cbv_cfg rift_pluto.yaml \
    --mode train_cbv --seed ${SEED} --repetitions 2 --no_resume

  CUDA_VISIBLE_DEVICES=0 python scripts/run.py \
    --ego_cfg pdm_lite.yaml --cbv_cfg joint_rift_pluto_stage1.yaml \
    --mode train_cbv --seed ${SEED} --repetitions 2 --no_resume

  CUDA_VISIBLE_DEVICES=0 python scripts/run.py \
    --ego_cfg pdm_lite.yaml --cbv_cfg joint_rift_pluto_stage2.yaml \
    --mode train_cbv --seed ${SEED} --repetitions 2 --no_resume

done
```

Do not run the three training stages for the same seed concurrently: Stage 1 depends on the completed RIFT checkpoint and Stage 2 depends on the completed Stage-1 checkpoint.

## 8. Key implementation changes

1. **Strict adversarial bootstrap**: Stage 1 refuses IL-Pluto-only initialization and requires a trained single-CBV RIFT checkpoint.
2. **Fresh Stage 2 collection**: separate Stage-2 model path prevents accidentally resuming the Stage-1 route/data-loader position and forces new on-policy collection.
3. **Time-aware interaction**: a shared spatial conflict point receives `exp(-|arrival gap| / tau)` weight (`tau=1.0 s` by default), preserving multi-crossing onset clustering and RIFT temporal discounting.
4. **CBV-CBV safety scale**: pair collision penalty default is raised from `1.0` to configurable `20.0`; audit this empirically using the logged reward scales.
5. **Per-environment fallback**: one vector environment without a valid two-CBV pair no longer forces every other environment to abandon joint action selection.
6. **Baseline API compatibility**: only joint policies receive `ego_nominal_trajectories` from `CarlaRunner`.
7. **Feasibility diagnostics**: oracle joint gain, both-necessary rate, oracle candidate rank/NLL, and non-factorizable reward ratio are emitted for go/no-go decisions.
8. **Eval diagnostics**: audited joint policies can compute the same diagnostics during evaluation and append JSONL records.

## 9. Explicit checkpoint override

If your downloaded/trained checkpoints are stored outside the repository's standard directory layout, edit these optional fields:

Stage 1 config:

```yaml
base_rift_ckpt_path: /absolute/path/to/rift_checkpoint.ckpt
```

Stage 2 config:

```yaml
stage1_ckpt_path: /absolute/path/to/stage1_joint_checkpoint.ckpt
```

Leaving them empty uses automatic latest-checkpoint discovery.
