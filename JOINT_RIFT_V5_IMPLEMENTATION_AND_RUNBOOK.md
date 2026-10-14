# Joint-RIFT V5：固定 nominal 协同排序的实现、运行与分析

日期：2026-10-13。本版按用户明确的范围实现：固定 Ego nominal 下的双 CBV 联合候选决策，不要求“一辆车改变 Ego 响应后另一辆车再利用响应”。

## 1. 目前支持到哪一步

| 环节 | 修改前 | 本版 |
|---|---|---|
| 定义与统计 | Full-Q、union/CG、NFR、oracle 等 | 增加质量预算下的协同目标、机会率、可行率、交互 regret；支持可切换的前向 braid 标签 |
| 完整采集 | 旧 NPZ 缺 latent/raw head 输入 | schema-v2 保存现有 Stage1 评分头全部输入、质量标签、事件 trace/方向和 checkpoint/scorer/reward 元数据 |
| 固定数据拟合 | 无独立入口 | `joint_rift_offline.py stats/verify/fit`，不依赖 CARLA/Lightning |
| RIFT 式训练消融 | Full-Q + uniform pair dual clip | 保留旧目标；增加合作目标、old-policy weighting、可选 soft ranking 辅助项、路线分组验证 |
| 幅度/先验消融 | residual 绑定 std(base) | 温度、std 下限、固定 residual scale、lambda 都可单独设置并写入 checkpoint |
| 闭环测试 | policy/independent/Full-Q oracle_gated | 增加明确标注的 `cooperation_oracle`，以及执行动作的协同诊断和路线聚类统计 |

原配置默认仍为 legacy Q、uniform loss、无辅助项、原标定及空间时间 crossing。新行为通过新生成的配置启用。

“完整缓存”专指**冻结 Pluto/latent 的现有 Stage1 head 输入**。没有保存原始 PlutoFeature，因此不能拿它验证 generator 重前向或做 Stage2 pi-head 训练；现有在线训练路径仍通过原始 observation 重前向。没有加入 learned Ego response model，也没有加入 nominal encoder；后者应由固定数据拟合结果决定。

## 2. 训练目标如何突出协同

### 2.1 unary 改为门槛，不继续当作无限增长的排名收益

设 independent pair 为 `(b1,b2)`，质量预算为 `budget`，构造可行集合：

```text
F = valid
    AND predicted CBV-CBV collision-free
    AND each candidate collision-free against nominal Ego/background
    AND each candidate offroad-free
    AND U1(i) >= U1(b1) - budget
    AND U2(j) >= U2(b2) - budget
```

使用**每辆车各自**的 unary 门槛，避免一辆车高分抵消另一辆车低分。默认budget=0；预算只是原RIFT虚拟回报单位，不是米或秒。可选 `require_comfortable` 要求40帧内无 `|acc|>4` 或 `|angular_acc|>4`，与当前DenseReward舒适度判据相对应；它不是完整的交通舒适度认证。

在F内按 `H=P12+coalition_weight*CG` 排序。可选 `require_both` 进一步要求CG>0.1，即当前固定nominal定义的双边际都为正。

这里的双边际是coverage意义，不是因果反事实。不要求两车都抢行、不把正braid符号简单当作“好协同”。

### 2.2 数值目标

```text
score(i,j) = 1 + (H(i,j)-min_F H) / max(max_F H-min_F H, interaction_scale_floor), pair in F
           = -1, valid pair outside F
advantage  = state-local standardize(score)
```

可行pair分数在[1,2]，不合格pair为−1。unary再高也不会压过F内的交互排序。标定下限避免把极小数值差放大成完整动态范围。它仍是受约束的代理目标，必须检验事件语义和真实执行质量。

若F为空，显式记录 `coop_has_feasible=0`，不虚构oracle、不自动放宽门槛；该state advantage为0，辅助loss跳过它。`cooperation_oracle`执行模式在F为空时回退independent并记录未干预。若这种state很多，应先修采样/质量预算/候选覆盖，而不是继续训练。

可行性是**监督目标**，当前learned policy执行时仍直接选head argmax，没有偷偷用evaluator硬过滤，所以必须报告 `selected/executed_feasible`，不能宣称部署时有硬安全保证。

## 3. BeTop-inspired 标签如何逐步验证

`reward.interaction_semantics` 有两个选择：

- `spatial_temporal`：保留旧的XY相交及arrival-gap权重，方便同数据比较目标与loss。
- `forward_braid`：在固定Ego初始坐标下检查同步的横向位置—时间顺序交叉，保存交叉时的纵向相对顺序符号；增加20m局部性门槛和距离权重，避免远处无关车辆也被视为有效事件。

后者是受BeTop启发的显式标签，不是复制整个BeTopNet，也不证明真实让行响应。新实现的符号表示**几何前后顺序**，没有加入硬编码assert/yield角色。

先在同一场景分布分别采集两种语义，检查：无关车辆、平行不交互、近距离前向顺序交叉、相反顺序、安全错峰、预测碰撞。已有单元测试覆盖顺序符号和局部性，但真实场景仍要人工查看候选图/视频及trace。

第一次目标/loss消融先固定同一种标签；切换braid应另开实验，不能把目标变化与标签变化混为一个原因。

## 4. 本版文件

- `RIFT/rift/cbv/planning/fine_tuner/rlft/joint_rift/cooperation_objective.py`：质量门槛、目标、机会判定。
- `nominal_braid.py`：可切换的同步前向braid事件。
- `joint_losses.py`：从原trainer移出的同一份dual-clip实现，增加weighting和soft ranking辅助项；旧trainer导入接口保留。
- `joint_rift_audited.py`、evaluator、datamodule、trainer和training_builder：接入采集、advantage、分组及checkpoint元数据。
- `RIFT/scripts/prepare_joint_rift_v5.py`：生成独立capture/train/eval YAML和manifest，不删除模型。
- `RIFT/scripts/joint_rift_offline.py`：统计、重算验证、固定数据拟合与head导出。
- `RIFT/scripts/summarize_joint_cooperation.py`：执行动作的协同指标，按路线等权统计。
- `RIFT/scripts/compare_joint_v5_outcomes.py`：匹配eval seed/route/repetition，路线聚类bootstrap。
- `RIFT/tools/test/test_joint_v5_pipeline.py`：真实源码方法级与数值测试。

## 5. 服务器环境

下列是服务器命令；本次没有连接该服务器或启动CARLA。先同步本版源文件。全部run使用新目录。

**2026-10-13 运行方式修正：沿用旧实验审计文档 §14.1 的 `riftuser` 身份。不要在 root 的 `(base)` shell 直接启动 `scripts/run.py`。** runner 会自行启动 CARLA；root 启动会出现 `Refusing to run with the root privileges.`。从 root shell 先单独执行下面这一行，进入非 root shell：

```bash
runuser -u riftuser -- bash --noprofile --norc
```

**以下本节以及第6–12节的命令全部在这个 `riftuser` shell 内执行**，变量在该 shell 中重新设置；不要依赖 root shell 中的 export。如果已经以 `riftuser` 登录，直接从下一块开始。可用 `id -un` 核对，必须输出 `riftuser`。这是旧文档 `runuser -u riftuser -- bash -lc '…'` 的持续交互形式，便于各步骤共用变量。

```bash
id -un
test "$(id -un)" = riftuser || { echo '请先进入 riftuser shell'; exit 1; }
set -o pipefail
export PROJECT_ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop_2
export RIFT_ROOT="$PROJECT_ROOT/RIFT"
export PY=/zeron-vepfs/tjqc/njl/envs/rift/bin/python
export CARLA_ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop/carla
export SCENARIO_ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop/scenario-dreamer
export PYTHONPATH="$RIFT_ROOT:$SCENARIO_ROOT:$CARLA_ROOT/PythonAPI/carla:$(find "$CARLA_ROOT/PythonAPI/carla/dist" -maxdepth 1 -name 'carla-*.egg' -print -quit)"
export WANDB_MODE=offline PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export BASE_CKPT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop/RIFT/rift/cbv/planning/model_ckpt/rift_pluto/pdm_lite-rule-seed0/carla_episode=427-epoch=12-val_loss=0.183.ckpt
test -f "$BASE_CKPT"
test -w "$PROJECT_ROOT/outputs" || { echo 'outputs 对 riftuser 不可写，请由目录 owner 修复'; exit 1; }
cd "$RIFT_ROOT"
test -r data/map_data/Town03_HD_map.npz
test -r data/speed_limits/Town03_speed_limits.npy
"$PY" -c 'import torch, carla, agents; print(torch.__version__, torch.cuda.is_available())'
"$PY" scripts/validate_joint_rift_core.py
"$PY" tools/test/test_joint_v5_pipeline.py
```

`test_joint_v5_pipeline.py` 的历史兼容测试需要Git HEAD可用；若服务器是无Git源码目录，测试会跳过该一项并说明。其他测试不依赖Git。map/speed数据仍需按原项目环境准备。

本次粘贴日志中的两个 `TypeError: 'type' object is not subscriptable` 是测试抽取源码时未保留延迟注解的 Python 3.8 兼容问题；已在测试文件加入 `from __future__ import annotations`。同步修正后的测试再运行，不能将旧版 `FAILED (errors=2)` 当作通过。无 Git 目录的配置生成仍可工作，manifest 记录 `git_head: unavailable`；修正版不再打印该命令的 fatal 提示。Gym/skimage/Lightning 的上述弃用提示不等于本次启动失败，不需要为此先升级依赖。

若 map/speed 检查失败，按 `JOINT_RIFT_EXPERIMENT_AUDIT_2026-10-11.md` §14.3 由目录 owner 建立数据映射。源码同步也由源码目录 owner 执行，避免给 `riftuser` 假定源码写权限。

## 6. 第一步：宽采集，先不按机会筛掉负样本

仓库已有 `drivetransformer_bench2drive_dev10.xml`，就是此前开发过的10条路线，仅用于开发诊断。

```bash
export RUN="$PROJECT_ROOT/outputs/v5_capture_spatial_seed0_r1"
"$PY" scripts/prepare_joint_rift_v5.py \
  --base_ckpt "$BASE_CKPT" --run_dir "$RUN" --variant coop \
  --interaction_semantics spatial_temporal \
  --max_states 1000 --stride_steps 10 --states_per_episode 30

mkdir -p "$RUN/capture_eval"
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/run.py \
  --mode eval --ego_cfg pdm_lite.yaml --cbv_cfg "$RUN/capture.yaml" \
  --routes rift/scenario/route/drivetransformer_bench2drive_dev10.xml \
  --seed 0 --repetitions 2 --base_port 35100 --base_tm_port 55100 \
  --output_dir "$RUN/capture_eval" \
  2>&1 | tee "$RUN/capture_eval/console.log"
```

你这次日志已经成功生成了原 `v5_capture_spatial_seed0` 的 YAML，随后 CARLA 因 root 身份退出；上面用新目录 `_r1` 保留这次记录。如果 `_r1` 已存在，再换新后缀。先做单路 smoke 时，在上述 `run.py` 命令追加 `--routes_subset 28198` 并将 `--repetitions` 设为1；只做 smoke 的 cache 不能用来宣称 route 泛化，正式多路线采集另开新 run。

capture执行independent，冻结同一个checkpoint。每10步采一次，每episode最多30个state，保留正负样本；恢复了旧forensic配置的rep/时间/压力filters，但capture_all不使用这些机会过滤。

输出：`$RUN/cache/state_*.npz` 和对应JSON。保存latent/raw/common轨迹、原始unary logits、old joint logits、Q分量、quality flags、nominal、事件trace和可选braid符号。

同一cache禁止覆盖。训练期间也支持同样的cache导出，但训练更新后checkpoint不同；固定数据拟合要求从一个冻结快照单独采集，工具会拒绝混合快照/配置。

## 7. 第二步：统计与采集分数重算验证

```bash
"$PY" scripts/joint_rift_offline.py stats \
  --cache_dir "$RUN/cache" --output_dir "$RUN/stats"
"$PY" scripts/joint_rift_offline.py verify \
  --cache_dir "$RUN/cache" --backbone_ckpt "$BASE_CKPT" \
  --output_dir "$RUN/verify"
"$PY" scripts/joint_rift_offline.py stats \
  --cache_dir "$RUN/cache" --require_both \
  --output_dir "$RUN/stats_strict_both"
```

重点看：

| 字段 | 判断 |
|---|---|
| routes / by_route.states | 是否来自多个独立路线，而不是一个长片段 |
| no_feasible_states | 质量门槛下是否有可选pair |
| opportunity_states / by_route.opportunity_rate | 是否存在比independent交互更好的合格pair |
| verification_max_error | 同head输入重算是否复现采集分数，默认容差5e-4 |
| stats_strict_both | 强双车互补机会是否足够；若几乎为0，先别做strict-both训练 |

旧36状态NPZ会被明确拒绝，因为没有latent/raw输入，不能原样拟合当前head。旧包的分解继续用 `audit_joint_rift_evidence.py`。

验证失败先查checkpoint SHA、候选身份、坐标和标定，不通过就不要训练。此验证还没有覆盖PlutoFeature重前向的一致性；服务器在线第一批训练的ratio应接近1，并应查看候选数、来源checkpoint和frame context。

## 8. 第三步：固定小数据过拟合与路线验证

先验证可学性，明确标记overfit：

```bash
"$PY" scripts/joint_rift_offline.py fit \
  --cache_dir "$RUN/cache" --backbone_ckpt "$BASE_CKPT" \
  --output_dir "$RUN/overfit_distill" --max_states 32 --overfit \
  --objective cooperation --loss distill --steps 500 --lr 0.001
```

再在同一完整cache上进行按route划分的对照：

```bash
"$PY" scripts/joint_rift_offline.py fit \
  --cache_dir "$RUN/cache" --backbone_ckpt "$BASE_CKPT" \
  --output_dir "$RUN/fit_legacy_rift" \
  --objective legacy --loss rift --steps 1000

"$PY" scripts/joint_rift_offline.py fit \
  --cache_dir "$RUN/cache" --backbone_ckpt "$BASE_CKPT" \
  --output_dir "$RUN/fit_coop_rift" \
  --objective cooperation --loss rift --steps 1000

"$PY" scripts/joint_rift_offline.py fit \
  --cache_dir "$RUN/cache" --backbone_ckpt "$BASE_CKPT" \
  --output_dir "$RUN/fit_coop_aux" \
  --objective cooperation --loss rift_aux --steps 1000

"$PY" scripts/joint_rift_offline.py fit \
  --cache_dir "$RUN/cache" --backbone_ckpt "$BASE_CKPT" \
  --output_dir "$RUN/fit_coop_oldpolicy" \
  --objective cooperation --loss old_policy --steps 1000
```

所有fit输出`summary.json`和`joint_head.ckpt`，保留原checkpoint中的backbone tensors，只替换joint_head。工具不加载/训练Pluto。默认80/20 route split；只有显式overfit才训练验证重叠。

offline reference logits在初始标定后的head上固定重算，便于比较同一数据上的优化行为；这不是on-policy CARLA RL，不能与在线PPO概念混用。

核心指标：`initial/final_train`、`initial/final_validation`中的regret、feasible、h_gain、opportunity_regret、opportunity_improvement_rate。不要跨不同objective比较loss大小。若没有机会，机会指标为null，而不是“成功率0”的混淆表述。

### 幅度诊断

先只改lambda：

```bash
"$PY" scripts/joint_rift_offline.py fit \
  --cache_dir "$RUN/cache" --backbone_ckpt "$BASE_CKPT" \
  --output_dir "$RUN/fit_coop_aux_lambda2" \
  --objective cooperation --loss rift_aux --lambda_delta 2 --steps 1000
```

再单独检查解耦先验和残差的配置：

```bash
"$PY" scripts/joint_rift_offline.py fit \
  --cache_dir "$RUN/cache" --backbone_ckpt "$BASE_CKPT" \
  --output_dir "$RUN/fit_coop_fixedscale" \
  --objective cooperation --loss rift_aux \
  --base_temperature 3 --residual_scale_mode fixed \
  --fixed_residual_scale 10 --lambda_delta 2 --steps 1000
```

固定尺度10、温度3、lambda2只是诊断起点，不是已验证最优参数。要归因，应进一步一次只改变一个因素。

关键数学事实：旧模式`Δ=lambda*std(B/T)*tanh(raw)`中，如果scale floor未生效，温度T会同时缩小base gap和residual上限；同一head参数的argmax保持不变。**温度单独不能解除这一幅度限制。** 新fixed模式将两者解耦；scale floor也修复均匀prior时Δ恒为0的问题。

若小数据仍拟合不了，先测幅度界及标定，再检查特征是否足以区分标签。若小数据能拟合、分组验证差，优先检查场景覆盖、过拟合和缺失nominal intent；本次没有提前增大网络。

## 9. 第四步：固定nominal的前向braid消融

另开数据集，只改变标签语义：

```bash
export BRAID_RUN="$PROJECT_ROOT/outputs/v5_capture_braid_seed0"
"$PY" scripts/prepare_joint_rift_v5.py \
  --base_ckpt "$BASE_CKPT" --run_dir "$BRAID_RUN" --variant coop \
  --interaction_semantics forward_braid

CUDA_VISIBLE_DEVICES=0 "$PY" scripts/run.py \
  --mode eval --ego_cfg pdm_lite.yaml --cbv_cfg "$BRAID_RUN/capture.yaml" \
  --routes rift/scenario/route/drivetransformer_bench2drive_dev10.xml \
  --seed 0 --repetitions 2 --base_port 35200 --base_tm_port 55200 \
  --output_dir "$BRAID_RUN/capture_eval"

"$PY" scripts/joint_rift_offline.py stats \
  --cache_dir "$BRAID_RUN/cache" --output_dir "$BRAID_RUN/stats"
```

对照原数据集的事件率、正负顺序、机会率和质量可行率，查看至少若干代表案例后，才确定正式目标采用哪种语义。braid和spatial数据不能混合成一次固定标签拟合。

## 10. 第五步：在线RIFT式训练

离线关卡通过后生成单独配置，例如不带辅助项的合作目标：

```bash
export TRAIN_RUN="$PROJECT_ROOT/outputs/v5_coop_online_trainseed0"
"$PY" scripts/prepare_joint_rift_v5.py \
  --base_ckpt "$BASE_CKPT" --run_dir "$TRAIN_RUN" --variant coop \
  --interaction_semantics spatial_temporal --opportunity_weight 1

CUDA_VISIBLE_DEVICES=0 "$PY" scripts/run.py \
  --mode train_cbv --ego_cfg pdm_lite.yaml --cbv_cfg "$TRAIN_RUN/train.yaml" \
  --routes rift/scenario/route/routes_training.xml \
  --seed 0 --repetitions 2 --base_port 35300 --base_tm_port 55300 \
  --output_dir "$TRAIN_RUN/train_output"
```

`routes_training.xml`只作为当前仓库可用的开发训练入口；正式实验应恢复/重新划分train XML并确认与最终测试分离，包括原RIFT prior的来源。

消融分别另建run_dir，改`--variant legacy / coop / coop_aux / coop_calibrated`，或改`--pair_weighting old_policy`。其中calibrated是温度/固定尺度/lambda的组合诊断，不能用它单独归因某一因素。生成器各variant共用geometry_horizon=40和scale floor=1；legacy指旧Q目标，不是历史全配置逐字节一致。

先保持opportunity_weight=1，再单独试4。新的机会条件是F中最大H比independent提高至少0.1，不使用Full-Q NFR或全矩阵碰撞率<=5%来代替机会判定。它是state加权，不是假称已经实现了复杂的分层replay sampler。

训练日志增加target_regret、selected_feasible和辅助loss；rollout audit增加cooperation机会/可行率。新配置按route分组验证；若一个4096-state buffer只有一条route会明确失败，可扩大采集多样性或显式设`training.split_by: state`仅做调试，不能据此报告泛化。

本版保留原Lightning checkpoint callback按val_loss保存的行为。离线head按开发集目标regret挑选；在线checkpoint仍应在独立dev routes比较regret/可行性和闭环结果，不要直接跨不同buffer比较val_loss认定最佳。

## 11. 第六步：闭环policy、independent和合作oracle

先用离线head进行闭环开发验证：

```bash
export EVAL_RUN="$PROJECT_ROOT/outputs/v5_coop_head_eval"
"$PY" scripts/prepare_joint_rift_v5.py \
  --base_ckpt "$BASE_CKPT" --fit_ckpt "$RUN/fit_coop_aux/joint_head.ckpt" \
  --run_dir "$EVAL_RUN" --variant coop_aux

for SEED in 0 1 2; do
  PORT=$((36000 + SEED * 100))
  TM=$((56000 + SEED * 100))
  CUDA_VISIBLE_DEVICES=0 "$PY" scripts/run.py \
    --mode eval --ego_cfg pdm_lite.yaml --cbv_cfg "$EVAL_RUN/eval.yaml" \
    --routes rift/scenario/route/drivetransformer_bench2drive_dev10.xml \
    --seed "$SEED" --repetitions 2 --base_port "$PORT" --base_tm_port "$TM" \
    --joint_eval_action_mode policy --output_dir "$EVAL_RUN/policy_seed$SEED"
  CUDA_VISIBLE_DEVICES=0 "$PY" scripts/run.py \
    --mode eval --ego_cfg pdm_lite.yaml --cbv_cfg "$EVAL_RUN/eval.yaml" \
    --routes rift/scenario/route/drivetransformer_bench2drive_dev10.xml \
    --seed "$SEED" --repetitions 2 --base_port "$((PORT+10))" --base_tm_port "$((TM+10))" \
    --joint_eval_action_mode independent --output_dir "$EVAL_RUN/independent_seed$SEED"
done
```

不要加`--no_resume`；新配置使用explicit checkpoint，原runner的该选项会跳过加载。

生成器从fit checkpoint恢复标定、目标与reward配置，避免部署时悄悄用错温度/尺度/标签。在线V5 Lightning checkpoint则用`--eval_ckpt /absolute/selected.ckpt`代替fit_ckpt，读取新保存的元数据。

如果要先检验可行候选上界，另跑同一配置并使用`--joint_eval_action_mode cooperation_oracle`。它执行当前质量门槛内H最大的pair，是evaluator-assisted上界，不能混入learned主结果。原`oracle_gated`仍是旧Full-Q oracle。

A/A复跑应使用同一个配置和同一checkpoint，只改变output目录和端口，重复相同seed。先量化重跑差异，再解释小幅策略效果。

## 12. 闭环统计命令

```bash
"$PY" scripts/summarize_joint_cooperation.py \
  --inputs "$EVAL_RUN"/policy_seed*/eval/*/joint_rift_eval_metrics.jsonl \
  --output_json "$EVAL_RUN/policy_cooperation_summary.json"

"$PY" scripts/summarize_joint_cooperation.py \
  --inputs "$EVAL_RUN"/independent_seed*/eval/*/joint_rift_eval_metrics.jsonl \
  --output_json "$EVAL_RUN/independent_cooperation_summary.json"

"$PY" scripts/compare_joint_v5_outcomes.py \
  --policy "$EVAL_RUN"/policy_seed*/eval/*/simulation_results.json \
  --independent "$EVAL_RUN"/independent_seed*/eval/*/simulation_results.json \
  --output_json "$EVAL_RUN/paired_outcomes.json"
```

前两项报告**各自访问状态**上的执行动作指标，不能把两个不同rollout中的同step强行当成同一世界状态。第三项匹配seed/route/repetition，按路线cluster bootstrap并保留该路线所有seed/rep；区间条件于当前固定训练模型，不代表training-seed泛化。

应联合查看：coop_executed_gain、coop_executed_feasible、机会利用、CBV碰撞/越界/不舒适比例、Ego driving score与失败状态。Ego score越高不是“CBV协同越好”的同义词；本工具只写差值，不机械标胜负。

## 13. 本次实际验证与尚未执行

实际完成：

- V5方法级/数值测试：14项通过，包括旧loss值/梯度回归、门槛与排名分离、空可行集、符号/局部性、缓存真实导出重算、真实head小样本拟合、在线辅助loss与padding、route分组及匹配统计。
- 原scorer/geometry/topology/pair测试14项、此前audit测试8项通过。
- 合成输入的真实head/真实导出方法，score重算误差0，目标regret从1降到0；CLI stats/verify/fit和配置生成入口已运行。RIFT、cooperation-RIFT、RIFT+aux三个合成拟合均可运行。
- Python语法编译、pyflakes、diff检查通过。

本机没有CARLA/Lightning，因此**没有执行新版本的真实缓存采集、完整Lightning训练或闭环实验**。上面的服务器命令是已接线的运行入口，仍需依次做单route smoke和真实数据验证。合成head拟合成功不等于研究方法已work。

当前首先运行第6、7节。它们能决定是否有足够质量合格的协同机会，以及缓存是否可信；通过后再运行第8节，避免一开始就投入全量在线训练。
