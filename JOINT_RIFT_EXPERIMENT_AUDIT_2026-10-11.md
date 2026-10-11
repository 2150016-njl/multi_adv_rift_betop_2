# Joint-RIFT 实验审计、结果汇总与后续研究交接（截至 2026-10-11）

## 0. 文档用途、结论等级与阅读规则

本文是当前 Joint-RIFT 分支的独立交接文档，供新的聊天/研究会话在**不依赖本对话上下文**的情况下继续审计、复现实验、分析失败原因及决定后续研究路线。

本文明确区分三类信息：

- **文件已验证**：当前工作区的代码、JSON、route 文件或 checkpoint 文件可直接复查。
- **终端历史已验证**：本轮实验终端输出中出现过，路径和数值已记录；个别远端日志可能未完整下载到本地。
- **推断/建议**：由前两者导出的解释，不能写成已证明的事实。

截至本文日期的总体结论：

> Joint-RIFT 的候选空间中确实存在非可分解联合回报和 oracle 改善空间；Stage-1 的联合 residual 头、审计、训练与 matched evaluation 流程已能稳定运行；高价值状态重加权（HV4）也确实被执行。**但现有训练 seed=0 的四个 CARLA 评估 seed 并未给出 HV4 相对 Uniform 的稳健优势，因此不能进入 Stage 2，也不能声称联合策略或 HV4 已经优于独立选择。**

一个必须优先澄清的评估问题是：`simulation_results.json` 中的 `score_composed` 是 CARLA 路线/ego 结果，尚未在实验协议中明示“分数更大是否就是 adversarial CBV 目标更好”。现有比较脚本将更大值机械称为 `policy_wins`，但这只是数值差的命名，**不是任务语义证明**。后续必须预注册指标方向。

---

## 1. 项目目标与方法边界

### 1.1 研究问题

RIFT 将 Pluto 产生的单 CBV 候选轨迹进行虚拟评估，并学习候选排序。Joint-RIFT 的目标不是重新生成轨迹，而是：在两个可控 CBV 的候选笛卡尔积中，选择带有额外交互价值的**联合轨迹对**。

对 CBV1 候选 `i` 与 CBV2 候选 `j`，当前设计为：

```text
joint_logit(i, j) = log p_RIFT_1(i) + log p_RIFT_2(j) + delta_theta(i, j, s)
```

其中：

- 前两项是冻结的单车 RIFT 因子化先验；
- `delta_theta` 是学习到的 joint residual；
- 当 `delta_theta = 0` 时，联合分布严格退化为两个独立 RIFT 分布的乘积；
- Stage 1 冻结 RIFT/Pluto 主体，只训练 residual；Stage 2（尚未启动）才允许训练 `pi_head`，轨迹生成器始终冻结。

因此 Joint-RIFT 要成立，需要满足两个不同层面的条件：

1. **机会存在**：候选矩阵中存在比独立候选组合更好的联合 pair，而且该奖励不能写成两个 unary 奖励的简单相加。
2. **可学习性存在**：冻结 RIFT 先验在这些 pair 上仍有足够支持，且 residual head 能从采样 rollout 中学到稳定选择。

### 1.2 V1 已固定的边界

- 固定双 CBV，不动态换 coalition；
- 完整保留两个 CBV 的候选 Cartesian product，不先做 top-k 截断；
- 使用 common-frame candidate trajectory + Pluto latent 构造 pair residual；
- 使用 PDM-Lite nominal intent，而非完整 ego closed-loop response surrogate；
- 使用 arrival-time-aware spatial conflict、coalition gain、leave-one-out/both-necessary 等诊断；
- 第一阶段不引入 reachable tube、复杂反事实 ego surrogate 或人工 assert/yield 角色。

这些边界降低了 V1 变量数，但也意味着结论只针对当前双 CBV、PDM-Lite、40-step virtual horizon 和当前 reward 定义。

---

## 2. 环境、路径、权限与可复现约束

### 2.1 实际使用路径

| 用途 | 远端路径 |
|---|---|
| 当前代码根目录 | `/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop_2/RIFT` |
| CARLA（不带 `_2`） | `/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop/carla` |
| scenario-dreamer（不带 `_2`） | `/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop/scenario-dreamer` |
| 共享 RIFT 原始模型根目录（不带 `_2`） | `/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop/RIFT/rift/cbv/planning/model_ckpt` |
| 可由 `riftuser` 使用的 Python | `/zeron-vepfs/tjqc/njl/envs/rift/bin/python` |
| 当前实验输出根目录 | `/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop_2/outputs` |

### 2.2 已解决的环境问题

1. 初始 `/root/miniconda3/envs/rift/bin/python` 不能被 `riftuser` 执行，报 `Permission denied`；已 clone 到共享可读写路径。验证为 Python 3.8.20、PyTorch 2.0.1+cu117、CUDA 可用。
2. `riftuser` 的登录 PATH 中无 `python`/`python3`；必须使用绝对 `PY` 路径。
3. CARLA Python API 必须同时加入 `PYTHONPATH`：

   ```text
   $RIFT_ROOT:$SCENARIO_ROOT:$CARLA_ROOT/PythonAPI/carla:<carla egg>
   ```

   否则会出现 `ModuleNotFoundError: agents`。
4. `_2/RIFT/data/map_data` 和 `_2/RIFT/data/speed_limits` 是空目录，而可用源数据位于不带 `_2` 的原项目目录。运行时 Pluto `CarlaMap` 使用相对路径 `data/map_data/TownXX_HD_map.npz`，因此需要将实际 map/speed 文件以可读取的方式映射至当前代码目录；目录 root-owned 时由有权限的用户创建链接/复制。
5. 源代码、config 和 outputs 的 owner 不完全相同。`riftuser` 负责运行；需要写 config/源码时使用实际 owner。不要试图让 `riftuser` 覆盖 root-owned 源文件。

### 2.3 运行稳定性约束

- CARLA 使用 `-nullrhi`，日志中 “Running Carla without GPU!” 在该 headless 模式下正常；模型仍可使用 CUDA。
- `Successfully sent SIGKILL to process group ...` 是 runner 退出时清理 CARLA 子进程的正常日志，不能单独判断成功或失败；必须同时检查 traceback、route 数和结果文件。
- CARLA RPC/Traffic Manager 端口不可复用。训练/评估使用显式、不同的 `--base_port` 与 `--base_tm_port`；曾出现 `bind: Address already in use`，通过避开遗留端口解决。
- 一切新 run 使用独立 `output_dir`。`--no_resume` 会清理配置的**模型目录**，但不保证清理同名 output 下旧 audit JSON，见第 8 节的污染问题。

---

## 3. 当前代码与审计实现

### 3.1 关键代码目录

```text
RIFT/rift/cbv/planning/fine_tuner/rlft/joint_rift/
RIFT/rift/carla_runner.py
RIFT/rift/gym_carla/envs/carla_env.py
RIFT/scripts/run.py
RIFT/rift/cbv/planning/config/joint_rift_pluto_stage1.yaml
RIFT/rift/cbv/planning/config/joint_rift_pluto_stage2.yaml
```

当前注册的 policy 名为 `joint_rift_audited`；原 `joint_rift_pluto` 保留作原型/调试对照。注册位置：

```text
rift/cbv/planning/__init__.py
```

### 3.2 当前实现的核心模块

| 模块 | 职责 |
|---|---|
| `joint_score_head.py` | 候选 latent、common-frame trajectory 和 pair geometry 的 residual scorer |
| `joint_traj_evaluator.py` / `audited_joint_traj_evaluator.py` | unary/pair 虚拟 rollout、pair safety、联合 reward |
| `interaction_topology.py` / `temporal_interaction.py` | 空间冲突事件、arrival-time closeness、multiple-crossing 聚合 |
| `joint_diagnostics.py` | oracle、non-factorizable residual、both-necessary、oracle gating |
| `joint_buffer.py` / `joint_datamodule.py` | state-level joint rollout 存储、collate、state weight |
| `joint_trainer.py` | masked joint log-softmax、dual-clipped objective、state-weighted loss |
| `joint_rift_audited.py` | policy selection、checkpoint load、eval mode、HV eligibility/weight |
| `joint_rift_pluto.py` | Stage 1/2 训练、checkpoint、rollout audit 落盘 |

### 3.3 已实现并通过核心验证的机制

`python scripts/validate_joint_rift_core.py` 已在远端输出：

```text
Joint-RIFT core validation: PASS
```

验证覆盖：

- residual 零初始化时退化为 independent factorized policy；
- pair swap symmetry；
- 时间对齐 crossing 比大时间差的同一空间 crossing 权重更高；
- additive reward 的 non-factorizable residual 接近零，非可分解合成奖励不接近零；
- state-weighted joint loss 的数值/梯度基本性质。

### 3.4 最重要的已修复代码问题

| 问题 | 表现 | 修复/状态 |
|---|---|---|
| Stage 1 prior 不严格 | 原路径可能从 IL Pluto 而不是 RIFT 起步 | audited Stage 1 要求/显式 bootstrap RIFT checkpoint |
| 空间 crossing 无时间语义 | 很早/很晚经过同一冲突点仍被当作同等 interaction | 采用 `exp(-abs(arrival_gap)/tau)`，默认 `tau=1.0` |
| pair collision 太弱 | CBV-CBV 碰撞可成为 reward hack | `lambda_pair_collision=20.0`，同时诊断 pair collision |
| baseline API 回归 | runner 向非 Joint policy 传 extra argument | 只对 Joint policy 传 PDM nominal trajectory |
| buffer 收到 `None` | `CBVRolloutBuffer.process_data_dict` 中 `NoneType` 下标错误 | runner/joint handoff 与 state payload 审计修复 |
| inference tensor 原地写 | PID `reset()` 在 inference tensor 外 `zero_()` | reset buffer 改为正常可写 tensor/clone 路径 |
| PyTorch 2.0 不支持 `any(dim=(1,2))` | `TypeError` 终止 Lightning sanity validation | 改为 `valid_mask.flatten(1).any(dim=1)` |
| CPU/CUDA device 不一致 | Joint score head Linear 收到 CPU `trajectory_samples` | scorer restore/输入 device 对齐修复 |
| 非显式 checkpoint eval | 自动按数值最新 CARLA episode 加载，非最佳模型 | `joint_ckpt_path` 显式 override |
| high-value 仅有实际权重记录 | Uniform 的 `high_value_rate=0` 无法区分“未启用”或“无合格状态” | 记录 `high_value_eligible` 与 `high_value_rate` 两套字段 |

### 3.5 当前运行/评估接口

`scripts/run.py` 支持：

```text
--routes
--routes_subset
--oracle_eval_mode {disabled, diagnose, oracle_gated}
--joint_eval_action_mode {policy, independent, oracle_gated}
```

含义必须严格区分：

- `policy`：运行 learned joint residual policy；
- `independent`：在**同一 candidate batch**上运行 factorized independent action；这是局部、matched 的消融对照；
- `oracle_gated`：仅为带门控的 upper-bound/诊断，不是可报告的 learned-policy 结果；
- `diagnose`：记录 oracle 和高价值诊断，但不改变 policy 执行。

---

## 4. 数据采集、候选诊断与 oracle 证据

### 4.1 早期全量 Joint-RIFT 评估诊断

文件：

```text
outputs/joint_rift_stage1_seed0_eval/eval/
  pdm_lite-joint_rift_audited-rule-seed0/joint_rift_eval_metrics.jsonl
```

本地可见 713 行；终端历史中与该阶段对应的完整诊断汇总为 447 个带 pair 的状态。严格 gate 的结果：

```text
states = 447
strict oracle gate pass = 0
high-value states (relaxed diagnostic) = 13 / 447 = 2.908%
```

代表性高价值 cluster：

| route | town | step | states | 最大 oracle joint gain |
|---|---|---:|---:|---:|
| 27494 | Town04 | 138–159 | 10 | 29.34 |
| 26405 | Town15 | 161 | 1 | 26.45 |
| 2091 | Town12 | 63–67 | 2 | 11.16 |

该诊断的总体现象：

- `oracle_joint_gain` 平均约 17.44，中位数约 12.88，最大约 98.43；候选矩阵中确有比 independent pair 更高的 evaluator 回报。
- `nonfactorizable_reward_ratio` 平均约 0.0553，中位数约 0.00208；平均不为零，但大量状态接近可分解。
- `both_necessary_rate` 平均约 0.0257、中位数 0；真正两个 CBV 都不可缺的状态很稀少。
- 严格 gate 的主要失败来源包括 both-necessary、pair collision、joint gain 与 nonfactorizable 条件。

**含义**：idea 的“候选空间机会存在”条件得到支持，但机会稀疏，且严格双车协同证据很少。把所有状态均匀训练会被大量低信息状态淹没，这是 HV 重加权的动机；同时也意味着任何整体均值都可能被少数 route/state 主导。

### 4.2 oracle 的边界

oracle 使用的是当前虚拟 evaluator，而不是真实 CARLA future/ego closed-loop 最优响应。因此：

- oracle gain 是 candidate-support / objective-support 证据；
- 不是 learned policy 的真实性能；
- 不是因果 ego 反应的 upper bound；
- 不可用 oracle-gated 结果替代 learned policy 主实验。

---

## 5. 早期 baseline、原始 Stage-1 与其局限

### 5.1 早期全量基线结果（仅作运行 sanity check）

当前工作区存在以下结果：

| 输出 | `score_composed` | `score_route` | `score_penalty` | 注释 |
|---|---:|---:|---:|---|
| IL Pluto baseline seed0 | 71.604 | 83.604 | 0.880 | 单次早期基线 |
| independent RIFT baseline seed0 | 94.679 | 94.679 | 1.000 | 单次早期基线 |
| 早期 Joint Stage-1 eval seed0 | 93.557 | 93.557 | 1.000 | 单次早期 joint eval |

对应文件：

```text
outputs/il_pluto_baseline_seed0/eval/pdm_lite-pluto-rule-seed0/simulation_results.json
outputs/rift_baseline_seed0/eval/pdm_lite-rift_pluto-rule-seed0/simulation_results.json
outputs/joint_rift_stage1_seed0_eval/eval/pdm_lite-joint_rift_audited-rule-seed0/simulation_results.json
```

这些数字不能作为 Uniform/HV4 主结论，原因包括：没有严格 train/holdout 分离、单 seed、不同早期代码状态、以及 score 方向尚未预注册。它们只证明 CARLA/RIFT/Joint 路径可以端到端产生结果。

### 5.2 原始 Stage-1 训练

早期训练使用 220-route bench 集（重复后 440 scenarios），输出在：

```text
outputs/joint_rift_stage1_seed0_train/
outputs/joint_rift_models/stage1/pdm_lite-rule-seed0/
```

可见多个 checkpoint，例如 episode 31、69、113、...、405。此训练在功能上完成并生成 offline W&B logs，但存在根本实验设计问题：后续用于开发/评估的 10 个 route 都来自原 220 集合。因此不能用它做不泄漏的泛化结论。

### 5.3 早期 route 27494 局部 matched 实验

终端历史中，使用早期模型在 route 27494 的四个 evaluation seeds 得到 policy 对 independent 的 `score_composed` 差：

```text
[12.88, 3.39, 13.91, 8.42], mean +9.65
```

但各 run 均存在 failed/blocked/timed-out 结果，且该 route 是从早期诊断中挑选出来的高价值 cluster；这不是独立测试，属于**探索性、选择偏差很大的局部正信号**。不能和后文 holdout-10 结果混合，也不能用于论文主结论。

---

## 6. 无泄漏训练集/holdout 的建立

### 6.1 split

从 `bench2drive220.xml` 中显式留出 10 条 route：

```text
3514, 3255, 26405, 25381, 25378,
25424, 2091, 27494, 17569, 28198
```

生成：

```text
rift/scenario/route/bench2drive_train210_excluding_dev10.xml  # 210 routes
rift/scenario/route/bench2drive_holdout10.xml                 # 10 routes
```

文件由：

```text
scripts/make_route_split.py
```

生成，且已检查 route 数为 210 和 10。

### 6.2 必须保留的设计解释

- train210 与 holdout10 的 route ID 无交集；这修正了早期实验的 route leakage。
- 训练使用 `repetitions=2`，故每个完整训练 run 应有 `210 × 2 = 420` 条 route records。
- holdout-10 使用 `repetitions=2`，每个 eval seed 每个 action mode 应有 `10 × 2 = 20` 个 matched route records。
- holdout10 已用于模型比较，因此它是当前意义上的 external test/selection set；以后若根据它反复调 threshold、架构或 checkpoint，就不再能称为最终无偏 test。应另留真正 test set 或冻结决策。

---

## 7. 正式 Stage-1 训练：Uniform 与 HV4

### 7.1 共同条件

两组使用：

- 相同 train210 route XML；
- 相同 PDM-Lite ego、rule recognition、RIFT bootstrap；
- 相同训练 seed `0`；
- `repetitions=2`、buffer capacity 4096；
- 相同 Joint-RIFT Stage-1 backbone，冻结 unary RIFT/Pluto，只训练 joint head；
- 唯一计划性变量：`high_value_reweight`。

共享 RIFT bootstrap 为：

```text
.../model_ckpt/rift_pluto/pdm_lite-rule-seed0/
carla_episode=427-epoch=12-val_loss=0.183.ckpt
```

### 7.2 Uniform 对照

路径：

```text
outputs/joint_rift_stage1_train210_uniform_seed0_train/
outputs/joint_rift_models/stage1_train210_uniform/pdm_lite-rule-seed0/
```

已核对：

```text
route_info lines = 420
complete rollout audits = 11
states per audit = 4096
checkpoints = 11
training route reward mean = -47.453762
```

Uniform config 中 `high_value_reweight.enabled=false`，因此所有训练 state weight 均为 1。汇总中：

```text
high_value_rate = 0
state_weight_mean = 1.0
```

这只能证明没有施加权重；由于 Uniform 完成早于 raw eligibility 字段加入，不能从其旧 audit 倒推出“无 HV 合格状态”。

Uniform checkpoint 的选择规则：仅按训练期文件名中的最小 `joint_val_loss`，不查看 holdout 结果。

```text
carla_episode=31-epoch=15-joint_val_loss=-0.031.ckpt
```

该规则并不证明该 checkpoint 全局最佳；它只是两个 ablation 使用的统一、holdout-independent 选择规则。

### 7.3 HV4 处理组

路径：

```text
outputs/joint_rift_stage1_train210_hv4_seed0_train/
outputs/joint_rift_models/stage1_train210_hv4/pdm_lite-rule-seed0/
```

HV4 config：

```yaml
high_value_reweight:
  enabled: true
  min_joint_gain: 10.0
  min_nonfactorizable_ratio: 0.01
  max_pair_collision_rate: 0.05
  weight: 4.0
```

候选 eligibility（未要求 `both_necessary`）为：

```text
oracle_joint_gain >= 10
AND nonfactorizable_reward_ratio >= 0.01
AND oracle_pair_collision <= 0
AND pair_collision_rate <= 0.05
```

省略 both-necessary 是明确的**放宽训练消融**：严格 both-necessary 在早期诊断中过于稀少，不能支撑足够 Stage-1 buffer。它不等于宣称所有被加权状态都证明了真正双车不可缺。

### 7.4 HV4 干净训练审计

HV4 output 目录中曾混入前一次中断 run 的 7 个 audit JSON；`--no_resume` 清理模型目录但不清理旧 audit。当前分析通过 checkpoint episode 对齐，使用下面 11 个 audit：

```text
31, 69, 112, 147, 183, 221, 251, 293, 333, 368, 404
```

干净审计文件：

```text
outputs/joint_rift_stage1_train210_hv4_seed0_train/
  hv4_final_run_audit_summary.json
```

结果：

| 项 | 值 |
|---|---:|
| 完整 route records | 420 |
| 训练更新数 | 11 |
| 每次更新 states | 4096 |
| HV eligible rate（update 平均） | 3.1361% |
| HV eligible states/update（平均） | 128.45 |
| HV eligible states/update（范围） | 63–242 |
| 实际 reweighted rate | 3.1361% |
| 平均 state weight | 1.094083 |
| oracle joint gain（update mean 的平均） | 21.6312 |
| nonfactorizable ratio（update mean 的平均） | 0.01956 |
| both-necessary rate（update mean 的平均） | 0.000927 |

校验：`1 + 3 × 0.03136097 = 1.0940829`，与 state weight 平均完全一致，证明合格 state 被赋予 4，其他 state 为 1。

HV4 checkpoint 选择结果：

```text
carla_episode=31-epoch=14-joint_val_loss=-0.034.ckpt
```

### 7.5 训练阶段的解释与问题

已证明：

- HV4 不是配置未生效或数据管道丢失；它确实对约 3.14% state 进行了四倍 state-level loss 加权。
- 联合 reward 的 oracle gain 和 NFR 在训练采样中不为零。

尚未证明：

- 加权目标与真实 CARLA route 指标方向一致；
- 这些 3.14% state 覆盖足以改变 residual 学习；
- HV eligibility 在不同训练 seeds、不同 policy rollout distribution 下稳定；
- `joint_val_loss` 与 holdout 行为相关。

关键风险：`both_necessary_rate` 约 0.093%，远低于 relaxed HV rate。这表明目前的 HV4 主要强调“有 oracle gain 且 reward 非可分解的安全状态”，而不是纯粹的“两个 CBV 都必要”的强 coordination state。

---

## 8. Holdout-10 matched 评估协议

### 8.1 固定 checkpoint 的必要性

为防止模型自动选取数值最新 checkpoint，创建只用于评估的 config，并设置：

```yaml
joint_ckpt_path: <absolute selected checkpoint>
```

评估不可使用 `--no_resume`，因为当前显式 checkpoint 加载逻辑在 `resume=True` 分支中执行。每次启动日志必须包含：

```text
Joint-RIFT explicit evaluation checkpoint: carla_episode=31-...
```

### 8.2 四条件、四评估 seeds

固定的是训练模型（训练 seed 0）；变化的是 CARLA evaluation seed `0,1,2,3`。每个 seed 对每个模型运行两种 action mode：

| 模型 | `policy` | `independent` |
|---|---|---|
| Uniform selected checkpoint | learned joint residual | 同 candidate batch 的 factorized action |
| HV4 selected checkpoint | learned joint residual | 同 candidate batch 的 factorized action |

每个结果包含 10 route × 2 repetitions = 20 records；四个 eval seed 共 80 records/条件，但 route records 在同一 seed 内相关，主统计单位应是**seed-level global matched delta**，而不是把 80 条 route 当独立样本。

相关工具：

```text
scripts/compare_joint_rift_eval.py
scripts/summarize_matched_comparisons.py
scripts/summarize_joint_rift_hv_ablation.py
```

`compare_joint_rift_eval.py` 对齐 route IDs，记录每 route 的 policy-minus-independent；`summarize_joint_rift_hv_ablation.py` 计算同一 eval seed 内：

```text
HV4(policy - independent) - Uniform(policy - independent)
```

这是当前最合适的差分中的差分（DiD）估计量：它消除该 eval seed 中 independent 基线的共同变化。它不能消除训练 seed 只为 0 的不确定性。

---

## 9. Holdout-10 的最终统计结果

### 9.1 原始每 seed `score_composed` matched delta

| eval seed | Uniform: policy−independent | HV4: policy−independent | HV4−Uniform interaction |
|---:|---:|---:|---:|
| 0 | +1.5540 | +1.2540 | −0.3000 |
| 1 | −3.2720 | +1.3310 | +4.6030 |
| 2 | −6.7425 | −6.4720 | +0.2705 |
| 3 | −2.4810 | −2.6455 | −0.1645 |

汇总：

| 指标 | Uniform | HV4 | DiD（HV4−Uniform） |
|---|---:|---:|---:|
| `score_composed` mean | −2.7354 | −1.6331 | +1.1023 |
| sample SD | 3.4063 | 3.7221 | 2.3465 |
| DiD median | — | — | +0.0530 |
| DiD min / max | — | — | −0.3000 / +4.6030 |

粗略以 n=4 的 t 区间估算，DiD 平均的 95% 区间约为 `[-2.63, +4.84]`。样本极小且得分并不应假设正态，因此该区间仅作不确定性提示；它明确包含有意义的负效应和正效应。

### 9.2 分项解释

| DiD 项 | mean | median | 解读 |
|---|---:|---:|---|
| `score_composed` | +1.1023 | +0.0530 | 均值几乎由 seed 1 的 +4.603 驱动，不稳健 |
| `score_route` | −1.0866 | −1.6147 | 路线推进项更差，不能忽略 |
| `score_penalty` | +0.0260 | +0.0320 | 仅有微弱 penalty 项改善 |

因此，`score_composed` 的小均值改善并非路线项一致提升，而更像 penalty 项的微小变化与单一 seed 事件共同造成。

### 9.3 route-level 现象（仅描述性，不作独立样本显著性）

- seed-0 两个模型都有 17/20 平局；早期 +40 分事件来自 `RouteScenario_2091_rep0`。
- 多 seed 下 `2091`、`27494`、`28198`、`3255` 等少数 route-repeat 贡献了大部分变化；大多数 route-repeat 的 delta 为 0。
- `27494` 经常失败或超时，且 policy−independent 变化不稳定；它既是早期 oracle 高价值 cluster，又是 CARLA route outcome 高方差来源。
- 这说明当前 10-route holdout 的有效信息量远低于“80 条 records”所暗示的数量，原因是大量完全平局、route 内重复相关以及少数事件主导。

### 9.4 status 变化的辅助观察

Uniform 的 policy-vs-independent 总计 80 route-repeat status 转换：

```text
Completed -> Completed: 68
Completed -> Failed - Agent got blocked: 2
Failed - Agent got blocked -> Completed: 1
Failed - Agent got blocked -> Failed - Agent got blocked: 4
Failed - Agent timed out -> Failed - Agent timed out: 5
```

HV4：

```text
Completed -> Completed: 70
Completed -> Failed - Agent got blocked: 1
Failed - Agent got blocked -> Failed - Agent got blocked: 3
Failed - Agent timed out -> Failed - Agent got blocked: 1
Failed - Agent timed out -> Failed - Agent timed out: 5
```

这些计数提示 HV4 可能改变少数失败转移，但没有足够样本或明确目标方向来证明它改善系统表现。

### 9.5 主结论

当前可说：

- HV4 的实现和数据采集正确；
- 与 Uniform 相比，HV4 的 `policy−independent` 数值差分平均更高 `+1.10`；
- 该差异高度不稳定，median 近零，且 route component 为负；
- learned joint policy 对 independent 的优势本身也不稳定：Uniform 仅 1/4 eval seeds 为正，HV4 仅 2/4 为正。

当前不可说：

- “HV4 有稳定提升”；
- “Joint-RIFT 已超过 independent RIFT”；
- “两 CBV 协同已被 learned policy 稳定利用”；
- “可以继续 Stage 2”。

---

## 10. 当前数据、代码、训练、评估的问题清单

### 10.1 数据问题

1. **真正强协同状态稀疏**：strict both-necessary 几乎为零；HV4 采用 relaxed label。当前结果更像 rare non-factorizable opportunity mining，而不是已证实的双车共同必要性学习。
2. **rollout distribution 依赖当前 policy**：HV eligibility 是 on-policy 收集的属性；Uniform/HV4 随着模型改变会采到不同状态，不能把 rate 差异直接归因为方法能力。
3. **output 目录不是 run manifest**：历史 audit 可残留，不能只按 glob 聚合。必须按 checkpoint episode 或 run UUID 对齐。
4. **数据路径/权限脆弱**：map/speed 数据跨项目共享，root/riftuser owner 混合，重现前必须运行完整 preflight。

### 10.2 代码问题/风险

1. 重要修复尚处于脏工作树；没有提交 hash/immutable manifest。新会话不得假定远端和本地源文件完全一致。
2. `joint_ckpt_path` 依赖 `resume=True` 的当前实现细节，错误加入 `--no_resume` 会悄悄改为 bootstrap 而不是评估目标模型。
3. score comparison 工具把 `policy_minus_independent > 0` 名为 win，但未编码任务方向。应重命名为 signed delta，或将方向作为显式 CLI/metadata 参数。
4. audit JSON 当前是 summary，未保存每 state 的 eligibility、pair index、reward decomposition 与 selector logits；对失败 route 做深入回放时证据不足。
5. `joint_val_loss` 来自每轮 on-policy buffer 的内部 90/10 split；不是固定验证集，也不是跨 checkpoint 可完全等价的泛化指标。

### 10.3 训练问题

1. 仅训练 seed 0。四个 CARLA eval seed 只评估环境/traffic 随机性，**不评估 learned model 的训练随机性**。
2. 选中的 episode 31 checkpoint 很早；虽然选择规则在两组一致且未使用 holdout，但可能错过后期学习，或说明训练 drift/validation mismatch。
3. 只训练 11 个 buffer update，HV state 平均每 buffer 约 128 个。若 residual head 信号弱，4 倍权重可能不足；若加大权重，又有过拟合 rare simulator artifacts 的风险。
4. Stage 1 residual 只能重排当前 RIFT candidate support；当 oracle candidate unary NLL 很高时，冻结 unary prior 会限制可学习性。Stage 2 是可能的机制，但只有在 Stage 1 信号稳健后才合理。

### 10.4 评估问题

1. **目标方向未冻结**：若任务是 adversarial scenario generation，较低 ego route score 可能更“强”；若任务是安全/route 成功，则较高分更好。当前的正负 DiD 不能脱离这一语义解释。
2. holdout 仅 10 条 route，且已经用于探索性决策；不应一边看 holdout 一边持续调 HV threshold，再称其为最终 test。
3. independent 是同一 joint model candidate batch 的 counterfactual action mode，适合回答“residual 是否带来额外选择价值”，但不等价于完整 external baseline（IL/independent RIFT/其他 planner）的最终比较。
4. route records 不是独立观测。主单位应是 seed-level global delta；n=4 极小。
5. 大量平局、少数 high-variance failure route 主导均值。应报告 per-seed、median、route-level waterfall，而不是只报全局平均。

---

## 11. 当前版本模型的机制性诊断

### 11.1 已经成立的部分

- 代码能构造/评分 full joint candidate matrix；
- objective 中可检测到 non-factorizable reward 和 oracle gain；
- high-value reweight 真实进入训练 loss，而不是只记录 log；
- explicit checkpoint、independent counterfactual 与 route-matched JSON 已建立；
- 当前结果排除了若干工程性假阳性：错误 prior、错误 map、错误 device、stale checkpoint、错误端口、缺失 pair payload 等。

### 11.2 当前 bottleneck 的最可能解释

按优先级：

1. **信号稀疏且异质**：真正/近似联合机会只占约几个百分点，route 间差异极大。
2. **目标与评估错配**：虚拟 joint reward 的 oracle 优势不必然转化为 CARLA route score；PDM nominal 不是 CBV pair 后的真实 ego response。
3. **residual 表达或监督不足**：4 倍 state reweight 只改变 loss 权重，不改变 candidate coverage、pair reward 或 counterfactual training target；可能不足以学习 rare event。
4. **冻结 unary prior 限制**：有价值 pair 可能在 unary RIFT 下概率太低，Stage 1 residual 的可调范围不够。
5. **checkpoint/训练动态不稳定**：最小内部 validation loss 均出现在很早 episode，后续 checkpoint 损失总体变大；需区分 overfitting、buffer distribution drift、学习率/epoch schedule 问题。
6. **CARLA 评价离散且噪声高**：blocked/timeout 与 actor spawn 差异可造成大分数跳变，掩盖小 residual 变化。

### 11.3 不应做的错误解释

- 不应因为 oracle gain 大就宣称 learned policy 一定有效；
- 不应因为 HV4 实际加权就宣称 HV4 有性能提升；
- 不应把 evaluation seeds 当作 independent training replications；
- 不应把早期挑选出的 route 27494 局部正结果视为泛化；
- 不应把 raw `score_composed` 的增大自动称为 adversarial 成功；
- 不应在当前证据下启动 Stage 2 并将失败归因为 Stage 1 不够长。

---

## 12. 下一阶段建议（按优先级）

### P0：冻结现状并补齐实验可审计性

在改模型前：

1. 提交/导出当前 patch，记录 `git diff`、commit hash、全部 config 副本、Python/CARLA/CUDA 版本和绝对 checkpoint 路径。
2. 建立每 run 的 `manifest.json`：训练/评估时间、source hash、config hash、route XML hash、seed、ports、checkpoint、有效 audit episode 列表。
3. 修改或包装 comparison 脚本：不再称 `policy_wins`，而输出 `signed_delta` 与一个显式 `metric_higher_is_better` 字段。
4. 冻结现有 holdout10 结果；后续调参尽量使用新的 development split，保留 holdout10 只做确认。

### P1：先判定目标语义，再做高价值 route 机制审计

对 2091、27494、28198、3255，保存并检查每个选择时刻：

- candidate pair IDs、policy pair、independent pair、oracle pair；
- unary log-prob、joint residual、joint reward 的完整分解；
- oracle gap、NFR、pair collision、both-necessary、arrival-time interaction trace；
- CARLA actor state、PDM nominal、最终 failure/status；
- BEV/video（已有 `--save_bev`/render 能力时使用）。

目标是回答：模型是没有选到 oracle pair、选到但 simulator outcome 不对应、还是 candidate/evaluator 本身产生伪高价值标签。若不回答这个问题，调 weight/threshold 只是盲调。

### P2：正确的重复实验层级

若继续验证 HV idea，优先做**训练 seed 重复**而非只增加 eval seed：

```text
training seeds: 0, 1, 2（至少）
per training seed: Uniform 与 HV4 各自独立训练
evaluation seeds: 至少 0, 1, 2；使用同一冻结 holdout protocol
```

分析采用 nested/paired 结构：训练 seed 为主随机单位，eval seed 为该模型的环境重复；报告每个 train seed 的平均 matched delta，再比较 HV4 vs Uniform。当前只有 training seed 0，不能估计训练方差。

### P3：仅在 P1/P2 支持后考虑的模型/训练消融

建议按顺序，一次只改一个变量：

1. **HV threshold sweep**：固定 weight 4，开发集上比较 `min_joint_gain`、NFR 门槛和 collision gate，报告 eligible coverage 与稳定性。
2. **weight sweep**：1、2、4、8；只有当 eligibility 稳定且有 route-level 因果证据时做。
3. **严格与放宽 label 消融**：有/无 both-necessary，明确比较稀疏性与效果，不能把二者混称为 coordination。
4. **固定 validation replay buffer**：从训练 route 中冻结一批 state，评价 checkpoint 的 joint loss、oracle rank、policy-oracle gap，避免仅依赖 rolling on-policy val loss。
5. **residual 校准诊断**：记录 `delta` 的均值、方差、相对 unary logit 尺度、argmax 与 independent action 的改变率。若几乎不改变 action，先解决 scale/optimization；若频繁改变且性能不稳，先解决 reward alignment/safety。
6. **Stage 2 的 go/no-go**：只有多 training seed 上 Stage 1 的 joint-vs-independent 效应方向一致、且 route metric 与目标对齐后，才允许 Stage 2。Stage 2 必须使用新目录、fresh on-policy rollout、显式 Stage-1 checkpoint。

---

## 13. 关键文件与结果索引

### 代码/配置

```text
README_APPLY_AND_RUN.md
RIFT/scripts/validate_joint_rift_core.py
RIFT/scripts/run.py
RIFT/scripts/make_route_split.py
RIFT/scripts/compare_joint_rift_eval.py
RIFT/scripts/summarize_matched_comparisons.py
RIFT/scripts/summarize_joint_rift_hv_ablation.py
RIFT/scripts/summarize_joint_rollout_audits.py
RIFT/rift/cbv/planning/config/joint_rift_pluto_stage1.yaml
RIFT/rift/cbv/planning/config/joint_rift_pluto_stage2.yaml
```

远端另有 run-specific config（由当前实验生成）：

```text
joint_rift_pluto_stage1_train210_uniform.yaml
joint_rift_pluto_stage1_train210_hv4.yaml
joint_rift_pluto_eval_uniform_selected.yaml
joint_rift_pluto_eval_hv4_selected.yaml
```

### 正式 train210 实验

```text
outputs/joint_rift_stage1_train210_uniform_seed0_train/
outputs/joint_rift_stage1_train210_hv4_seed0_train/
outputs/joint_rift_models/stage1_train210_uniform/
outputs/joint_rift_models/stage1_train210_hv4/
```

### 最终 holdout 统计

```text
outputs/holdout10_uniform_multiseed_summary.json
outputs/holdout10_hv4_multiseed_summary.json
outputs/holdout10_hv4_vs_uniform_multiseed.json
outputs/holdout10_statistics_terminal.log
```

---

## 14. 完整命令账本：版本、命令、检查点与结果路径

本节用于新会话追溯。除非另有说明，命令均在远端执行；代码根目录永远是带 `_2` 的目录，而 CARLA 和 scenario-dreamer 永远是不带 `_2` 的目录。

### 14.1 统一变量与运行身份

所有 CARLA/Python run 以 `riftuser` 运行。需要写源码、创建 config、修正 root-owned 目录时，使用实际 owner；不要假定 `riftuser` 能写 `RIFT/rift/...`。

```bash
ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop
RIFT_ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop_2/RIFT
CARLA_ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop/carla
SCENARIO_ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop/scenario-dreamer
PY=/zeron-vepfs/tjqc/njl/envs/rift/bin/python
OUT_ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop_2/outputs
MODEL_ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop/RIFT/rift/cbv/planning/model_ckpt
```

CARLA 子进程 run 的最低环境模板：

```bash
export CARLA_ROOT="$CARLA_ROOT"
export CARLA_PYTHONAPI="$CARLA_ROOT/PythonAPI/carla"
export CARLA_EGG="$(find "$CARLA_ROOT/PythonAPI/carla/dist" -maxdepth 1 -name "carla-*.egg" -print -quit)"
export PYTHONPATH="$RIFT_ROOT:$SCENARIO_ROOT:$CARLA_PYTHONAPI:$CARLA_EGG"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
```

验证环境（历史上已通过）：

```bash
runuser -u riftuser -- bash -lc '
RIFT_ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop_2/RIFT
CARLA_ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop/carla
SCENARIO_ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop/scenario-dreamer
PY=/zeron-vepfs/tjqc/njl/envs/rift/bin/python
export PYTHONPATH="$RIFT_ROOT:$SCENARIO_ROOT:$CARLA_ROOT/PythonAPI/carla:$(find "$CARLA_ROOT/PythonAPI/carla/dist" -maxdepth 1 -name "carla-*.egg" -print -quit)"
cd "$RIFT_ROOT"
"$PY" -c "import torch, carla; from agents.navigation.local_planner import RoadOption; print(torch.__version__, torch.cuda.is_available(), RoadOption)"
'
```

预期：`2.0.1+cu117 True <enum 'RoadOption'>` 或等价导入成功输出。

### 14.2 静态验证与版本审计

每次同步源码后先运行：

```bash
runuser -u riftuser -- bash -lc '
set -euo pipefail
RIFT_ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop_2/RIFT
CARLA_ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop/carla
SCENARIO_ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop/scenario-dreamer
PY=/zeron-vepfs/tjqc/njl/envs/rift/bin/python
export CARLA_ROOT
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$RIFT_ROOT:$SCENARIO_ROOT:$CARLA_ROOT/PythonAPI/carla:$(find "$CARLA_ROOT/PythonAPI/carla/dist" -maxdepth 1 -name "carla-*.egg" -print -quit)"
cd "$RIFT_ROOT"
"$PY" -m compileall -q rift scripts
"$PY" scripts/validate_joint_rift_core.py
'
```

预期最后一行：

```text
Joint-RIFT core validation: PASS
```

应保存的源码版本证据：

```bash
cd "$RIFT_ROOT"
git status --short > "$OUT_ROOT/code_status_<date>.txt"
git diff > "$OUT_ROOT/code_diff_<date>.patch"
git rev-parse HEAD > "$OUT_ROOT/code_head_<date>.txt" 2>&1 || true
```

注意：当前远端可能不是 Git worktree；这种情况下 `git rev-parse` 失败是预期的，`code_diff` 也无法生成。至少复制 `RIFT/rift/`、`RIFT/scripts/`、所有 eval config 和本报告作为不可变归档。

### 14.3 数据可用性与 map/speed 映射

当前 `_2/RIFT/data/map_data` 与 `_2/RIFT/data/speed_limits` 曾为空。源数据为：

```text
$ROOT/RIFT/data/map_data/HD-Map
$ROOT/RIFT/data/speed_limits/Speed-Limits
```

先检查（这些 `test` 成功时无输出）：

```bash
test -f "$ROOT/RIFT/data/map_data/HD-Map/Town03_HD_map.npz"
test -f "$ROOT/RIFT/data/speed_limits/Speed-Limits/Town03_speed_limits.npy"
```

如当前代码目录仍没有可读的平铺文件，使用有写权限的 owner 建立链接；不要把下列命令误放进 `runuser` 后而遇到 permission denied：

```bash
find "$ROOT/RIFT/data/map_data/HD-Map" -maxdepth 1 -type f -name "*.npz" \
  -exec ln -s {} "$RIFT_ROOT/data/map_data/" \;

find "$ROOT/RIFT/data/speed_limits/Speed-Limits" -maxdepth 1 -type f -name "*.npy" \
  -exec ln -s {} "$RIFT_ROOT/data/speed_limits/" \;
```

结果检查：

```bash
test -f "$RIFT_ROOT/data/map_data/Town03_HD_map.npz"
test -f "$RIFT_ROOT/data/speed_limits/Town03_speed_limits.npy"
```

### 14.4 Route split：train210 / holdout10

创建 split 的命令（已执行）：

```bash
cd "$RIFT_ROOT"
PYTHONDONTWRITEBYTECODE=1 "$PY" scripts/make_route_split.py \
  --input_xml rift/scenario/route/bench2drive220.xml \
  --holdout_ids 3514,3255,26405,25381,25378,25424,2091,27494,17569,28198 \
  --train_output rift/scenario/route/bench2drive_train210_excluding_dev10.xml \
  --holdout_output rift/scenario/route/bench2drive_holdout10.xml

grep -c '<route id=' rift/scenario/route/bench2drive_train210_excluding_dev10.xml
grep -c '<route id=' rift/scenario/route/bench2drive_holdout10.xml
```

预期输出：

```text
ROUTE_SPLIT_OK train=210 holdout=10 ...
210
10
```

输入/结果路径：

```text
input:  rift/scenario/route/bench2drive220.xml
train:  rift/scenario/route/bench2drive_train210_excluding_dev10.xml
test:   rift/scenario/route/bench2drive_holdout10.xml
```

### 14.5 历史版本 A：早期 baseline 与原始 Stage-1

这些命令的完整原始 shell 包装未作为单一文件保留；下列是根据 `README_APPLY_AND_RUN.md` 和输出目录重建的等价命令。它们只用于复现实行/定位，不应重写正式结果。

```bash
# IL Pluto early baseline
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/run.py \
  --mode eval --output_dir "$OUT_ROOT/il_pluto_baseline_seed0" \
  --ego_cfg pdm_lite.yaml --cbv_cfg pluto.yaml --seed 0 --repetitions 1

# Independent RIFT early baseline
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/run.py \
  --mode eval --output_dir "$OUT_ROOT/rift_baseline_seed0" \
  --ego_cfg pdm_lite.yaml --cbv_cfg rift_pluto.yaml --seed 0 --repetitions 1

# 早期 audited Joint Stage-1（历史，不是 leakage-free train210 protocol）
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/run.py \
  --mode train_cbv --output_dir "$OUT_ROOT/joint_rift_stage1_seed0_train" \
  --ego_cfg pdm_lite.yaml --cbv_cfg joint_rift_pluto_stage1_shared.yaml \
  --seed 0 --repetitions 2 --no_resume
```

对应结果：

```text
outputs/il_pluto_baseline_seed0/eval/pdm_lite-pluto-rule-seed0/
outputs/rift_baseline_seed0/eval/pdm_lite-rift_pluto-rule-seed0/
outputs/joint_rift_stage1_seed0_train/
outputs/joint_rift_models/stage1/pdm_lite-rule-seed0/
outputs/joint_rift_stage1_seed0_eval/eval/pdm_lite-joint_rift_audited-rule-seed0/
```

历史 early Joint eval diagnostics：

```bash
"$PY" scripts/summarize_oracle_gated_diagnostics.py \
  "$OUT_ROOT/joint_rift_stage1_seed0_eval/eval/pdm_lite-joint_rift_audited-rule-seed0/joint_rift_eval_metrics.jsonl" \
  --output_json "$OUT_ROOT/joint_rift_stage1_seed0_eval/oracle_diagnostics_summary.json"
```

若该输出 JSON 不存在，不应通过重新运行旧模型来替代正式 train210 分析；旧流程有 route leakage。

### 14.6 正式版本 B：创建 Uniform/HV4 train210 configs

下列命令需要对 config 目录有写权限，已执行。使用专属 `model_path` 防止与旧模型混用：

```bash
UNIFORM_CFG="$RIFT_ROOT/rift/cbv/planning/config/joint_rift_pluto_stage1_train210_uniform.yaml"
HV_CFG="$RIFT_ROOT/rift/cbv/planning/config/joint_rift_pluto_stage1_train210_hv4.yaml"

cp "$RIFT_ROOT/rift/cbv/planning/config/joint_rift_pluto_stage1_shared.yaml" "$UNIFORM_CFG"
cp "$UNIFORM_CFG" "$HV_CFG"

sed -i "s|^model_path:.*|model_path: '$OUT_ROOT/joint_rift_models/stage1_train210_uniform'|" "$UNIFORM_CFG"
sed -i "s|^model_path:.*|model_path: '$OUT_ROOT/joint_rift_models/stage1_train210_hv4'|" "$HV_CFG"

cat >> "$HV_CFG" <<'EOF'
high_value_reweight:
  enabled: true
  min_joint_gain: 10.0
  min_nonfactorizable_ratio: 0.01
  max_pair_collision_rate: 0.05
  weight: 4.0
EOF

install -d -o riftuser -g riftuser -m 775 \
  "$OUT_ROOT/joint_rift_models/stage1_train210_uniform" \
  "$OUT_ROOT/joint_rift_models/stage1_train210_hv4"
```

`base_rift_model_path` 必须指向共享旧项目中实际存在的 RIFT prior 路径：

```text
/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop/RIFT/
rift/cbv/planning/model_ckpt/rift_pluto
```

### 14.7 正式版本 B1：Uniform Stage-1（已完成）

精确执行命令：

```bash
runuser -u riftuser -- bash -lc '
set -euo pipefail
RIFT_ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop_2/RIFT
CARLA_ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop/carla
SCENARIO_ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop/scenario-dreamer
PY=/zeron-vepfs/tjqc/njl/envs/rift/bin/python
OUT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop_2/outputs/joint_rift_stage1_train210_uniform_seed0_train
export CARLA_ROOT
export PYTHONPATH="$RIFT_ROOT:$SCENARIO_ROOT:$CARLA_ROOT/PythonAPI/carla:$(find "$CARLA_ROOT/PythonAPI/carla/dist" -maxdepth 1 -name "carla-*.egg" -print -quit)"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 WANDB_MODE=offline
mkdir -p "$OUT"
cd "$RIFT_ROOT"
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/run.py \
  --mode train_cbv \
  --output_dir "$OUT" \
  --routes rift/scenario/route/bench2drive_train210_excluding_dev10.xml \
  --ego_cfg pdm_lite.yaml \
  --cbv_cfg joint_rift_pluto_stage1_train210_uniform.yaml \
  --seed 0 --repetitions 2 \
  --base_port 31000 --base_tm_port 51000 \
  --no_resume \
  2>&1 | tee "$OUT/console.log"
'
```

输出与验收：

```text
run:    outputs/joint_rift_stage1_train210_uniform_seed0_train/
models: outputs/joint_rift_models/stage1_train210_uniform/pdm_lite-rule-seed0/
audit:  <run>/train_cbv/pdm_lite-joint_rift_audited-rule-seed0/joint_rollout_audit_episode=*.json
result: 420 route lines, 11 checkpoints, 11 × 4096-state updates
selected checkpoint: carla_episode=31-epoch=15-joint_val_loss=-0.031.ckpt
```

仅验证/统计，不会改动模型：

```bash
RUN="$OUT_ROOT/joint_rift_stage1_train210_uniform_seed0_train"
MODEL="$OUT_ROOT/joint_rift_models/stage1_train210_uniform"
wc -l < "$RUN/train_cbv/pdm_lite-joint_rift_audited-rule-seed0/route_info.txt"
find "$MODEL/pdm_lite-rule-seed0" -type f -name "*.ckpt" | wc -l
"$PY" "$RIFT_ROOT/scripts/summarize_joint_rollout_audits.py" \
  "$RUN/train_cbv/pdm_lite-joint_rift_audited-rule-seed0" \
  --output_json "$RUN/uniform_rollout_audit_summary.json"
```

### 14.8 正式版本 B2：HV4 Stage-1（已完成）

精确执行命令：

```bash
runuser -u riftuser -- bash -lc '
set -euo pipefail
RIFT_ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop_2/RIFT
CARLA_ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop/carla
SCENARIO_ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop/scenario-dreamer
PY=/zeron-vepfs/tjqc/njl/envs/rift/bin/python
OUT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop_2/outputs/joint_rift_stage1_train210_hv4_seed0_train
export CARLA_ROOT
export PYTHONPATH="$RIFT_ROOT:$SCENARIO_ROOT:$CARLA_ROOT/PythonAPI/carla:$(find "$CARLA_ROOT/PythonAPI/carla/dist" -maxdepth 1 -name "carla-*.egg" -print -quit)"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 WANDB_MODE=offline
mkdir -p "$OUT"
cd "$RIFT_ROOT"
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/run.py \
  --mode train_cbv \
  --output_dir "$OUT" \
  --routes rift/scenario/route/bench2drive_train210_excluding_dev10.xml \
  --ego_cfg pdm_lite.yaml \
  --cbv_cfg joint_rift_pluto_stage1_train210_hv4.yaml \
  --seed 0 --repetitions 2 \
  --base_port 32000 --base_tm_port 52000 \
  --no_resume \
  2>&1 | tee "$OUT/console.log"
'
```

输出与验收：

```text
run:    outputs/joint_rift_stage1_train210_hv4_seed0_train/
models: outputs/joint_rift_models/stage1_train210_hv4/pdm_lite-rule-seed0/
selected checkpoint: carla_episode=31-epoch=14-joint_val_loss=-0.034.ckpt
```

因残留 audit，必须通过当前 checkpoint episode 创建 filtered audit 目录，再汇总：

```bash
runuser -u riftuser -- bash -lc '
set -euo pipefail
RIFT_ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop_2/RIFT
PY=/zeron-vepfs/tjqc/njl/envs/rift/bin/python
RUN=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop_2/outputs/joint_rift_stage1_train210_hv4_seed0_train
MODEL=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop_2/outputs/joint_rift_models/stage1_train210_hv4/pdm_lite-rule-seed0
SOURCE_AUDITS="$RUN/train_cbv/pdm_lite-joint_rift_audited-rule-seed0"
FILTERED_AUDITS="$RUN/final_checkpoint_matched_audits"
test ! -e "$FILTERED_AUDITS"
mkdir "$FILTERED_AUDITS"
for CKPT in "$MODEL"/*.ckpt; do
  NAME=$(basename "$CKPT")
  EPISODE=${NAME#carla_episode=}
  EPISODE=${EPISODE%%-*}
  AUDIT="$SOURCE_AUDITS/joint_rollout_audit_episode=$EPISODE.json"
  test -f "$AUDIT"
  ln -s "$AUDIT" "$FILTERED_AUDITS/$(basename "$AUDIT")"
done
cd "$RIFT_ROOT"
"$PY" scripts/summarize_joint_rollout_audits.py "$FILTERED_AUDITS" \
  --output_json "$RUN/hv4_final_run_audit_summary.json"
'
```

如果 filtered audit 目录已存在，**不要直接重跑上述命令**（`test ! -e` 会故意失败以保护已有证据）；只读取既有 summary，或新建有日期后缀的新目录。

### 14.9 正式版本 C：固定 checkpoint 的 holdout evaluation configs

创建 config（有 config 写权限的用户运行）：

```bash
UNIFORM_CKPT="$OUT_ROOT/joint_rift_models/stage1_train210_uniform/pdm_lite-rule-seed0/carla_episode=31-epoch=15-joint_val_loss=-0.031.ckpt"
HV4_CKPT="$OUT_ROOT/joint_rift_models/stage1_train210_hv4/pdm_lite-rule-seed0/carla_episode=31-epoch=14-joint_val_loss=-0.034.ckpt"
UNIFORM_CFG="$RIFT_ROOT/rift/cbv/planning/config/joint_rift_pluto_eval_uniform_selected.yaml"
HV4_CFG="$RIFT_ROOT/rift/cbv/planning/config/joint_rift_pluto_eval_hv4_selected.yaml"

test -f "$UNIFORM_CKPT"
test -f "$HV4_CKPT"
cp "$RIFT_ROOT/rift/cbv/planning/config/joint_rift_pluto_stage1_train210_uniform.yaml" "$UNIFORM_CFG"
cp "$RIFT_ROOT/rift/cbv/planning/config/joint_rift_pluto_stage1_train210_hv4.yaml" "$HV4_CFG"

if grep -q "^joint_ckpt_path:" "$UNIFORM_CFG"; then
  sed -i "s|^joint_ckpt_path:.*|joint_ckpt_path: \"$UNIFORM_CKPT\"|" "$UNIFORM_CFG"
else
  printf "\njoint_ckpt_path: \"%s\"\n" "$UNIFORM_CKPT" >> "$UNIFORM_CFG"
fi
if grep -q "^joint_ckpt_path:" "$HV4_CFG"; then
  sed -i "s|^joint_ckpt_path:.*|joint_ckpt_path: \"$HV4_CKPT\"|" "$HV4_CFG"
else
  printf "\njoint_ckpt_path: \"%s\"\n" "$HV4_CKPT" >> "$HV4_CFG"
fi
grep -nE "^model_path:|^joint_ckpt_path:" "$UNIFORM_CFG" "$HV4_CFG"
```

### 14.10 正式版本 C：holdout10 四模型条件、四 evaluation seeds（已完成）

下列命令是已执行协议的完整可重放版本。**不得附加 `--no_resume`**，否则 explicit checkpoint 不会被加载。每次 log 必须确认 `Joint-RIFT explicit evaluation checkpoint`。

```bash
runuser -u riftuser -- bash -lc '
set -euo pipefail
RIFT_ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop_2/RIFT
CARLA_ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop/carla
SCENARIO_ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop/scenario-dreamer
PY=/zeron-vepfs/tjqc/njl/envs/rift/bin/python
OUT_ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop_2/outputs
export CARLA_ROOT
export PYTHONPATH="$RIFT_ROOT:$SCENARIO_ROOT:$CARLA_ROOT/PythonAPI/carla:$(find "$CARLA_ROOT/PythonAPI/carla/dist" -maxdepth 1 -name "carla-*.egg" -print -quit)"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 WANDB_MODE=offline

run_eval () {
  SEED="$1"; LABEL="$2"; CFG="$3"; ACTION_MODE="$4"; PORT="$5"; TM_PORT="$6"
  OUT="$OUT_ROOT/holdout10_${LABEL}_${ACTION_MODE}_seed${SEED}"
  mkdir -p "$OUT"
  cd "$RIFT_ROOT"
  CUDA_VISIBLE_DEVICES=0 "$PY" scripts/run.py \
    --mode eval --output_dir "$OUT" \
    --routes rift/scenario/route/bench2drive_holdout10.xml \
    --ego_cfg pdm_lite.yaml --cbv_cfg "$CFG" \
    --seed "$SEED" --repetitions 2 \
    --base_port "$PORT" --base_tm_port "$TM_PORT" \
    --oracle_eval_mode diagnose \
    --joint_eval_action_mode "$ACTION_MODE" \
    2>&1 | tee "$OUT/console.log"
}

for SEED in 0 1 2 3; do
  BASE_PORT=$((33000 + SEED * 100))
  BASE_TM_PORT=$((53000 + SEED * 100))
  run_eval "$SEED" uniform joint_rift_pluto_eval_uniform_selected.yaml policy      "$BASE_PORT" "$BASE_TM_PORT"
  run_eval "$SEED" uniform joint_rift_pluto_eval_uniform_selected.yaml independent "$((BASE_PORT + 10))" "$((BASE_TM_PORT + 10))"
  run_eval "$SEED" hv4 joint_rift_pluto_eval_hv4_selected.yaml policy              "$((BASE_PORT + 20))" "$((BASE_TM_PORT + 20))"
  run_eval "$SEED" hv4 joint_rift_pluto_eval_hv4_selected.yaml independent         "$((BASE_PORT + 30))" "$((BASE_TM_PORT + 30))"
done
'
```

结果目录 pattern：

```text
outputs/holdout10_{uniform,hv4}_{policy,independent}_seed{0,1,2,3}/
  eval/pdm_lite-joint_rift_audited-rule-seed{0,1,2,3}/
    simulation_results.json
    live_results.txt
  console.log
```

注：seed0 曾使用端口 33000–33030；seeds1–3 使用 34100–34330。上面的统一 loop 对 seed0 可能与当时历史端口略有不同，但不改变模型/路线/随机种子协议；若仅追溯历史，以实际输出目录与 console log 为准。

### 14.11 比较与最终统计（已完成）

每个 seed 的 matched comparison：

```bash
cd "$RIFT_ROOT"
for SEED in 0 1 2 3; do
  SUFFIX="eval/pdm_lite-joint_rift_audited-rule-seed${SEED}/simulation_results.json"
  "$PY" scripts/compare_joint_rift_eval.py \
    --policy "$OUT_ROOT/holdout10_uniform_policy_seed${SEED}/$SUFFIX" \
    --independent "$OUT_ROOT/holdout10_uniform_independent_seed${SEED}/$SUFFIX" \
    --output_json "$OUT_ROOT/holdout10_uniform_policy_vs_independent_seed${SEED}.json"
  "$PY" scripts/compare_joint_rift_eval.py \
    --policy "$OUT_ROOT/holdout10_hv4_policy_seed${SEED}/$SUFFIX" \
    --independent "$OUT_ROOT/holdout10_hv4_independent_seed${SEED}/$SUFFIX" \
    --output_json "$OUT_ROOT/holdout10_hv4_policy_vs_independent_seed${SEED}.json"
done
```

为避免终端 JSON 过长，实际最终统计全部重定向到文件：

```bash
OUT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop_2/outputs
runuser -u riftuser -- bash -lc '
set -euo pipefail
RIFT_ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop_2/RIFT
PY=/zeron-vepfs/tjqc/njl/envs/rift/bin/python
OUT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop_2/outputs
cd "$RIFT_ROOT"
"$PY" scripts/summarize_matched_comparisons.py \
  --input_glob "$OUT/holdout10_uniform_policy_vs_independent_seed*.json" \
  --output_json "$OUT/holdout10_uniform_multiseed_summary.json"
"$PY" scripts/summarize_matched_comparisons.py \
  --input_glob "$OUT/holdout10_hv4_policy_vs_independent_seed*.json" \
  --output_json "$OUT/holdout10_hv4_multiseed_summary.json"
"$PY" scripts/summarize_joint_rift_hv_ablation.py \
  --uniform "$OUT"/holdout10_uniform_policy_vs_independent_seed*.json \
  --hv4 "$OUT"/holdout10_hv4_policy_vs_independent_seed*.json \
  --output_json "$OUT/holdout10_hv4_vs_uniform_multiseed.json"
' > "$OUT/holdout10_statistics_terminal.log" 2>&1
```

最终需要下载/归档的精确文件：

```text
outputs/holdout10_uniform_policy_vs_independent_seed0.json
outputs/holdout10_uniform_policy_vs_independent_seed1.json
outputs/holdout10_uniform_policy_vs_independent_seed2.json
outputs/holdout10_uniform_policy_vs_independent_seed3.json
outputs/holdout10_hv4_policy_vs_independent_seed0.json
outputs/holdout10_hv4_policy_vs_independent_seed1.json
outputs/holdout10_hv4_policy_vs_independent_seed2.json
outputs/holdout10_hv4_policy_vs_independent_seed3.json
outputs/holdout10_uniform_multiseed_summary.json
outputs/holdout10_hv4_multiseed_summary.json
outputs/holdout10_hv4_vs_uniform_multiseed.json
outputs/holdout10_statistics_terminal.log
```

### 14.12 结果路径—版本总表

| 版本 | 是否可作主结论 | 主要输入 | 结果/日志路径 |
|---|---|---|---|
| 环境 smoke/import | 否 | CARLA API、共享 env | 仅终端历史；建议以后保存 `preflight.log` |
| IL Pluto early seed0 | 否，仅 baseline sanity | 早期默认 bench route | `outputs/il_pluto_baseline_seed0/...` |
| RIFT early seed0 | 否，仅 baseline sanity | 早期默认 bench route | `outputs/rift_baseline_seed0/...` |
| 原始 Joint Stage-1 | 否，train/eval route leakage | bench220，440 records | `outputs/joint_rift_stage1_seed0_train/`、`outputs/joint_rift_models/stage1/` |
| Early joint oracle diagnostics | 可作 opportunity 证据，不作泛化结论 | 早期 Joint eval | `outputs/joint_rift_stage1_seed0_eval/.../joint_rift_eval_metrics.jsonl` |
| route27494 局部比较 | 否，探索性 | 被挑选高价值 route | 历史 `joint_rift_route27494_*` 输出；若未下载则仅终端历史 |
| Uniform train210 seed0 | 是，当前对照训练 | train210 × 2 | `outputs/joint_rift_stage1_train210_uniform_seed0_train/`、`outputs/joint_rift_models/stage1_train210_uniform/` |
| HV4 train210 seed0 | 是，当前处理训练 | train210 × 2 | `outputs/joint_rift_stage1_train210_hv4_seed0_train/`、`outputs/joint_rift_models/stage1_train210_hv4/` |
| Uniform/HV4 holdout10 eval seeds0–3 | 是，但仅训练 seed0 | holdout10 × 2 × 4 eval seeds | `outputs/holdout10_*_{policy,independent}_seed*/` |
| 多 seed 汇总/DiD | 是，当前主统计 | 8 个 comparison JSON | `outputs/holdout10_*_multiseed_summary.json`、`outputs/holdout10_hv4_vs_uniform_multiseed.json` |

### 14.13 下一版本的命名与归档要求

新实验禁止复用旧 output/model 目录。推荐格式：

```text
outputs/<YYYYMMDD>_<experiment>_trainseed<TRAIN_SEED>_<config-hash>/
outputs/<YYYYMMDD>_<experiment>_eval_trainseed<TRAIN_SEED>_evalseed<EVAL_SEED>/
outputs/manifests/<run-id>.json
```

每一个 manifest 至少记录：

```text
source revision / diff hash
config path + SHA256
route XML + SHA256
map/speed source path
train seed, eval seed, repetitions
GPU, ports, CARLA version, Python/Torch version
explicit checkpoint absolute path + SHA256
run start/end time, exit code
effective audit episodes and checkpoint episodes
```

### 推荐在新会话中的第一条指令

```text
请先完整阅读 JOINT_RIFT_EXPERIMENT_AUDIT_2026-10-11.md。
不要假设 score_composed 越高越好；先根据 RIFT/CBV 的实际任务目标确认
CARLA 指标方向。当前结论是 HV4 已正确训练但未证明稳健改进，不能启动
Stage 2。下一步先做 route-level mechanism audit 和可复现 manifest，再决定
是否进行多 training-seed 重训或调整 HV 定义。
```

