# Joint-RIFT：双 CBV 联合候选评分与协同交互训练方案

## 1. 文档目标

本文档集合把当前讨论收敛为一个可以逐步实现、逐步否证、逐步训练的研究方案。目标不是简单地“同时控制两个 CBV”，而是验证并实现：

> 在两个已经具备单车 realistic/adversarial 能力的 RIFT-CBV 上，增加一个 **联合候选 residual scorer**，显式学习两个未来行为模式之间不可由独立评分分解的协同关系，并用 joint counterfactual evaluation + RIFT 式 group-relative RL 训练该联合评分。

研究问题是：

> 相比两个 RIFT 独立运行，联合决策是否能选择出对 Ego 形成更强、持续且安全的联合交互压力的候选组合，并且这种增益确实来自 pair-wise coordination，而不是简单地把两个高风险单车轨迹叠加？

---

## 2. 当前已经拍板的初版设计

1. **暂定固定 2 个 CBV**，一个 joint segment 内锁定 pair identity。
2. 两辆 CBV 都从 **RIFT checkpoint** 出发，而不是纯 Pluto checkpoint。
3. Pluto/RIFT 仍分别为每个 CBV 生成全部有效 `R × 12` 候选，初版尽量不做 aggressive top-K pruning。
4. Joint policy 不是替换两个 RIFT scorer，而是：

   \[
   S_{ij}=\ell^1_i+\ell^2_j+\Delta_{ij}
   \]

   其中 `ℓ1, ℓ2` 是两个单车 RIFT 的 log-probability / calibrated score，`Δij` 是新增 joint residual。
5. **Stage 1：冻结 Pluto + RIFT pi_head，只训练 joint residual head。**
6. **Stage 2：继续训练 joint head，同时解冻 RIFT 的共享 `planning_decoder.pi_head`。** Pluto trajectory generator 和产生 candidate latent 的主体仍冻结。
7. 初版 Ego 固定使用 `pdm_lite`，并向 joint evaluator 暴露其**当前 nominal trajectory**；暂不做 ego-response surrogate，也暂不追求 ego-policy black-box 泛化。
8. Ego nominal trajectory 初版从 PDM-Lite 自己的 `remaining_route + current speed + target speed` 构造 40 帧、0.1 s 的时间参数化轨迹。当前 RIFT 代码中 PDM-Lite 没有现成 40-step trajectory 输出，因此这是一个显式 adapter，而不是 learned surrogate。
9. 交互语义初版保持简单：参考 BeTop 的 trajectory interaction / braid event，得到候选与 Ego nominal trajectory 的**交互事件时刻**；arrival order 只用于分析抢行/让行方向，不强制作为奖励目标。
10. 多次 crossing 不只取第一或最后：保留所有**分离的有效事件**，相邻连续 crossing 去重后，按 RIFT 当前 40-step return 的方式进行时间折扣累积。
11. 联合交互通过时间上的 union / coverage 构造，核心 pair reward 包括：
    - 两个 CBV 各自原有 RIFT realism/safety return；
    - joint interaction pressure；
    - coalition gain；
    - CBV1-CBV2 candidate-pair collision penalty。
12. 所有 joint pair 的 advantage 在当前 state 内归一化；loss **先 state 内平均，再 batch 平均**，避免 route 数量大、joint candidate 多的 state 占据不成比例的梯度。
13. 正式 RL 训练前先做 **candidate-pair feasibility / synergy diagnostic**；如果 joint reward 基本可分解、oracle joint gain 近零或 candidate space 缺乏协同行为，直接否定/修改 idea，而不是盲目训练。

---

## 3. 四份主体文档

- [01_METHOD_DESIGN.md](./01_METHOD_DESIGN.md)：方法定义、奖励、联合评分头、训练阶段、为什么它不同于多个 RIFT 独立运行。
- [02_CODE_IMPLEMENTATION.md](./02_CODE_IMPLEMENTATION.md)：严格基于当前 RIFT 代码调用链，逐文件、逐阶段说明怎么改。
- [03_EXPERIMENT_TEST_PLAN.md](./03_EXPERIMENT_TEST_PLAN.md)：从 unit test、离线 feasibility，到 Stage-1 / Stage-2 RL 和最终对比实验的完整顺序。
- [04_OPEN_QUESTIONS_AND_GO_NO_GO.md](./04_OPEN_QUESTIONS_AND_GO_NO_GO.md)：仍待实验确定的参数、风险、失败判据和下一阶段扩展。

---

## 4. 论文层面的最简核心表达

现有多个 RIFT 独立运行，相当于使用 factorized joint policy：

\[
\pi_{ind}(i,j|s)=\pi_1(i|s)\pi_2(j|s).
\]

本文方法学习：

\[
\pi_J(i,j|s)\propto
\pi_1(i|s)\pi_2(j|s)
\exp\{\Delta_\theta(i,j,s)\}.
\]

`Δθ` 只负责学习独立模型无法表示的 pair compatibility / synergy。

因此核心贡献不是“两个攻击车”，而是：

> **在 RIFT 的 realistic candidate manifold 上，显式学习非可分解的多车联合行为偏好。**

---

## 5. 最关键的 go / no-go 条件

在写 RL trainer 前，应先离线穷举 joint candidate pairs，回答：

1. 独立 RIFT 选择与 joint oracle 之间是否存在稳定、显著的 Oracle Joint Gain？
2. Joint reward 的二维矩阵是否具有明显的 non-factorizable residual，而不是近似 `f(i)+g(j)`？
3. 最优 joint pair 是否经常需要两辆 CBV 都贡献（Both-Necessary）？
4. Oracle pair 是否仍位于 RIFT 候选分布的合理支持区域，而不是永远由极低概率异常 mode 组成？
5. BeTop-inspired interaction event 是否足够密集；如果极度稀疏，是否需要软化为 PET/TTC 或距离相关的 continuous interaction signal？

只有这些条件成立，才进入正式 Joint-RIFT RL。

---

## 6. 主要代码与论文来源

### RIFT
- Repository: https://github.com/CurryChen77/RIFT
- `rift/carla_runner.py`
- `rift/ego/pdm_lite/pdm_lite.py`
- `rift/ego/pdm_lite/autopilot.py`
- `rift/cbv/planning/pluto/model/modules/planning_decoder.py`
- `rift/cbv/planning/pluto/model/pluto_model.py`
- `rift/cbv/planning/fine_tuner/rlft/rift_pluto/rift_pluto.py`
- `rift/cbv/planning/fine_tuner/rlft/rift_pluto/rift_trainer.py`
- `rift/cbv/planning/fine_tuner/rlft/rift_pluto/rift_datamodule.py`
- `rift/cbv/planning/fine_tuner/rlft/traj_eval/traj_evaluator.py`
- `rift/gym_carla/buffer/cbv_rollout_buffer.py`

### BeTop
- Paper: https://proceedings.neurips.cc/paper_files/paper/2024/hash/a862f5788fd09bb6843c694d8120d50c-Abstract-Conference.html
- Code: https://github.com/OpenDriveLab/BeTop
- 关键代码：`womd/betopnet/utils/topo_utils.py`

### AWM / Multi-agent coalition inspiration
- Paper: https://arxiv.org/abs/2607.10630
- 重点：Section 3.2 的 sparse coalition、leave-one-out counterfactual credit、pair gain / calibration。
