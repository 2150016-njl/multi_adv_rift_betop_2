
# Joint-RIFT / Multi-CBV 协同交互评分方案：完整技术背景、设计演化、代码实现与实验交接文档

> 本文用于在一个新的 ChatGPT 对话中完整恢复当前课题的技术背景、设计决策、代码状态与后续工作。  
> 目标不是做聊天摘要，而是形成一个可直接继续研发、实验和写论文的技术交接文档。

---

# 0. 项目目标与当前结论

本课题基于：

- RIFT: https://github.com/CurryChen77/RIFT
- BeTop: https://github.com/OpenDriveLab/BeTop
- 当前扩展代码仓库：
  https://github.com/2150016-njl/multi_adv_rift_betop_2/tree/main/RIFT

目标是在 RIFT 的单 CBV 轨迹候选排序基础上，扩展到 **2 辆协调控制的 CBV（Coordinated Background Vehicles）**，学习一个 **joint residual scoring head**，在不重新生成轨迹的前提下，从两个 CBV 各自的 Pluto/RIFT 候选笛卡尔积中选择更具有协同交互价值的联合轨迹对。

最终希望学习的联合分布为：

\[
\log \pi^J_\theta(i,j|s)
=
\log \pi_1^{RIFT}(i|s)
+
\log \pi_2^{RIFT}(j|s)
+
\delta_\theta(i,j,s)
-
\log Z
\]

或对应联合 score：

\[
S_{ij}
=
\ell_i^1
+
\ell_j^2
+
\delta_{ij}
\]

其中：

- \(i\)：CBV1 某个候选轨迹；
- \(j\)：CBV2 某个候选轨迹；
- \(\ell_i^1,\ell_j^2\)：两个单车 RIFT scorer 的 log-prob；
- \(\delta_{ij}\)：只表示两辆车联合组合带来的不可分解协同修正。

一个非常重要的设计要求是：

\[
\delta_{ij}=0
\]

时：

\[
\pi_J(i,j)
=
\pi_1(i)\pi_2(j)
\]

即 **严格退化为两辆独立 RIFT 的乘积分布**。

这使得论文能够清楚回答：

> “Joint-RIFT 相比两辆独立 RIFT，到底学到了什么？”

当前总体结论：

1. 这个 idea 在理论上是成立的；
2. 核心成立条件是 joint reward 必须具有显著 **non-factorizable / non-additive** 部分；
3. 当前代码架构已经实现大部分 Joint-RIFT；
4. 原实现存在两个核心方法偏差：
   - fresh Joint-RIFT 默认从 IL Pluto 而不是 RIFT checkpoint 初始化；
   - interaction reward 只看路径空间 crossing，不看到达冲突点的时间接近；
5. 已经准备了 audited 修复方案和完整实验流程；
6. 下一阶段最重要的不是盲目继续训练，而是通过 feasibility diagnostics 先验证：
   - 候选空间中是否真的存在联合协同机会；
   - 这些机会是不是独立 RIFT 难以选到；
   - joint reward 是否真的不可分解。

---

# 1. 原始 RIFT 的技术路线分析

## 1.1 RIFT 本质：不训练轨迹生成器，而训练 Pluto 候选评分头

RIFT 使用 Pluto 作为基础规划网络。

Pluto 对每个 CBV 当前状态构造若干 reference line：

\[
R = \text{当前局部候选 reference line 数量}
\]

对每条 reference line，Pluto 生成固定：

\[
M=12
\]

个轨迹 mode。

因此每个 CBV 每个 state 有：

\[
G = R \times 12
\]

个候选。

Pluto PlanningDecoder 输出：

\[
trajectory
\in
[B,R,12,80,6]
\]

和：

\[
logits
\in
[B,R,12]
\]

其中 future_steps 默认 80，对应 10 Hz 下约 8 秒。

RIFT 实际 virtual evaluator 只使用前 40 帧：

\[
T=40
\]

即约 4 秒。

---

## 1.2 RIFT 的候选 reference line

RIFT / Pluto 不是每个 reference line 都单独跑一次完整网络。

实际：

1. `CBVRoutePlanner.build_reference_line()`
2. `query_reference_lines()`
3. 根据当前 lane、相邻 lane、route topology 生成变量数量的局部 reference lines；
4. 一次 Pluto forward 同时处理所有 \(R\) 条 reference line；
5. 网络中存在：
   - route-to-route attention；
   - mode-to-mode attention；
   - scene cross-attention；
6. 输出所有 \(R\times12\) 候选。

因此：

> R 是局部 reference line 数，不是多个全局路线。

---

## 1.3 Pluto 推理执行与 RIFT virtual training 的区别

### 实际执行

原 Pluto/RIFT 推理 `_trim_candidates` 会：

1. flatten \(R\times12\)；
2. 按 raw score 排序；
3. 只保留 top-k=10；
4. top-k 内 softmax；
5. 选 argmax。

但由于全局 raw-score 最大值必然进入 top-k，因此最终实际执行轨迹仍等价于：

\[
\arg\max_{r,m} z_{r,m}
\]

只要 top-k 没有其它特殊过滤。

### RIFT virtual evaluator

RIFT 训练时 **不是只评 top-k**。

它会对所有 valid：

\[
R\times12
\]

候选计算 virtual rollout return。

这是 Joint-RIFT 第一版选择“尽量完整保留候选笛卡尔积”的重要依据。

---

## 1.4 RIFT TrajEvaluator

原 RIFT：

- `TrajEvaluator(dt=0.1, num_frames=40, sample_interval=5)`
- 候选 CBV：`TrackPropagate`
- 其它 nearby actor：`KinematicBicycleModel`
- 计算：
  - collision
  - offroad
  - lane relation
  - velocity
  - acceleration
  - comfort 等 dense reward
- 折扣：

\[
\gamma=0.98
\]

得到每个 candidate 的 discounted return：

\[
G_i
\]

然后在同一 state 内对所有候选做 group-relative 标准化：

\[
A_i
=
\frac{G_i-\mu_G}
{\sigma_G+10^{-5}}
\]

特点：

- 没有 critic；
- 没有 value network；
- 没有 GAE；
- advantage 完全是 virtual candidate group 内相对值。

---

## 1.5 RIFT buffer / trainer

关键训练逻辑：

- buffer capacity：4096；
- 原 RIFT 按 per-CBV trajectory 暂存；
- CBV done 后 flush；
- trajectory 长度 <=5 会忽略；
- train/val ≈ 90/10；
- batch 256；
- epoch 16；
- 每次更新后 inference model 同步当前 checkpoint。

最关键的是：

> RIFT 只训练 `planning_decoder.pi_head`，轨迹 generator 冻结。

因此 RIFT 本质可以理解为：

> 基于 frozen Pluto candidate set，学习一个 adversarial trajectory ranker。

---

## 1.6 RIFT loss

原 RIFT 把当前和 old logits：

\[
[R,12]
\]

flatten：

\[
G=R\times12
\]

然后：

\[
\log\pi
=
\log softmax(z_{1:G})
\]

得到：

\[
\rho_i
=
\frac{\pi_\theta(i|s)}
{\pi_{old}(i|s)}
\]

使用 PPO-style clipping：

\[
\epsilon=0.2
\]

并使用 negative advantage dual clip：

\[
c=3
\]

注意：

RIFT 虽然形式类似 PPO，但它并不是传统 sampled-action PPO。

因为同一个 state 下：

- 所有 candidate 都被 evaluator 枚举；
- 所有 candidate 都获得 advantage；
- 实际 CARLA 执行只是 argmax。

更严谨的表述是：

> RIFT-style group-relative clipped policy optimization / surrogate objective。

---

# 2. 为什么 Joint-RIFT 不是简单“两个独立 RIFT”

假设联合 reward 完全可加：

\[
Q(i,j)
=
Q_1(i)+Q_2(j)
\]

那么：

\[
\arg\max_{i,j}Q(i,j)
=
\left(
\arg\max_iQ_1(i),
\arg\max_jQ_2(j)
\right)
\]

此时训练 joint head 没有意义。

因此 Joint-RIFT idea 成立的数学必要条件是：

\[
Q_{joint}
=
Q_1
+
Q_2
+
Q_{coord}(i,j)
\]

其中：

\[
Q_{coord}(i,j)
\]

必须具有不可分解 pair interaction。

这也是整个论文 idea 最核心的逻辑。

---

# 3. 双 CBV 环境是否已有基础支持

RIFT 原环境本身已经支持同时 active 多个 CBV。

rule config 原来：

- train 最大 CBV 数可大于 1；
- eval 也支持 2；

当前 Joint-RIFT 代码已经把：

```yaml
max_agent_num: 2
```

固定为 2。

Observation 构造会：

- 按 ego 获取所有 CBV；
- 每个 CBV 单独构造 Pluto feature；
- nearby actor 中可以包含另一个 CBV 和 ego。

Action application 本身也循环所有 active CBV。

因此：

> 不需要重写 CARLA 环境才能控制两辆 CBV。

真正需要扩展的是：

- policy；
- joint evaluator；
- buffer；
- trainer；
- diagnostics。

---

# 4. Joint-RIFT 最终技术路线

## 4.1 Action space

CBV1：

\[
\mathcal I_1
=
\{\tau_i^1\}_{i=1}^{G_1}
\]

其中：

\[
G_1=R_1\times12
\]

CBV2：

\[
\mathcal I_2
=
\{\tau_j^2\}_{j=1}^{G_2}
\]

其中：

\[
G_2=R_2\times12
\]

联合 action：

\[
a=(i,j)
\]

联合候选空间：

\[
G_1\times G_2
\]

例如：

\[
G_1=60,\quad G_2=60
\]

则：

\[
3600
\]

个 candidate pairs。

这个量级第一版完全可以保留。

---

# 5. 为什么第一版不建议 top-K

用户最终同意：

> 第一版尽量完整保留 candidate set。

原因：

1. Pluto mode 数只有 12；
2. reference line 数通常有限；
3. \(60\times60=3600\) 不是不可接受；
4. 不应该在证明 joint opportunity 之前先把低单车概率但高协同 candidate 删除；
5. 如果 oracle pair 总在单车低 rank 位置，这本身就是很重要的 scientific diagnostic。

性能优化方式不是：

> 两层 for-loop 3600 次分别 rollout。

而是：

1. CBV1 全部 \(G_1\) candidate 各 rollout 一次；
2. CBV2 全部 \(G_2\) candidate 各 rollout 一次；
3. pair geometry broadcast：
   \[
   G_1G_2T
   \]
4. 比如：
   \[
   60\times60\times40
   =
   144000
   \]
   时间 pair entries。

这个规模可接受。

---

# 6. Joint score 的最终设计

最终核心：

\[
S_{ij}
=
\ell_i^1
+
\ell_j^2
+
\delta_\theta(i,j,s)
\]

其中建议：

\[
\ell_i^1
=
\log softmax(z_1)_i
\]

而不是直接用 raw logits。

原因：

- 两辆不同 CBV 的 raw logit scale 可能不同；
- log probability 更容易比较；
- 使独立 factorized prior 有清晰 probabilistic 解释。

---

# 7. 为什么 joint residual 不能只做两个 Pluto latent 点积

两个 CBV 的 Pluto feature 分别位于自己的 local frame。

即使网络参数共享：

\[
q_i^1
\]

和：

\[
q_j^2
\]

也不能保证几何坐标含义天然对齐。

因此最终设计为：

### Candidate representation

每个 candidate：

\[
e_i
=
Fuse(
q_i,
TrajEncoder(\tau_i^{common})
)
\]

其中：

- \(q_i\)：Pluto decoder candidate latent；
- \(\tau_i^{common}\)：转换到共同 frame 的 candidate trajectory；
- common frame 当前采用 ego-centric / common global frame。

### Pair interaction

当前实现：

\[
u_i=W e_i
\]

\[
v_j=W e_j
\]

共享投影后：

\[
compatibility_{ij}
=
\frac{u_i^\top v_j}{\sqrt d}
\]

另加入：

\[
d_{ij}^{min}
\]

即两条候选轨迹未来最小距离。

最后：

\[
raw\_\delta_{ij}
=
MLP(
compatibility_{ij},
geometry_{ij}
)
\]

这是一个较简单但合理的 V1。

---

# 8. BeTop 对 Joint-RIFT 的启发

BeTop 不是直接作为完整 reward 使用。

## 8.1 BeTop low-level braid signal

BeTop 的 topology utilities 会：

- 判断 trajectory segment 之间的拓扑 crossing；
- 底层保留 per-time segment signal；
- 最终常常压成 horizon-level Boolean interaction。

因此 BeTop 最重要的启发是：

> interaction 可以从未来轨迹之间的 topology / conflict relation 构造，而不一定依赖 ego response model。

## 8.2 BeTop JFPScorer

BeTop 的 `JFPScorer` 对两个 agent 的 multimodal trajectories：

1. 分别做 trajectory encoding；
2. 构建 mode × mode Cartesian product；
3. pairwise fuse；
4. 输出 joint score matrix。

这为 Joint-RIFT 的：

\[
G_1\times G_2
\]

联合评分矩阵提供了直接架构先例。

---

# 9. Ego trajectory 的最终 V1 决策

最初讨论过：

- ego reachable tube；
- nominal trajectory；
- ego response surrogate。

最终用户明确选择：

> V1 不使用 reachable tube。

使用：

> PDM-Lite 当前 nominal trajectory。

原因：

1. 方法先保持简单；
2. 不希望引入额外 ego-response surrogate；
3. 当前先允许 CBV evaluator 使用当前 PDM-Lite planning intent；
4. 后续再考虑把 ego policy 黑盒化。

---

# 10. PDM-Lite nominal trajectory 的实际实现

PDM-Lite 当前 controller 原本只直接输出：

- throttle；
- steer；
- brake。

并没有原生：

\[
40\times trajectory
\]

输出。

最初讨论过直接复用：

```python
forecast_ego_agent()
```

但审查后放弃。

原因：

该 function 明确带有：

> assume no hazard

等假设，而且 longitudinal extrapolation 并不能忠实表达当前正在 braking 的 PDM-Lite closed-loop future。

因此最终选择：

- 当前 ego state；
- `remaining_route`；
- current speed；
- 当前 frame 最终 `target_speed`；

做一个轻量 route-and-speed time parameterization。

输出：

\[
[40,4]
=
[x,y,heading,speed]
\]

其定位必须写清楚：

> 这是 PDM-Lite 当前 planning intent，不是两辆 hypothetical CBV 行为之后的真实 ego closed-loop response。

---

# 11. PDM nominal trajectory 当前时间参数化

当前 adapter：

\[
a_t
=
clip
\left(
\frac{v_{target}-v_t}{\tau},
-a_{decel},
a_{accel}
\right)
\]

\[
v_{t+1}
=
\max(0,v_t+a_t\Delta t)
\]

route arc-length：

\[
s_{t+1}
=
s_t
+
\frac{v_t+v_{t+1}}{2}\Delta t
\]

当前实现参数大致：

- horizon：40；
- dt：0.1s；
- speed response constant：1.0；
- max accel：2.0；
- max decel：4.0。

这些不是论文最终参数。

必须做实际 overlay sanity check。

---

# 12. Interaction semantics 的多轮讨论与最终决策

## 12.1 用户想要的语义

不是：

> 一辆 assertive、一辆 yielding。

而是：

> 两辆车对 ego 形成更高 joint interaction / pressure。

不预先规定方向。

因此应该分开：

### Interaction magnitude

\[
q_k(t)
\in[0,1]
\]

表示 interaction relevance / pressure intensity。

### Direction

可选：

\[
d_k(t)\in[-1,1]
\]

表示：

- CBV 更早到；
- Ego 更早到。

方向不作为主要优化目标。

---

# 13. 为什么不强制 assert/yield

若 reward 直接规定：

- CBV1 必须 assert；
- CBV2 必须 yield；

会把协同压成一个人工 scripted role pattern。

这可能：

1. 限制行为模式；
2. 错过：
   - relay pressure；
   - simultaneous bottleneck；
   - two-sided constraint；
3. 让方法变成 hand-crafted role assignment。

所以最终：

> high interaction 是主目标，arrival order 主要用于 diagnostic / direction annotation。

---

# 14. 40 帧里多个 crossing 怎么处理

用户问：

> 如果 40 帧中有多个 crossing，是否可以像 RIFT 当前 reward 一样按时间加权？

最终结论：

> 可以，而且这是比复杂 persistent behavior state 更适合 V1 的做法。

设计：

1. 保留所有真正分离的 crossing event；
2. 连续相邻 segment 的 crossing 合并为同一个 event；
3. 每个 cluster 只保留一次 onset；
4. 再做：

\[
P
=
\sum_e
\gamma^{t_e}w_e
\]

其中：

\[
\gamma=0.98
\]

与原 RIFT 一致。

这保持简单。

---

# 15. 原 interaction 实现的问题：只看空间 crossing

当前代码最初实现：

> CBV 当前 segment 与 Ego 未来任意 segment 在 XY 空间相交，就算 interaction event。

也就是说：

CBV 0.5s 到冲突点：

\[
t_C=0.5
\]

Ego 3.5s 才到：

\[
t_E=3.5
\]

依然记为 interaction。

这更像：

> future route/path conflict relevance

而不是：

> real temporal interaction pressure。

因此这是必须修复的 P0。

---

# 16. 最终 interaction V1：arrival-time closeness weighted crossing

保持原空间 conflict detector，但给每个 event 时间权重：

\[
\Delta t_e
=
|t_C-t_E|
\]

定义：

\[
w_e
=
\exp
\left(
-\frac{\Delta t_e}{\tau}
\right)
\]

V1 默认：

\[
\tau=1.0s
\]

然后：

\[
P_k
=
\sum_e
\gamma^{t_e}w_e
\]

好处：

- 保留 BeTop-inspired path/topology relevance；
- 保留 multiple crossings；
- 保留 RIFT 时间折扣；
- 不引入 TTC/PET 大杂烩；
- 不引入 reachable tube；
- 不引入 ego surrogate；
- 解决“路径交叉但时间差太大”的假 interaction。

---

# 17. Joint interaction / coalition reward

设：

\[
P_1(i)
\]

表示 CBV1 candidate 对 Ego 的 temporal interaction pressure；

\[
P_2(j)
\]

同理。

联合 temporal coverage：

\[
P_{12}(i,j)
=
\sum_t
\gamma^t
\left[
1-
(1-q_1(t))
(1-q_2(t))
\right]
\]

若 interaction trace 已是 weighted event：

\[
q_k(t)\in[0,1]
\]

即可直接使用 soft union。

---

# 18. Coalition Gain

定义：

\[
CG_{ij}
=
P_{12}(i,j)
-
\max(
P_1(i),
P_2(j)
)
\]

含义：

> 加入第二辆 CBV 后，联合 pressure 是否比最强单车 interaction 更高。

这个 metric 是 Joint-RIFT 与“双独立 RIFT”差异的重要证据。

---

# 19. Leave-one-out / Both-Necessary

定义：

\[
\Delta_1
=
P_{12}-P_2
\]

\[
\Delta_2
=
P_{12}-P_1
\]

如果：

\[
\Delta_1>\epsilon
\]

且：

\[
\Delta_2>\epsilon
\]

则两辆 CBV 都对联合 pressure 有必要贡献。

定义：

\[
BNR
=
P(
\Delta_1>\epsilon
\land
\Delta_2>\epsilon
)
\]

Both-Necessary Rate 是证明“coordination，而不是仅仅出现两辆车”的重要 metric。

---

# 20. AWM / sparse coalition 的启发

用户曾提出：

> coordination semantics 不一定只依赖 BeTop。

讨论过 2026 年的一类 “World Models as Adversaries / sparse coalition” 工作。

核心可借鉴思想：

- multi-agent adversary 有 credit ambiguity；
- 不应该默认所有 agent 都有价值；
- 应该通过 leave-one-out / counterfactual contribution 衡量 agent 对 coalition 的额外作用；
- coalition 应该是 sparse / scene-adaptive。

Joint-RIFT 当前采用：

\[
P_{12}-P_2
\]

和：

\[
P_{12}-P_1
\]

正好符合这种思想。

---

# 21. Joint reward 的最终 V1

建议：

\[
Q_{ij}
=
U_1(i)
+
U_2(j)
+
\lambda_P P_{12}(i,j)
+
\lambda_C CG_{ij}
-
\lambda_{pair}C_{ij}
\]

其中：

### \(U_1,U_2\)

直接复用 RIFT 原 TrajEvaluator：

- lane relation；
- speed；
- comfort；
- collision；
- offroad；
- background actors。

### \(P_{12}\)

联合 interaction pressure。

### \(CG\)

coalition gain。

### \(C_{ij}\)

CBV1 与 CBV2 candidate trajectories 的 pair collision penalty。

---

# 22. 为什么必须排除“另一辆受控 CBV”的 stale rollout

对 CBV1 candidate \(i\) 做 unary evaluation 时：

如果 nearby actor 中仍然把 CBV2 按当前 control 外推：

\[
\tilde\tau_2
\]

同时 joint candidate 又指定：

\[
\tau_j^2
\]

就会出现同一辆 CBV2 两套未来。

因此当前正确实现是：

CBV1 unary：

exclude：

- ego；
- CBV1 self；
- controlled CBV2。

CBV2 unary 同理。

CBV1-CBV2 interaction 只在 pair \(i,j\) 层计算。

这个设计是正确的，应该保留。

---

# 23. Pair collision 的计算

不重新 rollout 每个 pair。

两组 candidate 分别得到：

\[
V_1
\in
[G_1,T,4,2]
\]

\[
V_2
\in
[G_2,T,4,2]
\]

然后 broadcast：

\[
[G_1,G_2,T,4,2]
\]

用 SAT / oriented rectangle collision 判断：

\[
C_{ij}\in\{0,1\}
\]

这是高效且合理的。

---

# 24. Joint residual 的数值尺度问题

用户特别强调：

> \(\delta_{ij}\) 不能相比 \(\ell_i+\ell_j\) 小到几乎不起作用。

因此当前 scorer 使用：

\[
\sigma_{base}
=
Std(
\ell_i+\ell_j
)
\]

然后：

\[
\delta_{ij}
=
\lambda_\delta
\sigma_{base}
\tanh(raw\_\delta_{ij})
\]

使 residual 的 scale 与当前 state 的 base-score spread 同量级。

当前：

\[
\lambda_\delta=1
\]

优点：

- residual 不会天然过小；
- 也不会无界压过 RIFT prior。

需要监控：

\[
\frac{Std(\delta)}{Std(base)}
\]

即：

```text
residual_base_ratio
```

---

# 25. 一个 residual scale 后续风险

如果某 state：

\[
\sigma_{base}\approx0
\]

则：

\[
\delta\approx0
\]

且 gradient 也会被压缩。

因此后续应统计：

```text
base_std histogram
```

若大量 state 很小，再考虑：

\[
\sigma_{eff}
=
max(\sigma_{base},\sigma_{min})
\]

但第一版不建议直接加复杂 floor。

---

# 26. Pair symmetry / permutation invariance

为了避免 actor ID 产生 role bias：

- 两个 CBV 使用共享 candidate encoder；
- 共享 pair projection；
- compatibility 使用：
  \[
  u_i^\top u_j
  \]
- geometry 使用 symmetric distance。

这样交换两辆 CBV 后：

\[
S_{12}
\approx
S_{21}^T
\]

是合理的。

PairManager 按 actor ID 排序只是 canonical storage order，不是 behavior role。

---

# 27. PairManager

第一版 pair 逻辑：

1. active CBV >=2 时选 2 辆；
2. pair 在 segment 内锁定；
3. 只要两辆都 active，pair 不变；
4. 任一离开，当前 segment 结束；
5. 新 segment 再选新的 pair。

用户明确同意：

> 第一版不要做动态 coalition membership。

---

# 28. Stage 1 / Stage 2 训练设计

## Stage 1

必须：

- 从训练好的 RIFT checkpoint 初始化；
- freeze Pluto/RIFT 全部参数；
- 只训练 joint residual scorer。

因此：

\[
\theta_{RIFT}
\]

固定，

只训练：

\[
\theta_{joint}
\]

优点：

> 能明确证明提升来自 coordination residual，而不是重新做了一次单车 adversarial tuning。

---

## Stage 2

用户最终建议：

> 第二阶段应该让 RIFT 原 scorer 也训练，而不是永远 freeze。

最终决定：

Stage 2：

- joint head：train；
- `planning_decoder.pi_head`：train；
- trajectory generator：仍 freeze；
- encoder：freeze；
- loc/yaw/vel heads：freeze。

原因：

如果解冻 trajectory generator：

\[
\tau_i^{old}
\neq
\tau_i^{new}
\]

那么 old/new PPO ratio 的 candidate identity 不再一致。

因此 Stage 2 只能调整：

\[
z_i
\]

不能修改 candidate geometry。

---

# 29. Stage 2 学习率

当前建议：

joint head：

\[
lr_{joint}\approx10^{-4}
\]

pi head：

\[
lr_{pi}\approx10^{-5}
\]

即让单车 RIFT scorer 适配联合任务，但比 joint residual 更保守。

---

# 30. Joint PPO / RIFT-style loss

old score：

\[
S^{old}_{ij}
=
\ell_i^{1,old}
+
\ell_j^{2,old}
+
\delta^{old}_{ij}
\]

old distribution：

\[
\pi_{old}(i,j|s)
=
softmax_{ij}
(S^{old}_{ij})
\]

current：

\[
S^\theta_{ij}
=
\ell_i^{1,\theta}
+
\ell_j^{2,\theta}
+
\delta^\theta_{ij}
\]

\[
\pi_\theta
=
softmax_{ij}(S^\theta)
\]

joint return：

\[
Q_{ij}
\]

group advantage：

\[
A_{ij}
=
\frac{
Q_{ij}-\mu_Q
}{
\sigma_Q+\epsilon
}
\]

ratio：

\[
\rho_{ij}
=
\frac{
\pi_\theta(i,j|s)
}{
\pi_{old}(i,j|s)
}
\]

然后使用 RIFT 同样的：

- PPO clip 0.2；
- negative dual clip 3。

---

# 31. 为什么是 full joint softmax

不能：

- 对 i 单独 softmax；
- 对 j 单独 softmax；
- 分别训练两个 PPO。

Joint action 本身就是：

\[
(i,j)
\]

因此 categorical action space 是：

\[
G_1\times G_2
\]

所以必须：

\[
softmax(
S_{ij}
)
\]

flatten over all valid pairs。

---

# 32. Variable \(G_1,G_2\) / padding

每个 state：

\[
G_k
=
R_k\times12
\]

不同。

Joint collate：

- pad CBV1 candidate axis；
- pad CBV2 candidate axis；
- valid mask：

\[
M_{ij}
=
M_i^1
\land
M_j^2
\]

invalid pair score：

\[
-\infty
\]

不参与 softmax。

---

# 33. State-balanced loss

不能把 batch 中所有 valid pairs 直接平均。

否则 candidate 多的 state 权重大。

最终：

\[
L_s
=
-\frac{1}{|\mathcal A_s|}
\sum_{ij}
\psi_{ij}
\]

然后：

\[
L
=
\frac1B
\sum_sL_s
\]

当前实现已符合。

---

# 34. 为什么 Joint buffer 应按 state 存

原 RIFT buffer 是 per-CBV trajectory flush。

Joint-RIFT 没有：

- GAE；
- critic；
- actual multi-step on-policy return。

每个 state 已经完整包含：

- all pair candidates；
- all pair returns；
- all pair advantages。

因此直接存：

```text
feature_1
feature_2
old_joint_logits
joint_advantage
valid_mask_1
valid_mask_2
pair_ids
frame_context
diagnostics
collection_stage
collection_checkpoint
```

更加自然。

---

# 35. On-policy provenance

当前 Joint buffer 正确记录：

- `collection_stage`
- `collection_checkpoint`

如果：

Stage 1 buffer → Stage 2 trainer

直接拒绝。

如果：

checkpoint A collected buffer → checkpoint B trainer

也拒绝。

这是非常重要的 safeguard。

---

# 36. 当前代码仓库实现概况

仓库：

https://github.com/2150016-njl/multi_adv_rift_betop_2/tree/main/RIFT

当前已存在：

```text
rift/cbv/planning/fine_tuner/rlft/joint_rift/
├── __init__.py
├── geometry.py
├── interaction_topology.py
├── joint_action.py
├── joint_buffer.py
├── joint_datamodule.py
├── joint_rift_pluto.py
├── joint_score_head.py
├── joint_trainer.py
└── joint_traj_evaluator.py
```

说明：

> 当前已经不是“只有 Stage 1A 接口”的状态，而是完整 Joint-RIFT prototype。

---

# 37. 当前实现中已经正确的部分

## 37.1 Pluto candidate feature

PlanningDecoder 已暴露：

```python
candidate_feature = q
```

位置正确：

- decoder layers 后；
- `cat_x_proj` 后；
- `pi_head` 前。

这正是 joint scorer 应使用的 candidate semantic feature。

---

## 37.2 Full candidate groups

当前 Joint-RIFT 不调用原：

```python
_trim_candidates(topk=10)
```

而是保留：

\[
valid_R\times12
\]

全部 candidate。

符合最终设计。

---

## 37.3 Common frame

每个 candidate 都有：

- local trajectory；
- global trajectory；
- ego-centric/common trajectory。

Joint head 使用 common trajectory，而不是直接拿两个 local-frame latent 做几何比较。

正确。

---

## 37.4 Joint head

当前：

- candidate latent encoder；
- trajectory encoder；
- shared fusion；
- compatibility dot product；
- minimum-distance geometry；
- residual MLP；
- zero-init residual projection。

结构合理。

---

## 37.5 Residual initialization

最后 residual projection：

- weight zero；
- bias zero。

因此刚初始化：

\[
\delta=0
\]

严格退化为 factorized base policy。

正确。

---

## 37.6 Joint TrajEvaluator

复用 RIFT unary evaluator，并额外增加：

- PDM nominal Ego collision；
- interaction pressure；
- coalition gain；
- CBV-CBV pair collision。

正确方向。

---

## 37.7 另一受控 CBV 排除

当前 unary evaluation 已正确排除另一个 controlled CBV。

这是重要实现细节。

---

## 37.8 Pair collision

使用 rollout vertices + SAT broadcast。

不是 3600 次重新 rollout。

正确。

---

## 37.9 Joint buffer / datamodule / trainer

当前已经实现：

- state-level storage；
- padding；
- full joint softmax；
- dual-clip；
- state-inner mean → batch mean；
- Stage 1 freeze；
- Stage 2 only pi-head unfreeze；
- checkpoint provenance。

这些都符合最终技术路线。

---

# 38. 原实现发现的 P0 问题 1：fresh Joint-RIFT 从 IL Pluto 启动

原 `joint_rift_pluto.yaml`：

```yaml
ckpt_path:
  rift/cbv/planning/model_ckpt/pluto/pluto_1M_aux_cil.ckpt
```

原 `RLFTPluto.load_model()` 在当前 joint policy 自己没有 checkpoint 时：

```python
self.checkpoint = self.config['ckpt_path']
```

所以 fresh Joint-RIFT 实际上是：

\[
IL~Pluto
+
Joint~Residual
\]

不是：

\[
RIFT
+
Joint~Residual
\]

这是方法级错误。

因为 joint residual 会同时承担：

- 单车 adversarial ranking；
- pair coordination。

这会破坏论文中：

> “coordination residual 建立在单车 RIFT adversarial prior 之上”

的解释。

---

# 39. P0 修复 1：强制 RIFT prior

已设计 audited policy：

```text
joint_rift_audited
```

Stage 1：

自动从：

```text
rift/cbv/planning/model_ckpt/rift_pluto/
<ego>-<recognition>-seed<seed>/
latest *.ckpt
```

初始化。

如果找不到 trained RIFT checkpoint：

> 直接 fail。

不能 silent fallback to Pluto。

---

# 40. 原实现 P0 问题 2：interaction path crossing 不等于 temporal pressure

已经在第 15 节说明。

P0 fix：

\[
w_e
=
e^{-|\Delta t|/\tau}
\]

并继续用：

\[
\gamma^{t_e}
\]

折扣。

这一改动应该保留原：

- cluster crossing；
- multiple crossing；
- sparse event representation。

而不是改成复杂 persistent state。

---

# 41. 原实现 P1：pair collision penalty 太小

原：

```yaml
lambda_pair_collision: 1
```

pair collision 是：

\[
0/1
\]

而 RIFT unary collision cost 通常量级显著更大。

所以：

\[
-1
\]

可能不足以避免：

> 为了提高 interaction 而选择两辆 CBV 互撞。

audited V1 建议：

\[
\lambda_{pair}=20
\]

但这不是最终论文值。

必须通过 reward-scale audit 决定。

---

# 42. 原实现 P1：Runner 对所有 policy 传 ego nominal

当前 `CarlaRunner.train_cbv()` 无条件：

```python
self.cbv_policy.get_action(
    ...,
    ego_nominal_trajectories=...
)
```

而例如：

```python
GRPOPluto.get_action(...)
```

没有这个 keyword。

因此会破坏 baseline training。

修复原则：

只有：

- `joint_rift_pluto`
- `joint_rift_audited`

接收 nominal trajectory。

其它 policy 保持原接口。

---

# 43. 原实现 P1：vector batch 全体 fallback

原 Joint-RIFT：

如果任何 env 当前不是 strict 2-CBV pair：

```python
return super().get_action(...)
```

会让整个 vector batch 都回到 independent RIFT。

正确做法：

per-env：

```text
env0: 有2辆 → joint
env1: 只有1辆 → independent
```

不应相互影响。

---

# 44. 原实现 P1：eval 没有 joint diagnostics

原 eval 不传 nominal trajectory。

虽然 learned scorer 本身推理时不需要 nominal trajectory，但论文指标：

- \(P_1\)
- \(P_2\)
- \(P_{12}\)
- coalition gain
- Both-Necessary
- oracle gap
- temporal crossing gap

都需要 evaluator。

因此 audited eval 应输出：

```text
joint_rift_eval_metrics.jsonl
```

用于离线统计。

---

# 45. Feasibility diagnostics：正式训练前必须做

这是整个研究最重要的一步之一。

## 45.1 Candidate coverage

收集大量 2-CBV states。

例如：

\[
10k\sim50k
\]

但具体可根据算力调整。

对每个 state 生成所有：

\[
G_1G_2
\]

candidate pairs。

问：

> 候选笛卡尔积里有没有至少一个合理 safe/high-interaction pair？

如果没有：

> joint scorer 再强也没用。

此时应修改 candidate generator，而不是继续训练 scorer。

---

# 46. Non-factorizable residual diagnostic

对：

\[
Q_{ij}
\]

做 additive decomposition：

\[
\hat Q_{ij}
=
\bar Q_{i\cdot}
+
\bar Q_{\cdot j}
-
\bar Q
\]

定义：

\[
R_{ij}
=
Q_{ij}-\hat Q_{ij}
\]

统计：

\[
NonFactorizableRatio
=
\frac{
Var(R)
}{
Var(Q)+\epsilon
}
\]

如果长期：

\[
\approx0
\]

说明 reward 几乎 additive。

那么：

> Joint-RIFT 没有数学必要。

这是核心 go/no-go 指标。

---

# 47. Oracle Joint Gain

Independent RIFT：

\[
(i_{ind},j_{ind})
=
(
\arg\max_i \ell_i^1,
\arg\max_j \ell_j^2
)
\]

Oracle joint：

\[
(i^*,j^*)
=
\arg\max_{ij}Q_{ij}
\]

定义：

\[
OJG
=
Q_{i^*,j^*}
-
Q_{i_{ind},j_{ind}}
\]

如果：

\[
OJG\approx0
\]

说明独立 RIFT 已经足够。

Joint-RIFT 没必要。

---

# 48. Oracle pair rank / NLL

非常重要。

对 oracle pair：

\[
(i^*,j^*)
\]

记录：

\[
rank_1(i^*)
\]

\[
rank_2(j^*)
\]

以及：

\[
NLL_{oracle}
=
-\log\pi_1(i^*)
-\log\pi_2(j^*)
\]

作用：

> 判断 Stage 1 是否可能只靠 residual 从 RIFT prior 中把 oracle pair 拉起来。

如果 oracle pair 总是：

```text
rank1 = 50/60
rank2 = 58/60
```

说明 base RIFT 对真正 joint opportunity 极不认可。

则 Stage 2 解冻 `pi_head` 很有必要。

---

# 49. Coalition Gain diagnostic

在 oracle pair 或 policy-selected pair 上记录：

\[
CG
=
P_{12}-\max(P_1,P_2)
\]

如果：

\[
OJG>0
\]

但：

\[
CG\approx0
\]

说明 Joint policy 的增益可能来自：

- unary realism；
- pair safety；
- 其它 reward terms；

而不是 coordination。

那么论文 story 仍不成立。

---

# 50. Both-Necessary Rate

定义：

\[
BN
=
I[
P_{12}-P_2>\epsilon
\land
P_{12}-P_1>\epsilon
]
\]

统计：

\[
BNR
=
E[BN]
\]

这是证明“不是一辆车已经完成全部 interaction”的重要指标。

---

# 51. Factorization Gap

实际 learned joint policy：

\[
(i_J,j_J)
=
\arg\max\pi_J
\]

independent：

\[
(i_I,j_I)
=
(
\arg\max\pi_1,
\arg\max\pi_2
)
\]

定义：

\[
FG
=
Q(i_J,j_J)
-
Q(i_I,j_I)
\]

这个可以作为 learned policy coordination performance 指标。

---

# 52. Reward-scale audit

正式训练前必须统计：

```text
std(U1+U2)
std(P12)
std(CG)
pair_collision candidate rate
collision pair reward rank
```

尤其需要确认：

\[
Var(U_1+U_2)
\]

不会完全压过：

\[
Var(P_{12}+CG)
\]

否则 joint head 可能主要重新学习 single-agent ranking。

---

# 53. 为什么 RIFT prior 初始化能缓解 unary domination

若从 RIFT：

\[
\ell_i^{RIFT}
\]

已经学习过：

\[
U_i
\]

那么 joint residual 不必重新编码大部分 unary preference。

这使：

\[
\delta_{ij}
\]

更有机会专门拟合：

\[
Q_{coord}(i,j)
\]

这正是必须从 RIFT checkpoint 初始化的核心理由。

---

# 54. Ego collision 的语义

讨论中一个尚未完全做最终论文定稿的问题：

> Ego collision 是 adversarial success 还是 failure？

当前更推荐的 V1 解释：

> 目标是 high-pressure near-critical interaction，而不是简单撞车。

所以更适合：

- high interaction；
- near conflict；
- ego 被迫受约束；
- actual crash 作为 failure / heavy penalty。

原因：

如果 crash 是直接 success：

> scorer 很容易退化成 collision-seeking，而不是协调测试场景生成。

如果论文目标明确变成：

> crash-seeking adversarial attack

则 reward 语义需要重写。

当前方案默认：

> near-critical pressure 优于实际 crash。

---

# 55. Hard feasibility mask vs scalar penalty

曾讨论：

对于：

- offroad；
- CBV-CBV collision；
- extreme invalid behavior；

是否应该直接 hard-mask，而不是 scalar negative reward。

V1 目前仍以 scalar penalty 为主。

但如果实验出现：

- optimizer 经常把 pair collision 当成高 value；
- reward-scale 调参不稳定；

可以改成：

\[
M_{ij}=0
\]

直接 invalid。

这比无限调 penalty 更稳定。

第一版先通过 statistics 判断。

---

# 56. Interaction direction diagnostic

如果需要给论文可视化解释：

对于 conflict point：

\[
t_C
\]

和：

\[
t_E
\]

可以定义：

\[
d
=
\tanh
\left(
\frac{t_E-t_C}{\tau_d}
\right)
\]

或者离散：

- +1：CBV 更早；
- -1：Ego 更早；
- 0：接近 tie。

但要强调：

\[
|d|
\]

不能当 interaction strength。

因为双方相差很久时 direction 很明确，但 interaction 很弱。

---

# 57. 为什么没继续用 persistent behavior state

讨论过：

> crossing 后是否保持 +1/-1 状态？

最终 V1 放弃。

原因：

1. 逻辑会变复杂；
2. crossing 后永久保持会严重误导；
3. 还要定义 decay / event merge；
4. 当前 sparse event + temporal closeness + discount 已足够表达 V1 interaction。

所以 V1 采用：

> event-based interaction。

后续如果 sparse reward 太稀，再考虑连续状态。

---

# 58. 当前代码修复包

当前已经生成 audited patch package：

```text
joint_rift_audit_fix.zip
```

内容包括：

- patch application script；
- audited temporal interaction；
- audited policy；
- configs；
- validation scripts；
- experiment README。

当前 ChatGPT 环境里的路径：

```text
/mnt/data/joint_rift_audit_fix.zip
```

注意：

由于 GitHub integration 对仓库写操作返回 403，

> 修复代码没有直接推到远端 main。

必须本地应用 patch。

---

# 59. 为什么没有声称“代码已 push”

GitHub connector：

- repository metadata 显示 push permission；
- 但 create branch / create file API 实际返回：
  - `403 Resource not accessible by integration`

因此当前状态是：

> 已完成代码补丁与静态审核，但尚未远程 commit。

新聊天中不要误以为 remote repo 已经包含 audited 修复。

---

# 60. Apply audited patch

在本地：

```bash
unzip joint_rift_audit_fix.zip
```

进入：

```bash
cd multi_adv_rift_betop_2/RIFT
```

执行：

```bash
python /PATH/TO/joint_rift_audit_fix/apply_joint_rift_audit.py
```

然后：

```bash
git status --short
```

---

# 61. 静态 validation

先：

```bash
python -m compileall -q rift scripts
```

再：

```bash
python scripts/validate_joint_rift_core.py
```

期待：

```text
Joint-RIFT core validation: PASS
```

应测试：

1. \(\delta=0\) factorization；
2. independent argmax equivalence；
3. pair swap symmetry；
4. temporal weighting；
5. nonfactorizable ratio；
6. padding / masks；
7. loss state balance。

---

# 62. RIFT baseline checkpoint

先找：

```bash
find rift/cbv/planning/model_ckpt/rift_pluto \
  -name '*.ckpt' -print
```

如果已有对应：

```text
pdm_lite-rule-seed0
```

RIFT checkpoint，可以直接使用。

否则先训练：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run.py \
  --ego_cfg pdm_lite.yaml \
  --cbv_cfg rift_pluto.yaml \
  --mode train_cbv \
  --seed 0 \
  --repetitions 2 \
  --no_resume
```

---

# 63. Joint-RIFT Stage 1

运行：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run.py \
  --ego_cfg pdm_lite.yaml \
  --cbv_cfg joint_rift_pluto_stage1.yaml \
  --mode train_cbv \
  --seed 0 \
  --repetitions 2 \
  --no_resume
```

要求：

- 必须找到 RIFT prior；
- joint head zero-init；
- pi-head frozen；
- trajectory generator frozen。

---

# 64. Stage 1 训练后先做 feasibility 判断

重点指标：

```text
feasibility/oracle_joint_gain
feasibility/nonfactorizable_reward_ratio
feasibility/both_necessary_rate
feasibility/oracle_rank_1
feasibility/oracle_rank_2
feasibility/oracle_nll_1
feasibility/oracle_nll_2

reward/P12
reward/coalition_gain
reward/pair_collision_rate

reward_scale/P12_std
reward_scale/coalition_std
reward_scale/unary_return_std

joint/residual_base_ratio
```

---

# 65. Stage 1 go/no-go

### Kill idea / revise reward if：

\[
OracleJointGain\approx0
\]

and/or：

\[
NonFactorizableRatio\approx0
\]

长期成立。

### Candidate generator 不够 if：

oracle opportunity 很高，但 candidate coverage 很低。

### 需要 Stage 2 if：

oracle candidate：

- rank 很差；
- NLL 很高；

说明 frozen RIFT prior 给这些 candidate 的概率太小。

---

# 66. Joint-RIFT Stage 2

如果 Stage 1 feasibility 支持 idea：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run.py \
  --ego_cfg pdm_lite.yaml \
  --cbv_cfg joint_rift_pluto_stage2.yaml \
  --mode train_cbv \
  --seed 0 \
  --repetitions 2 \
  --no_resume
```

Stage 2 应：

- 从 Stage 1 最新 audited checkpoint 初始化；
- 重新采集 on-policy rollout；
- joint head train；
- pi-head train；
- generator frozen。

---

# 67. 为什么 Stage 1 / Stage 2 用不同 checkpoint 目录

不要 Stage 1 跑完后只改 config 再 resume 同一个 policy directory。

因为原训练流程：

- data loader progress；
- training resume；
- checkpoint detection；

耦合较强。

为了避免：

> Stage 2 实际没有重新从头采集足够新 rollout，

audited 方案使用独立目录：

```text
joint_rift_pluto_stage1_audited/
joint_rift_pluto_stage2_audited/
```

Stage 2 显式从 Stage 1 checkpoint 初始化。

---

# 68. Evaluation baselines

最少四组：

## Pluto ×2

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run.py \
  --ego_cfg pdm_lite.yaml \
  --cbv_cfg pluto.yaml \
  --mode eval \
  --seed 0 \
  --pretrain_seed 0 \
  --repetitions 1
```

## RIFT ×2

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run.py \
  --ego_cfg pdm_lite.yaml \
  --cbv_cfg rift_pluto.yaml \
  --mode eval \
  --seed 0 \
  --pretrain_seed 0 \
  --repetitions 1
```

## Joint Stage 1

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run.py \
  --ego_cfg pdm_lite.yaml \
  --cbv_cfg joint_rift_pluto_stage1.yaml \
  --mode eval \
  --seed 0 \
  --pretrain_seed 0 \
  --repetitions 1
```

## Joint Stage 2

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run.py \
  --ego_cfg pdm_lite.yaml \
  --cbv_cfg joint_rift_pluto_stage2.yaml \
  --mode eval \
  --seed 0 \
  --pretrain_seed 0 \
  --repetitions 1
```

---

# 69. 论文最终 baseline 建议

至少：

| Method | Single-CBV adversarial prior | Joint scorer | Joint RL |
|---|---:|---:|---:|
| Pluto ×2 | No | No | No |
| RIFT ×2 | Yes | No | No |
| Joint-RIFT Stage 1 | Yes | residual only | Yes |
| Joint-RIFT Stage 2 | Yes | residual + pi-head | Yes |

额外建议：

- Direct Joint Evaluator reranker；
- oracle joint reranker；
- supervised distillation of oracle evaluator（可选）；
- w/o coalition gain；
- w/o temporal weighting。

---

# 70. Direct joint reranker 为什么重要

如果：

\[
Q_{ij}
\]

在线计算本身很便宜，那么最直接就是：

\[
\arg\max_{ij}Q_{ij}
\]

根本不需要 RL 学 joint scorer。

所以一定要测：

```text
online joint evaluator latency
joint scorer latency
```

如果 evaluator 很贵：

> joint residual scorer 的价值是 amortized counterfactual evaluation。

这是论文 method motivation 中应该正面回答的问题。

---

# 71. Evaluation metrics

## Coordination-specific

1. Oracle Joint Gain
2. Factorization Gap
3. Coalition Gain
4. Both-Necessary Rate
5. Non-Factorizable Reward Ratio
6. selected pair oracle gap
7. joint same as oracle rate

## Interaction / safety

1. minimum TTC
2. PET
3. required deceleration / DRAC-like
4. interaction duration / weighted event pressure
5. Ego collision
6. CBV-CBV collision
7. offroad
8. comfort
9. lane deviation

注意：

TTC/PET 更适合作为 evaluation metric，而不是第一版全部塞进 reward。

---

# 72. Generalization

用户暂时决定：

> 第一版暂不验证跨 ego policy 泛化。

因此当前可以只用：

```text
PDM-Lite
```

完成方法验证。

但论文 limitation 应诚实写：

> 当前 interaction evaluator 使用 PDM-Lite nominal planning intent。

后续可以再测试：

- Plant；
- E2E AV；
- 其它 planner。

---

# 73. PDM nominal overlay 实验

必须做：

对无强交互场景：

1. 当前 frame 生成 nominal 4s trajectory；
2. 记录 PDM-Lite 后续真实 closed-loop 4s；
3. overlay；
4. 统计：
   - position error；
   - heading error；
   - speed error。

目的不是要求完全预测真实 future。

只需要证明：

> nominal adapter 合理表达当前 planning intent。

---

# 74. Interaction unit tests

构造人工轨迹：

## Case A

同一 conflict point，几乎同时到达：

\[
|\Delta t|\approx0
\]

应：

\[
w\approx1
\]

## Case B

同一 path crossing，但相差 3 秒：

应：

\[
w\ll1
\]

## Case C

连续 3 个 segment 因数值几何都检测到同一 crossing：

应只记：

\[
1
\]

个 cluster event。

## Case D

真实分离的两次 crossing：

应记：

\[
2
\]

个 events。

---

# 75. Scorer unit tests

## Factorization

zero-init：

\[
\delta=0
\]

必须：

\[
S_{ij}
=
\ell_i+\ell_j
\]

以及：

\[
\arg\max S
=
(\arg\max \ell_1,\arg\max \ell_2)
\]

## Symmetry

交换两组 CBV：

\[
S_{21}
\approx
S_{12}^T
\]

## Gradient Stage 1

```text
joint_head grad > 0
pi_head grad = 0
generator grad = 0
```

## Gradient Stage 2

```text
joint_head grad > 0
pi_head grad > 0
generator grad = 0
```

---

# 76. Loss unit tests

构造：

state A：

\[
2\times2
\]

state B：

\[
20\times20
\]

使用同结构 advantage。

验证：

> state B 不会因为 candidate 多 100 倍而对 loss 贡献约 100 倍。

---

# 77. Pair collision unit tests

人工 rectangle trajectories：

1. 完全不相交；
2. 同帧重叠；
3. 旋转 box 相交；
4. 只某一帧相交；
5. 擦边。

验证 SAT 实现。

---

# 78. 多 seed 实验

至少：

\[
seed=0,1,2
\]

推荐流程：

```bash
for SEED in 0 1 2; do

  CUDA_VISIBLE_DEVICES=0 python scripts/run.py \
    --ego_cfg pdm_lite.yaml \
    --cbv_cfg rift_pluto.yaml \
    --mode train_cbv \
    --seed ${SEED} \
    --repetitions 2 \
    --no_resume

  CUDA_VISIBLE_DEVICES=0 python scripts/run.py \
    --ego_cfg pdm_lite.yaml \
    --cbv_cfg joint_rift_pluto_stage1.yaml \
    --mode train_cbv \
    --seed ${SEED} \
    --repetitions 2 \
    --no_resume

  CUDA_VISIBLE_DEVICES=0 python scripts/run.py \
    --ego_cfg pdm_lite.yaml \
    --cbv_cfg joint_rift_pluto_stage2.yaml \
    --mode train_cbv \
    --seed ${SEED} \
    --repetitions 2 \
    --no_resume

done
```

同一 seed 内：

\[
RIFT\rightarrow Stage1\rightarrow Stage2
\]

必须串行。

不同 seed 可以不同 GPU 并行。

---

# 79. 当前最重要的 scientific question

不是：

> “loss 能不能下降？”

而是：

> “Joint candidate space 中，是否存在独立 RIFT 无法自然选到、但联合上明显更优的行为对？”

对应：

\[
OracleJointGain
\]

和：

\[
NonFactorizableRatio
\]

这是当前最重要的研究判据。

---

# 80. 什么情况下应该放弃这个 idea

下面任一长期成立，都应考虑 kill / revise：

### A

\[
OracleJointGain\approx0
\]

说明 joint selection 没额外空间。

### B

\[
NonFactorizableRatio\approx0
\]

说明 reward 基本可分解。

### C

candidate coverage 很低。

说明 Pluto candidate generator 根本不包含有用 coordination trajectories。

### D

Joint-RIFT 只提高 crash rate，但 coalition / both-necessary / interaction pressure 不升。

说明 scorer 在学单纯 aggressive collision。

### E

Joint-RIFT 与 direct evaluator reranker 几乎无 latency 差异。

此时 RL scorer 的必要性需要重新论证。

---

# 81. 什么情况下 idea 很有希望

如果同时看到：

1. Oracle Joint Gain 显著正；
2. NonFactorizableRatio 明显 >0；
3. Oracle pair 在 independent RIFT 下不是 trivial top-1；
4. Coalition Gain >0；
5. Both-Necessary Rate 显著；
6. Stage 1 joint residual 能缩小 oracle gap；
7. Stage 2 在 oracle rank 很差时进一步改善；
8. pair collision / offroad / ego crash 没有失控；

那么论文核心故事会非常清楚：

> Independent RIFT individually ranks adversarial trajectories well, but cannot model pair-specific interaction synergy. Joint-RIFT learns a residual energy over the candidate Cartesian product to recover non-factorizable cooperative adversarial behavior.

---

# 82. 当前 paper story 推荐表述

可以概括成：

### Problem

Existing RIFT independently scores each adversarial background vehicle.

With multiple controlled agents, independent factorization assumes:

\[
\pi(a_1,a_2|s)
=
\pi_1(a_1|s)\pi_2(a_2|s)
\]

which cannot model pair-specific synergy.

### Method

Retain pretrained RIFT as unary adversarial prior and learn:

\[
\delta_\theta(a_1,a_2,s)
\]

over the Cartesian product.

### Reward

Use:

- RIFT unary realism/adversarial return；
- time-aware Ego interaction；
- coalition gain；
- pair collision penalty。

### Training

RIFT-style group-relative clipped optimization on all joint candidates.

### Evidence

Show:

- oracle joint gain；
- nonfactorizable reward；
- coalition gain；
- both-necessary rate；
- learned factorization gap。

---

# 83. 当前不应做的事情

1. 不要马上把 reward 加成十几个 TTC/PET/DRAC/topology terms；
2. 不要重新引入 reachable tube；
3. 不要训练 ego response surrogate；
4. 不要一开始动态选 coalition membership；
5. 不要解冻 trajectory generator；
6. 不要为了省算力过早 top-K；
7. 不要只看 collision rate；
8. 不要只证明 joint scorer 与 independent scorer 选得不同；
9. 不要从 Pluto 而不是 RIFT 初始化；
10. 不要跳过 feasibility diagnostics 直接烧大规模 CARLA。

---

# 84. 后续开发优先级

## P0

1. 应用 audited patch；
2. 验证 RIFT checkpoint 初始化；
3. temporal interaction tests；
4. smoke test one route；
5. feasibility metrics 可正确写出。

## P1

6. Stage 1 小规模训练；
7. OJG / NonFactorizable / BNR 分析；
8. reward-scale audit；
9. candidate rank / NLL audit；
10. 决定是否需要 Stage 2。

## P2

11. Stage 2；
12. 多 seed；
13. baselines；
14. ablations；
15. evaluation plots。

---

# 85. 新聊天建议的第一条消息

建议把本文上传后，在新聊天里直接说：

> “这是我们关于 Joint-RIFT 双 CBV 协同评分项目的完整技术交接文档。请先完整阅读，并以文档中已确定的设计为准，不要重新从头发散。当前下一步是：在本地应用 audited patch 后，逐文件审核 patch 是否与仓库当前 main 一致，然后做 Stage 0 静态/unit tests 和单 route CARLA smoke test。在修改任何方法前先指出是否会破坏 factorized-RIFT prior、candidate identity、state-balanced loss 或 interaction semantics。”

这样最容易无缝继续。

---

# 86. 最终一句话状态

当前项目已经从：

> “想做两个 RIFT agent 的联合选择”

收敛成：

\[
\boxed{
\text{Pretrained RIFT unary prior}
+
\text{full two-CBV candidate Cartesian product}
+
\text{common-frame joint residual scorer}
+
\text{time-aware interaction / coalition reward}
+
\text{state-level RIFT-style group-relative training}
}
\]

真正下一步不是继续设计更多模块，而是：

\[
\boxed{
\text{通过 feasibility + smoke tests 证明这个 joint residual 确实有必要且能学到不可分解协同}
}
\]

---

# Appendix A. 关键代码路径

## RIFT / Pluto

```text
rift/cbv/planning/pluto/model/modules/planning_decoder.py
rift/cbv/planning/pluto/model/pluto_model.py
rift/cbv/planning/pluto/pluto.py

rift/cbv/planning/fine_tuner/rlft/rift_pluto/rift_pluto.py
rift/cbv/planning/fine_tuner/rlft/traj_eval/traj_evaluator.py
rift/cbv/planning/fine_tuner/rlft/rift_pluto/rift_datamodule.py
rift/cbv/planning/fine_tuner/rlft/config/rift_training.yaml
```

## Joint-RIFT

```text
rift/cbv/planning/fine_tuner/rlft/joint_rift/
```

重点：

```text
joint_rift_pluto.py
joint_score_head.py
joint_traj_evaluator.py
interaction_topology.py
joint_buffer.py
joint_datamodule.py
joint_trainer.py
pair_manager.py
geometry.py
```

## Ego

```text
rift/ego/pdm_lite/pdm_lite.py
rift/ego/pdm_lite/autopilot.py
```

## Runner

```text
rift/carla_runner.py
```

## Config

```text
rift/cbv/planning/config/rift_pluto.yaml
rift/cbv/planning/config/joint_rift_pluto.yaml
rift/cbv/recognition/config/rule.yaml
```

---

# Appendix B. 关键超参数初值

```text
joint horizon = 40
dt = 0.1
gamma = 0.98

num Pluto modes = 12

lambda_delta = 1.0

lambda_pressure = 1.0
lambda_coalition = 1.0

audited initial lambda_pair_collision ≈ 20
arrival_time_tau ≈ 1.0 s

Stage 1:
lr_joint = 1e-4
pi frozen

Stage 2:
lr_joint = 1e-4
lr_pi = 1e-5
trajectory generator frozen

PPO clip = 0.2
dual clip = 3
```

这些是 V1 起点，不是最终论文最优值。

---

# Appendix C. 最重要的公式合集

## Joint score

\[
S_{ij}
=
\ell_i^1
+
\ell_j^2
+
\delta_{ij}
\]

## Residual scaling

\[
\delta_{ij}
=
\lambda_\delta
\sigma_{base}
\tanh(raw\_\delta_{ij})
\]

## Temporal interaction

\[
w_e
=
\exp(-|\Delta t_e|/\tau)
\]

\[
P_k
=
\sum_e
\gamma^{t_e}w_e
\]

## Joint pressure

\[
q_{12}(t)
=
1-(1-q_1(t))(1-q_2(t))
\]

\[
P_{12}
=
\sum_t
\gamma^tq_{12}(t)
\]

## Coalition gain

\[
CG
=
P_{12}-\max(P_1,P_2)
\]

## Leave-one-out

\[
\Delta_1=P_{12}-P_2
\]

\[
\Delta_2=P_{12}-P_1
\]

## Joint reward

\[
Q_{ij}
=
U_1(i)+U_2(j)
+
\lambda_PP_{12}
+
\lambda_CCG
-
\lambda_{pair}C_{ij}
\]

## Joint advantage

\[
A_{ij}
=
\frac{Q_{ij}-\mu_Q}
{\sigma_Q+\epsilon}
\]

## Oracle Joint Gain

\[
OJG
=
\max_{ij}Q_{ij}
-
Q_{i_{ind},j_{ind}}
\]

## Nonfactorizable residual

\[
R_{ij}
=
Q_{ij}
-
\bar Q_{i\cdot}
-
\bar Q_{\cdot j}
+
\bar Q
\]

\[
NFR
=
\frac{Var(R)}
{Var(Q)+\epsilon}
\]

---

# Appendix D. 推荐最终实验表

| Experiment | Purpose |
|---|---|
| Pluto ×2 | IL baseline |
| RIFT ×2 | independent adversarial baseline |
| Joint Stage 1 | pure coordination residual |
| Joint Stage 2 | coordination + unary ranking adaptation |
| Oracle joint evaluator | upper-bound / candidate support |
| Direct evaluator reranker | test whether RL scorer is necessary |
| w/o coalition | verify coalition term |
| w/o temporal weighting | verify timing-aware interaction |
| multiple seeds | robustness |

建议至少 seed 0/1/2。

---

# Appendix E. 当前最大风险排序

1. **Oracle Joint Gain 实际≈0**  
   → idea 无必要。

2. **NonfactorizableRatio≈0**  
   → joint reward 本质可分解。

3. **candidate coverage 不足**  
   → scorer 无法创造轨迹。

4. **oracle candidates 在 RIFT prior 下概率极低**  
   → Stage 1 不够，需要 Stage 2。

5. **pair collision / crash reward hacking**  
   → 调 pair collision / hard mask。

6. **interaction reward 太稀**  
   → 先软化 time weight，不要立即加复杂 persistent state。

7. **PDM nominal 偏离 planning intent 太大**  
   → 调 nominal adapter。

8. **Joint scorer 只是重学 unary ranking**  
   → 通过 RIFT prior + NFR + CG + BNR 诊断识别。

---

# Appendix F. 当前代码审核结论简表

| 模块 | 结论 |
|---|---|
| Candidate latent 暴露 | 正确 |
| Full Cartesian candidates | 正确 |
| Common frame | 正确 |
| Joint scorer | 基本正确 |
| Residual zero-init | 正确 |
| Residual scale calibration | 正确，需 histogram |
| Pair symmetry | 基本正确 |
| PairManager | 正确 |
| RIFT unary reuse | 正确 |
| Exclude other controlled CBV | 正确 |
| Pair collision | 正确 |
| Joint buffer | 正确 |
| Joint softmax | 正确 |
| State-balanced loss | 正确 |
| Stage 1 freeze | 正确 |
| Stage 2 pi-head only | 正确 |
| Generator freeze | 正确 |
| RIFT prior init | 原实现错误，audited fix |
| Temporal interaction | 原实现错误，audited fix |
| Runner baseline compatibility | 原实现有回归，audited fix |
| Per-env fallback | 原实现次优，audited fix |
| Feasibility diagnostics | 原实现不完整，audited fix |
| Eval joint metrics | 原实现不完整，audited fix |

---

# Appendix G. 术语建议

为避免论文中过度 claim：

推荐：

- “Joint-RIFT”
- “joint residual scorer”
- “pairwise non-factorizable coordination”
- “time-aware interaction pressure”
- “coalition gain”
- “RIFT-style group-relative clipped optimization”

避免未经证明直接写：

- “causal ego response”
- “true ego reaction prediction”
- “BeTop topology reward”  
  当前更准确是：
  “BeTop-inspired path/topology conflict event”
- “PPO” 单独称呼  
  更严谨：
  “PPO-style / RIFT-style clipped surrogate”

---

# Appendix H. 新聊天中不要重复讨论的已定事项

以下已经多轮讨论并基本定稿：

1. 第一版固定 2 CBV；
2. 第一版不动态换 coalition；
3. 第一版不做 reachable tube；
4. 第一版使用 PDM-Lite nominal intent；
5. 第一版不做 ego response surrogate；
6. 第一版尽量保留 full candidate Cartesian product；
7. joint head 不只做 latent dot product；
8. 使用 common-frame trajectory；
9. Stage 1 从 pretrained RIFT 开始；
10. Stage 1 freeze base RIFT；
11. Stage 2 只额外解冻 pi-head；
12. generator 始终 freeze；
13. state-inner mean → batch mean；
14. multiple crossing cluster + RIFT temporal discount；
15. crossing 要加入 arrival-time closeness；
16. coordination 不强制 assert/yield；
17. coalition gain / leave-one-out 是主 coordination evidence；
18. 泛化到其它 ego policy 暂不作为 V1 必做；
19. feasibility diagnostics 必须先于大规模正式实验；
20. idea 的宗旨是“必须证明有必要做，而且能 work”，不是堆复杂模块。

