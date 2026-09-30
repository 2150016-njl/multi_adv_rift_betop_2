# Joint-RIFT 方法设计

## 1. 研究目标：学习“不可由两个单车 RIFT 独立解释”的联合行为偏好

两辆 CBV 分别记为 `C1` 与 `C2`。在当前 CARLA state `s` 下，Pluto/RIFT 分别产生：

\[
\mathcal I_1=\{\tau^1_i\}_{i=1}^{G_1},\qquad G_1=R_1\times12
\]

\[
\mathcal I_2=\{\tau^2_j\}_{j=1}^{G_2},\qquad G_2=R_2\times12.
\]

这里每个 `τ` 都是 Pluto 生成的 realistic future trajectory candidate；`R1/R2` 是各自当前有效 reference-line 数量。

多个 RIFT 独立运行时，等价于：

\[
\pi_{ind}(i,j|s)=\pi_1(i|s)\pi_2(j|s).
\]

这种策略虽然两个 CBV 都能观察周围车辆当前状态，但**不知道另一辆 CBV 最终选择哪个 future mode**，所以无法显式表达：

- 某个 `C1 mode` 与某个 `C2 mode` 是否互补；
- 两辆车是否在同一时间段做冗余交互；
- 是否形成时间接力；
- 两辆候选未来是否互撞；
- 第二辆车的加入是否真的增加 Ego interaction pressure。

Joint-RIFT 的目标就是只学习这部分 residual：

\[
\boxed{
\pi_J(i,j|s)\propto
\pi_1(i|s)\pi_2(j|s)\exp\{\Delta_\theta(i,j,s)\}
}
\]

---

## 2. Base score：从 RIFT 而不是纯 Pluto 出发

两辆 CBV 使用同一个共享 RIFT 模型参数，分别对自身 candidate 输出 logits：

\[
z^1_i,\qquad z^2_j.
\]

先各自转换成合法 marginal log-probability：

\[
\ell^1_i=\log\operatorname{softmax}(z^1)_i,
\]

\[
\ell^2_j=\log\operatorname{softmax}(z^2)_j.
\]

联合 base score：

\[
B_{ij}=\ell^1_i+\ell^2_j.
\]

因为：

\[
\exp(B_{ij})=\pi_1(i)\pi_2(j),
\]

所以当 residual 为 0 时，joint policy **严格退化为两个独立 RIFT**。

这给方法提供了一个非常干净的 baseline / initialization。

---

## 3. Joint residual 的数值尺度：不能让它相对 base score 微乎其微

用户提出的担忧是成立的。如果：

\[
|\Delta_{ij}|\ll|\ell^1_i+\ell^2_j|,
\]

那么 joint head 即使有梯度，也无法明显改变行为模式排序。

初版建议：

\[
S_{ij}=B_{ij}+\lambda_\Delta\,\sigma_B(s)\,\tanh(\hat\Delta_{ij}),
\]

其中：

\[
\sigma_B(s)=Std\{B_{ij}: (i,j)\text{ valid}\}.
\]

解释：

- `hatΔ` 是 joint head 的 raw output；
- `σB(s)` 自动把 residual 的有效尺度校准到当前 state 的 base-score spread；
- `tanh` 防止一开始 residual 完全覆盖 RIFT prior；
- `λΔ` 初版设 `1.0`，之后根据离线诊断调整；
- 最后一层权重和 bias **zero initialization**，使初始：

\[
\Delta_{ij}=0,
\]

从而 joint policy 初始严格等于 independent-RIFT policy。

### 必须记录的训练诊断

每个 training step / epoch 记录：

\[
r_{scale}
=
\frac{Std(\Delta_{ij})}
{Std(B_{ij})+\epsilon}.
\]

工程判断：

- 长期 `<0.05`：residual 实际没影响，应检查 LR / `λΔ` / zero-init 后梯度；
- 约 `0.2 ~ 1.0`：通常是合理区域；
- 持续 `>2`：joint head 可能完全覆盖 RIFT prior，需检查 realism 是否下降。

这些不是论文定理，而是训练健康监控。

---

## 4. Ego nominal trajectory：初版采用 PDM-Lite 当前规划，不做 reachable tube

### 4.1 当前代码事实

当前 `rift/ego/pdm_lite/pdm_lite.py` 的 `PDM_LITE.get_action()` 只返回：

```python
{
    "ego_actions": {env_id: [throttle, steer, brake]}
}
```

没有直接暴露 `[40, ...]` 的 future trajectory。

但 `rift/ego/pdm_lite/autopilot.py` 在每帧 `_get_control()` 内部已经维护：

- `self.remaining_route`
- `self.remaining_route_original`
- `self.target_speed`
- Ego 当前位置、当前速度

所以初版增加一个**非学习式 nominal-plan adapter**：

```python
planner.get_nominal_trajectory(num_frames=40, dt=0.1)
```

输出：

```text
[T=40, C]
C 至少包含 x, y, heading, speed
```

### 4.2 时间参数化建议

不要只把 spatial route points 当 trajectory，因为 interaction reward 需要时间对齐。

初版采用：

1. 对 `remaining_route` 计算 arc-length；
2. 当前速度 `v0`；
3. PDM-Lite 当前 `target_speed`；
4. 用简单受限加速度将 `v0` 平滑趋向 `target_speed`；
5. 积分得到未来 0.1 s 间隔的 longitudinal progress；
6. 在 `remaining_route` 上按 arc-length 插值得到未来 40 帧坐标和 heading。

这不是 ego-response surrogate：它不学习 `CBV pair -> Ego response`，只是把 PDM-Lite 当前 nominal intention 时间化。

### 4.3 初版含义的边界

因此训练学到的是：

> 哪两个 CBV candidate 对 **PDM-Lite 当前 nominal plan** 形成更强的联合交互压力。

不是：

> 精确预测 PDM-Lite 遭遇这两个 CBV 后会如何重新规划。

这一区别应在论文中诚实说明。

---

## 5. Interaction 语义：初版采用简单的 event-based temporal signal

不建议初版同时混入复杂 reachable set、learned interaction model、动态 ego response、复杂多图拓扑。

### 5.1 BeTop 提供什么

BeTop 官方实现会先产生 trajectory segment 上的 `raw_braid_mask`，然后默认沿时间 `any` 聚合为 trajectory-level topology edge。

Joint-RIFT 不直接使用它的最终 trajectory-level binary edge，而保留底层的**事件时刻**。

对候选 `τ^k_g` 与 Ego nominal trajectory `τ^E` 得到：

\[
b^k_g(t)\in\{0,1\},\qquad t=1,\ldots,40.
\]

其中 `1` 表示该时间 segment 上检测到 topology / crossing interaction event。

### 5.2 抢行 / 让行只作为分析标签

对一个有效 conflict event，利用 spatial conflict point 的 arrival time：

\[
t_C,\quad t_E.
\]

定义：

\[
d=
\begin{cases}
+1,&t_C<t_E-\delta\\
-1,&t_C>t_E+\delta\\
0,&|t_C-t_E|\le\delta.
\end{cases}
\]

它用于：

- 可视化；
- 分析联合行为是抢行 / 让行 / 混合；
- 后续扩展。

初版 **不强制 reward 偏好 +1 或 -1**。目标只是高 interaction。

---

## 6. 多次 crossing：采用 RIFT 风格的时间折扣累积

这一点采用用户提出的思路，比构造复杂 persistent behavior state 更适合第一版。

### 6.1 相邻 raw crossing 先去重

`raw_braid_mask` 有时会出现连续 True：

```text
... 0 0 1 1 1 0 ...
```

这些应视为同一个 crossing event，而不是三个奖励。

将连续 segment cluster 成 event：

```text
e1 = [t_start, ..., t_end]
```

使用 event onset 或 cluster center 作为该事件的 reward time。初版建议使用 onset `t_start`，因为 RIFT discount 本身偏好更早发生的 interaction。

### 6.2 多个分离 crossing 全部保留

如果：

```text
e1 @ t=8
e2 @ t=25
```

两个事件都参与 return，不只取 first / last。

对第 `k` 辆 CBV 某个 candidate：

\[
G^{int}_k
=
\sum_{e\in\mathcal E_k}
\gamma^{t_e-1}w_e,
\]

初版：

- `γ = 0.98`，与现有 RIFT candidate evaluator 一致；
- `we=1` 可作为最简单版本；
- 如果离线实验发现 binary event 太稀疏，再把 `we` 软化成由 PET / arrival-time gap 得到的 `[0,1]` severity。

### 6.3 为什么不先做“持续 behavior state”

优点：

- 与 RIFT 现有 40-step discounted return 结构一致；
- 实现简单；
- 多次 crossing 自然可累积；
- 不需要人为定义“crossing 后 +1 状态持续几帧”。

持续 state 可以作为后续 ablation：如果 event-based signal 太稀疏，再在每个 event 周围做时间核平滑，而不是第一版直接引入。

### 6.4 Reward hacking 防护

初版只做两层简单防护：

1. adjacent crossing 去重；
2. RIFT 原有 lane / comfort / collision / offroad realism penalties 保留。

如果 diagnostic 中发现 candidate 通过来回摆动产生异常多 crossing，再增加：

- event-count saturation；或
- `clip(G_int, 0, percentile_95)`。

不要第一版先加过多规则。

---

## 7. 单车 interaction trace 与联合 interaction pressure

对 `C1 candidate i` 和 `C2 candidate j`，构造时间序列：

\[
r^1_i(t)\in[0,1],
\qquad
r^2_j(t)\in[0,1].
\]

最简单版本是去重后的 0/1 event impulse；软化版可以使用 `[0,1]` event severity。

### 7.1 Joint temporal union

定义：

\[
r^{12}_{ij}(t)
=
1-[1-r^1_i(t)][1-r^2_j(t)].
\]

性质：

- 任意一辆车与 Ego 强交互时，joint interaction 高；
- 两辆同时高时仍有额外贡献，但存在 diminishing return；
- 两辆在不同时间形成 interaction 时，会扩大未来 horizon 的 interaction coverage；
- 不强制指定“同时夹击”还是“先后接力”。

因此符合当前“高协同交互但不规定方向/角色”的目标。

### 7.2 RIFT 风格时间折扣

\[
P_1(i)=\sum_t\gamma^{t-1}r^1_i(t),
\]

\[
P_2(j)=\sum_t\gamma^{t-1}r^2_j(t),
\]

\[
P_{12}(i,j)=\sum_t\gamma^{t-1}r^{12}_{ij}(t).
\]

这里 `P12` 是 joint interaction pressure / exposure。

---

## 8. Coalition Gain：证明不是两个 RIFT 独立叠加

定义：

\[
\boxed{
CG_{ij}=P_{12}(i,j)-\max(P_1(i),P_2(j))
}
\]

含义：

> 两辆车共同出现，相比其中最强的一辆，额外增加了多少 Ego interaction pressure。

该指标借鉴 AWM 的 pair gain / counterfactual coalition 思想，但这里不复制其 self-play 框架。

同时记录 leave-one-out contribution：

\[
\Delta_1=P_{12}-P_2,
\]

\[
\Delta_2=P_{12}-P_1.
\]

如果：

\[
\Delta_1>\epsilon,\quad\Delta_2>\epsilon,
\]

说明两辆车对 joint pressure 都有非平凡贡献。

这可以定义 `Both-Necessary Rate`。

---

## 9. Joint reward：第一版保持简单

### 9.1 继续复用 RIFT 的 realistic trajectory return

现有 `TrajEvaluator.get_rollout_return()` 已经计算：

- collision
- offroad
- comfort
- lane alignment
- lane center
- velocity
- timestep cost

并以 `γ=0.98` 累积。

记两辆 CBV 各自的 unary return：

\[
R^1_i,\qquad R^2_j.
\]

### 9.2 但 unary rollout 要改两个地方

**Ego future：**

当前 RIFT 对附近 vehicle 用 kinematic model 传播。Joint 版本中，对 Ego 不再沿用 current-control rollout，而替换为当前 PDM-Lite nominal trajectory。

**另一辆受控 CBV：**

评估 `C1 candidate i` 时，不应把 `C2` 当作“保持当前 control 的 ordinary nearby actor”；否则会和 joint candidate `j` 不一致。

因此：

- `C1 unary evaluator`：exclude `C2`；
- `C2 unary evaluator`：exclude `C1`；
- `C1-C2` 关系单独基于 `(i,j)` candidates 计算。

### 9.3 Pair collision penalty

计算：

\[
C^{12}_{ij}=I(\tau^1_i\text{ 与 }\tau^2_j\text{ 在40帧内碰撞}).
\]

可以进一步使用 earliest collision time 做时间折扣，但初版 binary penalty 已足够。

### 9.4 第一版最终 joint return

\[
\boxed{
Q_{ij}
=
R^1_i+R^2_j
+\lambda_P P_{12}(i,j)
+\lambda_C CG_{ij}
-\lambda_{12}C^{12}_{ij}
}
\]

说明：

- `R1 + R2` 保住 RIFT realism / road / comfort / ego-collision safety；
- `P12` 直接推动高联合 interaction；
- `CG` 明确推动“第二辆车带来额外价值”；
- `C12` 防止 joint head 用两个 CBV 互撞刷交互。

### 9.5 权重不要拍脑袋

正式训练前，从离线 candidate-pair dataset 统计：

```text
std(R1 + R2)
std(P12)
std(CG)
collision rate
```

先让几项的有效 variation 进入可比较尺度，再小范围搜索 `λP, λC, λ12`。

最终 `Qij` 还会 state 内做 mean/std normalization，因此绝对尺度不是最关键，但各项相对尺度仍然重要。

---

## 10. Joint group advantage

对当前 state 的所有 valid pairs：

\[
\mathcal A_s=\{(i,j):i\text{ valid},j\text{ valid}\}.
\]

计算：

\[
\mu_s=Mean\{Q_{ij}\},
\]

\[
\sigma_s=Std\{Q_{ij}\}+10^{-5},
\]

\[
\boxed{
A_{ij}=\frac{Q_{ij}-\mu_s}{\sigma_s}
}
\]

这和 RIFT 当前 group-relative advantage 的核心思路一致，只是 group 从：

```text
单 CBV 的 R×12 candidates
```

变成：

```text
双 CBV 的 G1×G2 candidate pairs
```

。

---

## 11. Joint-RIFT policy loss

保存 collection 时的 old joint logits：

\[
S^{old}_{ij}.
\]

训练时重新 forward 得：

\[
S^\theta_{ij}.
\]

联合 categorical policy：

\[
\log\pi_\theta(i,j|s)
=
\log softmax(vec(S^\theta))_{ij}.
\]

importance ratio：

\[
\rho_{ij}
=
\exp[
\log\pi_\theta(i,j|s)-
\log\pi_{old}(i,j|s)
].
\]

沿用 RIFT：

\[
L_1=\rho A,
\]

\[
L_2=clip(\rho,0.8,1.2)A.
\]

`A >= 0`：标准 PPO min clip；

`A < 0`：沿用 RIFT dual clip `3A`。

---

## 12. Loss averaging 必须修改为 state-balanced

不同 state：

\[
G_1G_2
\]

可以差很多。

不能直接把整个 batch 所有 valid pair flatten 后做一次 mean，否则 candidate 多的 state 权重更大。

必须：

\[
L_s
=
-\frac1{|\mathcal A_s|}
\sum_{(i,j)\in\mathcal A_s}
L_{ij},
\]

再：

\[
\boxed{
L=\frac1B\sum_sL_s
}
\]

。

---

## 13. Joint residual head：结构建议

### 13.1 输入一：Pluto candidate latent

`PlanningDecoder` 在：

```python
loc = loc_head(q)
yaw = yaw_head(q)
vel = vel_head(q)
pi  = pi_head(q)
```

之前已经有每个 `(reference line, mode)` 的 candidate latent `q`。

修改模型，使其额外输出：

```python
candidate_feature = q
```

trajectory generation 不变。

### 13.2 输入二：共同坐标系下的 candidate geometry

不能只拿两个 `q` 点积，因为两个 CBV 的 Pluto feature 都是各自 local-centered 的。

将两辆 candidate trajectory 转到同一 common frame，建议初版统一到 **Ego-centric right-handed frame**。

对每条 candidate，从 40 帧中下采样，例如每 5 帧取一次：

```text
8 points × [x, y, heading, speed]
```

通过小 MLP 得到：

\[
g_i^1,\quad g_j^2.
\]

组合：

\[
h_i^1=Fuse(q_i^1,g_i^1),
\]

\[
h_j^2=Fuse(q_j^2,g_j^2).
\]

### 13.3 为了计算效率，使用低秩 pair compatibility

共享 projection：

\[
u_i=f(h_i^1),\quad v_j=f(h_j^2).
\]

base compatibility：

\[
c_{ij}=u_i^Tv_j/\sqrt d.
\]

再加入少量 pair geometry scalars，例如：

- candidate pair 最小距离；
- 最小 CBV-CBV bbox gap；
- 两辆 candidate 与 Ego 的第一个 interaction time 差；
- 两辆 candidate interaction duration / event time summary。

注意：pair geometry head 保持很小，不直接输入 `[G1,G2,T,D]` 大 tensor 到大型 MLP。

最终：

\[
\hat\Delta_{ij}
=w_c c_{ij}+MLP(g^{pair}_{ij}).
\]

### 13.4 对称性

当前任务没有固定 `CBV1=leader / CBV2=follower` 语义。

应尽量满足：

\[
\Delta(C1_i,C2_j)
\approx
\Delta(C2_j,C1_i).
\]

可通过：

- shared candidate encoder；
- dot-product compatibility；
- symmetric pair features；

来实现。

并增加 swap unit test。

---

## 14. Stage 1 与 Stage 2

### Stage 1：只训练 joint head

冻结：

- Pluto encoder；
- trajectory decoder；
- candidate latent generation；
- RIFT `planning_decoder.pi_head`。

训练：

- `JointResidualScorer`。

目的：

> 单独验证 joint coordination 是否存在，不让“单车 scorer 变激进”混淆结论。

### Stage 2：joint head + 原 RIFT pi_head 一起训练

解冻：

```text
planning_decoder.pi_head
JointResidualScorer
```

注意 RIFT 的 `pi_head` 是**共享参数**，不是每辆 CBV 各有一套。

推荐：

```text
joint head LR: 1e-4 量级
pi_head LR:    1e-5 ~ 3e-5 量级
```

具体由 Stage-1 diagnostic 调整。

### 为什么不解冻 trajectory generator

如果解冻生成 `q / trajectory` 的主体：

- 同一个 buffer state 再 forward 时 candidate trajectories 会改变；
- collection 时计算的 `Aij` 不再对应当前 pair；
- PPO old/new ratio 失去清晰 candidate identity。

因此第一篇工作不要这么做。

---

## 15. 推理流程

当前时刻：

1. PDM-Lite 先产生 Ego action + nominal trajectory；
2. 两辆 CBV Pluto/RIFT 一次 batched forward；
3. 得到各自：
   - all valid trajectories；
   - RIFT logits；
   - candidate latent `q`；
4. 构建 joint residual matrix：

   \[
   \Delta\in\mathbb R^{G_1\times G_2};
   \]

5. 构建：

   \[
   S_{ij}=\ell^1_i+\ell^2_j+\Delta_{ij};
   \]

6. 执行：

   \[
   (i^*,j^*)=argmax_{ij}S_{ij};
   \]

7. 两条 trajectory 分别经过原 PID controller；
8. 两个 CBV action 同一 CARLA tick 施加。

---

## 16. 为什么这一版没有过度设计

第一版**明确不做**：

- Ego response surrogate；
- reachable tube；
- min-max self-play；
- 动态 coalition size；
- 全图 GNN；
- 强制抢行 / 让行 label；
- full Pluto generator joint fine-tuning；
- 复杂 CVaR / tail risk。

只保留：

```text
RIFT candidate manifold
+ Ego nominal plan
+ interaction event timing
+ coalition gain
+ joint residual scorer
+ RIFT-style RLFT
```

这足以验证核心假设。

---

## 17. 与 BeTop / AWM 的关系

### BeTop

使用其核心思想：future trajectory 的 behavioral topology / braid interaction，而不是直接复制其最终 trajectory-level binary edge。

本方法保留 event time，使其能进入 RIFT 的 40-step discounted return。

### AWM

借鉴的不是完整 self-play，而是：

- multi-agent reward sharing 不足以证明 coordination；
- 第二个 adversary 应有额外 pair gain；
- leave-one-out / team contribution 能支撑 coalition credit。

AWM Section 3.2 明确定义了 leave-one-out contribution 和 pair gain；这可以作为 Joint-RIFT 的 coalition metric / motivation 来源。

---

## 18. 论文核心可验证假设

### H1：联合动作空间存在有效增益

\[
Q(i_{oracle},j_{oracle})
>
Q(i_{ind},j_{ind}).
\]

### H2：增益不是 additive

`Qij` 存在显著 pair-wise residual，而不是近似：

\[
f(i)+g(j).
\]

### H3：joint head 能学会逼近这种 preference

Joint-RIFT 的 selected pair 在：

- `P12`
- Coalition Gain
- Both-Necessary Rate

上优于 independent RIFT。

### H4：不是靠不真实/不安全行为取得增益

同时：

- collision 不显著恶化；
- CBV-CBV collision 低；
- offroad 低；
- lane / comfort metrics 不明显下降。

如果 H1/H2 不成立，idea 应该被否定或重新定义，而不是继续堆模型。
