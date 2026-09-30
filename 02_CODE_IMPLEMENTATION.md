# Joint-RIFT 代码实现流程

本文按当前 RIFT 实际调用链给出实现顺序。原则是：每一步先能独立测试，再继续下一步，避免一次性改完整 pipeline 后无法定位问题。

---

# 0. 当前 RIFT 主调用链

训练入口大致为：

```text
scripts/run.py
  -> CarlaRunner.run()
    -> train_cbv()
      -> env.reset()
      -> ego_policy.set_ego_and_route()
      -> while not done:
           ego_policy.get_action()
           cbv_policy.get_action()
           env.step()
           buffer.store()
      -> if buffer_full:
           cbv_policy.train()
```

关键代码：`rift/carla_runner.py`。

当前顺序已经是：

```python
ego_actions_dict = self.ego_policy.get_action(...)
CBVs_actions_dict = self.cbv_policy.get_action(...)
```

所以正好可以在 Ego action 计算后，把 PDM-Lite nominal trajectory 显式传给 Joint-CBV policy。

---

# Step 1. 增加 PDM-Lite nominal trajectory 输出

## 1.1 修改文件

```text
rift/ego/pdm_lite/autopilot.py
rift/ego/pdm_lite/pdm_lite.py
```

## 1.2 当前代码可利用的数据

`AutoPilot._get_control()` 每帧已经更新：

```python
self.remaining_route
self.remaining_route_original
self.target_speed
```

并可通过 `CarlaDataProvider` 得到 Ego 当前：

```text
location
heading
velocity
```

当前没有现成的 40-step trajectory。

## 1.3 新增 API

建议：

```python
class AutoPilot:
    def get_nominal_trajectory(
        self,
        num_frames=40,
        dt=0.1,
    ) -> np.ndarray:
        ...
```

输出约定：

```text
shape [40, 4]
[x, y, heading, speed]
```

必须明确坐标系。

推荐先输出 CARLA/global 原始 frame，后续统一由 Joint evaluator 转 common frame。

## 1.4 时间参数化

伪代码：

```python
route = self.remaining_route
v = current_speed
v_target = self.target_speed
s = 0

for t in range(40):
    a = clip((v_target - v) / tau_v, a_min, a_max)
    v = max(0, v + a * dt)
    s = s + v * dt
    xy, heading = interpolate_route_by_arclength(route, s)
    traj[t] = [xy.x, xy.y, heading, v]
```

第一版的目标不是复刻所有 PDM-Lite future feedback，而是提供“当前 nominal intention”。

## 1.5 修改 `PDM_LITE.get_action()`

建议输出：

```python
data = {
    "ego_actions": actions,
    "ego_nominal_trajectories": nominal_trajs,
}
```

`nominal_trajs[env_id] -> [40,4]`。

---

# Step 2. 把 Ego nominal trajectory 传入 CBV policy

## 2.1 修改 `CarlaRunner.train_cbv()`

原：

```python
ego_actions_dict = self.ego_policy.get_action(...)
CBVs_actions_dict = self.cbv_policy.get_action(...)
```

改为显式数据流：

```python
ego_actions_dict = self.ego_policy.get_action(...)

CBVs_actions_dict = self.cbv_policy.get_action(
    CBVs_obs_list,
    info_list,
    deterministic=False,
    ego_nominal_trajectories=ego_actions_dict.get(
        "ego_nominal_trajectories"
    ),
)
```

不要依赖隐式 global variable，便于后续替换其他 Ego policy。

同时修改：

```text
CBVBasePolicy.get_action interface
所有被调用的子类兼容 optional kwarg
```

如果不想改所有子类，可在新 `JointRIFTPluto` 单独定义接受额外 kwarg，Runner 根据 policy type 分支；但长期建议统一 interface。

---

# Step 3. 确保一个 joint segment 内固定两辆 CBV

## 3.1 当前基础

RIFT 已支持一个 Ego 下多个 CBV：

```python
CarlaDataProvider.get_CBVs_by_ego(ego_id)
```

rule recognition config 已有：

```yaml
max_agent_num: 3  # train
max_agent_num: 2  # eval
```

初版建议训练也改成：

```yaml
max_agent_num: 2
```

## 3.2 新增 PairManager

建议文件：

```text
rift/cbv/planning/fine_tuner/rlft/joint_rift/pair_manager.py
```

功能：

```python
pair_ids[env_id] = (cbv_id_a, cbv_id_b)
```

规则：

1. 当前 env 第一次同时存在 >=2 CBV 时选定 pair；
2. segment 内保持 ID 不变；
3. 某成员 done / destroyed / reach goal，则 joint segment 结束；
4. 下一个合法时刻重新初始化新 pair。

因为 joint head 尽量对称，所以 pair storage 可简单按 actor id 排序来保持 deterministic ordering，不把排序当行为语义。

---

# Step 4. 暴露 Pluto candidate latent `q`

## 4.1 修改

```text
rift/cbv/planning/pluto/model/modules/planning_decoder.py
rift/cbv/planning/pluto/model/pluto_model.py
```

当前 decoder：

```python
loc = self.loc_head(q)
yaw = self.yaw_head(q)
vel = self.vel_head(q)
pi = self.pi_head(q).squeeze(-1)
```

增加：

```python
return traj, pi, q
```

上层 model output：

```python
output["candidate_feature"] = q
```

shape：

```text
[B, padded_R, 12, D]
```

不要 detach，因为：

- Stage 1 上游全部冻结，实际上无梯度；
- Stage 2 只解冻 pi_head，`q` 仍固定；
- 后续如果想独立训练 trajectory encoder，可直接使用。

---

# Step 5. 新建 JointRIFTPluto policy

建议目录：

```text
rift/cbv/planning/fine_tuner/rlft/joint_rift/
  __init__.py
  joint_rift_pluto.py
  joint_score_head.py
  joint_traj_evaluator.py
  joint_buffer.py
  joint_datamodule.py
  joint_trainer.py
  config/
```

并在：

```text
rift/cbv/planning/__init__.py
```

注册新 policy，例如：

```text
joint_rift_pluto
```

---

# Step 6. Joint policy forward：先得到两辆车各自全部 candidates

当前 `RIFTPluto.get_action()` 会 batch 多辆 CBV feature，然后每辆独立 `_get_action()`。

新版本不要立即逐辆 argmax，而应：

1. 仍然把两辆 CBV Pluto feature batched forward；
2. 对每辆取：

```text
raw trajectory [valid_R, 12, T, 6]
candidate trajectory [valid_R, 12, T, 3]
logits [valid_R, 12]
candidate_feature [valid_R, 12, D]
```

3. flatten：

```text
traj1 [G1,T,...]
logit1 [G1]
feat1 [G1,D]

traj2 [G2,T,...]
logit2 [G2]
feat2 [G2,D]
```

**不要使用现有 `_trim_candidates(topk=10)` 来构造 RL joint group。**

实际执行也建议初版直接在全部 valid joint pair 上 argmax，而不是先各自 top-10；否则会损失模式多样性。

---

# Step 7. 坐标统一

这是容易出 bug 的地方，必须单独实现并测试。

建议所有 joint evaluator / joint head geometry 均统一到：

```text
Ego-centric right-handed frame
```

步骤：

```text
Pluto candidate local frame
 -> CBV / global frame
 -> Ego-centric common frame

PDM nominal global frame
 -> Ego-centric common frame
```

建议建立单独 utility：

```text
rift/cbv/planning/fine_tuner/rlft/joint_rift/geometry.py
```

API：

```python
to_common_frame_trajectory(...)
```

并禁止 joint evaluator 内部散落不同版本的坐标变换。

---

# Step 8. JointResidualScorer

## 8.1 Candidate representation

对每条 candidate：

```text
Pluto latent q
+
common-frame trajectory embedding
```

trajectory 初版只使用前 40 帧，并每 5 帧采样：

```text
8 × [x, y, heading, speed]
```

小 MLP：

```python
traj_embed = TrajectoryEncoder(sampled_traj)
```

然后：

```python
h = fuse(q_proj, traj_embed)
```

## 8.2 高效 pair score

不要构建大规模：

```text
[B,G1,G2,2D]
```

再走重 MLP。

推荐：

```python
u1 = pair_proj(h1)   # [B,G1,d]
u2 = pair_proj(h2)   # [B,G2,d]

bilinear = torch.einsum("bid,bjd->bij", u1, u2) / sqrt(d)
```

再加非常小的 pair geometry head：

```text
min pair distance
first interaction time 1
first interaction time 2
interaction time difference
...
```

输出：

```text
raw_delta [B,G1,G2]
```

## 8.3 Zero initialization

joint residual 最终 projection：

```python
nn.init.zeros_(last.weight)
nn.init.zeros_(last.bias)
```

确保训练开始：

```text
delta = 0
joint policy == independent RIFT policy
```

## 8.4 Residual scaling

当前 state：

```python
base = logp1[:, :, None] + logp2[:, None, :]
base_std = masked_std(base).detach()

delta = lambda_delta * base_std * torch.tanh(raw_delta)
joint_logits = base + delta
```

记录：

```text
std(delta) / std(base)
```

---

# Step 9. Interaction event detector

建议先单独写：

```text
interaction_topology.py
```

基于 BeTop `topo_utils.py` 的核心 segment-crossing 逻辑，但不要整包复制依赖。

API：

```python
def get_interaction_events(
    cbv_traj,       # [G,T,2]
    ego_nominal,    # [T,2]
    valid_mask=None,
):
    # return [G, N_event] or event_mask [G,T-1]
```

第一版输出：

```text
raw_event_mask [G,T-1]
clustered_event_mask [G,T-1]
```

对连续 True cluster 只保留 onset：

```text
0 1 1 1 0 -> 0 1 0 0 0
```

之后：

```python
weights = gamma ** torch.arange(T-1)
interaction_return = (event_mask * weights).sum(-1)
```

`gamma=0.98`。

### arrival order

额外函数：

```python
get_arrival_order_label(...)
```

仅用于 logging / metric；初版不进入主要 reward。

---

# Step 10. JointTrajEvaluator：复用 RIFT unary evaluator

建议从：

```text
rift/cbv/planning/fine_tuner/rlft/traj_eval/traj_evaluator.py
```

继承或拆 utility。

不要复制所有逻辑两份后长期分叉。

## 10.1 Unary rollout

分别对 `C1` 和 `C2`：

```text
candidate rollout
lane relation
off-road
comfort
velocity
collision with background actors
collision with Ego nominal trajectory
```

其中：

- Ego future = supplied PDM nominal trajectory；
- 另一辆 controlled CBV 从 nearby actor 列表排除。

得到：

```text
unary_return_1 [G1]
unary_return_2 [G2]
interaction_trace_1 [G1,T]
interaction_trace_2 [G2,T]
P1 [G1]
P2 [G2]
```

## 10.2 Pair-wise broadcast

构造：

```python
r12 = 1 - (1-r1[:, None, :]) * (1-r2[None, :, :])
P12 = (r12 * temporal_weight).sum(-1)

CG = P12 - torch.maximum(P1[:, None], P2[None, :])
```

## 10.3 CBV1-CBV2 candidate collision

输入两组已经 rollout 的 bounding polygons / centers：

```text
[G1,T,...]
[G2,T,...]
```

利用 broadcasting / 分块计算：

```text
pair_collision [G1,G2]
```

不要在 Python `for i for j` 中重新 rollout 两辆车。

## 10.4 Final Q

```python
Q = (
    unary_return_1[:, None]
    + unary_return_2[None, :]
    + lambda_p * P12
    + lambda_c * coalition_gain
    - lambda_pair_collision * pair_collision.float()
)
```

然后当前 state 内：

```python
adv = (Q - masked_mean(Q)) / (masked_std(Q) + 1e-5)
```

返回：

```python
{
    "joint_advantage": adv,
    "joint_q": Q,                # 可选，debug/log
    "p12": P12,
    "coalition_gain": CG,
    "pair_collision": ...,
    "valid_mask": ...,
}
```

---

# Step 11. 实际 joint action 选择

collection / eval 时：

```python
joint_logits = joint_head(...)
flat_index = masked_joint_logits.view(-1).argmax()
i_star, j_star = unravel(flat_index)
```

得到：

```text
trajectory1 = candidate1[i_star]
trajectory2 = candidate2[j_star]
```

分别沿用 Pluto 当前：

```python
get_control(...)
```

输出：

```python
CBVs_action[cbv1_id] = [throttle1, steer1, brake1]
CBVs_action[cbv2_id] = [throttle2, steer2, brake2]
```

环境 `VectorWrapper.step()` 已支持 action dict 对多 CBV 同 tick 应用，不需要重写 CARLA stepping 机制。

---

# Step 12. 新 JointRolloutBuffer

不要强行复用当前 per-CBV trajectory buffer 语义。

新增：

```text
joint_buffer.py
```

每个 joint state 存：

```python
{
    "feature_1": ...,             # PlutoFeature
    "feature_2": ...,
    "old_joint_logits": [G1,G2],
    "joint_advantage": [G1,G2],
    "valid_mask_1": [G1],
    "valid_mask_2": [G2],
    "pair_ids": ...,
    "diagnostics": {             # 可选
        "p12": ...,
        "coalition_gain": ...,
    }
}
```

建议按 **joint state** 直接存，不必像当前 `CBVRolloutBuffer` 那样等待单个 CBV trajectory done 才 flush，因为 Joint-RIFT advantage 已经在当前 state 完整计算，不依赖 GAE。

容量初版可仍从 `4096 state samples` 开始。

---

# Step 13. JointDataModule padding

每个 sample 的 `G1,G2` 不同。

batch 中分别 pad 到：

```text
max_G1
max_G2
```

得到：

```text
joint_old_logits [B,max_G1,max_G2]
joint_advantage  [B,max_G1,max_G2]
joint_valid_mask [B,max_G1,max_G2]
```

其中：

```python
joint_valid_mask = valid1[:, :, None] & valid2[:, None, :]
```

训练/验证仍可先沿用当前 RIFT：

```text
90% / 10%
batch size 初版不要直接照搬 256
```

因为 joint matrix 更大，建议先 profile 后决定，可能从 `32/64` 开始。

---

# Step 14. JointTrainer Stage 1

## 14.1 参数冻结

```python
for p in pluto_model.parameters():
    p.requires_grad = False

for p in joint_head.parameters():
    p.requires_grad = True
```

## 14.2 Forward

对 batch 的两个 feature 分别 forward（可合并成一个大 batch 再 split）：

```text
logits1, q1, traj1
logits2, q2, traj2
```

trajectory generator 不需要 gradient。

joint head 得：

```text
new_joint_logits
```

## 14.3 Loss

```python
new_logp = masked_log_softmax(new_joint_logits.flatten(1))
old_logp = masked_log_softmax(old_joint_logits.flatten(1))
ratio = exp(new_logp - old_logp)
```

之后复用 RIFT dual-clip 公式。

## 14.4 State-balanced mean

不要：

```python
objective[mask].mean()
```

而要：

```python
state_loss = masked_sum(objective, pair_dims) / valid_count
loss = -state_loss.mean()
```

---

# Step 15. Stage 2：解冻共享 RIFT pi_head

Stage 1 通过后，再：

```python
for p in model.planning_decoder.pi_head.parameters():
    p.requires_grad = True
```

其它 Pluto params 保持 freeze。

optimizer parameter groups：

```python
AdamW([
    {"params": joint_head.parameters(), "lr": lr_joint},
    {"params": pi_head.parameters(), "lr": lr_pi},
])
```

推荐起点：

```text
lr_joint = 1e-4
lr_pi    = 1e-5 ~ 3e-5
```

注意：Stage 2 必须重新收集 on-policy buffer。

不要把 Stage-1 old logits / advantage 原封不动继续训练，因为当前 joint policy 已改变。

---

# Step 16. Config

建议：

```text
rift/cbv/planning/config/joint_rift_pluto.yaml
rift/cbv/planning/fine_tuner/rlft/config/joint_rift_training.yaml
```

核心字段：

```yaml
policy_name: joint_rift_pluto
buffer_capacity: 4096
num_joint_cbv: 2
num_modes: 12
joint_horizon: 40
gamma: 0.98

joint:
  lambda_delta: 1.0
  latent_dim: 64
  traj_sample_interval: 5
  stage: 1

reward:
  lambda_pressure: ...
  lambda_coalition: ...
  lambda_pair_collision: ...

training:
  lr_joint: 1.0e-4
  lr_pi: 1.0e-5
  train_pi_head: false
```

Stage 2：

```yaml
train_pi_head: true
```

---

# Step 17. 日志必须一开始就加

每个训练 epoch：

```text
joint/loss
joint/policy_entropy
joint/residual_std
joint/base_std
joint/residual_base_ratio
joint/ratio_mean
joint/ratio_clip_fraction

reward/unary_1
reward/unary_2
reward/P12
reward/coalition_gain
reward/pair_collision_rate

selection/independent_same_as_joint_rate
selection/oracle_gap_if_debug
```

否则 joint head “训了但没影响行为”很难发现。

---

# Step 18. 不建议在第一轮代码中做的修改

先不要：

```text
改 Pluto trajectory head
改 encoder
训练 Ego
引入 learned ego response model
动态图多 CBV coalition
把 BeTopNet 整个模型搬进来
引入 CVaR
```

这些都会极大增加故障定位难度。
