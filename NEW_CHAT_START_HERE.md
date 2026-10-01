
# 新聊天快速接手说明

请先阅读同目录的 `JOINT_RIFT_FULL_HANDOFF.md`。

当前课题是：在 RIFT 单 CBV scorer 基础上，对 2 辆 CBV 的 Pluto/RIFT 候选笛卡尔积学习 joint residual scorer：

\[
S_{ij}=\ell_i^{RIFT}+\ell_j^{RIFT}+\delta_{ij}.
\]

已定技术路线：

- 2 CBV 固定 pair；
- full \(G_1\times G_2\) candidate product；
- candidate latent + common-frame trajectory；
- PDM-Lite nominal trajectory；
- temporal crossing weight：
  \[
  e^{-|\Delta t|/\tau}
  \]
- coalition gain：
  \[
  P_{12}-\max(P_1,P_2)
  \]
- Stage 1：只训 joint head；
- Stage 2：joint head + RIFT pi-head；
- trajectory generator 永远 freeze；
- joint state-level RIFT-style dual-clip training；
- 必须先做 OJG / NonFactorizableRatio / oracle rank/NLL / Both-Necessary feasibility diagnostics。

当前 original joint implementation 已有大部分架构，但曾发现：
1. fresh joint 默认从 IL Pluto 而不是 trained RIFT 起步；
2. interaction 原来只看空间 crossing；
3. Runner nominal 参数破坏 GRPO 等 baseline；
4. vector env fallback 粒度过粗；
5. feasibility/eval metrics 不完整。

已经生成 audited patch package，但因为 GitHub integration 写权限返回 403，没有直接 push 到远端。

新聊天下一步应优先：
1. 应用 audited patch；
2. 逐文件 diff 审核；
3. `py_compile` / unit tests；
4. 单 route CARLA smoke test；
5. Phase-1 feasibility；
6. 决定是否进入 Stage 2。

不要重新从 reachable tube / dynamic coalition /复杂 reward 发散。
