# Joint-RIFT：版本复盘、代码机制审计与改进实验方案

日期：2026-10-13。审计工作区 HEAD：`6f7a921`。本文中的“本次复算”来自本地文件的实际运行；“历史记录”来自交接材料及归档文件；“建议”尚未训练或闭环验证。

后续实现说明：本文主要审计上述历史HEAD；其后已在工作区实现可切换的V5目标、完整Stage1缓存和离线拟合入口，详见 [V5实现与运行手册](JOINT_RIFT_V5_IMPLEMENTATION_AND_RUNBOOK.md)。本文中“未修改生产代码”仅描述最初审计阶段。

## 1. 核心判断

**当前系统已经能运行，但现有证据没有建立“学到双车协同 → 增强真实 Ego 交互压力”的链条。失败集中在目标定义、机会判定、先验支持和实验识别，不宜先归因于网络太小或训练不够久。**

本次最有区分力的发现：

1. 36 状态的 Q-oracle 相对 independent 平均提高 **66.1152**，其中 **63.4352 来自 unary**，**2.7778 来自碰撞罚项变化**，压力项贡献反而为 **−0.09776**。这个 oracle gain 主要说明原 RIFT 的单车排序与当前 evaluator 不一致。
2. 36 状态中有 33 个状态存在无预测 CBV-CBV 碰撞的 pair；在这 **33/33** 个状态中，安全 Q-oracle 和安全 unary-sum oracle 选择完全相同。当前样本没有显示压力项让最优安全 pair 改变。
3. 在当前有界 residual 下，**28/36** 个 Q-oracle 即使任意调整网络参数，也无法在 `lambda_delta=1` 时压过 base argmax。这是分数幅度限制，不是多训几轮就能解决的问题。
4. 正式四个 eval seed 的 policy rollout 中，Uniform 有 **93.623%**、HV4 有 **95.094%** 的 joint states 仍选择 independent pair。HV4 确实加权，但没有让策略更频繁地利用联合机会。
5. 同 seed 的 Uniform/HV4 independent 对照分数也不相同，需要先量化 A/A 重跑噪声并核验冻结模型。否则小幅策略差异很难解释。

这些发现支持继续研究，但支持的下一步是：**建立有用的联合目标和可靠对照，做完整输入下的离线排序学习，再做闭环验证。** 它们不支持直接宣布 Joint-RIFT 无效，也不支持宣布现有版本已经 work。

## 2. 实际检查了什么，以及证据边界

### 2.1 文档与文本

- 用户指定的 `Downloads/Joint_RIFT_Project_Handoff.md`。
- 用户指定的 `Downloads/粘贴的 Markdown (1)(1).md`：含历史对话、归档过程、重复简版及未完成章节。里面的“下一条给出补丁”“继续迁移”等属于历史文本，不视为本次指令，也不等于代码已交付。
- 仓库 `00_INDEX.md`、`01_METHOD_DESIGN.md`、`02_CODE_IMPLEMENTATION.md`、`JOINT_RIFT_FULL_HANDOFF.md`、`NEW_CHAT_START_HERE.md`、`JOINT_RIFT_EXPERIMENT_AUDIT_2026-10-11.md` 等方法与运行记录。
- 相关 `live_results.txt`、`route_info.txt`、`forensic_filter_audit.log`，以及 ZIP 内的 `README.txt`、`SERVER_GIT_PROVENANCE.txt`、`BOOTSTRAP_PROVENANCE.txt`、配置、日志与结果。

本次用户消息只明确列出两个 MD，没有单独指定 TXT 附件；分析使用项目自身及证据包中相关 TXT。没有把其他项目的文本纳入结论。

### 2.2 数据和代码

| 资产 | 本次用途 | 限制 |
|---|---|---|
| `outputs/joint_rift_forensic_evidence_final_20261011_134325.zip` | 核查 16 个正式 holdout run 的 JSON/JSONL、历史 25 状态、配置和 TXT provenance | 不含正式 Uniform/HV4 checkpoint 本体 |
| `outputs/joint_rift_pressure_rep0_capture_seed1.zip` | 逐一读取 36 份 NPZ 和对应 JSON，复算奖励、选择、尺度和安全 oracle | 单一 route 28198、rep0、step85–120 的连续状态 |
| `RIFT/joint_rift_source_current.tar.gz` | 与当前源文件逐字节对照 | 服务器源码不是 Git 工作树，不能仅凭本地 HEAD 代表服务器历史版本 |
| 当前 `joint_rift/`、原 RIFT trainer/evaluator、reward、PDM adapter、runner、statistics scripts | 对照实际调用链、目标和指标方向 | 本地缺 CARLA/Lightning，未运行整套仿真训练 |

最新源码包的 Python 文件与当前工作区相比，只有 `joint_rift_audited.py` 不同：源码包有后加的 forensic repetition/pressure/collision filters，当前文件没有。其余对应 `.py` 一致。因此核心数学代码审计适用，但不能把当前文件当作最新取证配置的完全复现版本。

归档中的 train210 和 holdout10 XML 已实际解析：分别 210 和 10 条，ID 交集为 0。当前裸仓库没有这两份 XML及部分正式 YAML，需要从归档恢复或重新生成并校验 hash。

**更严格的泄漏边界**：共享 RIFT bootstrap 的 route 训练历史仍不可核实；TXT 只提供 checkpoint 路径、大小和 SHA256，没有上游 route/log。因此可以说“Joint head 的 train210 与 holdout10 分开”，尚不能证明整个模型从未接触 holdout。

### 2.3 本次产物

- [离线证据审计脚本](RIFT/scripts/audit_joint_rift_evidence.py)：只需 NumPy，直接读 ZIP，不解压执行归档代码，不加载 pickle。
- [真实源码数学探针](RIFT/scripts/probe_joint_rift_math.py)：只需 Torch，对 scorer 和 loss 的实际源代码做隔离检查。
- [诊断数值测试](RIFT/tools/test/test_evidence_audit.py)。
- [最终复算摘要](outputs/joint_rift_review_20261013/final/summary.json)。
- [逐状态分解](outputs/joint_rift_review_20261013/final/states.csv)、[尺度扫描](outputs/joint_rift_review_20261013/final/scale_sweep.csv)、[正式路线配对差值](outputs/joint_rift_review_20261013/final/formal_route_comparisons.csv)。
- [数学探针及原有测试结果](outputs/joint_rift_review_20261013/math_probe_verified.json)。

未修改生产 policy、reward 或 trainer，未覆盖历史结果、checkpoint 和取证数据。本次交付是可复查的分析工具与实验决策依据。

## 3. 各版设计究竟改了什么、为什么没有形成可靠提升

这里区分“模块版本”与“实际实验版本”，避免把写了 temporal reward 等同于完成了一次有效新方法验证。

| 版本 | 动机与实际改动 | 已有结果 | 问题与解释 |
|---|---|---|---|
| V0：Pluto / independent RIFT | Pluto 生成每个 reference line 的 12 模态，RIFT 学单车偏好 | 早期 Ego driving score：Pluto 71.604、RIFT 94.679 | 原 RIFT dense reward 侧重自然驾驶与避碰；独立策略没有显式 pair compatibility。不能据分数给它贴“强对抗先验”标签 |
| V1：joint residual | `Sij=log p1(i)+log p2(j)+Δij`；零初始化、共享 encoder、交换对称 | 数学契约测试通过 | 能表达非加性分布不代表已学到；冻结 base 极尖锐，有界 residual 的覆盖范围受限 |
| V2：时间交互 | 空间 crossing 加 `exp(-abs(arrival_gap)/tau)`；保留多事件 onset | 合成 near/late crossing 可区分 | 仍是中心线相交代理，遗漏跟车、并行、占道、近失；union 使用 CBV onset 时间，未建立真实 Ego 响应 |
| V3：joint reward | `Q=U1+U2+P12+CG−20C` | Q 矩阵存在非加性结构及 evaluator oracle gap | 大部分高收益可由 unary/避碰解释；CG 与 both-necessary 都是固定 nominal 下的代数代理 |
| V4a：早期 Stage1 | 冻结主干，只训 residual；220 routes ×2 | 有 11 个左右 checkpoint；早期 Joint score 93.557 | 开发 route 来自训练集；历史 prototype 还出现过 IL/RIFT bootstrap 混淆，不同阶段结果不能合并 |
| V4b：audited Uniform | 强制 RIFT bootstrap、修复接口和数值问题；train210 ×2 | 420 records、11 次 4096-state 更新；按 train-time val_loss 选 episode31 | 修复运行正确性，但仍沿用 Full-Q 及 uniform pair ratio loss；不能仅按 val_loss 推断行为最佳 |
| V4c：HV4 | 给 gain≥10、NFR≥.01、oracle 安全、全矩阵碰撞率≤.05 的 state 四倍权重 | 干净审计 HV 约 3.1361%，平均权重 1.094083；规则确实执行 | 规则主要筛 Full-Q 的改进而非 pressure-specific coordination；整体碰撞率门槛排掉很多存在安全 pair 的状态 |
| V4d：正式 holdout10 | 固定 Uniform/HV4 checkpoint，各做 policy/independent，eval seed0–3，各20 records | 未建立稳健的 HV4 优势；单一训练 seed | 指标方向、模拟噪声和 route-level 机制未识别；反复用该10条调设计后，应把它们视为开发集 |
| V4e：25 状态取证 | 在若干选定路线导出 dense matrices | README/inventory 显示25/25 policy=independent，0/25 policy=oracle | 样本不是所有状态；多数无 pressure，不足以直接否定整个 joint 空间 |
| V4f：pressure-safe 与36状态 | 严格过滤失败后放宽采样，捕获 route28198 连续36状态 | 本次已完整复算 | 有 reward 非加性，但安全 Q 最优未超出安全 unary 最优；仍不足以支撑强协同主张 |
| Stage2 | 预备解冻共享 pi_head，生成器继续冻结 | 有实现与配置，未找到正式训练/评估证据 | 不是已经跑过但失败的版本；应等机会、可学性和闭环代理验证后决定 |

历史 `route27494` 局部比较曾记录 policy−independent 为 `[12.88,3.39,13.91,8.42]`。它来自挑选的高价值路线，且包含 blocked/timed-out 结局；正 driving-score 差首先是 Ego 驾驶结果更好，不能直接称为双 CBV 对抗增强。

## 4. 36 状态复算：最关键的证据

### 4.1 重构正确，研究解释需要改变

按该 run 配置重构：

```text
Uij = U1(i) + U2(j)
Hij = P12(i,j) + CG(i,j)
Qij = Uij + Hij - 20 Cij
```

最大重构误差 `1.42e-14`。NPZ 选择与 JSON 的 independent/policy/oracle 索引一致。

| 指标 | 本次复算 |
|---|---:|
| 平均 NFR(Q) | 1.94457% |
| 平均 NFR(H) | 0.83301% |
| 平均 NFR(Δ) | 0.21122% |
| 平均 oracle Q gain | 66.11518 |
| 同一 oracle pair 的 unary gain | +63.43517 |
| 同一 oracle pair 的 pressure H gain | −0.09776 |
| 同一 oracle pair 的 collision penalty gain | +2.77778 |
| 平均 std(unary-sum) | 28.5953 |
| 平均 std(H) | 0.13549 |
| policy=independent | 28/36 |
| policy=Q-oracle | 0/36 |
| Q-oracle both-necessary（metadata） | 0/36 |
| policy pair 预测 CBV-CBV 碰撞 | 10/36 |
| Q-oracle pair 预测 CBV-CBV 碰撞 | 5/36 |
| 当前 HV4 规则 eligible | 0/36 |

unary 与 H 的矩阵标准差之比约 211；对 Full-Q 统一 standardize 不会改变这个相对量级。既有 loss 主要面对单车排序、避碰与可驾驶性；给整个 state 四倍权重并不会单独增强压力成分。

NFR 是“最佳行列加性拟合解释不了的方差占比”，不是协同率，也不是 rank 优势。应同时看非加性绝对幅度、pair 排名和安全区域，不能只看比例。

### 4.2 必须加入更强的 oracle 对照

原来的 `Q-oracle − RIFT argmax` 混合了三类差距：

- RIFT 的单车排名与当前 unary evaluator 的差距；
- 两车碰撞兼容性；
- 压力引起的额外联合选择收益。

本次在 `safe = valid & ~pair_collision` 上分别求：

```text
safe_unary_pair = argmax_safe (U1 + U2)
safe_joint_pair = argmax_safe Q
```

33 个存在 safe pair 的状态里，两者索引 **全部相同**，Q 差全部为 0；另外 3 状态没有 safe pair，需要明确标记不可行，不能虚构“安全最优”。5 个不受硬约束的 Q-oracle 碰撞状态中，有2个其实存在非碰撞候选，说明 `−20C` 并不保证避碰。

这是一条局部而明确的证据：**当前36状态的 Full-Q 上界主要靠修正 unary 与安全选择实现，并不需要 pressure head 才能达到。** 它不证明其他路线也没有联合机会，应把这一对照扩展到独立多路线数据。

后续针对用户核心假设的补充复算：**这也不能推出“没有比independent更好的交互候选”。** 在independent本身预测无CBV-CBV碰撞的26个状态中，26/26存在另一组同样无预测CBV-CBV碰撞、unary总回报不下降且H更高的候选，H改善均值约0.28347。其中19状态有H改善超过0.1的候选，且这些19状态至少存在一个改善候选通过lambda=1的乐观幅度界检查；learned policy在这26状态中没有一次同时实现安全、unary总回报不下降和H提高。安全Full-Q最优等于安全unary最优，与存在更好H候选完全可以同时成立。详见 [机会重检摘要](outputs/joint_rift_review_20261013/opportunity_recheck_v2/summary.json)。0.1只是敏感性检查，不是统计显著性门槛；幅度界通过不是实际网络可实现的充分条件。“安全”仅指当前模型预测CBV-CBV无碰撞，不能代替所有交通质量约束。

具体例子：route28198、rep0、step90，independent和learned都选(16,16)，Q=−88.52089、H=0.39470、CG=0.08862；候选(2,3)同样无预测CBV-CBV碰撞，Q=−18.23356、H=1.02273、CG=0.32000，unary总回报提高69.65930。目标pair的lambda必要下界约0.6941，因此不能把该例的未选中单独归因于lambda=1的硬上限。CG>0.1符合当前nominal“双边际为正”定义，但不构成BeTop一致性或真实闭环协同证明。

另一项补充：当前nominal both-necessary可以由`CG>epsilon`精确恢复，因为`min(P12−P1,P12−P2)=CG`。最初复算摘要中“缺p1/p2所以无法重新计算both-necessary”的限制表述过强；缺失的仍是分别的两个边际及真实闭环反事实。新摘要已更正。

安全 mask 有孔洞；本次 safe-H NFR 使用 row+column 最小二乘拟合。不能把碰撞位置填0后直接 double-center，也不能把安全约束本身引入的联合相关性解释成压力协同。

### 4.3 residual 的硬能力边界

实际代码：

```text
Δij = lambda_delta × std(B) × tanh(raw_deltaij)
B = log p1 + log p2
```

令 base 最优 pair 为 b、目标 pair 为 o，`g=Bb−Bo`。目标严格胜过 b 的必要条件是：

```text
g < 2 × lambda_delta × std(B)
```

这是必要条件，不是网络能同时实现所有目标的充分条件；等号情况下至多接近打平，不能保证 argmax 改变。

这36状态的 `std(B)` 平均25.4569，oracle gap 平均62.5825，所需 lambda 下界平均1.22885；**28/36 的下界大于1**。oracle 的 base probability 中位数仅约 `1.81e-30`。

因此，保留极尖锐先验且把 residual 幅度固定为当前尺度，同时要求其选中低支持 oracle，三者相互冲突。

### 4.4 放大旧 residual 的结果

这是 `B + alpha × 已训练Δ` 的后处理，**不是重训不同 lambda 的实验**。

| alpha | 改变 independent 的状态数 | oracle 命中 | 平均 Q 增益 | 平均 H 增益 | 预测碰撞状态数 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 | 10 |
| 0.5 | 2 | 0 | 1.2074 | 0 | 10 |
| 1 | 8 | 0 | 1.5244 | −0.00885 | 10 |
| 2 | 20 | 0 | 17.0393 | +0.00490 | 11 |
| 5 | 31 | 0 | 33.5295 | +0.00669 | 11 |
| 10 | 34 | 0 | 35.1460 | +0.00610 | 12 |

放大可以改变行为和提高虚拟 Q，但增加的 H 很小，预测碰撞还会上升。它适合诊断“分数够不够翻转”，不能单独作为协同修复。

所有36状态属于同一个连续片段；`36×576=20,736` 不是20,736次独立试验，本报告没有据此计算显著性。

## 5. 正式评估重读：目前能说什么

### 5.1 分数方向

`rift/scenario/statistics_manager.py` 明确定义：

```text
score_composed = max(score_route × score_penalty, 0)
```

它是 Ego driving score，数值大代表 Ego 路线/违规综合结果好。`compare_joint_rift_eval.py` 把正差命名为 `policy_wins`，HV 汇总脚本还写“positive interaction supports ablation”，这些名称没有证明它与 CBV 的研究目标一致。

本课题需要分开报告：

- Ego 交互难度：合适的 RTTC/PET、制动响应、受迫减速/延误、有效交互持续时间；
- 场景质量约束：CBV 碰撞、越界、逆行、加速度与 jerk、轨迹跟踪误差；
- 协同机制：在单车干预、独立选择和安全 unary 对照下仍存在的增益。

如果目标是受约束的 adversarial 场景，降低 Ego driving score 可以是辅助现象；无原因的 blocked/timeout 并不自动是成功。若目标是自然可控的多车互动，分数方向又不同。应在下一轮训练前冻结这一定义。

### 5.2 四 eval seed 原始差值

以下均为 `policy−independent` 的 Ego score 差，不贴胜负标签。

| eval seed | Uniform | HV4 | HV4−Uniform 的差分 |
|---:|---:|---:|---:|
| 0 | +1.55400 | +1.25400 | −0.30000 |
| 1 | −3.27200 | +1.33102 | +4.60302 |
| 2 | −6.74250 | −6.47200 | +0.27050 |
| 3 | −2.48100 | −2.64550 | −0.16450 |
| 均值 | −2.73538 | −1.63312 | +1.10225 |

差分中位数为 +0.053，均值主要受 seed1 影响；仍只有1个 training seed，无法推出对训练随机性的稳健性。若目标是增加 adversarial difficulty，正差也未必是希望的方向。

16个正式 run 都有20个唯一的 `RouteScenario_<id>_rep<r>` 记录，现有 comparator 在这批结果上没有把rep0/rep1覆盖掉。复算 CSV 保留了每条记录的状态、Ego分数差和CBV质量差，便于逐路线查因。

### 5.3 策略到底有没有改变

在正式 **policy rollout** 上按实际 joint-state 数统计：

| 处理 | joint states | 与 independent 同选择 | 命中 Q-oracle | Q-oracle both-necessary |
|---|---:|---:|---:|---:|
| Uniform | 5,363 | 93.623% | 0.2051% | 0.0932% |
| HV4 | 5,850 | 95.094% | 0.2222% | 0.1880% |

这是描述性统计，不能把状态数当独立样本做显著性。HV4 更接近 independent 是事实，但不单独证明更差：有用策略可能只应在少数关键状态变化。需要看“有明确安全协同机会的状态上是否正确改变”。

### 5.4 同策略 A/A 是当前缺失的必要对照

两组 independent 模式理论上都只看相同冻结 RIFT base；若 tensor 和初始世界状态一致，应得到相同初始动作。但实际同seed的 `HV4 independent−Uniform independent` Ego分数差为：

```text
seed0: +0.25050
seed1: −3.28602
seed2: +0.78450
seed3: +4.28700
```

这与要解释的效果量同量级。可能来自 CARLA 重跑差异、初始化/配置不一致，或冻结模型存在漂移；当前正式 checkpoint 未随证据包提供，无法在本地排除后两项。不能只凭同 seed 就假定严格匹配，也不能直接断言这些差异全是模拟噪声。

下一步先比较 `model.*` state_dict hash、初始actor状态、前N步候选与控制，再做重复 A/A。仅通过两个 noisy independent 结果做差分并不能消除噪声。

## 6. 为什么不 work：按因果链排序

### 6.1 优化目标与课题贡献没有对齐

`DenseRewardModel` 的碰撞、越界、舒适度、车道对齐和速度项构成 unary，且 unary evaluator 把 nominal Ego 也作为碰撞障碍。当前系统并非直接优化“让 Ego 更困难”的闭环奖励。

`Q-oracle > independent` 因而容易来自更自然、更快、离中心线更近或更安全的单车选择。新增 joint 网络有时只是替冻结 unary scorer 纠错。应加入安全 unary oracle/可学习 unary 纠错对照，否则提升难以归因于协同。

### 6.2 CG 的非线性会制造“看起来像协同”的结构

设 `O=sum_t gamma^t x_i(t)y_j(t)`，则代码的软 union 满足：

```text
P12 = P1 + P2 − O
CG  = P12 − max(P1,P2) = min(P1,P2) − O
H   = P1 + P2 + min(P1,P2) − 2O
```

即使两辆车在互不相关的两个时间各自相交，`min(P1,P2)` 也能产生非加性矩阵。

对 union 自身，二阶增益 `P12−P1−P2+Pempty=−O≤0`；它表达的是覆盖/去重，无法描述某辆车改变 Ego 反应后使另一辆车价值增加的响应型正协同。覆盖可以是合理的协同目标，但需要按覆盖目标评价，不能把它升级解释为因果配合。

当前 `both_necessary` 只是 `P12−P2>eps` 且 `P12−P1>eps`。它没有重新运行移除一辆 CBV 后的 Ego，更没有验证另一辆车行为是否必要，宜叫“nominal coverage 双边际为正”。

### 6.3 采样门槛把“候选有碰撞”误当“状态不可用”

253-state strict forensic 记录实际显示：253 route匹配 → 91 repetition匹配 → 32 time-window匹配 → coalition std过滤后0。这证明 writer 没拿到合格样本；不证明学习失败，也不证明采样空间一定有足够协同。

36-state broad capture 平均候选pair碰撞率39.80%，但33状态仍有safe pair。使用全矩阵碰撞率≤5%直接拒绝状态，会排掉正需要协调避碰的交互片段。

建议按“是否有足够高质量 safe candidates、是否存在压力排序收益”筛样本；报告不可行状态比例，不要求所有候选都安全。

### 6.4 frozen prior 和 residual 上限不匹配

低支持 oracle 的 base NLL 极大。tanh cap 使大部分不可翻转，饱和还会减弱梯度。另一方面 `std(B)=0` 时，无论 raw head 输出什么，Δ都被乘为0；本次数学探针已复现均匀先验完全无法产生residual的边界。

可试验的改法是先验温度 `B/T_base`、与base独立的有下限残差尺度、适量 unary 校准。它们分别改变信任先验、幅度和单车错误，应分别做消融。不是直接同时解冻所有网络。

### 6.5 当前 loss 是 RIFT 式全候选 surrogate，不是标准 PPO 期望

`joint_dual_clip_loss()` 对每个state内所有valid pair等权平均：

```text
L_uniform = − mean_a dual_clip( (pi_new(a)/pi_old(a)) × A(a) )
```

它继承原 RIFT 的设计；不能简单称为写错了PPO，更不能假设当前在用MSE。但如果将其解释为标准旧策略期望，则应是：

```text
L_oldpolicy = − sum_a pi_old(a) × dual_clip(ratio × A)
```

本次用真实函数在同一非均匀先验合成state上比较初始梯度，方向余弦仅 **0.34974**；加上旧概率权重的未裁剪一阶梯度与精确 `−E_pi[A]` 一致。

当state内mean(A)=0、new=old时，现有uniform目标的logit梯度为 `−A/N`；精确期望的梯度为 `−pi(a)(A(a)−E_pi A)`。两者强调的候选不同。前者可能主动纠正低概率候选，后者可能因oracle支持极低而几乎无梯度。

因此，**不能仅把旧概率权重补上就宣称修好**。应在固定数据上对比：现有surrogate、精确期望/信任区间目标、soft-target ranking/distillation，评价真正的safe regret与压力排序，而不是比较不同loss的数值。

当前收集实际argmax执行，候选回报由虚拟evaluator穷举，并非实际采样所有joint动作得到闭环回报。“on-policy buffer”保证的是state与checkpoint出处，不等于这些reward已经来自真实闭环策略分布。

### 6.6 表征和评价的几何语义还没统一

实际代码中的两个明确问题：

- `joint_score_head.py` 的encoder采样前40帧，但 minimum-distance geometry 对输入完整80帧取最小值；evaluator目标只有40帧。探针只改变40帧之后的位置，Δ改变 **1.7966**。输入含更远未来本身不必然错误，但与“40-frame pair geometry”约定不一致，需显式统一或做80帧上下文消融。
- `temporal_interaction.py` docstring 写连续cluster保留最大weight，实际只取onset的weight；并且事件保存在CBV segment时间轴，未输出conflict对应的Ego时间/位置。两车在相同Ego冲突机会前后经过，可能被当成两个时间覆盖机会。

还有三个待验证表征假设：

- scorer没有显式输入 Ego nominal，只有候选latent、common trajectories和两车最小距离；latent可能隐含Ego观测，但未证明包含nominal规划与速度意图。
- 单个最小距离不能区分接近后退出、长时间阻挡、追尾或相互绕行。
- exact中心线交点无法反映平行挤压、跟车、占据可行通道等主题相关交互。

先做输入可学性测试；只有补充信息显著降低held-out regret时，才值得扩大交互网络。

### 6.7 验证loss与checkpoint选择不能代表协同行为

当前 datamodule 对连续state做random_split，同一route/actor片段相邻时间很可能跨train/val。loss是相对收集策略、归一化advantage的surrogate，不同buffer的难度和参考策略不同。跨buffer选最小val_loss不一定有可比性。

正式Uniform/HV4都选中episode31，即第一轮附近的checkpoint。全run虽完成11轮更新，但主评估没有证明后10轮是否改进行为。建议报告学习曲线，并用独立开发route的safe regret、pressure ranking及闭环验证选checkpoint，最终test保持封存。

## 7. 下一版怎么做：最小、可否证的路线

### 7.1 第一关：实验和数据可信

先完成以下小实验，比继续跑420个scenario更便于查因：

1. **A/A复跑**：相同checkpoint、seed、路线、初始配置，3条代表路线各重复5次，输出目录独立；核验base模型tensor、candidate与前N步controls。报告路线score、压力和CBV质量的重复差异。端口可不同，其他条件记录清楚。
2. **collection/re-forward一致**：完整缓存至少32个state，重新前向得到candidate顺序、common frame、logits、Δ，比较CPU/GPU/独立collate、train/eval模式；训练前ratio应接近1。若改变了候选身份，再漂亮的loss也无效。
3. **恢复正确配置资产**：对照tar中的forensic helper修复或恢复过滤逻辑；增加route、rep、time窗口和缺诊断字段的明确检查。当前代码忽略相关YAML键，会使取证样本与预期不同。
4. **冻结确认**：generator与latent-producing模型tensor hash在训练前后一致；不能只检查requires_grad。

建议验收：配置与hash完整，数据来源一致，无静默候选置换；A/A噪声已量化。容差按实际dtype/device设定，不要求不同CARLA run逐浮点一致。

### 7.2 第二关：证明有“压力特有”的联合机会

采多路线、多个独立episode的宽数据，覆盖：双车无关、单车强/另一车弱、同冲突重复、时间互补、避碰协调、两车都可能约束Ego，以及无安全pair。

至少比较：

| 对照 | 回答的问题 |
|---|---|
| independent RIFT | 当前先验的实际表现 |
| unary evaluator oracle | 冻结单车排序还差多少 |
| safe unary oracle | 加了避碰约束后，是否已解释全部收益 |
| safe Full-Q oracle | 当前目标的有效上界 |
| 在unary质量匹配的safe集合上最大化pressure | 压力收益是否只靠牺牲自然性获得 |
| learned joint / 可学习unary纠错 | 增益是否确实需要pair交互 |

核心量应从 `Q_oracle−RIFT` 改为：

```text
G_pressure = Q_safe_joint − Q_safe_unary
以及相同unary质量预算下的 H 改善
```

阈值应依据重复评价噪声、候选质量尺度和最小有意义行为变化确定。只统计NFR>0不够；要有跨独立episode的安全排序改变、两车边际及可解释事件。

若多路线仍重复出现本次33/33现象，就应修改交互目标或场景分布；不要用更大scorer拟合没有排序增益的代理目标。

### 7.3 第三关：固定数据上学会排序

**现有36份NPZ不足以原样训练现有scorer**：缺candidate latent和raw PlutoFeature。不能把它们拼上另一rollout的latent，也不能用一个trajectory-only toy network的成功替代原head成功。

下一次缓存至少保存：

- immutable `run_id, route_id, rep, env, episode, step, pair_ids`；
- feature_1/2或完整candidate latent，raw/local/common trajectories，candidate identity与valid masks；
- base logits、raw_delta、scaled_delta、old joint logits；
- Q和全部分量，p1/p2、event traces、冲突点、CBV/Ego arrival times；
- 预测rollout/实际执行轨迹及collision/offroad/comfort标签；
- nominal current-state alignment、dt/horizon、reward参数、checkpoint/source/config hashes。

先做小样本过拟合，再按route/episode分组验证，避免random split连续帧。

建议只开展三个主目标对照：

1. 现有RIFT式全pair surrogate，作为可复现基线；
2. 有明确prior/KL约束的dense reward排序蒸馏或精确期望，验证full-Q可学性；
3. 在同样质量与安全约束下增加pressure-specific pair辅助目标，验证协同信号可学性。

可采用soft teacher：

```text
pi_teacher(a) ∝ pi_prior(a)^beta × exp(Q_normalized(a)/T_teacher)
L_distill = KL(pi_teacher || pi_theta)
```

beta与T分别控制先验和目标锐度，采用独立开发集选择。强先验可能把teacher也锁死，必须同时报告目标pair支持和teacher regret。

interaction辅助目标优先比较2×2差分：

```text
H(i,j) − H(i,j') − H(i',j) + H(i',j')
```

它去除unary行列影响；只在四个角都有效、质量可比较的支持上使用。碰撞是独立安全目标，不应把Full-Q中碰撞引起的非加性当作压力标签。

建议的初步验收标准（设计建议，不是已有事实）：小训练子集safe-Q regret明显接近0；在独立route上超过可学习unary对照；压力收益和安全约束共同改善。具体数值门槛应在看最终test前根据dev噪声固定。

### 7.4 第四关：最小结构调整

保留Pluto generator与候选笛卡尔积，先试：

```text
Sij = calibrated_base(i,j)
    + unary_correction_1(i) + unary_correction_2(j)
    + pair_interaction(i,j,ego_intent)
```

这是把当前混在residual中的任务分开：

- unary correction解释本次很大的单车排序差；
- pair term承担压力/兼容性的不可加部分；
- safety作为显式可行性或风险约束单独评估。

先验温度、残差尺度下限、nominal encoder、pressure辅助loss分别加开关做消融。采用纯pair差分或行列投影约束时，要明确其支持；有孔洞的safe mask用加权最小二乘。

若无safe pair，设置有记录的最小风险fallback，报告fallback率。虚拟碰撞预测有误差，不能在还未校准时把新hard mask当成真实安全保证。

只在这些小实验支持下再解冻pi_head。现在解冻它很可能主要修正unary，表面获得更高Q却仍没有双车贡献；这可以作为有效baseline，但不能混写为协同成果。

### 7.5 第五关：检验虚拟代理与闭环是否一致

固定nominal的评分不受candidate pair影响，所以无法直接表现“CBV1先让Ego制动，CBV2利用其响应”。先在少量可复建episode对候选进行短闭环分支回放，验证代理排序：

- 按safe-Q/H分层选pair，而非只选oracle；
- 记录nominal与真实Ego在0.5/1/2/4秒的位移、速度、arrival gap和交互事件误差；
- 比较虚拟压力排序与真实制动/延误/RTTC等的rank相关和top-pair regret；
- 区分candidate跟踪偏差、Ego响应偏差、背景预测偏差。

CARLA世界不是可随意clone的Python对象；分支实验需要从相同episode种子和动作历史重建，先检验重建前缀误差。

反事实推荐保留车辆并替换为自然/neutral轨迹，而非随意删除actor。删除会改变交通拓扑，混入“少一辆车”效应。至少做both、only1-intervenes、only2-intervenes、neither四条件，Ego每次真实响应：

```text
both必要：F(both) > F(only1)+eps，且 F(both) > F(only2)+eps
响应型二阶增益：F(both)−F(only1)−F(only2)+F(neither)
```

覆盖型协同不一定有正二阶增益，应另报互补覆盖；先明确主张的协同类型，避免用指标定义强行“证明”。

若nominal与真实响应排序弱，再加入少量response-aware评估或校准。此时放宽历史“不用ego surrogate”的方案边界是有证据支持的调整，仍保留RIFT/Pluto候选选择主线。

### 7.6 最后才做正式重训与泛化

在上面关卡通过后，至少3个training seeds，配多个evaluation seeds，固定训练预算和checkpoint选择规则；所有policy使用一致候选空间及质量约束。

主结果同时报告压力、CBV质量、协同消融和Ego outcome；按路线/episode聚类统计，另分training seed，不把成千上万相邻state当独立试验。

现有holdout10已深度参与机制分析，应转作dev。真正final test需要新路线/新episode分布，或从头建立包括RIFT prior在内可审计的分割。仅从已用于Joint训练的210条再挑20条不能得到新的无泄漏最终测试。

## 8. 推荐实验表与决策规则

| 顺序 | 实验 | 输出 | 失败后做什么 |
|---|---|---|---|
| E0 | hash/配置/候选identity核验 + A/A复跑 | manifest、动作前缀对照、噪声范围 | 先修实验基础，不重训 |
| E1 | 多路线safe unary vs safe joint oracle | 分量gain、压力排序改变、不可行率 | 调整目标或采样场景 |
| E2 | 32-state过拟合 + route分组离线验证 | safe regret、pair contrasts、oracle支持 | 分开诊断幅度、信息缺失、模型容量 |
| E3 | prior温度/尺度/nominal/辅助目标消融 | 同state同budget对照 | 保留最少有效改动，不全组合盲扫 |
| E4 | 少量短闭环分支与反事实 | proxy-to-real相关、真实双车边际 | 需要时校准/改造evaluator |
| E5 | 多train-seed正式训练及封存test | 质量受约束的协同增益、学习曲线 | 若仍无增益，收缩主张或改题中子问题 |

E1的机会证据不足时，E2学不会不能归因网络；E2小样本都学不会时，E5训练更多路线不能解决输入或能力边界；E4代理无效时，E2离线loss再漂亮也不能保证闭环work。

## 9. 本次测试结果与可重跑命令

### 9.1 已执行

- 新增offline audit数值测试：**7/7通过**，包括有安全孔洞时加性目标不应被误判、XOR非加性、masked oracle、空支持、非有限值、重复路线保留和不匹配记录拒绝比较。ZIP测试使用内存fixture，不依赖系统临时目录写权限。
- 现有scorer、geometry、interaction_topology、pair_manager测试：**14/14通过**。为避开包级CARLA注册依赖，直接装载对应真实源模块后运行原有unittest；未替换函数实现。
- 数学探针：交换对称误差0；旧概率加权的一阶梯度与精确期望一致；均匀prior残差为0、80帧geometry影响40帧目标两个边界已复现。
- 36状态全量audit完成：Q重构误差约`1.42e-14`，索引与metadata一致。
- 16正式run完整读取，320条route/repetition记录，160条policy/control配对。

本机使用 `C:/Users/Lenovo/anaconda3/python.exe`，Python3.12.7、Torch2.5.1；归档服务器是Python3.8/Torch2.0.1。没有运行Lightning trainer或CARLA，不能把隔离数学测试写成服务器集成/闭环验证通过。

### 9.2 Windows本地复算

从项目根目录运行。每次使用新的output目录，脚本会拒绝覆盖已有目录。

```powershell
$ReviewPython = 'C:\Users\Lenovo\anaconda3\python.exe'
& $ReviewPython RIFT/scripts/audit_joint_rift_evidence.py `
  --dense_zip outputs/joint_rift_pressure_rep0_capture_seed1.zip `
  --formal_zip outputs/joint_rift_forensic_evidence_final_20261011_134325.zip `
  --output_dir outputs/joint_rift_review_new_run

& $ReviewPython RIFT/scripts/probe_joint_rift_math.py --run_source_tests `
  --output_json outputs/joint_rift_review_new_run/math_probe.json

& $ReviewPython RIFT/tools/test/test_evidence_audit.py
```

### 9.3 Linux服务器离线复算

不需要启动CARLA，也不需要配置CARLA egg：

```bash
cd /zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop_2
PY=/zeron-vepfs/tjqc/njl/envs/rift/bin/python
"$PY" RIFT/scripts/audit_joint_rift_evidence.py \
  --dense_zip outputs/joint_rift_pressure_rep0_capture_seed1.zip \
  --formal_zip outputs/joint_rift_forensic_evidence_final_20261011_134325.zip \
  --output_dir outputs/joint_rift_review_new_run
"$PY" RIFT/scripts/probe_joint_rift_math.py --run_source_tests \
  --output_json outputs/joint_rift_review_new_run/math_probe.json
"$PY" RIFT/tools/test/test_evidence_audit.py
```

原始完整CARLA训练/评估命令见 `JOINT_RIFT_EXPERIMENT_AUDIT_2026-10-11.md` 第14节。本次不把尚未实现的新objective写成可运行flag。恢复旧正式实验时，必须先恢复归档XML/YAML并核验explicit checkpoint；eval不能附加会跳过加载的`--no_resume`。

## 10. 当前最值得做的具体改进

**优先做safe-unary对照、完整特征缓存与小样本排序实验。** 它们分别回答“联合机会是否存在”“是否有足够输入”“是否能学会”，投入小且能决定后续方向。

如果这三步成立，最有希望的下一版是：保留Pluto候选、校准冻结先验、显式分开unary纠错与pair交互、用质量受约束的pressure排序监督，再用闭环反事实确认。当前证据尚不足以保证这一版成功，但已经足以排除“只加HV权重/放大旧residual就能证明协同”的路线。
