# Joint-RIFT V5：候选、目标、学习与闭环结果的证据分析

日期：2026-10-14。本文依据用户提供的两个 ZIP、当前代码、逐状态重算和本次补做的本地可学性实验。历史 Markdown 中的指令作为背景，不作为本轮执行指令。

**结论：当前不是“所有候选都无法胜过 independent”，也不是“只要把联合评分头训练好就能证明双车协同”。有较好候选，但当前合格的显著机会很少，主要可由单车候选纠错获得；部分机会又被评分残差幅度排除，现有 RIFT surrogate 和数据组成不足以把剩余机会学好。闭环 oracle 的改善混有质量恢复与并列时任意换候选，尚不能归因于双车协同。**

应优先把实验变成：有明确双车语义的候选机会 → 与质量纠错分开的教师排序 → 已验证能学到的评分头 → 对照质量 oracle 和强独立评分的闭环实验。用户不要求 Ego 响应链，本分析也不把响应模型作为先决条件。

## 1. 证据范围与可信度

输入：

- `outputs/v5_capture_spatial_seed0_20261013_151751/v5_cache_20261014_040945.zip`：144 个 schema-v2 NPZ 和144 个对应 JSON。
- 同目录 `v5_results_20261014_040945.zip`：98 个源码、配置、日志与结果文件。

主闭环来自：`analysis_20261013_162119/closed_loop_20261013_162756`。

| 运行 | 完成的 route/repetition 记录 | 判断 |
|---|---:|---|
| 主 policy | 20/20 | 可用于本轮开发对照 |
| 主 independent | 20/20 | 可用于本轮开发对照 |
| 主 cooperation_oracle | 20/20 | 可用于本轮开发对照 |
| capture independent | 20/20 | 可辅助估计重复运行波动 |
| `closed_loop_20261013_161646` | 1/20 | 中途结果，不能当完整实验 |
| `closed_loop_20261013_162603` | 1/20 | 中途结果，不能当完整实验 |

主运行 manifest 中19个 joint 模块的源码哈希与结果包相符；结果包的24个 Python 文件与本地对应文件字节一致。缓存/重算最大误差为 `3.814697265625e-6`，低于 `5e-4` 容差。因此，本次现象不能简单解释为“服务器没更新到当前代码”。

`analysis_20261013_161044`、`analysis_20261013_162119`、`analysis_20261014_035510` 的同名离线拟合历史完全相同：相同数据、seed 和训练入口的重复，不是三个独立训练 seed。

限制：只有 eval seed0；路线是反复使用的 dev10；离线验证仅2条路线、1个显著机会。ZIP 不含训练后 head 权重，不能逐张量检查服务器模型的饱和程度；本次本地探针从零残差的真实 head 重新训练，并明确与服务器模型区分。没有新跑 CARLA，也没有新跑完整 Lightning RL。

## 2. 先区分三种“候选更好”

### 2.1 Full-Q 更好：144/144，但几乎都在纠正 unary

旧目标为 `Q=U1+U2+P12+CG−20*Cpair`。在144状态上，Full-Q oracle 相对 independent 的平均增益：

| 分量 | 平均增益 |
|---|---:|
| Q 合计 | 16.582230 |
| U1+U2 | 15.605955 |
| CBV-CBV 碰撞惩罚恢复 | 0.972222 |
| H=P12+CG | 0.004053 |

三个分量相加与 Q 增益一致。所有状态都存在 Q 更高候选，仍然主要说明单车质量评估可纠错，不能直接说明协同。

基础RIFT prior与当前teacher也不完全是同一评价问题：当前joint evaluator以PDM nominal替代普通背景预测中的Ego轨迹，并排除另一受控CBV后另算pair collision；单车prior是训练得到的近似排序。因此unary纠错收益既可能来自原模型误差，也可能来自teacher定义/环境分布变化，不能全算joint interaction建模收益。

### 2.2 质量合格且 H 显著更好：6/144

本轮 F：预测 CBV-CBV 无碰撞、两辆 CBV 各自对 nominal Ego/背景无碰撞、无越界、每辆车 U 不低于自己的 independent U。显著机会为 `max_F H−H(independent)>=0.1`。

| 状态条件 | 数量 |
|---|---:|
| 至少存在 CBV-CBV 无碰撞 pair | 144 |
| 再要求两车 unary collision-free，仍有支持 | 97 |
| 再要求两车 offroad-free，仍有支持 | 97 |
| 再加每车 U 门槛，F非空 | 93 |
| independent 本身在F内 | 53 |
| F非空，但 independent 不在F内 | 40 |
| F为空 | 51 |
| F内H有差异 | 20 |
| F内H完全并列 | 73 |
| 合格且H增益至少0.1 | 6 |

所以144状态中，51状态没有合作训练的有效区分信号；73状态只有“哪些pair合格”的信号；20状态才有F内交互排序信号，其中6个达到预设显著门槛。V5虽然把 unary 从连续收益改成门槛，数据仍让它主要学习质量分类。

这与之前36状态审计不能直接作机会率升降比较：旧数据刻意截取压力片段，旧审计的部分结论只要求CBV-CBV pair-safe和合计U不下降；本轮采用每车U分别不下降、对nominal/background无碰撞、无越界等更强门槛，且使用宽采集。旧压力片段发现较多H改善候选与当前宽采集仅6个合格显著机会可以同时成立。变化不能被简单解释为“V5把候选变差了”。

候选集合不是固定12×12：80状态为12×12，27状态为36×36，29状态为24×24，8状态为12×24。源码把有效参考线与12个 mode 展平。采样和损失统计要保留这个差异；每状态等权也不意味着每条路线或每个物理事件等权。

### 2.3 必须由双车联合决策才能获得的收益：当前未证明

| 路线 | 缓存状态 | 显著机会 | repetition覆盖 |
|---|---:|---:|---|
| 27494 | 52 | 1 | 0、1 |
| 2091 | 17 | 2 | 0、1 |
| 17569 | 17 | 0 | 0、1 |
| 25381 | 10 | 0 | 0、1 |
| 25378 | 9 | 0 | 1 |
| 3255 | 11 | 1 | 1 |
| 26405 | 7 | 2 | 1 |
| 28198 | 21 | 0 | 0、1 |

六个机会只有四个 route/repetition/pair 片段，其中两组是相邻时间步。全部六个都有只改变一个坐标的合格候选，即 `(i,b2)` 或 `(b1,j)`，其H增益已达到0.1。因此，机会率4.17%不等于“双车联合决策的必要性达到4.17%”。

| route / rep / step | independent | F内H oracle | H增益 | oracle CG | 最佳只换一车的H增益 |
|---|---|---|---:|---:|---:|
| 27494 / 0 / 320 | (1,4) | (7,7) | 0.175282 | 0.046077 | 0.166308 |
| 2091 / 0 / 110 | (2,5) | (0,11) | 0.132443 | 0 | 0.132443 |
| 2091 / 0 / 120 | (1,5) | (0,11) | 0.189663 | 0 | 0.189663 |
| 3255 / 1 / 170 | (4,4) | (4,9) | 0.164633 | 0 | 0.164633 |
| 26405 / 1 / 120 | (8,10) | (8,11) | 0.119591 | 0 | 0.119591 |
| 26405 / 1 / 130 | (8,6) | (8,11) | 0.212296 | 0 | 0.212296 |

2091 的 oracle 两个下标都变了，不代表H增益需要两车变化：另一坐标可能仅在并列或质量门槛中被重新选择。27494 的两车共同变化比最佳单坐标调整额外多0.008974，说明存在小的联合补益，但不足以把它描述成强双车协同证据。

这不意味着理论上任何独立模型都能解决全部联合问题；确定性独立模型也可能利用共享场景信息产生协调选择。需要相同输入、预算和监督下的强独立模型对照，不能用分布是否factorized直接证明联合网络必要。

## 3. 强双车候选为何消失：主要在 collision gate，不在 unary budget

CG在全部候选中并非处处为零：11状态存在CG>0，5状态存在CG>0.1。但逐步加门槛：

| 搜索集合 | 存在CG>0.1的状态 |
|---|---:|
| 全部有效pair | 5 |
| CBV-CBV collision-free | 3 |
| 再加两车各自 unary collision-free | 0 |
| 再加offroad-free和U预算 | 0 |

F中只有27494/rep0/step320这一状态保留正CG，且其最大可行CG低于0.05。strict-both全为空，是当前候选与标签/门槛组合的真实结果，不是强化学习尚未训练好的问题。

固定缓存门槛消融：

| 修改 | F非空状态 | H显著机会 |
|---|---:|---:|
| 当前设置 | 93 | 6 |
| 每车U预算1、2、5、10 | 93 | 6 |
| 完全取消U预算 | 97 | 7 |
| 取消 unary collision-free gate | 143 | 15 |
| 取消offroad gate | 93 | 6 |
| 再要求当前comfortable flag为真 | 3 | 0 |
| H增益阈值降到0.05 | 93 | 9 |
| H增益阈值降到0.01 | 93 | 14 |
| H增益阈值提高到0.2 | 93 | 1 |

含义：单纯放宽U预算基本无效；降显著阈值只能增加弱机会。取消碰撞门槛带来的机会不能直接当作有效协同。comfort 的“40帧内任意一次阈值超限即不合格”非常苛刻，不宜直接作为下一版硬门槛；先看加速度/角加速度幅值、持续时间、PID预测与实际执行的差异。

当前 `unary_collision` 合并 nominal Ego 与背景碰撞。缓存没有分开的 Ego/background 标志、逐时碰撞对象和PID rollout中心，不能从这批缓存断言究竟是谁排除了强CG。下一次必须分别保存。

项目有两种可能的研究定义，需要保持明确：

1. 若要求固定nominal下几何兼容的正向协同，就保留该约束，转向前向顺序、让行/抢行组合、错峰覆盖等能在无碰撞条件下成立的语义和候选。
2. 若目标仍是双CBV协同制造有意义的交互挑战，nominal Ego重叠可以是冲突暴露信号，不能未经分析全部与CBV-CBV/背景实际碰撞同等处理。但也不能把“nominal重叠”直接认证为安全协同；应分开定义和闭环检验。该选项不要求学习Ego响应链。

本次不擅自取消碰撞门槛。合理调整是先分解标签、人工核对，再明确实验目标。

## 4. BeTop语义与目前实际测试的差距

主运行实际配置为 `interaction_semantics: spatial_temporal`：XY空间轨迹交点按arrival gap加权。`forward_braid`虽然已接入，但本轮没有采集或测试它。空间交点的时间接近权重偏好较接近的过冲突点时刻，与“不允许对固定nominal碰撞”的门槛容易形成张力；这与上述CG消失结果相符，但不能据此断言所有标签都错。

BeTop原文在各agent局部横向坐标—时间投影中定义intertwine，并讨论assertive/passive的行为关系；它没有给出当前 `P12+CG` 这个标量合作奖励。不能把当前空间交点数量或CG直接等同于BeTop正向协同。[BeTop §3.1](https://arxiv.org/html/2409.18031v1#S3.SS1)

现有 `nominal_braid.py` 是带局部性限制的启发式前向投影事件；符号是几何前后关系，不是经过人工认证的assert/yield语义。应在同一批状态的PID rollout轨迹上重标两套语义、抽查再决定；目前缓存只存生成器轨迹与teacher trace，没有保存teacher的完整PID轨迹，不能拿原始候选轨迹直接重标并宣称仅改变了语义。

建议联合目标除coverage外增加明确的双车结构：两车各自与Ego的有效关联、局部共同冲突区域、时间/顺序兼容、两车互不冲突，以及相对强独立基线的联合补益。正负braid不是好坏标签；不要求两车都抢行，更不要求一辆先改变Ego响应后另一辆跟进。

## 5. 残差幅度：本轮有可计算的结构性限制

当前head为 `z=B+lambda*s*tanh(raw)`，`s=max(std(B),1)`。若一个候选a连

`B(base)−B(a) < 2*lambda*s`

都不能满足，则不论网络raw值如何，它都无法超过independent。反之只是通过最乐观的必要条件，不保证同一个共享head能同时实现各候选的极值。

在每个显著机会的候选集合中，求最小必要lambda下界：

| route / step | 任一H增益>=0.1合格候选的lambda下界 | 最佳H oracle的lambda下界 |
|---|---:|---:|
| 27494 / 320 | 1.0074 | 1.6363 |
| 2091 / 110 | 1.0149 | 1.2878 |
| 2091 / 120 | 1.1243 | 1.2078 |
| 3255 / 170 | 0.8488 | 0.8488 |
| 26405 / 120 | 0.3187 | 0.3187 |
| 26405 / 130 | 0.9342 | 0.9342 |

前三个在lambda1下不可达；lambda2使六个均通过幅度必要条件。各机会的初始prior总概率多数在1e-16至1e-19量级，只有26405/120约1.63e-5。这同时解释了old-policy weighting为什么难以关注它们。

注意：uniform pair surrogate并不是在初始时完全不给低prior候选梯度；它的问题不能用old-policy加权的梯度解释替代。其后续固定reference clipping、共享参数干扰与状态分布需要分别分析。

保持旧 `std(B/T)` 标定且floor不生效时，温度同时缩小base gap与残差上限，不能解除上述相对限制；fixed scale才能解耦。当前 `fixed_residual_scale: 10` 虽存在配置中，但 `residual_scale_mode: base_std`，实际并未使用固定10。

## 6. 服务器拟合结果：有改进，但不能据此认定已经学会合作

训练112状态、6路线，验证32状态、2路线（3255、28198）；训练显著机会5个，验证显著机会仅1个。

| 目标/损失 | 导出模型验证regret | 验证可行率 | 唯一验证机会是否改善 |
|---|---:|---:|---|
| 初始head | 0.535718 | 10/32 | 否 |
| legacy Q / RIFT | 0.535718 | 10/32 | 否 |
| cooperation / RIFT | 0.535718 | 10/32 | 否 |
| cooperation / old-policy | 0.535718 | 10/32 | 否 |
| cooperation / RIFT+aux | 0.379468 | 12/32 | 是 |

前三组采用“开发集regret严格改善才保存”的规则，导出时回退到初始权重。**这不等于整个训练没有梯度、也不等于所有中间权重都等于初始值。** loss记录发生了变化，部分末期行为甚至变差。

aux在step100取得最佳验证regret；step1000已退化到0.598218。保存step100的规则起作用了，但在1个显著验证状态上挑模型，选择噪声很大。它在训练集的regret从0.660920变为0.705474，可行数从43变为41，训练机会改善仍为0/5。

32状态overfit：regret从0.893052降至0.278774，合格数从13升至22，但唯一显著机会仍没改善。lambda2的同样32状态训练regret为0.292913，机会仍没改善。此前把这称为“可学性关卡通过”会过于宽松：整体质量纠错能学，不等于机会排序已学会。

`metrics.regret`还有一个口径问题：F为空时全部有效pair分数相同，regret恒0。验证32状态里14个F为空、18个F非空，其中15个F内H完全并列。大部分开发指标在衡量质量恢复，不能单独作为合作checkpoint选择标准。

`opportunity_improvement_rate`当前只要求H增益>1e-6，并不要求重新达到定义机会时的0.1。后续应同时报告任意正增益、>=0.1捕获率、捕获的oracle收益比例和质量违规率。

## 7. 本次补做的真实head可学性实验

为了区分“幅度不足”与“幅度足够后仍学不好”，本机对缓存输入补做了500步诊断。使用当前真实 `JointResidualScorer`，冻结cached features，不加载CARLA/Lightning；初始零残差分数与缓存一致，最大误差<=3.81e-6。每步平衡使用所有指定机会状态，AdamW lr0.001。

这是本机Torch环境的小样本训练，不是服务器训练后权重的复现，也不是路线验证或闭环实验。

| 数据 | 配置/损失 | >=0.1机会捕获 | 最终平均target regret |
|---|---|---:|---:|
| 单个27494/320机会 | lambda1 / distill | 0/1 | 0.993457 |
| 同一个机会 | lambda2 / distill | 1/1 | 0 |
| 六个机会 | lambda1 / distill | 1/6 | 1.832243 |
| 六个机会 | lambda2 / distill | 6/6 | 0.006091 |
| 六个机会 | lambda2 / RIFT | 0/6 | 1.983781 |
| 六个机会 | lambda2 / RIFT+aux | 6/6 | 0.020555 |
| 六个机会 | T3、lambda2、fixed10 / distill | 6/6 | 0 |

由此可以作出比猜测更明确的判断：

- lambda1确实阻断了部分排序；在27494单状态实验中直接得到验证。
- 当前head和cached features并非连这六个状态都无法区分，幅度足够且直接监督时能够记住。
- lambda2并不能独自修复纯RIFT loss；同数据同初始化下，distill/aux显著更容易学到。
- 服务器32状态lambda2仍失败，与混合大量质量状态、机会仅1个和优化干扰相容，不能据此宣布lambda无效。
- 没有证明当前表征足以泛化，也没有证明新架构必需。优先修采样、目标分离和优化，再以路线验证判断是否要增加nominal输入。

本地lambda1六状态distill的 `|raw_delta|>3` 比例平均约61.7%，lambda2 distill约0.23%；这说明本地幅度受限时确有饱和。但服务器checkpoint不在ZIP中，不能把这个百分比直接写成服务器模型的性质。

RIFT式loss的具体局限：离线脚本1000步一直用初始old logits；正advantage动作概率比超过1.2后该分支变平，负advantage分支也会被clip。它不是最终绝对概率的硬约束，但低prior目标只获得少量相对增加就可能进入无进一步推动的区域。old-policy weighting又额外按原始极小概率降低这些机会的作用。不能由这一固定reference诊断直接证明所有on-policy PPO都失败；在线更新会重新采集reference，但其在每个buffer内的学习仍需验证。

## 8. 学习不足的同状态证据，比跨rollout均值更可靠

independent模式仍会计算同一个训练后head的反事实policy选择。因此，可以在它访问的1481个完全相同状态上比较：

| 指标 | independent | learned head反事实选择 |
|---|---:|---:|
| 合格选择数 | 429/1481，28.97% | 361/1481，24.38% |
| 从不合格恢复为合格 | — | 102 |
| 从合格变为不合格 | — | 170 |
| 99个显著机会中取得合格正增益 | — | 13/99 |
| 99个机会中取得>=0.1合格增益 | — | 8/99 |

这直接证明现有head没有充分学会教师质量/交互排序，且质量回退超过恢复。它不是“因为两个策略访问了不同状态所以看起来差”。但这里比较的是同状态教师预测，不是从同一CARLA快照分叉后的真实长期结果；99帧也不是99个独立场景。

在policy自身1428个状态上，independent反事实有466个合格，实际policy仅402个；恢复62次、破坏126次。其19个显著机会只捕获1个。两个访问分布都显示质量回退问题。

## 9. 闭环表现与oracle解释

主运行每种模式20条记录，均17条Completed，另3条blocked/timeout。

| 指标 | independent | policy | cooperation_oracle |
|---|---:|---:|---:|
| Ego平均composed score | 89.2075 | 87.2005 | 88.1625 |
| Ego平均route score | 89.2075 | 89.2005 | 88.1625 |
| 总CBV collision count | 10 | 10 | 3 |
| joint诊断状态 | 1481 | 1428 | 1098 |
| 有joint诊断的路线 | 8 | 9 | 8 |
| 自身状态中的显著机会 | 99 | 19 | 32 |
| 实际选出的合格正H增益动作 | 0 | 3 | 58 |

不同策略joint诊断覆盖路线数不同，原合作summary分别对8或9条路线求均值，不能直接当作相同路线总体的处理效应。无joint状态的路线必须单列，不能随意补成可行率0或1。真实episode outcomes仍可在相同10条路线/2个repetition上配对。

policy相对independent：Ego composed −2.007，CBV碰撞差0，越界比例+0.001342，不舒适比例−0.019240。Ego composed差主要由2091/rep0的一次vehicle collision产生−40分，再在20条记录上平均；碰撞对象5831也曾是该episode的CBV成员。但仅凭这些汇总不能证明是协同造成或训练预期内的行为。

Ego分数下降若被视为对抗挑战，仍需区分有意义交互与碰撞/场景偶然性；若目标是安全正向协同，则该碰撞不能算收益。不要只挑一个Ego分数方向定义成功。

oracle相对independent：总CBV碰撞少7次，其中27494/rep0少4次、2091/rep0少2次、28198/rep0少1次。路线均值差−0.35次，当前路线bootstrap区间约[-0.8,-0.05]；这只在本模型、eval seed0和该开发路线集合条件下成立，没有跨seed稳健性或机制归因。

**关键混杂：当前cooperation oracle不只在有显著合作机会时执行。** 只要F非空就对H取argmax；H并列时直接取扁平矩阵第一个候选。

在oracle的1098个joint状态中：

- F非空692次，日志把它们都记作intervention，但真正相对independent改变动作的只有602次。
- 602次改变中，529次H增益为0，15次为负，58次为正；其中达到0.1的32次。
- 260次是在independent本来合格且H完全相同的情况下，仅因并列候选索引而改动作。
- 284次是independent不合格、oracle恢复合格，但H没有正增益；属于质量恢复。
- 全部oracle状态的 `coop_nominal_both_rate` 都为0，说明本次oracle没有在F内执行CG>0.1的强双边际协同。

因此，“oracle碰撞从10降到3”很可能混有质量过滤、候选模式变化和状态分布变化；不能当成强协同教师已验证成功。它只是当前状态代理H的受限最优，不是长期真实闭环性能的理论上界。

另外，当前 `cbv_collision_count` 是汇总CBV碰撞，不能自动解释为CBV-CBV互撞次数。后续需要按碰撞对象分组。

## 10. 重复运行波动已经存在

capture与主independent使用相同基础轨迹生成器、独立选择规则与seed0；当前离线导出代码保留backbone张量。这构成一个有参考价值的近似A/A，虽然不是专门设计并逐张量核验的正式A/A。

capture−主independent的Ego composed差−1.5835，CBV碰撞总数均为10，但各路线碰撞分布变化；不舒适比例差+0.016352。该波动与policy相对independent的部分差值处在相近量级。

所以必须补同一eval.yaml、同一checkpoint、同seed的独立重复。不要把重复训练同一seed的相同history当成这项验证。也不要把1428或1481个相关时间步当成有效样本量来计算显著性。

## 11. 根因排序与证据等级

| 问题 | 本轮证据 | 判断 |
|---|---|---|
| 没有任何更好候选 | 144/144有更高Q；6/144有合格显著H增益 | 否 |
| 缺少强双车合格机会 | F内CG>0.1为0；六机会都能只换一车达门槛 | 当前数据下明确成立 |
| 目标仍主要落在质量纠错 | 73个F内H平坦、51个F空；oracle大量零H变化 | 明确成立 |
| 残差幅度限制 | 三机会必要lambda>1；单状态lambda2恢复可学 | 明确成立 |
| 训练目标/loss/采样不足 | 同状态质量退化；lambda2纯RIFT0/6、aux6/6本地probe | 明确成立，具体全局贡献需消融 |
| 当前head连机会状态都不能表示 | lambda2直接监督可记住6/6 | 不支持这种强断言 |
| 表征缺失nominal导致泛化差 | evaluator依赖nominal而head无显式nominal输入，开发结果不稳 | 合理假设，尚未隔离证明 |
| 仅靠增加训练时长即可解决 | 100步较好、1000步退化；strict-both无支持 | 不支持 |
| 整个2CBV联合评分idea无效 | 数据/目标/基线不足以排除合理方案 | 不能据此否定 |

当前实现是可以运行的V5诊断管线，但还不是已验证work的联合协同方法。之前“有接口即可按全量runbook做下去”的表述不够严格：新的证据要求在机会语义与可学性处设置明确关卡。

## 12. 下一轮最小改进方案

### 12.1 先修统计与oracle实验，不增加网络

建议新增以下显式模式；**当前代码尚无这些模式，不能直接把名字传给run.py**：

- `quality_oracle`：在F内按base prior选择；independent已合格时保留它，F为空时明确fallback。只回答质量恢复的价值。
- `interaction_oracle`：F内最大H；H并列时优先independent，其次最大base prior；仅当增益>=预先固定阈值时作为交互干预。质量恢复另记，或另模式执行。
- 现有 `cooperation_oracle` 保留为历史对照，但不再直接叫“纯合作上界”。

比较质量oracle与interaction oracle的增量，再与policy和强独立评分比较；记录真实动作变化，不用F非空代替intervention。将hard-safety、U预算、H排序、双边际各自计数。

新增报告：F非空条件下的可行率/质量恢复与回退；机会条件下>=0.1捕获率与H regret；无机会且independent合格时的无益变化率；同状态policy vs independent；按route/repetition/pair片段聚合。模型选择优先检查质量回退，再选机会指标，不只用全体平均regret。

### 12.2 先用现有缓存验证优化，再采新数据

保留零残差初始head和相同输入。至少分开：六机会专门过拟合、机会+质量困难负样本、完整144状态；在lambda1与lambda2下比较RIFT、distill、RIFT+aux。固定reference的离线拟合明确标注离线诊断，不当作on-policy实验。

下一版损失建议把质量与F内交互排序分开归一化：

`L = L_RIFT + alpha*L_quality + beta*I(has_interaction_support)*L_rank + eta*L_preserve`

`L_rank`只在合格且H有差异的集合里对比；`L_quality`负责不合格动作；`L_preserve`只在independent合格且没有显著交互增益时抑制无益换候选。F为空不能凭空构造合作正样本。teacher quality flags只用于训练，若推理使用它们，必须标为evaluator-assisted，不能混入learned结果。

先让六机会稳定可拟合；随后以多个训练seed做路线划分验证。若仍在完整数据中被质量负样本冲掉，采用机会/质量恢复/无机会合格/无F四类分层采样，限制相邻帧重复，保留代表性的普通分布验证集。不能只过滤负样本后用富集数据的机会率冒充总体机会率。

### 12.3 重做围绕协同机会的采集与配对

当前rule recognition选最近车辆，PairManager锁定当前成员并按ID约定存储，没有按“两个成员都能与Ego构成有效交互、且合格联合候选有补益”选pair。capture_all只扩大保存范围，并不创造协同机会。

在附近候选车辆中，先按两车各自关联、共同局部冲突区域、可行候选数量和结构差异筛pair；锁pair以保持段内连续。先判断是“没有合适第二辆车”还是“有合适车但生成器没有有效模式”。若冻结Pluto生成器缺少时序/路线多样性，再评估候选重采样、时间缩放等可控扩展，每次保持公平候选预算。不能一开始就解冻generator或扩大joint head。

逐段保存：全部有效参考线/mode身份、nominal、PID rollout中心/朝向/速度、Ego/background/CBV-CBV分开碰撞标志及对象、逐时距离/时间裕度、舒适度严重程度、拓扑事件/顺序、checkpoint与配置。现有schema-v2足够重训Stage1 head，但不够独立重算全部teacher语义，也不够重前向Pluto。

固定保存间隔与每episode上限之外，记录总帧数、单CBV/双CBV可用数、nominal可用数、F非空数和有效机会片段数。pilot按独立route/repetition/pair片段计预算，先争取得到几十个经人工核对的有效双车片段，而不是把一个片段存成几百帧。

### 12.4 需要时增加nominal条件化，不以大网络替代证据

当前head输入包含两组candidate latent、CBV common轨迹与raw速度；没有显式Ego未来nominal。latent可能间接含有Ego历史，但不能保证足以恢复PDM未来意图。evaluator又基于PID rollout，head显式轨迹主要是生成器轨迹，存在待测的表示差异。

可优先增加小型nominal encoder与候选相对nominal事件特征，再做有/无nominal消融；同时设单车质量头、单车交互头和真正pair残差。相比直接堆大MLP，这样更容易解释单车纠错、交互预测与联合补益各自的作用。缺失nominal是泛化候选原因，不是本次六状态记忆失败的必要解释。

### 12.5 最后做RIFT在线训练和正式闭环

前述语义、候选和固定数据关卡通过后，比较legacy-Q、合作目标RIFT、合作目标RIFT+rank；固定生成器、候选预算、训练步数、标定和路线，至少多个训练seed。在线检查首批ratio、candidate身份和重前向一致性，不把离线latent重算验证当成完整在线验证。

闭环必须包含independent、质量oracle、交互oracle、learned强独立head、learned联合head及A/A。所有oracle只作为诊断。分别报告有意义交互、CBV-CBV/背景/Ego碰撞、越界、舒适度和Ego结果，按预先定义的研究目标解释。开发集选定方案后冻结，再用另留路线做最终测试。

**进入下一阶段的依据应是可验证的条件，而不是训练时长：有效双车机会确实存在；幅度覆盖这些机会；机会小集可拟合；路线验证没有明显质量回退；交互oracle相对质量oracle有增量价值。当前只部分满足前两项，尚不宜直接扩大在线训练。**

## 13. 本轮产物与复现

本轮只新增分析脚本、图和报告，没有修改生产policy、训练loss或历史结果。

- `RIFT/scripts/audit_joint_rift_v5_results.py`：直接读取两个ZIP，重算F、机会、预算消融、幅度下界，检查源码及闭环同状态指标。
- `RIFT/scripts/probe_joint_v5_cached_learning.py`：真实head的单机会/六机会可学性探针，支持`--seed`。
- `outputs/joint_rift_v5_review_20261014/repro_v2/audit.json`：可复现审计JSON。
- 同目录`states.csv`：144状态逐项数值。
- 上一级`extra.json`：CG逐级门槛、数据组成与近似A/A。
- `counterfactual_same_state.json`：三种rollout上的同状态反事实比较。
- `local_learning_probe.json`：本地500步探针全历史及逐状态结果，seed0。
- `v5_diagnostic_overview.png/.svg`、`validation_regret.png`：诊断图。

服务器同步两个新增脚本后，可在root终端运行以下非CARLA诊断；路径已使用实际存在的capture。每次输出文件名带新时间戳：

```bash
runuser -u riftuser -- bash -lc '
set -euo pipefail
PROJECT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop_2
PY=/zeron-vepfs/tjqc/njl/envs/rift/bin/python
CAPTURE="$PROJECT/outputs/v5_capture_spatial_seed0_20261013_151751"
cd "$PROJECT/RIFT"
for SEED in 0 1 2; do
  "$PY" -u scripts/probe_joint_v5_cached_learning.py \
    --cache_zip "$CAPTURE/v5_cache_20261014_040945.zip" \
    --output_json "$CAPTURE/opportunity_probe_seed${SEED}_$(date +%Y%m%d_%H%M%S).json" \
    --steps 500 --seed "$SEED"
done
'
```

这只复核优化与可学性，不需另跑CARLA。新增quality/interaction oracle、分层采样、nominal encoder和新语义采集仍是后续实现项，不冒称当前代码已全部支持。
