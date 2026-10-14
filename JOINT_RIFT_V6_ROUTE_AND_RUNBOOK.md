# Joint-RIFT V6：先验证联合规划价值，再蒸馏评分头

日期：2026-10-14。V6是技术路线调整，不是V5再增加训练轮数。

## 1. 这次实质改了什么

主路线变成：**冻结RIFT/Pluto候选生成 → 精确几何/固定nominal交互评价 → 分开质量恢复与交互增益的联合教师 → nominal条件化直接评分蒸馏 → 学生访问状态再标注与数据聚合。**

不引入用户不要求的Ego响应链。V6主要使用密集奖励教师监督，不将其冒称为on-policy PPO。旧V5/RIFT训练入口保留；新direct head不接旧PPO trainer，误用train_cbv会明确报错。

执行提供两条明确区分的路径：

- `policy`：纯学生，用预测质量置信度与预测交互增益决定是否替换independent；推理不读取teacher标签。
- `verified_policy`：学生提案，evaluator复核，失败则回退教师。属于模型化混合规划，不是纯learned策略。能给出当前代理评价下的条件保证，但不能保证真实长期闭环或论文效果。

## 2. 新发现并修正的碰撞问题

原 `traj_eval/traj_evaluator.py::get_collision_matrix` 调用 `STRtree.query(polygon)` 后，只要返回候选就置collision=True。Shapely无predicate的query只检索轴对齐包围盒相交候选，不能等同于真实polygon相交。[官方STRtree说明](https://shapely.readthedocs.io/en/stable/strtree.html)

V6增加旋转矩形SAT检测，用于CBV对nominal Ego/背景的碰撞trace；原CBV-CBV碰撞仍由既有精确SAT模块计算。`collision_geometry=exact_sat` 是新preset默认，旧配置默认仍保留 `legacy_aabb`，历史结果不会被悄悄重定义。

已用旧生产方法实际复现“两个斜向长矩形包围盒重叠、车框不相交却被判碰撞”的案例，并验证SAT与Shapely真实polygon.intersects在多帧/分块测试中一致。**实际场景能恢复多少有效候选，仍须新采集量化。旧缓存没有完整车框，不能反推一个真实修复比例。**

新缓存同时保存exact采用的碰撞trace与AABB trace，可以统计候选快照误报。nominal Ego/背景分别记录，不再混为一个不可解释的标志。

## 3. 四种教师与可计算的增量

质量集合F沿用每辆车U预算、无越界、CBV-CBV无碰撞及配置要求的unary碰撞约束。默认不要求严格comfort flag。

默认交互分数仍为 `J=P12+CG`，用于保持与原问题的连续性；新数据默认标签为forward_braid。它是固定nominal覆盖代理，不是已经验证完整的BeTop行为语义。

| 模式 | 作用 |
|---|---|
| `quality_oracle` | F内最大base prior；independent已经合格时保持它，只做质量恢复 |
| `factorized_oracle` | F内最大P1+P2，作为使用真实单车标签与共同质量过滤的强诊断对照 |
| `interaction_oracle` | 相对quality_oracle，J增益至少0.05才替换；并列保持reference |
| `joint_oracle` | 两车各自有事件、CG至少0.01，且J相对quality至少提高0.05、相对factorized至少提高0.01才干预；否则回退quality |

这些阈值是显式pilot设置，不是已验证的最优值。与V5的H>=0.1/CG>0.1口径不同，不能直接比较“机会率升高”。

factorized_oracle使用evaluator可行集合，属于privileged诊断，不能冒称可部署的独立神经网络。学习对照提供 `head_kind=unary_guarded`：action logits只由两车共享单车头相加，但共享预测质量guard；名字明确表示存在共同质量过滤，并非完全独立执行策略。

旧 `cooperation_oracle` 保留用于历史比较。新教师不会在无增益并列时随意切换到矩阵第一个候选。

## 4. 直接评分头与训练

新 `NominalDirectScorer` 输入：两车latent、生成轨迹、速度、完整nominal采样，以及相对nominal和independent候选的表示。输出三个相互区分的量：无界action logits、pair可行性logits、交互分数预测；另有单车pressure辅助预测。

`z=learned_action_logit+0.05*centered_base_prior`。没有 `lambda*std(B)*tanh` 上限，所以不再有V5的结构性“网络完美也越不过prior”限制。推理置信阈值0.8只是预测门槛，不能当安全认证。

训练主损失为：teacher动作交叉熵 + 正负类分别归一化的可行性BCE + 交互回归辅助。按joint机会、普通interaction机会、质量恢复、保持原动作、无F五类采样，再按物理片段采样，减少连续帧和长路线支配。

checkpoint在开发集按“质量恢复−2×质量回退＋3×interaction捕获＋5×joint捕获”的显式pilot效用选择。该权重不是理论最优，也不是测试集指标；所有原始计数都会输出。路线划分保证所训练机会类型在train/validation都有支持；默认会按开发机会分布调整划分并明确记录，不能拿这个开发集当无偏最终测试集。

`--teacher joint` 没有真实joint支持时拒绝训练，避免把质量纠错冒称协同。`--teacher interaction` 可以作为明确标记的较弱交互控制实验。

## 5. 这次实际验证结果与边界

旧144状态按新教师分成：无F51、质量恢复37、保持47、额外交互9、严格joint机会0。9个额外交互状态来自3个片段。

使用真实上传缓存、当前直接评分头，本地500步机会小集训练：step100首次达到9/9合格改善，保存模型9/9 teacher一致、质量恢复3、质量回退0。**这是明确标注的overfit，不是泛化或真实闭环成功。**

还做了路线验证：训练105状态，验证39状态（26405、28198、3255），验证3个interaction机会：

| 模型 | 选中step | 验证interaction捕获 | 验证质量恢复/回退 |
|---|---:|---:|---:|
| direct joint | 0，回退初始 | 0/3 | 0 / 0 |
| unary_guarded | 250 | 1/3 | 4 / 3 |

这说明**旧数据上泛化仍未解决**，不能宣称新联合头已胜过强独立头。也正因此，V6先验证新教师/候选价值，允许模型化verifier执行，不让尚未泛化的学生独自承担主效果。

本地已覆盖：SAT真/假碰撞、旧生产query误报复现、教师并列保持、联合增量与单车改善区分、split标签要求、verifier接受/拒绝、nominal条件化、换车对称性、预测guard、真实生产score绑定、真实fit/export/backbone合并、完整与部分闭环分离统计、聚合语义一致性、PID标签回放。

本机没有CARLA/Lightning，未新跑真实V6采集和闭环。服务器Python3.8必须先跑preflight。原V5测试继续保留并回归通过。

## 6. verified_policy究竟保证什么

学生提案合格且相对quality有足够J增益时采用；否则采用interaction教师。若配置 `verification_target: joint`，还要求双车参与与相对factorized的增量，否则采用joint教师。

在当前evaluator和有限候选集合下：

1. F非空时，执行候选一定在F中。
2. 相对quality教师，没有足够交互增益的提案不会被当作交互改善接受。
3. F为空时明确回退independent并记录，不虚构合格动作。

上述保证不覆盖evaluator误差、真实Ego响应、控制器执行偏差和长期回报。模型化模式必须独立报告，不能与纯policy混合声称learned RL成功；它当前也不是计算加速证明，因为现有实现会完整计算候选评价。

## 7. 新采集与廉价标签消融

新preset默认：精确SAT、forward_braid、shared_conflict配对。shared_conflict在原候选车中优先共同冲突区域和预计到达接近的双车，最多检查最近12个有效候选，保持已有成员与段内锁定。它是采集启发式，不是双车机会存在的保证。旧最近车辆规则可单独切回对照。

新缓存保存PID rollout中心/朝向、相位对齐的nominal、Ego/background分别碰撞、AABB对照trace、latent/raw输入、独立候选身份、backbone与collection head哈希、gamma/dt/horizon。仍不包含完整PlutoFeature，不能宣称能用它训练generator。

`stats`阶段在同一缓存上运行：

- 原标签重新计算验证，最大误差要求<=5e-5；
- 固定nominal碰撞拒绝的主定义；
- `label_only`诊断：背景和CBV-CBV碰撞仍拒绝，只把nominal Ego重叠当冲突标签，观察机会是否主要被该门槛消除；
- 同一PID轨迹重标spatial_temporal，隔离标签语义而非重跑世界。

label_only只是显式的目标消融，不能自动认证安全或正向协同；其中U仍使用采集时的质量回报，因此不是完全取消nominal冲突代价。主闭环默认仍reject。

## 8. 服务器只需一个启动入口

将V6 ZIP按目录结构同步到项目。保留历史实验和模型；不要删除旧outputs。脚本可从当前root终端调用，会自动以riftuser重启，设置共享Python和CARLA环境。所有新run使用新目录。

先做教师pilot，**不启动长训练**：

```bash
cd /zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop_2/RIFT
export V6_RUN=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop_2/outputs/v6_braid_sat_$(date +%Y%m%d_%H%M%S)
bash scripts/run_joint_v6.sh pilot "$V6_RUN"
```

pilot依次：生成配置 → V5/V6测试 → independent宽采集 → 原标签回放与三个定义统计 → 五种教师/独立闭环 → 配对统计 → 两个ZIP打包。任一步出错即停止；输出以stage名和真实目录为准，不再手动猜时间戳路径。

### 8.1 2026-10-14 预检权限修复与继续运行

若日志为 `V6 production imports: PASS`、`Joint-RIFT core validation: PASS`，随后在 `outputs/joint_rift_v5_validation/<uuid>` 报 `PermissionError`，表示测试仍写入公共历史目录；并未开始CARLA采集。公共目录不可写的具体所有者需另查，不能由该日志直接断定。

同步 `joint_rift_v6_patch_20261014_r1.zip` 中的文件到服务器项目目录。此修订把V5/V6测试产物写入本次run的 `preflight_tests/v5`、`preflight_tests/v6`，通过 `JOINT_RIFT_TEST_OUTPUT_DIR` 指定；单独运行测试且未指定变量时使用系统临时目录。无需修改公共历史目录的所有权。新增 `continue_pilot` 跳过已完成的prepare，从预检继续：

```bash
cd /zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop_2/RIFT
export V6_RUN=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop_2/outputs/v6_braid_sat_20261014_072303
bash scripts/run_joint_v6.sh continue_pilot "$V6_RUN"
```

该命令适用于本次prepare已完成、预检被权限错误中断的run，可从root终端调用，脚本仍自动切换riftuser。它依次执行preflight、capture、stats、teachers、summarize、package。不要对这个已有目录重新执行pilot；prepare要求新目录。若后续某个阶段单独失败，修复后从对应stage继续，避免重复已经完成的统计或实验。Gym弃用提示不是此次失败原因。

### 8.2 r2：修复共享冲突配对的Vector3D速度接口

若预检通过、CARLA加载成功，随后在 `rule_cbv.py` 报 `float() argument must be a string or a number, not 'Vector3D'`，根因是新增配对代码错误地把本项目 `CarlaDataProvider.get_velocity()` 的返回值当成标量。该接口实际返回CARLA速度向量；Ego和候选CBV两处均改为计算 `sqrt(x*x + y*y + z*z)`，单位m/s，再应用原有2m/s的ETA分母下限。缓存未就绪返回None时取0，再应用下限。不能直接调用项目misc.get_speed而不换算，因为那个函数返回km/h。

同步 `joint_rift_v6_patch_20261014_r2.zip` 后，继续使用8.1中的实际run路径与 `continue_pilot` 命令，无需重新prepare或删除outputs。新预检增加真实 `carla.Vector3D` 的速度接口校验；本地新增生产配对方法回归，验证Ego和CBV的ETA单位、零速度/None、保留活跃成员及最近车辆回退。方法回归不等于完整CARLA闭环验证。

本次日志的首次地图加载超时已在attempts=1恢复，不是最终退出原因；运行器已报告终止CARLA进程组。该错误发生在首条场景早期，日志未显示完成任何路线。继续运行会使用项目既有checkpoint机制；首次重试应重新执行该未完成场景。

只做接口smoke时，先在命令前设置：

```bash
export V6_SUBSET=26405
```

smoke不能用于路线泛化。正式pilot另开新目录并 `unset V6_SUBSET`。评估seed默认0，接口稳定后可 `export V6_SEEDS="0 1 2"` 在新run中扩展；不要把同一已完成目录重复当新实验。

默认模型路径与此前一致；若需要替换：

```bash
export V6_BASE_CKPT=/absolute/path/to/trained_single_RIFT.ckpt
```

不要加 `--no_resume`。V6 direct策略通过显式导出checkpoint加载，沿用项目的resume=True路径。

## 9. 先看哪些结果

pilot末尾打印 `v6_results_*.zip` 和 `v6_cache_*.zip` 的实际路径。下载这两个文件即可继续分析。主要检查：

| 文件/字段 | 意义 |
|---|---|
| `stats_*/summary.json → strata` | 质量、交互与joint分别多少，不混口径 |
| `by_route` | 机会是否跨路线存在 |
| `collision_audit` | exact数据中AABB会误报多少候选；null表示没测到，不表示0误报 |
| `same_semantics_replay_max_error` | PID回放是否复现原标签 |
| `stats_nominal_conflict_ablation_*` | nominal门槛是否切掉了主要机会 |
| `stats_spatial_replay_*` | 只改标签后机会变化 |
| `teachers_summary_* → interaction_oracle_vs_quality_oracle` | 合作排序是否超出质量恢复 |
| `teachers_summary_* → joint_oracle_vs_factorized_oracle` | 严格联合增量是否兑现 |
| `decision_executed_changed` | 实际动作变化，避免用F非空冒充干预 |

配对结果新增Ego vehicle/pedestrian/layout collision与blocked/timeout计数，继续保留CBV碰撞、越界、舒适度和Ego分数。只将progress完整的run用于配对；policy自己的逐状态均值仍不能和另一rollout当同状态处理。

若joint支持为0，不能通过多训练创造不存在的机会。检查SAT修复、nominal约束和forward语义消融，再决定扩场景/候选或明确将论文范围改为联合兼容交互规划。若只有quality oracle改善，则结论仍是质量恢复，不能换名叫合作。

## 10. 教师有价值后再拟合

纯joint训练要求有joint支持。先小集过拟合，再路线验证：

```bash
# 仍可在root终端直接调用。
export V6_TEACHER=joint
export V6_STEPS=500
bash scripts/run_joint_v6.sh overfit "$V6_RUN"
export V6_STEPS=1500
bash scripts/run_joint_v6.sh fit "$V6_RUN"
```

若当前只有普通interaction支持，必须明确作为控制实验：

```bash
export V6_TEACHER=interaction
bash scripts/run_joint_v6.sh overfit "$V6_RUN"
bash scripts/run_joint_v6.sh fit "$V6_RUN"
```

强学习对照另建输出：

```bash
export V6_HEAD_KIND=unary_guarded
bash scripts/run_joint_v6.sh fit "$V6_RUN"
unset V6_HEAD_KIND
```

所有fit目录带真实新时间戳，不能拿overfit头当最终模型。完整fit开发集没有改善时会保存初始头，并在summary写 `best_step=0`；这不是训练成功。

## 11. 学生闭环、verifier与A/A

选定一个fit导出后，在新目录生成匹配配置。小head导出会自动与哈希匹配的服务器backbone合并，不要求下载整个大模型。

```bash
export V6_STUDENT_CKPT=/absolute/actual/fit_directory/joint_head.ckpt
export V6_EVAL=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop_2/outputs/v6_student_eval_$(date +%Y%m%d_%H%M%S)
bash scripts/run_joint_v6.sh prepare "$V6_EVAL"
bash scripts/run_joint_v6.sh eval "$V6_EVAL"
bash scripts/run_joint_v6.sh summarize "$V6_EVAL"
bash scripts/run_joint_v6.sh package "$V6_EVAL"
```

eval分别执行纯policy、independent、同seed independent A/A重复、verified_policy。不要把verified改善归于纯学生；检查 `decision_student_accepted` 与teacher修复比例。

## 12. 数据聚合而非盲目延长PPO

```bash
bash scripts/run_joint_v6.sh recollect "$V6_EVAL"
```

它在学生访问的状态上保存完整teacher输入与标签。随后聚合初始cache和新cache，在固定验证路线条件下继续蒸馏：

```bash
runuser -u riftuser -- bash -lc '
set -euo pipefail
RIFT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop_2/RIFT
PY=/zeron-vepfs/tjqc/njl/envs/rift/bin/python
cd "$RIFT"
"$PY" -u scripts/joint_rift_v6.py fit \
  --cache /actual/initial_run/cache \
  --additional_cache /actual/student_eval/recollect_cache \
  --config /actual/student_eval/eval.yaml \
  --resume_head /actual/previous_fit/joint_head.ckpt \
  --head_only --teacher joint \
  --validation_routes ACTUAL_FROZEN_DEV_ROUTE_IDS \
  --steps 1500 --output_dir /absolute/new/aggregation_fit
'
```

这一块的actual路径和验证route必须按真实fit summary填写，不能直接粘贴占位符。聚合允许collection head不同，但拒绝backbone、reward/几何、gamma/dt/horizon和质量定义混合。它属于dataset aggregation，不是on-policy policy-gradient；正式测试路线必须另留。

## 13. 验收与不能提前保证的部分

本版能落实并测试：旧包围盒误报被精确几何区分；评分头无旧幅度上限；密集监督的小集可拟合；无益并列变化被教师抑制；质量恢复和交互/联合增量分别统计；有条件的代理验证执行；一条入口正确使用riftuser。

必须由新服务器实验回答：有效双车机会有多少、教师相对质量/单车对照是否有闭环增量、学生能否在新路线逼近教师、是否符合人工核对后的正向交互语义。任何“保证真实闭环改善/保证论文idea成立”的承诺都不符合现有证据。

第一轮只运行第8节pilot，先验收教师，不要同时投长训练。若教师确实有增量，再做第10–12节；若没有，就据SAT误报、split冲突和标签回放结果调整候选/场景，而不再把失败都归于head训练。
