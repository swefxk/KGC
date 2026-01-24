论文标题
Semantics-Preserving Top-K Injection with Confidence Gating and Residual Correction for Efficient Knowledge Graph Completion
（中文可在论文内写：面向高效知识图谱补全的语义保持型 Top-K 注入：置信门控与残差纠错）

摘要
知识图谱补全（KGC）要求在全实体集合上进行 filtered 排名，结构模型擅长全实体检索但语义能力有限，文本模型语义强但难以承受全实体打分开销。本文提出一种高效的“检索-重排”式框架：以 RotatE 作为结构召回器获取 Top-K 候选，在候选内引入文本 Bi-Encoder 的语义分数，并通过置信度门控控制语义注入强度，避免对高置信样本的负扰动。在此基础上，我们提出候选内残差纠错项 Δ，对结构候选的局部排序错误进行定向修正。关键地，我们识别并修复了 Top-K 纠错与全实体 filtered 排名统计之间的“阈值错位”问题，提出语义保持型阈值（保留语义注入、排除 Δ）以保证 rank 统计一致性。基于 `fb15k_custom`（SimKGC 风格划分与文本）测试集，Top-200 候选下主系统 D 相较 C 提升 AVG MRR +0.0396（95% CI [0.0375, 0.0416]），Recall@200 保持不变，表明增益来自候选内重排/纠错而非候选覆盖变化。

引言
背景与动机
标准 filtered link prediction 要求对每个查询在全实体集合上排序。结构化嵌入模型（如 RotatE）能够高效完成全实体检索，但对实体/关系文本语义利用有限；文本模型具备语义判别能力，却通常无法在全实体规模上高效推理。自然思路是“先结构检索 Top-K，再用语义重排”，但工程上有两类关键风险：
- 语义注入不稳定：对高置信样本过度注入可能破坏原本正确的排序。
- 评测口径错配：Top-K 内纠错若与全实体 filtered 排名统计结合不当，会产生阈值错位，导致 rank 统计偏移甚至崩溃。

本文要解决的问题
在保持标准 filtered 评测可比性的前提下，如何以“候选内可控”的方式注入语义，并对候选内排序错误做定向纠错，同时保证不掉点或可解释地控制风险。

方法概览
框架由四个部件组成：
1) 结构检索器 RotatE：全实体打分并产生 Top-K 候选。
2) 语义 Bi-Encoder：对候选内实体做语义匹配打分。
3) 置信度 Gate：基于结构 Top-K 分数统计与关系/方向信息，预测语义注入强度 g。
4) 候选内残差纠错 Δ（Refiner-R2）：只在 Top-K 内施加的结构残差项，修复局部错序。
关键机制：语义保持型阈值保证全实体 rank 统计一致。

方法细节
问题定义与评测口径
给定查询 (h,r,?) 或 (?,r,t)，在实体集合 E 上进行 filtered 排名，评测 RHS、LHS 与 AVG=(RHS+LHS)/2 的 MRR、Hits@{1,3,10} 与 Recall@K。

结构检索与 Top-K 候选
结构模型给出全实体结构分数：
s_struct(q,e), e ∈ E
对每个查询取 Top-K 候选集合 C_K(q)。

语义候选重排（Bi-Encoder）
语义分数定义为点积匹配：
s_sem(q,e) = ⟨f_q(q_text), f_e(e_text)⟩,  e ∈ C_K(q)
其中实现采用 SimKGC 风格 bi-encoder，关键细节如下：
- 文本输入为预先缓存的 entity/relation text embeddings。
- 实体编码：v = f_e(e_text)，MLP 投影并 L2 归一化。
- 关系编码：r = f_r(r_text + dir_emb)，dir_emb 表示 RHS/LHS 方向。
- Query 融合：q = f_q([a, r, a ⊙ r, a − r])，再 L2 归一化。
- 只对 RotatE 的 Top-K 候选计算 s_sem；RHS 用 (h,r→t)，LHS 用 (t,r→h)。

置信度门控与语义注入
Gate 仅使用推理可得特征，避免 gold 泄漏。输入由两部分组成：
1) 结构 Top-K 分数统计（5 维）：m12(=s1−s2)、mean、std、gap(=s1−mean)、entropy(softmax 温度可调)；
2) 关系 id 与方向 id 的嵌入。
输出采用 softplus + clamp：
g = clip(softplus(MLP([stats, rel_emb, dir_emb]) + init_bias), g_min, g_max)
语义注入项：
s_inj(q,e) = b · g(q) · s_sem(q,e),  e ∈ C_K(q)
其中 b 可按 RHS/LHS 分别设置，g 是 query 级别的自适应缩放（对该 query 的所有候选共享）。

候选内残差纠错 Δ（Refiner-R2）
Refiner 仅在 Top-K 内输出残差项 Δ：
s_topK(q,e) = struct_w · s_struct(q,e) + s_inj(q,e) + γ · Δ(q,e),  e ∈ C_K(q)
其中 γ 可按 RHS/LHS 分别设置（当前 LHS 默认 γ=0）。

语义保持型阈值（修复 rank 统计错位）
全实体 filtered rank 统计中，使用阈值：
s_thresh = struct_w · s_gold_struct + (b · g) · s_gold_sem
即阈值保留语义注入、排除 Δ。这样 Δ 只改变候选内相对次序，不会扭曲全局 rank 统计。

实验设置
数据集
- `fb15k_custom`：采用 SimKGC 风格划分与文本设置。

评测指标
- RHS / LHS / AVG 的 MRR、Hits@{1,3,10}
- Recall@K（gold 是否进入 topK）
- Paired bootstrap：对 ΔMRR（D−C）给出 CI

对比对象与消融
- 结构基线：RotatE（R0，full-entity）
- C：Sem+Gate（无 Δ）
- D：Sem+Gate+Δ（γ>0）
- Safety：entropy gating（保守版，可选）
- 关键消融：错误阈值 vs 语义保持阈值；TopK 敏感性；Δ 的 γ 曲线；bucket（head/torso/tail）

核心结果（run_id=200, test）
System-Protocol Comparison（TopK injection exact-rank, K=200）：
| Setting | AVG MRR | H@1 | H@3 | H@10 | Rec@200 |
|---|---|---|---|---|---|
| RotatE@TopK (strict) | 0.3075 | 0.2030 | 0.3521 | 0.5126 | 0.8363 |
| C: Sem+Gate | 0.3204 | 0.2245 | 0.3564 | 0.5136 | 0.8363 |
| D: Sem+Gate+Δ | 0.3599 | 0.2728 | 0.3922 | 0.5367 | 0.8363 |
| Safety: entropy q=0.6 | 0.3361 | 0.2408 | 0.3733 | 0.5282 | 0.8363 |

显著性（paired bootstrap, AVG）：
- D − C: +0.0396 (95% CI [0.0375, 0.0416])
- D − RotatE@TopK: +0.0525 (95% CI [0.0504, 0.0547])

解读要点：
- Recall@200 不变，说明增益来自候选内重排/纠错，而非候选覆盖变化。
- D 在同一 system 口径下超过最强单体基线（RotatE@TopK 或 Sem-only@TopK）。

结论
本文提出一种面向 KGC 的高效 Top-K 语义注入与候选内纠错框架，在标准 filtered 评测口径下实现可控、稳定的性能提升。置信度门控抑制高置信样本上的语义负扰动，候选内残差纠错 Δ 进一步修复局部排序错误。更重要的是，语义保持型阈值修复了 Top-K 纠错与全实体 rank 统计的错位问题，保证评测一致性。实验表明，在 `fb15k_custom` 上 Top-200 设置下方法取得显著增益，且 Recall@K 不变，提升主要来自候选内排序质量。

作者自述贡献点（逐条）
1) 语义保持型 Top-K filtered 排名一致性：提出并验证“阈值保留语义注入、排除 Δ”的 rank 统计方式，避免评测错位与崩溃。
2) 置信度门控的候选内语义注入：基于结构 Top-K 分数统计与关系/方向信息学习注入强度，优先保证“不掉点”。
3) 候选内残差纠错 Δ（Refiner-R2）：在 Top-K 内对结构分数施加定向残差修正，显著提升排序质量。
4) 可解释诊断与显著性证据：报告 Recall@K、bucket、paired bootstrap CI，明确增益来源与不确定性。
5) 可复现工程交付：提供主线脚本、统一 artifacts/run_id 管理、以及一键等价验收与 bootstrap 接口。

目标会议/期刊
ISWC 2026

领域关键词
Knowledge Graph Completion; Link Prediction; Retrieve-and-Rerank; Top-K Reranking; Semantic Injection; Confidence Gating; Calibration; RotatE; Bi-Encoder
