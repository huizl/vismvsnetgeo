# 论文方法图绘制指南

对应稿件：[PAPER_DRAFT_LARGE_DISPARITY_OCCLUSION.md](PAPER_DRAFT_LARGE_DISPARITY_OCCLUSION.md)。

本文件提供六张方法图的绘制说明，不生成位图，也不包含实验结果图。图中关系与正文公式草案保持一致；最终结构确定后同步调整。

## 统一绘图规范

- 画布：白底，优先使用矢量绘图，最终导出 SVG/PDF。总体框架适合通栏，其余图可根据细节使用单栏或通栏。
- 主干：浅灰填充、深灰描边；A 使用蓝色（建议 #2563EB），B 使用橙色（#D97706），C 使用青绿色（#0F766E）。
- 不能只用颜色表达身份；每个模块都写明 A/B/C 和完整名称，黑白打印也应能识别。
- 实线箭头：推理信息流；虚线箭头：训练标签或辅助监督；反向梯度若需要展示，用另一种细虚线并注明 Gradient。
- 图中的方框表示操作，叠片表示图像或特征，薄立方体表示候选深度上的匹配体。
- 可用中文标注绘制初稿；正式投稿按全文语言统一，避免中英文随意混排。
- 图中文字在缩放到排版尺寸后仍清楚可读；不通过极小字号塞入所有公式。
- 不写未经确认的通道数、卷积层数、候选数、耗时或显存。使用 C_t、N_t、H_t、W_t 表示。
- 所有示意概率、采样位置和权重热力图标注“示意 / Schematic”，不能呈现为实测网络输出。
- GT Depth 只出现于训练监督或评测诊断。不能从 GT 直接连入 B 或 C 的推理输入。

## 图1 总体框架图

**建议位置：** 第3.1节。通栏横向，宽高比约 2.3:1。

**图名：** 面向大视差与复杂遮挡的三级误差抑制框架。

### 布局

整张图分上下两条带。上方约占 70%，为推理主干；下方为训练监督。

在推理主干上方增加一条简洁的问题链，使用浅红色细框表示：`Large Disparity / Complex Occlusion → Coarse-stage Ambiguity → Restricted Candidate Coverage / Unreliable View Evidence → Cascaded Error`。分别用细虚线从 A、B、C 指向其作用位置，使读者在阅读网络细节前先看到“问题—模块”的对应关系。这条问题链只表达研究动机，不画成真实数据流。

上方从左到右布置：

1. 输入：一张参考图像和叠放的源图像，旁边独立小框写 Camera Parameters。
2. Shared Feature Pyramid：画三个分辨率层级，用灰色表明继承的基础环节。
3. Stage 1：Global Depth Hypotheses → Geometric Warping → Pair Matching → C: Hypothesis-aware Fusion → Regularization & Regression。
4. 阶段输出：Depth、Probability、Std。显示为三个小图标或小框，不画为真实结果。
5. B: Adaptive Range → Stage 2。同样包含匹配、C 和回归。
6. 第二个 B: Adaptive Range → Stage 3。
7. Final Depth。

“共享特征提取”指同一特征提取器应用于多个输入图像，不代表所有阶段其他模块共享参数。

下方布置：

1. GT Depths（参考和源）与相机参数 → Geometric Visibility Labels。
2. 三个阶段的逐源匹配特征各自连接 Visibility Prediction。
3. Visibility Prediction 与 Labels → A: Visibility Supervision。
4. 各阶段深度与参考 GT → Multi-stage Depth Supervision。
5. 两类监督汇入 Total Loss。

### 必须保留的箭头

- 相机参数分别进入各阶段 Geometric Warping。
- 共享特征分别送入对应阶段，而不是只送 Stage 1。
- Stage t 的预测均值与标准差送入 Stage t+1 的 B。
- B 输出候选深度，进入当前阶段 Warping。
- C 输入基础可靠性和候选相关匹配信息，输出融合特征。
- GT 到标签构造、GT 到深度损失使用虚线。
- A 的梯度作用于共享匹配表示及可见性头；无需画 A 向 C 直接传送 GT 或概率。
- Stage 1 不从 GT 获得搜索范围；全局范围是相机/数据协议给定的输入设定。

### 框内英文建议

- Reference / Source Images
- Shared Feature Pyramid
- Global Depth Hypotheses
- Geometric Warping
- Pair Matching Features
- A: Source-wise Visibility Supervision
- B: Adaptive Depth Range
- C: Hypothesis-aware View Fusion
- Depth / Probability / Std
- Final Depth

### 可直接使用的图注

**图1. 面向大视差与复杂遮挡的三级误差抑制框架。** 大视差与复杂遮挡会增加粗阶段匹配歧义，并通过后续候选范围和多视图融合传播误差。在原始 Vis-MVSNet 的三级主体上，A 以逐源几何可见性监督增强遮挡关系学习，B 根据前级估计状态自适应生成后续候选，C 在不同候选深度处调整源视图贡献。实线表示推理数据流，虚线表示训练监督；真值深度不参与推理。

### 绘制检查

- Stage 2/3 的 B 位于候选生成位置，不画在深度输出后直接修补结果。
- C 属于各阶段内部融合操作，不画成最终深度后处理。
- 问题链中的“三级误差传播”与网络信息流在视觉上分层，避免让读者误以为它是一个可计算模块。
- 三个阶段的可见性头可以使用相同图形，但不要据此标为共享权重，除非最终设计明确如此。
- 如果细节过多，将阶段内部操作放进图4，总体图只保留 Match → C → Regress。

## 图2 A：几何标签与可见性监督

**建议位置：** 第3.2节。左右两个子图。

**左侧子图：几何关系。**

1. 画参考相机 C0、源相机 C1/C2，以及前景挡板和背景表面。
2. 参考相机观察背景点 X。
3. C1 到 X 的射线无遮挡，标 Visible。
4. C2 到 X 的射线先遇到前景点 Y，标 Occluded。
5. 在源相机坐标旁标 z_proj 和 z_src，说明被遮挡时 z_proj > z_src + τ。
6. 可在角落加越界投影/缺失深度图标，标 Ignore，不把它涂成遮挡标签。
7. 这是几何示意，不暗示大视差与遮挡必然同时发生。

**右侧子图：标签流程。**

GT Reference Depth + Cameras → Back-project & Reproject → Source GT Sampling → Validity Check → Visible / Occluded / Ignore。

与此平行：Pair Features → Visibility Head → q_s。

Visible/Occluded 标签及有效掩码，与 q_s 一起送入 Focal/BCE Loss，输出 L_A。

**对应公式：** 式(1)、式(3)—(7)。

**掩码小图：** 可画三个 6×6 方格，用蓝色表示可见、深灰表示遮挡、白底斜线表示忽略；图例写明 Schematic。

**图注：**

**图2. 逐源可见性标签构造与监督。** 对参考点进行几何投影，并根据源视图真值深度区分可见、被遮挡和不参与监督的位置。有效标签约束逐源可见性预测，几何标签仅用于训练监督。

**不要画的内容：**

- 不把参考图有效掩码复制成所有源视图的相同可见性标签。
- 不把所有不一致点统一画成遮挡。
- 不从 GT 标签直接连到推理融合权重。
- 不把可见性标签图画成当前模型实测输出。

## 图3 B：自适应范围与候选采样

**建议位置：** 第3.3节。上流程、下三个案例。

**上方流程：**

Previous Probability → Mean & Std → Upsample / Align → Half-width Constraint → Global Range Intersection → N_t Depth Hypotheses。

框旁放简式 h = clip(κσ, h_min, h_max)，完整公式保留正文。

**下方三个案例：** 每个案例均使用相同方向的深度轴。

### 案例(a)：较小标准差

- 用短竖线画 N_t 个候选，中心标 μ。
- 画较窄分布作为示意，标 Small Std。
- 显示受 h_min 限制，范围不会任意收缩到零。
- 不必画 GT；如果画，用虚线注明仅作示意参照，不进入范围计算。

### 案例(b)：较大标准差

- 候选数量与(a)保持相同。
- 画更宽范围，标 Large Std。
- 明确候选间距随范围增大，不画成范围更宽但点数和间距都不变。
- 文字：“更大搜索空间 / Larger search span”，不写“Guaranteed accuracy”。

### 案例(c)：全局边界

- 标明全局下界 d_min，中心 μ 靠近左侧。
- 先用虚线画未裁剪区间，再用实线画与全局范围的交集。
- 展示真实宽度会缩小。
- 若最终设计采用保持宽度的平移而非交集，此子图和正文式(9)必须一起修改。

**对应公式：** 式(8)—(11)、式(18)。

**图注：**

**图3. 不确定性驱动的候选范围生成。** B 根据前级估计状态调整范围半宽，并在全局深度边界内采样。固定候选数量下，范围宽度与采样间距共同变化；边界截断会影响实际搜索跨度。图中为方法示意。

## 图4 C：深度假设感知源视图融合

**建议位置：** 第3.4节。左右分块。

**左侧：两种权重表达的对比。**

- 画“源视图 × 候选深度”矩阵。
- 左矩阵代表基础像素级可靠性：同一行颜色相同，不同行可以不同。
- 右矩阵代表候选相关权重：同一行随深度列变化。
- 统一色标，标 Weight (Schematic)。
- 在两个矩阵下方写“每个像素均有一张这样的权重矩阵”，避免误解为全图只有一个权重。
- 此处比较权重表达能力，不把图中颜色作为实际效果证据。

**右侧：残差融合流程。**

1. Pair Features + Pair Probability + Normalized Depth Coordinate → Residual Predictor → η tanh → r_s,k。
2. Base Reliability → b_s。
3. b_s 与 r_s,k 相加 → Valid-source Mask → Softmax over Views → α_s,k。
4. α_s,k 与 Pair Features 相乘 → Sum over Views → Fused Features。
5. Fused Features → Regularization → Softmax over Depth → P(k) → Mean / Std。

**特别标注：**

- 第一个 softmax 沿源视图维度，第二个 softmax 沿候选深度维度。
- Valid-source Mask 来自候选投影有效性，不是 GT 遮挡掩码。
- r=0 时恢复基础权重的归一化形式。
- 不将 A 的标签箭头连接到该推理流程。

**对应公式：** 式(12)—(18)。

**图注：**

**图4. 深度假设感知的源视图融合。** C 以候选深度相关残差修正基础源视图可靠性，并在源视图维度归一化后聚合匹配特征。融合结果进一步生成深度概率。示意矩阵用于说明权重表达方式。

## 图5 训练与推理流程

**建议位置：** 第3.5节。双泳道流程图。

左泳道 Training，右泳道 Inference。两个泳道共用相同前向模块：

Images + Cameras → Features → Stage Loop → Final Depth。

训练泳道额外包含 GT Labels、L_A、Depth Loss、Baseline Auxiliary Loss → Total Loss → Backpropagation。

推理泳道在 Final Depth 结束。可见性辅助头是否保留用于诊断应依据最终需求，不能画 GT 输入。

**可复制 Mermaid：**

~~~mermaid
flowchart TB
  subgraph TRAIN[Training]
    TI[Images and Cameras] --> TF[Feature extraction]
    TF --> TS[Stages with B and C]
    TS --> TD[Depth predictions]
    TG[GT depths] --> TV[Geometric visibility labels]
    TS --> TQ[Visibility prediction]
    TV --> TA[A loss]
    TQ --> TA
    TG --> TL[Depth loss]
    TD --> TL
    TA --> TOTAL[Total training objective]
    TL --> TOTAL
    AUX[Retained baseline auxiliary losses] --> TOTAL
    TOTAL --> BP[Backpropagation]
  end
  subgraph INFER[Inference]
    II[Images and Cameras] --> IF[Feature extraction]
    IF --> IS[Stages with B and C]
    IS --> ID[Final depth]
  end
~~~

**图注：**

**图5. 训练与推理的数据使用范围。** 两者共享深度估计主干，训练额外利用真值深度构造逐源可见性标签和监督目标。推理仅使用图像、相机参数及规定的全局深度范围。

## 图6 问题、模块与证据的对应关系

**建议位置：** 引言末尾或讨论部分；若篇幅紧张可用表格代替。

采用三列：Problem → Module → Evidence。

三条主行：

1. 几何标签能否补充已有可靠性学习 → A → 遮挡区域误差、可见性预测诊断。
2. 估计状态能否改善候选覆盖 → B → coverage、width、最近候选误差。
3. 深度相关权重能否改善匹配信息利用 → C → 深度相关权重、困难区域误差。

下方用括号汇总三个模块 → Joint Method → 大视差、遮挡及交集评价。

顶部增加灰色基础框：Vis-MVSNet — Uncertainty-guided Fusion。从该框向三项研究问题连线，表明本文是在已有方法上进一步改进。不要将基础框标成“无可见性处理”。

**文字要求：**

- 左侧写研究问题，不写“原始方法完全没有某功能”。
- 右侧写需要测量的证据，不直接写“显著提升”。
- 图中不填未经确认的百分比。
- 用 A/B/C 字母与颜色保持和图1一致。

**图注：**

**图6. 研究问题、模块设计与评价证据。** 三个模块分别从监督、搜索及融合角度服务于大视差和遮挡改进目标，对应诊断与区域指标用于验证其作用。

## 推荐排版组合

- 正文篇幅较充足：图1总体、图2监督、图3范围、图4融合进入方法，图5放训练设置，图6放引言或讨论。
- 正文篇幅较紧：保留图1；图2—4合成一张三子图；图5放补充材料；图6改为表格。
- 图号随最后排版调整，但公式符号、模块颜色和名称保持一致。
- 结果图片另行准备：本指南中的方法示意不能代替深度误差图、覆盖诊断和真实权重输出。

## 完稿前一致性检查

1. 原始 Vis-MVSNet 是唯一 Base，A/B/C 使用全文统一名称。
2. 框架与损失公式一致：A 是训练监督，B 是候选生成，C 是多视图融合。
3. GT 没有进入推理范围生成和融合。
4. 标准差不是源视图可见性概率，不使用同一符号或图例混表示。
5. 两类 softmax 的维度清楚。
6. 相机内参与图像/特征尺度一致。
7. 候选数量固定时，范围变宽不同时声称间距不变。
8. 图中“仅训练”文字与虚线图例一致。
9. 图示中的超参数、参数共享和层数均来自最终确定方案。
10. 所有概率、热力图、射线和采样点示意均未被标作实测结果。
