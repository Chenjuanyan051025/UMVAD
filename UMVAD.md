# UMVAD 复现会话记录

## 1. 论文方法核心逻辑（来自用户描述）

目标算法核心由三部分组成：

- 视角内解耦（分离共性特征与特性特征）
- 隐式体素（Implicit Voxel）进行多视角特性信息融合
- 去噪知识蒸馏（Denoising Knowledge Distillation）训练

### 1.1 去噪知识蒸馏（Teacher-Student，参考 DeSTSeg）

- 教师网络 `T`
  - 使用 ImageNet 预训练 `ResNet18`
  - 参数冻结，不更新
  - 输入：正常的 `V` 个视角图像 `x^(v)`
  - 输出：教师特征 `f_tea^(v)`
- 学生网络 `S`
  - 可训练
  - 输入：伪异常图像 `pseudo(x^(v))`（通过在正常图像上添加合成噪声如 DTD 纹理生成）
  - 任务：去噪重构教师网络的正常特征（从异常输入恢复到正常特征）

### 1.2 关键模块一：视角内解耦（Intra-view Decoupling）

目的：将单视角特征分解为：

- 视角共性特征 `f_c`（低维、跨视角共享冗余信息）
- 视角特性特征 `f_s`（视角特有、互补信息，用于后续融合）

实现：一致性瓶颈（Consistency Bottleneck, CB）

- 使用变分信息瓶颈（VIB）思想
- 编码器将 `f_E` 映射到低维空间得到 `f_c`
- 解耦公式：

`f_s^(v) = f_E^(v) - f_c`

### 1.3 关键模块二：视角间融合（Inter-view Fusion）

实现：隐式体素构建（Implicit Voxel Construction, IVC）

- 初始化一个可学习的 3D 体素原型 `P0`（与样本无关）
- 通过多层 Fusion Block 融合多视角特性特征（自注意力 + 交叉注意力）
  - `Q`：3D 体素特征
  - `K/V`：各视角特性特征 `f_s`
  - 使用 cross-attention 将 2D 特征填入 3D 体素
- 融合完成后得到 `Pn`
- 通过 STN（含可学习角度参数 `alpha^(v)`）投影回 2D 平面，得到 `f_p^(v)`

重构：

- 将 `f_p^(v)` 与 `f_c` 相加
- 输入 Decoder 得到学生输出 `f_stu^(v)`

### 1.4 训练策略：View-wise Dropout

目的：增强视角缺失/相机损坏场景的鲁棒性

- 训练时随机丢弃部分视角
- 在 IVC 的交叉注意力层使用 `Masked-View Cross Attention`
- 若视角 `j` 被丢弃，则 attention mask 对应位置设为 `-inf`

公式（用户给出）：

`MVCA(Q, K, V) = softmax(M + QK^T / sqrt(d)) V`

其中 `M` 为掩码矩阵（丢弃位置为 `-inf`，其余为 `0`）

### 1.5 损失函数

总损失：

`L_Total = alpha * L_Distill + beta * L_IB`

1. 蒸馏损失 `L_Distill`
   - 用余弦相似度约束教师/学生特征一致
   - `L_Distill = sum_v (1 - cossim(f_tea^(v), f_stu^(v)))`
2. 瓶颈损失 `L_IB`
   - 用 KL 散度约束 CB 模块的共性特征分布接近先验

### 1.6 推理与异常评分

- 测试时输入测试图像（不做伪异常处理，或视为噪声为 0）
- 计算教师输出 `f_tea` 与学生输出 `f_stu`
- 基于余弦距离计算差异图作为异常图
- 上采样得到像素级异常定位图
- 取最大值得到样本级异常得分

### 1.7 用户总结（原意）

- `CB` 去除多视角间冗余信息
- `IVC` 利用 3D 先验融合互补信息
- `View-wise Dropout` 提高抗干扰能力
- 在 Real-IAD 等数据集上取得 SOTA 性能（需后续严格复现验证）

---

## 2. 当前复现目标与约束（已确认）

- 目标：`严格复现论文结果`
- 实现方式：`从零实现`
- 主数据集：`Real-IAD`
- 评估范围：`全部指标`
- 运行策略：`单次跑出接近论文结果`（当前不做多随机种子统计）
- 实验范围（当前阶段）：`先只做主结果表`（暂不优先消融）
- 官方代码：`未开源`

---

## 3. 已确认本地资源路径（已确认）

- 论文 PDF：
  - `D:\UMVAD\33349-Article Text-37417-1-2-20250410 (1).pdf`
- Real-IAD 数据路径：
  - `D:\UMVAD\realiad`

---

## 3.1 论文版本与来源（已从本地 PDF 提取）

- 论文标题：`Unveiling Multi-View Anomaly Detection: Intra-view Decoupling and Inter-view Fusion`
- 会议：`AAAI 2025 (AAAI-25)`
- 本地 PDF：`D:\UMVAD\33349-Article Text-37417-1-2-20250410 (1).pdf`
- 已提取文本文件：`D:\UMVAD\paper_extracted.txt`

注意：

- PDF 首页文字包含 `Code – https://github.com/Kerio99/IDIF` 的代码链接声明。
- 但当前用户反馈为“官方没开源”，后续需要实际验证该链接是否可用/是否完整。
- 当前复现策略仍按用户要求：`从零实现`。

---

## 4. 已确认硬件条件（已确认）

- GPU：`1-2 张 48GB A6000`

这意味着后续可尝试较高分辨率/更大 batch/更大的隐式体素规模，但仍需根据论文设置优先对齐超参数。

---

## 5. 会话中提出但尚未确认的问题（待补充）

以下问题仍未在当前对话中得到回答，后续开始复现前建议补齐：

1. 论文的正式标题、年份、会议/期刊信息（当前仅有 PDF 文件）
2. 论文版本是否为最终版（会议版 / 扩展版 / arXiv 版）
3. 是否完全不参考任何第三方复现代码（官方未开源，但是否允许参考社区实现）
4. `DTD`（或论文使用的伪异常纹理源）数据是否已准备好，以及路径
5. 是否先做小规模 sanity check（少类别）再扩展到 Real-IAD 全类别，还是直接全量
6. 训练环境偏好：Windows 原生 / WSL / Linux
7. 软件栈版本要求：Python / PyTorch / CUDA
8. 是否需要完整实验追踪（TensorBoard/W&B、配置快照、固定 seed 记录等）
9. “全部指标”具体清单是否包含：
   - image-level 指标
   - pixel-level 指标
   - PRO
   - 类别平均与总体汇总
10. 训练时长是否有硬性限制（几天内完成）
11. 当前工程 `D:\UMVAD` 中已有代码的使用策略（复用/重构/仅作占位）
12. 伪异常生成策略是否必须严格对齐论文细节（混合方式、mask、纹理采样）
13. IVC 实现是否允许先做近似版本进行验证（如果严格复现阶段不允许，需要直接按论文实现）

---

## 6. 从 PDF 中抽取到的关键实现细节（已确认）

以下信息为论文正文明确给出，可作为硬约束：

### 6.1 训练框架与监督

- 教师网络：`ImageNet 预训练 ResNet18`（冻结）
- 蒸馏监督形式：`Denoising Knowledge Distillation`（参考 DeSTSeg）
- 学生输入：`pseudo(x^(v))`（伪异常图像）
- 蒸馏损失：逐视角 `cosine similarity`（论文 Eq.1）
- 总损失：`L_Total = α * L_Distill + β * L_IB`（论文 Eq.10）

### 6.2 学生结构（论文描述）

- 先做 `Intra-view Decoupling`（CB）
- 再做 `Inter-view Fusion`（IVC）
- IVC 使用：
  - learnable `3D voxel prototype P0`
  - 多层 Fusion Block（`Self-Attention + Cross-Attention + FFN`）
  - `STN(v)` + learnable angle `α^(v)` 将 `Pn` 投影回各视角 2D 特征
- 重构：`f_stu^(v) = Decoder^(v)(f_p^(v) + f_c)`

### 6.3 View-wise Dropout（论文描述）

- 训练时随机丢弃部分视角
- IVC 中将 cross-attention 替换为 `Masked-View Cross Attention (MVCA)`
- 被丢弃视角对应 attention mask 置为 `-inf`（论文 Eq.11/12）
- 论文说明示例：每视角 dropout 概率设为 `0.2`

### 6.4 实验设置（论文明确给出）

- 输入尺寸：`256 x 256`
- Batch size：`8`
- 教师监督层：`ResNet18 block1 (64x64), block2 (32x32), block3 (16x16)`
- 伪异常纹理源：`DTD (Describable Textures Dataset)`
- Real-IAD：每个样本 `5 views`，共 `30 categories`

### 6.5 评估指标（论文明确给出）

- `S-AUROC`：样本级异常检测
- `P-AUROC`：像素级异常定位
- 论文说明：样本只有当所有视角均正常时才视为正常

---

## 7. 从 PDF 中抽取到的目标结果（对齐目标）

### 7.1 Real-IAD 主结果（Table 1，Ours）

- `Average All = 97.0 / 98.9`（`S-AUROC / P-AUROC`）

说明：

- 这是当前 Real-IAD 主结果表的首要对齐目标。
- 论文称在 30 个类别中有 22 个类别达到最佳。

### 7.2 MVTec 3D-AD 与 Eyecandies（Table 2，Ours）

- `MVTec 3D-AD Mean S-AUROC = 95.6`
- `Eyecandies Mean S-AUROC = 94.2`

### 7.3 Real-IAD 消融（Table 3）

- `DKD ensemble baseline = 94.0`
- `DKD + 3DConv = 95.8`
- `DKD + IVC = 96.1`
- `DKD + IVC + CB (IDIF) = 97.0`

### 7.4 缺失视角测试（论文正文）

- 随机缺失 1 / 2 / 3 / 4 个视角时，`S-AUROC` 分别为：
  - `95.9`
  - `95.7`
  - `95.1`
  - `94.0`
- 论文报告：
  - 每缺失一个视角平均性能下降约 `0.57% S-AUROC`
  - 使用 View-wise Dropout 后全视角性能从 `97.0` 略降到 `96.4`

---

## 8. 严格复现面临的信息缺口（来自论文正文缺失）

当前 PDF 未明确给出（至少已提取文本中未出现）的关键实现细节：

- 优化器类型（如 Adam / AdamW / SGD）
- 初始学习率、权重衰减、学习率调度器
- 训练 epoch / iteration 数
- `α`、`β`（总损失权重）具体取值
- CB 模块 bottleneck 维度、重参数化细节、`q(f_E|f_c)` 的具体实现
- IVC 中体素尺寸（`D x H x W`）、Fusion Block 层数 `n`、注意力头数
- STN 的具体参数化方式（单角度/多角度、旋转轴定义）
- Decoder 的具体结构
- Segmentation Head 的结构与训练损失（正文提到存在该模块，但未给出明确训练公式）
- 伪异常生成的具体超参数（混合比例、mask 生成方式等，正文仅说明遵循 DeSTSeg）

这些缺口需要通过以下方式补齐：

1. 严格按论文文字实现可确定部分
2. 将未公开细节做成配置项并通过主结果对齐反推
3. 必要时参考 DeSTSeg 的公开实现细节（仅作“论文未公开参数”的合理补全依据）

---

## 9. 实施过程中发现的 Real-IAD 数据协议细节（关键）

在实际读取 `Real-IAD` JSON 并按多视角分组时，发现：

- 同一个多视角样本（同一 `sample_key`）的不同视角 `anomaly_class` 可能不一致
  - 例如同一样本某些视角为 `OK`，某些视角为异常类（如 `HS`）
- 因此，不能假设“组内所有视角标签完全一致”

这与论文中的样本级定义是兼容的（样本只要任一视角异常即可视为异常），但会影响实现：

1. 数据加载器不能强制组内标签一致（已修正）
2. 样本级 `is_anomaly` 应按“任一视角异常”定义（已实现）
3. 样本级 `anomaly_class` 在组内不一致时应标记为 `MIXED`（当前实现采用该策略）

---

## 10. 当前已实现的复现代码范围（截至本轮）

已在 `D:\UMVAD` 中实现以下模块（第一版，可运行）：

### 10.1 模型与方法（核心）

- `src/umvad/models/backbones.py`
  - `ResNet18` 多层特征提取（教师/学生共用）
- `src/umvad/models/idif_modules.py`
  - `ConsistencyBottleneck (CB)`（变分瓶颈形式，含重参数化）
  - `FusionBlock`（Self-Attn + Cross-Attn）
  - `ImplicitVoxelConstruction (IVC)`（learnable voxel prototype + Fusion Blocks）
  - `STNProjector3D`（3D 旋转 + 投影回 2D）
  - `ViewFeatureDecoder`（多尺度特征解码到 `layer1/2/3`）
  - `Conv3DFusion`（用于 Table 3 的 3DConv 消融）
- `src/umvad/models/idif.py`
  - `IDIFModel`（冻结教师 + 可训练学生）
  - 多尺度余弦距离异常图与样本分数计算
  - 消融开关：
    - `use_cb: true/false`
    - `fusion_mode: ivc / 3dconv / none`

### 10.2 训练与伪异常生成

- `src/umvad/data/pseudo_anomaly.py`
  - `DTDTexturePool`（递归纹理采样）
  - `PseudoAnomalySynthesizer`（DTD/随机纹理 + 平滑 mask 混合）
- `src/umvad/engine/losses.py`
  - 多尺度蒸馏余弦损失
  - CB 瓶颈损失（重构 + KL）
- `src/umvad/engine/trainer.py`
  - `IDIFTrainer`
  - 训练循环、AMP、checkpoint、日志
  - `view-wise dropout` 训练
  - 测试时随机缺失视角评估

### 10.3 评估与指标

- `src/umvad/engine/metrics.py`
  - `S-AUROC`
  - `P-AUROC`
  - `PRO`（基于连通域 region overlap）
  - 分类别指标汇总

### 10.4 脚本与配置

- 训练脚本：`scripts/train_idif.py`
- 评估脚本：`scripts/eval_idif.py`
- 缺失视角扫点评估脚本：`scripts/eval_missing_views_sweep.py`
- 配置模板：
  - `configs/idif_realiad_main.yaml`
  - `configs/idif_realiad_missing_views.yaml`
  - `configs/idif_realiad_ablation_template.yaml`

### 10.5 已完成的本地 smoke test

- 随机输入前向（主模型、3DConv、no-CB、no-fusion）通过
- 基于 `Real-IAD` 子集的最小训练步（含反传）通过
- 基于 `Real-IAD` 子集的评估（含 `P-AUROC/PRO`）通过

---

## 11. 当前仍需完成的“严格复现”工作（后续）

1. 对齐论文未公开超参数（优化器/LR/epoch/损失权重/体素尺寸等）
2. 严格对齐 DeSTSeg 伪异常合成细节（当前为可运行近似实现）
3. 明确并实现论文提到但正文未给细节的 `Segmentation Head`（如确实参与训练）
4. 实现/验证 `DKD ensemble` 的完整对照流程（Table 3）
5. 在 `Real-IAD` 全类别上跑主结果表并对齐 `97.0 / 98.9`
6. 跑缺失视角实验并对齐 `95.9 / 95.7 / 95.1 / 94.0`
7. 若需“完整复现论文全部实验”，补齐 `MVTec 3D-AD` 与 `Eyecandies` 的多视角预处理与评估流程

---

## 12. 4060 Laptop 测试运行建议（新增）

用户问题：`4060 Laptop 可以跑吗？`

结论：

- 可以跑通、做单类/子集验证、做小规模训练与评估
- 不建议作为“严格复现论文完整结果（全量 Real-IAD 主表+消融）”的主力机器
- 主要限制来自：
  - 显存（4060 Laptop 常见 `8GB`）
  - 全量评估的耗时与内存开销

### 12.1 已提供测试配置（4060 友好）

- 配置文件：`configs/idif_realiad_4060_smoke.yaml`

主要设置：

- 单类 smoke test：`categories: ["audiojack"]`
- `batch_size: 1`
- `AMP: true`
- 缩小 IVC 规模：
  - `cb_latent_channels: 32`
  - `voxel_shape: [2, 4, 4]`
  - `num_fusion_blocks: 1`
  - `num_heads: 4`
- `epochs: 5`
- `eval_every: 5`（只在最后评估一次，节省时间）

### 12.2 预计计算时间（4060 Laptop，经验估计）

在以下假设下：

- GPU：`RTX 4060 Laptop`（约 `8GB` 显存）
- Windows 环境
- 配置使用 `configs/idif_realiad_4060_smoke.yaml`
- 单类 `audiojack`

预计总时长（5 epoch + 末次评估 1 次）：

- 约 `30 ~ 90 分钟`

更细分（粗估）：

- 训练：`20 ~ 60 分钟`
- 末次评估（含 `P-AUROC/PRO`）：`8 ~ 25 分钟`

注意：

- 若笔记本散热受限发生降频，时间可能进一步增加
- 若将 `num_workers` 从 `0` 提到 `2` 且系统稳定，通常会更快一些

### 12.3 运行命令（4060 smoke）

训练：

`python scripts/train_idif.py --config configs/idif_realiad_4060_smoke.yaml`

评估（训练结束后）：

`python scripts/eval_idif.py --config configs/idif_realiad_4060_smoke.yaml --checkpoint runs/idif_realiad_4060_smoke/best.pt`

### 12.4 严格复现前的建议

- 先用该配置确认：
  - 数据加载正常
  - 训练不 OOM
  - 指标计算链路正常
- 再迁移到 A6000 机器跑论文对齐配置

---

## 13. A6000 标准结构测试建议（新增）

用户澄清：需要的是 `标准（A6000）` 测试超参数，而非 4060 降配版本。

### 13.1 基于当前数据加载器统计的多视角样本规模（`drop_incomplete=True`）

- `Real-IAD train`: `7293` 个多视角样本
- `Real-IAD test`: `22917` 个多视角样本

当 `batch_size = 8` 时：

- 每个训练 epoch 步数：`7293 // 8 = 911` 步（`drop_last=True`）
- 测试评估步数：`ceil(22917 / 8) = 2865` 步

### 13.2 已提供 A6000 标准结构测试配置

- 配置文件：`configs/idif_realiad_a6000_standard_test.yaml`

特点：

- 使用论文风格主结构（不降模型规模）：
  - `batch_size: 8`
  - `256x256`
  - `5 views`
  - `CB + IVC`
  - `voxel_shape: [4, 8, 8]`
  - `num_fusion_blocks: 2`
  - `num_heads: 8`
- 仅缩短训练时长用于“先跑起来验证”：
  - `epochs: 10`
  - `eval_every: 10`（只在最后评估一次）

### 13.3 A6000 预计计算时间（单卡，经验估计）

以下为 `configs/idif_realiad_a6000_standard_test.yaml` 的粗估：

- `10 epoch` 训练 + `1 次完整评估`：约 `2 ~ 5 小时`

更细分（单卡 A6000 48GB，AMP 开启）：

- 训练部分：约 `1.5 ~ 3.5 小时`
- 末次评估（含 `S-AUROC / P-AUROC / PRO`）：约 `30 ~ 90 分钟`

补充说明：

- `PRO` 计算在 CPU 上会显著拉长评估时间（尤其全量 test）
- Windows 下 `num_workers` 与磁盘性能会明显影响速度
- 当前代码默认是单卡训练（尚未实现 DDP），即使有 2 张 A6000 也不会自动加速

### 13.4 若按“完整主跑”估计（非本测试配置）

若使用标准结构并把训练拉长到 `50 epoch`（当前 `configs/idif_realiad_main.yaml` 的占位值）：

- 单卡 A6000 粗估总时长：约 `8 ~ 20 小时`（取决于评估频率与 I/O）

为节省时间，建议：

- 训练过程中降低评估频率（如 `eval_every = 5` 或 `10`）
- 最终单独跑一次完整评估与缺失视角扫点

---

## 14. 当前阶段共识（流程层面）

- 已进入复现实现阶段，优先目标是先搭建“可训练、可评估、可做消融”的完整代码框架。
- 现已完成 PDF 文本提取与第一轮信息抽取，并完成核心模型/训练/评估管线第一版实现。
- 后续重点转为：对齐论文未公开超参数与实验协议细节，迭代逼近论文指标。

---

## 15. 运行问题记录（环境）

### 15.1 `scipy` 安装损坏导致训练脚本导入失败（用户环境 `mvad`）

用户报错（2026-02-23 会话）：

- `ImportError: The scipy install you are using seems to be broken`
- 触发路径：训练脚本导入 `trainer -> metrics -> scipy.ndimage`

已做代码侧修复：

- `src/umvad/engine/metrics.py` 中将 `scipy` 改为可选依赖
- 当 `scipy` 不可用/损坏时：
  - 训练可正常启动
  - `PRO` 指标返回 `NaN`
  - `S-AUROC / P-AUROC` 不受影响（仍可计算）

后续建议（环境层面）：

- 在 `mvad` 环境中重新安装 `scipy`，恢复 `PRO` 指标计算能力

### 15.2 `NumPy` / PyTorch NumPy bridge 不可用导致评估阶段失败（用户环境 `mvad`）

用户后续报错：

- `RuntimeError: Numpy is not available`
- 触发位置：`evaluate()` 中 `tensor.cpu().numpy()`
- 表现：训练流程能进入 `fit()`，在末次评估阶段失败

伴随警告：

- `Failed to initialize NumPy: DLL load failed while importing _multiarray_umath`

说明：

- 该问题通常表示 `numpy` 安装损坏或依赖 DLL 冲突（常见于 `conda` / `pip` 混装）
- 当前环境中不仅 `scipy`，`numpy` 也存在异常

代码侧临时绕过（已实现）：

1. `trainer.fit()` 支持 `eval_every = 0` 跳过验证评估
2. 即使不评估，也会额外生成 `best.pt`（便于脚本兼容）
3. 已将 `configs/idif_realiad_4060_smoke.yaml` 默认设置为 `eval_every: 0`

影响：

- 可继续完成训练与保存 checkpoint
- 暂时无法在该环境计算任何依赖 `.numpy()` 的评估指标（包括 `S-AUROC / P-AUROC / PRO`）

建议：

- 尽快重装 `numpy`（并一并修复 `scipy`），之后再恢复评估

---

## 16. 新 Conda 环境重建（用户决策）

用户决定重建新的 conda 环境以解决 `numpy/scipy` 损坏问题。

已提供环境文件：

- `environment.win-gpu.yml`

推荐命令（Windows / Anaconda Prompt）：

1. 创建环境
   - `conda env create -f environment.win-gpu.yml`
2. 激活环境
   - `conda activate mvad_clean`
3. 验证核心包与 CUDA
   - `python -c "import torch, torchvision, numpy, scipy; print(torch.__version__, torchvision.__version__); print(numpy.__version__, scipy.__version__); print('cuda', torch.cuda.is_available())"`
4. 验证 GPU 名称
   - `python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"`
5. 运行项目 smoke test
   - `python scripts/train_idif.py --config configs/idif_realiad_4060_smoke.yaml`

注意：

- 新环境中尽量不要再混用 `pip` 和 `conda` 安装 `torch/torchvision/numpy/scipy`
- 若 `pytorch-cuda=12.1` 因驱动问题不可用，可改为 `11.8`
