# 数据集说明（`realiad/`）

## 1. 数据集概览

该目录下的数据集主体为 `realiad/`，看起来是一个用于工业异常检测/分割（Anomaly Detection / Segmentation）的多类别多视角图像数据集。

基于本地文件统计（`d:\UMVAD\realiad`）：

- 类别数：30
- 图像文件（`.jpg`）：151,050
- 掩码文件（`.png`）：51,329
- 图像目录：`realiad/realiad_256/`
- 标注与划分目录：`realiad/realiad_jsons*`

说明：

- `realiad_256` 目录名表明这是一份 `256` 版本数据（通常表示 256 尺度/分辨率版本）。
- `.jpg` 为图像，`.png` 为像素级异常掩码（mask）。

## 2. 类别与目录结构

### 2.1 类别列表（30 类）

`audiojack`, `bottle_cap`, `button_battery`, `end_cap`, `eraser`, `fire_hood`, `mint`, `mounts`, `pcb`, `phone_battery`, `plastic_nut`, `plastic_plug`, `porcelain_doll`, `regulator`, `rolled_strip_base`, `sim_card_set`, `switch`, `tape`, `terminalblock`, `toothbrush`, `toy`, `toy_brick`, `transistor1`, `u_block`, `usb`, `usb_adaptor`, `vcpill`, `wooden_beads`, `woodstick`, `zipper`

### 2.2 图像目录结构（按类别）

每个类别位于：

- `realiad/realiad_256/<category>/`

常见结构为：

- `OK/`：正常样本
- `NG/`：异常样本（按缺陷类型再分子目录）

示例（以 `audiojack` 为例）：

- `realiad/realiad_256/audiojack/OK/...`
- `realiad/realiad_256/audiojack/NG/BX/...`
- `realiad/realiad_256/audiojack/NG/HS/...`
- `realiad/realiad_256/audiojack/NG/QS/...`
- `realiad/realiad_256/audiojack/NG/ZW/...`

全数据集中观察到的缺陷类型代码（并非每个类别都包含全部类型）：

- `AK`, `BX`, `CH`, `HS`, `PS`, `QS`, `YW`, `ZW`

文件名中可见多视角标识 `C1`~`C5`（例如 `..._C3_...jpg`），说明原始数据含 5 个视角。

## 3. JSON 标注格式说明

各标注文件位于 `realiad/realiad_jsons*.json`，通常按类别一个 JSON 文件，例如：

- `realiad/realiad_jsons/audiojack.json`

JSON 结构（抽样确认）：

- `meta`
- `train`
- `test`

其中 `train` / `test` 每条记录包含：

- `category`：类别名
- `anomaly_class`：标签（`OK` 或缺陷类型代码）
- `image_path`：相对路径（相对于 `meta.prefix` 指向的类别目录）
- `mask_path`：掩码相对路径；正常样本为 `null`

`meta` 中常见字段：

- `prefix`：类别前缀（如 `audiojack/`）
- `normal_class`：正常类（通常为 `OK`）
- `pre_transform`：部分版本包含（如 `realiad_jsons`、`realiad_jsons_sv`）

## 4. 标注版本（`realiad_jsons*`）说明

当前目录包含 6 套标注/划分：

- `realiad_jsons`（标准全量划分）
- `realiad_jsons_sv`（单视角划分，统计上仅使用 `C1`）
- `realiad_jsons_fuiad_0.0`
- `realiad_jsons_fuiad_0.1`
- `realiad_jsons_fuiad_0.2`
- `realiad_jsons_fuiad_0.4`

### 4.1 统计摘要（按全部 30 类汇总）

| 标注目录 | JSON数 | 类别数 | Train | Test | Train掩码数 | Test掩码数 | 视角特征 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `realiad_jsons` | 30 | 30 | 36,465 | 114,585 | 0 | 51,329 | `C1`~`C5` |
| `realiad_jsons_sv` | 30 | 30 | 10,408 | 19,802 | 0 | 9,415 | 仅 `C1` |
| `realiad_jsons_fuiad_0.0` | 30 | 30 | 57,840 | 30,000 | 0 | 9,843 | `C1`~`C5` |
| `realiad_jsons_fuiad_0.1` | 30 | 30 | 57,840 | 30,000 | 3,774 | 9,843 | `C1`~`C5` |
| `realiad_jsons_fuiad_0.2` | 30 | 30 | 57,840 | 30,000 | 7,605 | 9,843 | `C1`~`C5` |
| `realiad_jsons_fuiad_0.4` | 30 | 30 | 57,840 | 30,000 | 15,313 | 9,843 | `C1`~`C5` |

补充观察：

- `realiad_jsons` 的 `train + test = 151,050`，与磁盘上的 `.jpg` 总数完全一致，说明该版本基本覆盖了全部图像样本。
- `realiad_jsons` 的 `test` 中非 `OK` 标签数为 51,329，恰好等于 `.png` 掩码总数，说明异常标签样本对应像素级掩码。
- `fuiad_*` 与 `sv` 更像是基于同一图像池派生出的不同训练/测试划分方案。

### 4.2 关于 `fuiad_0.x` 的推断（基于统计）

从统计上看，`fuiad_0.1 / 0.2 / 0.4` 的训练集中，来自 `NG` 路径的样本占比分别约为：

- `0.1`：`5720 / 57840 ≈ 9.9%`
- `0.2`：`11500 / 57840 ≈ 19.9%`
- `0.4`：`23065 / 57840 ≈ 39.9%`

因此可以合理推断：目录名中的 `0.x` 很可能表示训练集中“污染样本”或“来自 NG 路径样本”的比例设定。

## 5. 使用注意事项（很重要）

### 5.1 不要只看路径判断标签

统计发现：JSON 中有不少样本虽然 `image_path` 位于 `NG/...` 下，但 `anomaly_class` 仍然是 `OK`。

例如在 `realiad_jsons` 汇总统计中：

- `test` 中 `anomaly_class = OK` 且 `image_path` 位于 `NG/...` 的样本数为 **26,881**

这意味着：

- 训练/评估时应以 **`anomaly_class` 和 `mask_path`** 为准
- 不应仅根据路径包含 `NG` 就直接当作异常样本

### 5.2 掩码文件与图像文件混在同一目录树

- 图像是 `.jpg`
- 掩码是 `.png`

如果你自己写数据加载器扫描文件目录，请过滤后缀，避免把掩码当作输入图像。

### 5.3 `sv` 版本是单视角（统计上仅 `C1`）

如果实验希望利用多视角信息，不应使用 `realiad_jsons_sv`。

## 6. 建议的数据加载方式

建议优先按 JSON 读取，而不是直接遍历目录：

1. 先选择一个标注版本（如 `realiad_jsons`）
2. 按类别读取对应 JSON
3. 使用 `meta.prefix + image_path` 拼接到 `realiad/realiad_256/`
4. 以 `anomaly_class` 判断类别，以 `mask_path` 是否为空判断是否有像素级标注

如果后续需要，我可以继续补一版：

- 面向训练代码的字段映射说明（PyTorch Dataset 示例）
- 各类别详细样本量统计表（每类 train/test、缺陷类型分布）
- 自动校验脚本（检查 JSON 路径是否全部存在）
