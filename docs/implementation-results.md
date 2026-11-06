# 首版实现与验证记录

日期：2026-11-04。已完成 **AnnData 预处理接口 + 百万细胞流式 PCA + 一个可选 TileLang 融合内核**，源码同步至 `cellfm:/home/naie/work/slai/user-liuweiheng/singlecellNPU`。旧 PCA/scops 源码、数据、结果未修改；未安装或升级共享环境。

**结论：优先保留数据通路优化与 CPU/NPU 分工，TileLang 暂不设为默认。** 单块融合明显快，但相同缓存预热政策下，本次完整流程仅比普通 NPU 路径少约 2.4 秒，尚不足以证明稳定的端到端优势。

## 已交付

- `pp`：复用 Scanpy CPU QC/HVG；CPU/NPU Normalize、Log1p；分块 PCA，支持层、复制、基因掩码及独立 scaling。
- `DeviceSession`：自有 FP32 快照，稀疏值可连续驻留；AnnData 继续保持 NumPy/SciPy 数据类型。
- `io` / `streaming`：一次规范化 CSR、源文件 SHA-256 绑定缓存、全局 int64 / 块内 int32 索引、按行数和 nnz 限制的流式处理。
- PCA：成熟 torch_npu MatMul 做 Gram/投影，CPU FP64 累加和 SciPy 部分特征分解；避免旧 AI CPU SVD。
- TileLang：HVG CSR→dense + 全基因总量归一化 + Log1p 的 AIV 原型；动态 nnz 不为每块重新编译，输出可直接供 Gram 使用。
- 28 项测试通过；wheel 成功生成：`singlecell_npu-0.1.0-py3-none-any.whl`。没有进行 Git commit。

使用方式见 [README](../README.md)、[API](api.md) 和 [架构说明](architecture.md)。尚未实现通用 NPU SpMM、低秩 PCA、多卡、异步双缓冲、neighbors/UMAP/Leiden。

## 百万细胞端到端结果

真实输入 **1,306,127 cells × 27,998 genes，2,624,828,308 nnz**；2,000 HVG、50 PC。协议保持原实验：全基因总量归一化至 10000、Seurat HVG 稳定取 k、Log1p、单位方差、无 clipping。这里不是默认参数的 Scanpy PCA 性能比较。

实际设备 **Ascend910_9382 / A3**，CANN 9.0.0，PyTorch 2.7.1+cpu、torch_npu 2.7.1.post4；**910B/A2 尚未验证**。

| 完整流程 | 缓存条件 | 外部进程 wall | 峰值 RSS |
|---|---|---:|---:|
| 原 PCA 原样复现 | 原 HDF5 两遍读取，页缓存未控制 | 331.63 s | 12.57 GiB |
| 新版首次构建缓存 | 包含约 21.0 GB canonical CSR 写入 | 248.60 s | 12.51 GiB |
| 新版 CPU FP64 | 已有 CSR，计时前显式读取预热 | 147.89 s | 10.89 GiB |
| 新版 NPU，host HVG 变换 | 同上 | 136.75 s | 12.16 GiB |
| 新版 NPU，TileLang HVG 变换 | 同上 | 134.32 s | 12.32 GiB |

**全部是单次完整运行，不是多次中位数或统计显著性结论。** wall 包括 Python/设备初始化、实际发生的编译加载、读取、计算、数据哈希、验证、H5AD/CSV/NPY 与两张图的导出。三次预热分别读取同样的 21,003,913,052 bytes，耗时 5.47 / 5.45 / 5.44 s，明确排除在表内；没有清空 OS 或 TileLang 编译缓存，因此也不是冷启动指标。源代码快照与具体命令保存在远程 `artifacts/warm-*.source.zip` 和对应 JSON。

同预热政策下，NPU 普通路径相对 CPU 本次约 **1.08×**，融合路径约 **1.10×**。不能把原流程→缓存复用后的整体改善宣称为 NPU Kernel 加速；其中包含数据格式、重复工作消除、CPU 求解器和缓存的共同收益。

保留了之前未控制页缓存的观测：CPU 186.19 s、NPU 129.72 s、融合 121.94 s。CPU 第一遍统计当时为 70.04 s，NPU 为 36.00 s，存在明显主机阶段差异，**不采用那一组数字计算设备加速比**。补测后 CPU/NPU 第一遍为 41.87 / 41.61 s，条件更接近。

证据：[基线](../benchmarks/evidence/legacy-20261104T105820Z.json)、[首次构建](../benchmarks/evidence/new-build.json)、[预热 CPU](../benchmarks/evidence/warm-cpu.json)、[预热 NPU](../benchmarks/evidence/warm-npu.json)、[预热融合](../benchmarks/evidence/warm-fused.json)。源文件哈希为 `255a36ee92de25cb3568faa2c27d31fe6d0db30f285c5c977be8d6245de14044`。

## 精度与旧基线差异

保留同一组预设门限，未因失败放宽。以完整 CPU FP64 Gram/eigh/投影为参考：

| 指标 | NPU 普通路径 | TileLang 融合路径 | 门限 |
|---|---:|---:|---:|
| 全量表达最大绝对差 | 0 | 9.54e-7 | ≤2e-4 |
| 特征值最大相对差 | 1.10e-7 | 9.54e-8 | ≤2e-4 |
| loading 子空间最小余弦 | >0.999999999999 | >0.999999999999 | ≥0.9999 |
| 对齐 score 最大绝对差 | 0.000514 | 0.000476 | ≤0.002 |
| 抽样 kNN 重合率 | 100% | 100% | ≥99% |

cells/HVG 身份及顺序一致，score 相对 Frobenius 差也通过。kNN 仅覆盖固定种子 2048 细胞、样本内部 k=15；没有声称验证完整百万细胞邻接图或聚类结果。参考与候选共享前处理，但 CPU PCA 独立计算；前处理另有 Scanpy 对照，API PCA 另有 sklearn 对照。

旧 SVD 与新流程的对齐 score 最大差曾达到 **0.003869**，门限失败。排查发现主要是旧 FP32 SVD 的小方向误差在极端细胞上放大；随后旧 SVD 对完整 CPU 参考也得到 **0.004052**，同一门限失败，而新流程通过。旧失败记录没有被删除，也没有为了复现它而降低新解的精度。详见 [误差诊断](accuracy-diagnosis.md)、[普通路径验证](../benchmarks/evidence/cpu-vs-new-reuse.json)、[融合验证](../benchmarks/evidence/cpu-vs-new-fused.json)、[旧 SVD 验证](../benchmarks/evidence/cpu-vs-legacy.json)。

## 性能机制与下一步

1. **小矩阵求解留 CPU。** 原 SVD 调用约 8.87 s，CPU top-50 eigh 约 0.22–0.26 s。真实块 Gram + 求解 + 投影的代表性 wall 中位数为 0.213 s（7 次）；L0 只见两次 MatMul 与一次 Sub，设备 kernel 合计约 1.20 ms，CPU 求解期间设备空闲是预期行为。该 L0 的 Core 字段为 N/A，不能从它推断核类型/核数。此段不是全流程计时。
2. **融合收益没有按单 kernel 倍数外推。** 真实块 TileLang 含 H2D 约 4.07 ms，本文 torch 组合路径约 21.76 ms；但同预热政策下，完整第二遍仅由 46.95 s 降至 41.29 s，整段流程仅下降约 1.8%。默认保持 host 变换，融合显式 opt-in。
3. **主机读取与重复统计仍优先。** 当前第一遍约 42 s、HVG/QC 导出约 17 s；下一步应研究参数绑定的统计缓存、I/O 预取和减少重复扫描，再验证 AIV/AIC 重叠是否值得做。不要先重写已经只占少量时间的 MatMul。
4. **下个验收平台是真正的 910B。** 复验环境、编译、精度和完整流程，再扩大数据集/密度范围。官方 `ops-sparse` SpMM 与低秩求解器在新的高维需求有证据时再接入。

原始 trace 在远端 `artifacts/dense-profile/`，小型摘要在 [profile JSON](../benchmarks/evidence/dense-profile-summary.json) 和 [原始 kernel CSV](../benchmarks/evidence/dense-profile-kernels.csv)。测试日志：[18 项接口/数据/集成测试](../benchmarks/evidence/all-unittest.log)、[10 项内核测试](../benchmarks/evidence/all-kernel-tests.log)。
