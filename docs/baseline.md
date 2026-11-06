# 旧 PCA 可复现基线

`benchmarks/legacy_baseline/` 原样保留已有 PCA 的两份源文件，作为新工具的语义和全流程对照，不随新实现调整。2026-11-04 已在服务器校验它们与 `/home/naie/work/slai/user-liuweiheng/PCA/` 下的原文件 SHA-256 一致。

| 文件 | SHA-256 |
| --- | --- |
| run_pca.py | `9e2fb8eabf681a3bd40f59162ce6756626b3894133e545dca6cbcc7e7b6a4695` |
| npu_backend.py | `73001238f685fa427278ad5e30a91bc0af010811b866cbb35a7f536c5b8457b0` |

## 运行

在 `cellfm` 的新项目根目录运行：

```bash
bash benchmarks/run_legacy_baseline.sh
```

脚本复用旧 PCA 虚拟环境及 `env_npu.sh`，设置 `PYTHONDONTWRITEBYTECODE=1`，不安装依赖、不修改旧实验。结果保存在新项目 `artifacts/legacy-baseline-<UTC timestamp>/`，测量证据保存在 `benchmarks/evidence/legacy-<UTC timestamp>.json` 和 `.log.txt`。每次运行新建输出目录；已存在时拒绝覆盖。支持 `PCA_ROOT`、`INPUT_PATH`、`OUTPUT_PATH`、`RUN_TAG` 环境变量。

## 语义与计时范围

- 输入 10x legacy HDF5：1,306,127 个细胞、27,998 个基因、2,624,828,308 个存储值；每块 8,192 行。只去除零计数细胞，基因最少在 3 个细胞中出现。
- 所有基因计数归一化到每细胞 10,000；Seurat dispersion HVG，20 个均值分箱，严格选 2,000 基因，边界同分按原基因顺序处理。
- HVG log1p 后按样本标准差缩放，无 clipping；50 主成分。FP32 NPU Gram 和 projection，主机 FP64 汇总；旧求解器是 `torch.linalg.svd`，此前 profiling 已确认该形状落 AI_CPU。
- `summary.elapsed_seconds` 从参数解析和输出目录创建之后开始，包含 NPU 初始化、两遍读取与预处理、精度参考检查、表达矩阵与结果写盘、绘图、输入 SHA-256；不包含 Python 导入，也不包含末尾 H5AD 维度回读与 SUCCESS 写入。
- 外层 `process_wall_seconds` 覆盖子 Python 进程全生命周期，包括上述导入和末尾回读。`child_peak_rss_gib` 来自 Linux `RUSAGE_CHILDREN`。
- 子步骤 `gram_wall_seconds` / `svd_wall_seconds` / `projection_wall_seconds` 包含设备传输和同步，但不包含其后 CPU 参考验证；三者之和不能视为完整 NPU 工作总成本。
- 不清空操作系统页缓存，记录为 `uncontrolled; no page-cache eviction`；单次全量结果仅用于首版回归与工程对照，不能替代多次交替运行的性能统计。
- 服务器实际硬件为 **Ascend910_9382（A3）**。这些数据不代表目标 **Ascend 910B（A2）** 性能。新旧流程都使用 NPU，比较结果也不代表相对 CPU 的 NPU 加速比。

## 复现记录

完整运行在 2026-11-04 10:58:20–11:03:52 UTC 成功结束，结果为 `artifacts/legacy-baseline-20261104T105820Z/`，保留全部表达矩阵和 PCA 产物。源日志与外层测量在 [legacy-20261104T105820Z.json](../benchmarks/evidence/legacy-20261104T105820Z.json) 和 [运行日志](../benchmarks/evidence/legacy-20261104T105820Z.log.txt)。

| 测量项 | 单次结果 |
| --- | ---: |
| 完整子进程 wall time | 331.628441 s |
| 程序内部 elapsed | 323.162471 s |
| 进程峰值 RSS | 12.568375 GiB |
| 160 次 Gram 含传输同步 | 2.465414 s |
| 1 次 SVD 含传输同步 | 8.872793 s |
| 160 次 projection 含传输同步 | 1.298150 s |
| Gram 首块相对 Frobenius 误差 | 6.19549e-7 |
| 前 50 个 eigenvalue 最大相对误差 | 1.74140e-6 |
| PCA 子空间最小 cosine | 0.999999888 |
| projection 首块最大绝对误差 | 1.30113e-4 |
| 全量 score 均值最大绝对值 | 3.98473e-7 |

Scanpy HVG 数值、sklearn 小样本 PCA、全量 score covariance、导出 H5AD 形状检查均通过。输入 SHA-256 为 `255a36ee92de25cb3568faa2c27d31fe6d0db30f285c5c977be8d6245de14044`；最终保留 1,306,127 细胞、2,000 HVG、50 PC，累计解释方差比例 0.26548529386520386。

软件：CANN 9.0.0、torch 2.7.1+cpu、torch_npu 2.7.1.post4、NumPy 2.4.6、SciPy 1.17.1、Scanpy 1.11.5、AnnData 0.12.19。所有声明阈值及软件版本也保存在证据 JSON 中。

报告新实现收益时，应保持输入、筛选、HVG tie-break、缩放、PC 数、精度阈值和产物相同；若省略绘图/哈希/CSV，必须分别报告共同计算范围与交付全流程，不能直接相除总时间。
