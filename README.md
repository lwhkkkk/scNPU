# scNPU

当前 Python 包名为 `singlecell_npu`，命令行为 `singlecell-npu`。

面向 Ascend 的单细胞预处理原型：CPU AnnData 兼容接口、显式设备快照，以及可复现的百万细胞流式 PCA。当前开发机实际为 **Ascend910_9382 / A3**；910B / A2 是目标平台，尚未验收。

## 已实现

| 能力 | 当前执行位置 |
|---|---|
| AnnData QC、HVG | 复用 Scanpy CPU 算法；不声称 NPU 加速 |
| Normalize / Log1p | CPU 或 torch_npu；CSR 仅传递非零值，支持连续驻留 |
| PCA | 分块 NPU FP32 Gram/投影 + CPU FP64 累加、SciPy 部分特征分解 |
| 百万细胞数据通路 | 10x legacy/v3 分块读取、一次规范化、可复用 canonical CSR 缓存 |
| HVG 变换融合 | 实验性 TileLang AIV CSR → dense、全基因分母归一化与 Log1p；需独立验证环境 |

没有对 Scanpy 做全局 monkeypatch，也不把 NPU tensor 写入 `AnnData.X`。归一化默认目标为 10000；PCA 默认不 scaling，复现旧百万细胞实验的 CLI 则明确采用 **unit variance、无 clipping**。两者不是同一组默认语义。

## 使用已有环境

当前验证使用原 PCA 虚拟环境，只读取其解释器和依赖，不安装或升级共享环境：

```bash
source /home/naie/work/slai/user-liuweiheng/PCA/env_npu.sh
cd /home/naie/work/slai/user-liuweiheng/singlecellNPU
export PYTHONPATH="$PWD"
export PYTHONDONTWRITEBYTECODE=1
PYTHON=/home/naie/work/slai/user-liuweiheng/PCA/.venv_npu/bin/python
```

已有依赖的独立环境可通过 `pip install --no-deps -e .` 安装；本轮没有修改共享环境。torch / torch_npu / CANN 必须按硬件与官方版本配套安装，不由此包自动下载。CPU 模式无需 torch_npu；`DeviceSession(device='cpu')` 也需要 PyTorch。

```python
from singlecell_npu import pp, DeviceSession

pp.calculate_qc_metrics(adata, percent_top=None, inplace=True)
with DeviceSession(adata, device="npu:0") as session:
    session.normalize_total(10000).log1p().writeback(adata)
adata.uns["log1p"] = {"base": None}
pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)
pp.pca(adata, n_comps=50, mask_var="highly_variable", backend="npu")
```

详细参数、内存预算与快照失效规则见 [接口文档](docs/api.md)。百万细胞不要整体上传为 session：逐非零行索引需要额外 `8 × nnz` 字节，应用分块流式管线。

## 复现百万细胞流程

```bash
$PYTHON benchmarks/run_pipeline.py --record artifacts/new-build.json -- \
  --input /home/naie/work/slai/user-liuweiheng/PCA/brain_1m.h5 \
  --output artifacts/new-build --cache cache/brain-canonical \
  --device npu --n-hvg 2000 --n-pcs 50
```

再次运行使用不同 `--output` / `--record` 与相同 `--cache`，会检验输入 SHA-256 并复用缓存。首次构建与后续复用须分别报告；缓存未完整写入时拒绝读取，应另选路径，程序不会覆盖或删除旧结果。可去掉 `--cache` 作直接读取对照，或 `--device cpu` 测试同流程 CPU 后端。

CLI 保留旧实验的输出文件名和内容，包括 `log1p_hvg.npy`、`pca_scores.npy`、`brain_1m_pca.h5ad`、QC/HVG CSV、两张图及 `summary.json` / `SUCCESS`。H5AD 的 `.X` 为 `None`，表达矩阵在独立 `.npy`；不要误认为它包含完整原始 counts。

`--hvg-backend tilelang` 是显式实验选项，需配套 TileLang 环境；默认 `host`。内核限制、测量口径及是否值得采用见 [内核记录](docs/hvg-kernel.md)。没有静默 CPU fallback 或默认启用尚未证明的加速。

当前服务器使用已有依赖组合运行融合实验：

```bash
$PYTHON benchmarks/run_pipeline.py --record artifacts/my-fused.json \
  --tilelang-site /home/naie/work/slai/user-liuweiheng/envs/tilelang-ascend/lib/python3.11/site-packages -- \
  --input /home/naie/work/slai/user-liuweiheng/PCA/brain_1m.h5 \
  --output artifacts/my-fused --cache cache/brain-canonical --device npu --hvg-backend tilelang
```

公平比较已有缓存时，可在 runner 的 `--` 前添加 `--warm-cache cache/brain-canonical`；预读时间单列，不能当成首次运行性能。

## 验证与证据

```bash
$PYTHON -m unittest discover -s tests -p 'test_io.py' -v
$PYTHON -m unittest discover -s tests -p 'test_streaming.py' -v
SINGLECELL_NPU_TEST_DEVICE=1 $PYTHON -m unittest discover -s tests -p 'test_pp.py' -v
$PYTHON benchmarks/compare_artifacts.py artifacts/旧结果 artifacts/新结果 --output artifacts/comparison.json
```

完整比较要求保留 cells/HVG 的身份与顺序，检查表达矩阵、方差、子空间、旋转对齐后的 scores，以及固定种子抽样 2048 细胞的 kNN 重合率。抽样 kNN 不等于完整百万细胞邻接图或聚类验证。门限在 `benchmarks/compare_artifacts.py` 固定，不因测试失败放宽。

研究依据见 [研究与设计报告](研究与设计报告.md)，旧实验复现见 [基线记录](docs/baseline.md)，新实现测量见 [实现验证记录](docs/implementation-results.md)，模块边界见 [架构说明](docs/architecture.md)。既有 PCA/scops 源码、数据与结果保持只读，本轮代码、canonical CSR 缓存、benchmark 日志和结果均位于新项目目录内。

## 当前边界

- 官方 `ops-sparse` SpMM、随机化/低秩 PCA、neighbors/UMAP/Leiden、多卡与跨块 H2D 重叠尚未接入。
- 协方差法需 `O(HVG²)` 存储，均值远大于方差时可能消减失精；不能作为任意矩阵的通用 SVD 替代。
- CSR 缓存以完整源文件 SHA-256 和块策略绑定；目前信任本地缓存文件，没有逐块内容 checksum。避免并发修改源文件或缓存。
- 支持单 genome legacy 10x 和纯 Gene Expression v3；混合 feature types 必须先明确选择。
- 所有设备性能必须标注实际硬件、环境、是否包含编译/传输、缓存状态与完整输出工作量。
