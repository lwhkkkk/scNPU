# AnnData 接口（首版）

`singlecell_npu.pp` 保留 CPU AnnData；不 monkeypatch Scanpy，不将 NPU tensor 填进 `.X`。依赖 Scanpy 的 QC/HVG **目前是明确的 CPU 路径**；Normalize/Log1p 和 PCA 可显式选择 `backend="npu"`。这些接口支持内存中的 NumPy/SciPy 数据；大规模磁盘数据请使用流式管线。

```python
from singlecell_npu import pp
from singlecell_npu.session import DeviceSession

pp.calculate_qc_metrics(adata, percent_top=None, inplace=True)  # Scanpy CPU
# 单独调用会各自上传/下载；连续处理建议使用独立设备快照。
with DeviceSession(adata, device="npu:0") as session:
    session.normalize_total(target_sum=10000).log1p().writeback(adata)
adata.uns["log1p"] = {"base": None}  # session 只管理数值，不隐式修改元数据
pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)  # Scanpy CPU
pp.pca(adata, n_comps=50, mask_var="highly_variable", backend="npu")
```

## 契约与范围

* `calculate_qc_metrics` / `highly_variable_genes` 直接转发到 Scanpy，参数、返回值、HVG flavor 与依赖要求沿用安装版本；没有近似 flavor 替换。
* `normalize_total` 默认 `target_sum=10000`，并非 Scanpy 的默认中位数。NPU 路径要求有限正目标；CPU 路径还支持 `None`。分母覆盖输入矩阵的全部列，若输入已切到 HVG，不会自动恢复全基因总量。融合 HVG 流式路径使用单独的全基因总量。
* `normalize_total` / `log1p` 支持 `layer` 和 `copy`；`copy=False` 原地修改返回 `None`，`copy=True` 返回新 AnnData。原地操作视图会报错。未支持的 Scanpy 参数因签名限制报错，不被忽略。
* 设备快照是独立所有权的 FP32 数据；源数据值的后续修改不会反映在快照中。`writeback` 检查形状和 obs/var 轴名，但**不能检测同轴值修改**，调用方应避免在快照期间修改源矩阵。关闭后不可再次访问。
* CSR 在上传前合并重复坐标、移除零、排序；NPU 保留稀疏值，使用一次构建的逐非零行索引实现行归约，不将整个矩阵变密。额外设备行索引约 `8 × nnz` 字节，适用于能驻留的批次；百万级请使用流式路径。
* Session 仅更新数值，调用方负责相关 `.uns` 元数据。`pp.log1p` 会写标准 `uns['log1p']`。

## PCA

`pca(..., zero_center=True, scale=False, mask_var=None, block_rows=8192)` 默认所有基因；若需 HVG 必须显式给出掩码。`scale=True` 表示样本标准差归一化、无 clipping；这是单独于普通 PCA 的选项。`zero_center=False` 为非中心化第二矩特征分解，不允许同时 `scale=True`。

NPU 使用 FP32 Gram 与投影，禁用 HF32；块间统计、累加及 SciPy 部分特征分解在 CPU FP64 完成。CPU 基准计算使用 FP64。NPU 精度选项使用上下文恢复，但它们是进程级设置；不要在同一进程用多个线程同时执行不同精度策略。协方差法对极大均值、极小方差输入存在消减误差，不适合作为通用高条件数 SVD 替代。

`max_covariance_bytes` 默认 512 MiB，按六个 FP64 方阵预估协方差/求解工作区；超限明确报错，推荐先选 HVG。此预算不包含 AnnData 原数据、单块 dense 数据和最终 scores。两次按行分块读取：Gram 后再投影，避免保留全体 dense 表达矩阵。公开输出为标准 `obsm['X_pca']`、`varm['PCs']`、`uns['pca']`，未选基因 loading 为零。`uns['pca']['singlecell_npu']` 标记实际 backend 相关数值策略和墙钟时间。

## 验证

`tests/test_pp.py` 使用 Scanpy 和 sklearn 为独立基准，覆盖 dense/CSR、重复坐标、零细胞、零基因、layer/copy、视图/掩码、中心化/非中心化及 scaled PCA。声明的 NPU 门限为逐元素 `atol=2e-4, rtol=2e-5`，PCA 方差相对误差 `2e-4`、子空间最小余弦 `>0.9999`；测试前固定门限。

```bash
python -m unittest discover -s tests -p test_pp.py -v
SINGLECELL_NPU_TEST_DEVICE=1 python -m unittest discover -s tests -p test_pp.py -v
```

当前可用服务器为 Ascend 910_9382/A3；通过的设备测试不等于 910B/A2 已完成验证。

2026-11-04 在 `cellfm` 使用原 PCA 环境（Python 3.11、torch 2.7.1、torch_npu 2.7.1.post4、Scanpy 1.11.5、AnnData 0.12.19），设置 `SINGLECELL_NPU_TEST_DEVICE=1` 后 **9/9 测试通过，unittest 墙钟 6.580 秒**。这是正确性测试时间，不是算子性能指标。首轮暴露该版本 `npu.config.allow_internal_format` 仅有 setter、无 getter；已去掉对这个进程级选项的修改，仅对可读取的 HF32 设置做临时覆盖和恢复，之后全套通过。
