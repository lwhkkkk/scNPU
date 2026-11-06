# HVG CSR → normalized log1p dense：TileLang 原型

## 接口与语义

```python
from singlecell_npu.kernels import hvg_to_dense
dense_npu = hvg_to_dense(
    values_npu, indices_npu, indptr_npu, all_gene_totals_npu,
    n_genes=2000, max_row_nnz=max_row_nnz, target_sum=10000.0,
)
```

输入是 CPU 已完成 HVG 选择的 canonical CSR：列索引是 HVG 子空间内的连续编号，行内排序且唯一。values/totals 为 FP32；indices/indptr 为 int32。必须先在 int64 中按块重基 indptr，再确认块 nnz 不溢出后转 int32。归一化分母来自 **HVG 选择之前的全基因 totals**，不是 HVG 行和。

输出定义为 `log1p(count / all_gene_total * target_sum)`，FP32，缺失项为精确 0。全基因 total 为 0 的合法空/零值行输出 0。wrapper 只做 tensor metadata 验证；非负、有限、CSR bounds、sorted/unique、max_row_nnz 等内容由受控 CPU 调用方验证，不为每块重复 D2H 扫描。

当前原型上限：1 ≤ HVG 数 ≤ 8192；每块 dense 元素数和 nnz ≤ int32 最大值。行长通过 256 nnz 分块循环，没有 4096 nnz 截断。nnz 使用已验证的 `T.symbolic` 动态维度，行内循环按实际长度执行，不因每块 nnz/max_row_nnz 改变重复 JIT。Rows/HVG 数仍静态，通常全块和末块有两种 specialization。它是 CSR packing/normalize/log1p 算子，不负责 HVG 统计或挑选。

## 实现选择

每个 AIV subcore 处理一行。连续 values/indices 以 1D DMA 读到 UB，在 UB 内按列索引 scatter 到清零 dense 行，然后执行 FP32 vector 除法、缩放、log1p，整行 1D DMA 写回 GM。没有按 nnz 从 GM 标量读取。两条 AIV subcore 分担相邻两行；这里没有能受益于 Cube 的矩阵乘，未为了形式上的 AIC/AIV 协同增加 Cube 工作。

当前安装的 `tilelang.language.ascend_tile` 有 gather，无公开 vector scatter，所以本版使用 **UB-local scalar scatter**。这可能成为长行瓶颈；不能仅凭算子融合推定它快于 PyTorch。之后可评估经验证的 Ascend C scatter 或按密度切换 dense packing。

`_copy_1d.py` 的兼容方法来自用户已有 `scops-tilelang/scops/kernels/copy_compat.py`：调用 wheel 内 `src/tl_templates/ascend/common.h` 的 `copy_gm_to_ub/copy_ub_to_gm`，绕过该 wheel 动态 region 的 lowering 问题。此处主动限制为 1D，避免已有 2D DMA 尾块错误。未修改编译器或全局环境。

log1p 小值保护参考同项目 `math_compat.py`：`x < 1e-4` 时使用 `x - x²/2`，截断误差 < 3.34e-13；其他元素执行 `ln(1+x)`。计算始终 FP32，没有 FP16/BF16/HF32 转换。三个 dense UB buffers 加 chunk/metadata/mask，8192 genes 时显式 buffer 总量约 102 KB，仍需以编译器实际 UB 分配为准。

## 验证和计时方法

`tests/test_hvg_kernel.py` 对照独立 FP64 CPU reference，检查空行、零 totals、奇数行尾、非 64 对齐列尾、255/256/257 nnz 边界、2000 genes、6001 nnz 长行、tiny positive log1p、fractional totals、零行 batch、动态 nnz 缓存复用、重复运行、metadata 错误。判据 `rtol=2e-5, atol=2e-6`，另要求零/非零 support 精确一致，避免绝对误差掩盖 tiny positive 丢失。

`benchmarks/bench_hvg_kernel.py` 比较同一输入的：

- CPU：FP32 稀疏归一化/log1p + SciPy dense packing。
- torch native：成熟 FP32 gather/divide/log1p/zeros/scatter，预生成 row/flat int64 索引；生成和上传额外索引的成本单列。
- TileLang：本文融合算子。

同时报告预驻留输入的 NPU event 时间、同步 host wall 时间、pageable H2D + 算子的同步 wall 时间。输出留在 NPU，不计 D2H。首次 JIT 单列，warmup 后计时；H2D native 路径包含额外索引重建与上传。该指标是块内子流程，并非端到端 PCA 加速。

## 环境和当前状态

环境：`/home/naie/work/slai/user-liuweiheng/envs/tilelang-ascend`；安装 wheel `tilelang-0.1.1.10+ubuntu.20.4.cann900`，运行时模块版本 `0.1.1.010+e909216995c2acc527bdb252039ec93a920c052c`；PyTorch `2.7.1+cpu` + `torch_npu 2.7.1.post4`；CANN 9.0.0。当前可用设备实际为 `Ascend910_9382`（A3），不能把测量直接宣称为 Ascend 910B 性能。

2026-11-04 实测：**10 tests passed in 32.90 s**。测试时没有修改共享环境。两个计时任务顺序运行，CPU 线程环境 `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=16`，各 warmup 3 次、正式 10 次，以下为中位数：

| 输入（8192 × 2000） | TileLang device | torch native device | CPU FP32 wall | TileLang H2D + wall | torch native H2D + wall |
|---|---:|---:|---:|---:|---:|
| synthetic 8.00%，含长/空行边界 | 1.748 ms | 18.889 ms | 87.692 ms | 3.257 ms | 25.088 ms |
| 真实第一块，8.12% | 1.767 ms | 18.612 ms | 90.765 ms | 4.065 ms | 21.756 ms |

真实输入来自 `/home/naie/work/slai/user-liuweiheng/PCA/brain_1m.h5` 第一块；HVG mask 来自本项目新生成的 `artifacts/legacy-baseline-20261104T105820Z/gene_statistics.csv.gz`，按 gene ID/order 显式校验。counts 为 1,330,083 nnz、最长行 604 nnz；先计算全基因 totals，随后选 HVG。TileLang 输出相对 FP64 reference 最大绝对误差 **5.18e-7**，零/非零 support 精确一致。synthetic 最大绝对误差 3.42e-7。

真实块 TileLang 同步 wall 1.861 ms，torch native 18.728 ms。纯 device 比值约 10.5×；H2D-inclusive 比值约 5.35×。这些只比较这里明确实现的块内路径，**不是全流程加速比，也不是与所有可能优化的 PyTorch/CPU 实现比较**。CPU 路径输出在 host，NPU 路径输出留在 device，表内已注明边界。

synthetic 首次调用含 JIT 为 5457.6 ms；真实块在相同 shape、不同 nnz 下复用磁盘缓存，进程内首次加载调用 410.5 ms。后续正式时间排除它们，完整流程应计入首次准备成本。原始 JSON 在远端新项目 `artifacts/hvg-kernel-synthetic.json`、`artifacts/hvg-kernel-real.json`；真实 input 为 `artifacts/hvg-real.npz`。

TileLang 环境没有 h5py。可用 PCA 的已有 CPU 环境做数据准备，避免修改任何共享安装：

```bash
# 在新项目根目录，数据准备不初始化 NPU。
/home/naie/work/slai/user-liuweiheng/PCA/.venv_npu/bin/python \
  -m benchmarks.bench_hvg_kernel --prepare-only \
  --tenx /home/naie/work/slai/user-liuweiheng/PCA/brain_1m.h5 \
  --hvg-table artifacts/legacy-baseline-20261104T105820Z/gene_statistics.csv.gz \
  --rows 8192 --save-input-npz artifacts/hvg-real.npz
source /home/naie/work/slai/user-liuweiheng/PCA/env_npu.sh
/home/naie/work/slai/user-liuweiheng/envs/tilelang-ascend/bin/python \
  -m benchmarks.bench_hvg_kernel --input-npz artifacts/hvg-real.npz \
  --repeats 10 --output artifacts/hvg-kernel-real.json
```

完整流程还可用显式实验 launcher 复用现有依赖：

```bash
source /home/naie/work/slai/user-liuweiheng/PCA/env_npu.sh
/home/naie/work/slai/user-liuweiheng/PCA/.venv_npu/bin/python \
  -m benchmarks.tilelang_launcher --inspect
# 执行任意项目 module 时，把 --inspect 换成 module 名和原有参数。
```

该 launcher 通过 `site.addsitedir` 追加受信任的 TileLang site-packages，保留 PCA interpreter 的依赖优先级，不安装依赖、不修改旧环境文件或 shell 配置。已验证 NumPy 2.4.6、SciPy 1.17.1、h5py 3.16.0 来自 PCA venv；torch/torch_npu 来自原 `/opt/conda`；TileLang 来自指定 env。随后在这个组合下重跑了全部 10 项内核测试，最终记录为 **10 passed in 13.31 s**，并完成百万细胞完整流程。

首轮完整融合流程（已有 canonical CSR 缓存）全进程 121.938 s，host 变换的 NPU 流程 129.722 s；两者均含完整输出、数据哈希及数值检查。融合版本对 CPU FP64 参考：表达最大差 9.54e-7，旋转对齐后的 score 最大差 4.76e-4，抽样邻居重合率 100%，全部预设门限通过。记录在 `benchmarks/evidence/cpu-vs-new-fused.json`。后续相同缓存预热政策的补测见 [实现验证记录](implementation-results.md)。完整流程包含首次模块加载及本次实际发生的 JIT，但此前已经运行同形状 kernel，**不属于清空编译缓存后的首次安装时间**。

保持可选 backend：当前只验证一个真实数据集和 A3 环境，单次全量性能不等于跨数据集、跨硬件的稳定收益。不同稀疏度、910B 编译支持与端到端重复测量仍是默认启用前的条件。
