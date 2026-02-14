# 基于 AMD Radeon AI PRO R9700 的 GPU-EFIT 平衡重建设计方案

---

**编制日期**: 2026年2月13日  
**文档版本**: V1.0  
**关联文件**: BEST 10MW ECRH 控制保护系统总体设计方案 V2.1 / GPU 加速射线追踪技术分析 V1.1

---

> **项目定位**
>
> 本方案基于中国 EAST 装置已成功部署的 P-EFIT（GPU 并行平衡重建代码），使用 AMD Radeon AI PRO R9700 GPU 和 HIP 编程模型进行复现和优化。目标是为 BEST 托克马克的等离子体控制系统（PCS）提供高分辨率实时平衡重建能力，同时打通"EFIT 平衡重建 → 射线追踪 → Launcher 角度优化"的全链条 GPU 加速。

---

## 目录

1. [硬件平台对比：从 TITAN X 到 R9700](#1-硬件平台对比从-titan-x-到-r9700)
2. [P-EFIT 算法回顾与 HIP 移植策略](#2-p-efit-算法回顾与-hip-移植策略)
3. [RDNA 4 架构特定优化设计](#3-rdna-4-架构特定优化设计)
4. [各网格分辨率性能估算](#4-各网格分辨率性能估算)
5. [混合精度与 AI 加速器利用](#5-混合精度与-ai-加速器利用)
6. [与 BEST PCS 集成方案](#6-与-best-pcs-集成方案)
7. [实施路线与风险分析](#7-实施路线与风险分析)

---

## 1. 硬件平台对比：从 TITAN X 到 R9700

### 1.1 硬件规格对比

P-EFIT 原始开发平台为 2016 年的 NVIDIA Pascal TITAN X。AMD Radeon AI PRO R9700 是 AMD 于 2025 年发布的 RDNA 4 架构专业 GPU。两者对比如下：

| 参数 | NVIDIA TITAN X (Pascal, 2016) | AMD Radeon AI PRO R9700 (RDNA 4, 2025) | 倍率 |
|------|-----|-----|:---:|
| **计算单元** | 28 SM | 64 CU | **2.3×** |
| **流处理器** | 3,584 CUDA Cores | 4,096 Stream Processors | 1.14× |
| **FP32 算力** | 11 TFLOPS | **~48 TFLOPS** | **4.4×** |
| **FP16 矩阵算力** | 无专用单元 | **191 TFLOPS** | **∞** (新能力) |
| **FP64 算力** | ~0.34 TFLOPS (1:32) | ~0.76 TFLOPS (1:64) | 2.2× |
| **AI 加速器** | 无 | **128 个二代 AI 加速器** | **∞** (新能力) |
| **本地数据共享 (LDS/Shared)** | 96 KB/SM (与 L1 共享) | **128 KB/CU (专用)** | **1.33×** |
| **寄存器文件 (VGPR)** | 256 KB/SM | **768 KB/CU** | **3×** |
| **L2 缓存** | 3 MB | 8 MB | 2.7× |
| **L3 缓存 (Infinity Cache)** | 无 | **64 MB** | **∞** (新能力) |
| **显存容量** | 12 GB GDDR5X | **32 GB GDDR6** | **2.67×** |
| **显存带宽** | 480 GB/s | 640 GB/s | 1.33× |
| **PCIe** | Gen 3 ×16 (~16 GB/s) | **Gen 5 ×16 (~64 GB/s)** | **4×** |
| **Wave/Warp 大小** | 32 (Warp) | 32 (Wave32, 可选 Wave64) | 相同 |
| **制程** | 16nm | 4nm | 4 代 |
| **功耗** | 250W | 300W | 1.2× |
| **编程模型** | CUDA | **HIP / ROCm 7.1+** | — |

### 1.2 对 EFIT 关键计算的影响分析

```
EFIT 关键计算环节与硬件优势匹配
═══════════════════════════════════════════════════════════════

环节                    计算特征              R9700 优势来源
──────────────────────  ────────────────────  ──────────────────────
Green 函数边界计算      O(N³) 矩阵-向量乘    FP32 4.4× + Infinity Cache
(pflux_ 的主要部分)     带宽敏感              64MB L3 可缓存 Green 矩阵

G-S 求解器 Step 1&5    矩阵-矩阵乘          FP32 4.4× + 128KB LDS
(特征分解)              计算密集              FP16 矩阵可达 191 TFLOPS

G-S 求解器 Step 3      并行三对角求解        CU 2.3× + Wave32
(前缀和)                延迟敏感              768KB VGPR 提高占用率

G-S 求解器 Step 2&4    矩阵转置              640 GB/s + Infinity Cache
(矩阵转置)              纯带宽               合并访问优化

整体迭代                所有环节              PCIe Gen5 4× host 传输

═══════════════════════════════════════════════════════════════
```

### 1.3 R9700 相对于 Instinct 系列的定位

| 对比项 | Radeon AI PRO R9700 | Instinct MI250X | Instinct MI300A |
|--------|:---:|:---:|:---:|
| **定位** | 工作站/边缘 AI | 数据中心 HPC | 数据中心 HPC+AI |
| **FP32** | ~48 TFLOPS | 47.9 TFLOPS | 61.3 TFLOPS |
| **FP64** | ~0.76 TFLOPS | **47.9 TFLOPS** | **61.3 TFLOPS** |
| **显存** | 32 GB GDDR6 | 128 GB HBM2e | 128 GB 统一内存 |
| **显存带宽** | 640 GB/s | 3,276 GB/s | 5,300 GB/s |
| **价格** | ~$1,500 (估) | ~$15,000+ | ~$15,000+ |
| **适用场景** | **P-EFIT (FP32)** | 射线追踪 (FP64) | 两者兼顾 |
| **部署环境** | 控制室工作站 | 专用机房 | 专用机房 |

> **关键洞察**：P-EFIT 使用 **单精度 FP32**，这正好是 R9700 的强项。RDNA 4 的 FP64 极弱（1:64），不适合需要双精度的射线追踪，但对 FP32 的 EFIT 平衡重建完全够用。R9700 的性价比（FP32 算力/价格）远优于 Instinct 系列，且可部署在控制室工作站中，无需专用机房。

---

## 2. P-EFIT 算法回顾与 HIP 移植策略

### 2.1 P-EFIT 原始算法结构

P-EFIT（Huang et al., ASIPP）的核心是一个 GPU 并行的 Grad-Shafranov 方程求解器，嵌入在 EFIT 迭代重建框架中：

```
P-EFIT 完整迭代流程
═══════════════════════════════════════════════════════════════

外层: Picard 迭代（通常 10-30 次收敛）
│
├─► 1. 电流密度拟合 (current_)
│      读取诊断信号 → 用基函数展开 Jφ → 线性最小二乘拟合
│      计算量: O(N²) × 基函数数, 占比 <10%
│
├─► 2. 极向磁通计算 (pflux_) ←── 最耗时环节 (57-92%)
│      │
│      ├─ 2a. Green 函数边界条件
│      │    ψ_boundary = G_matrix × J_plasma_vector
│      │    计算量: O(N³), 带宽密集
│      │    N=65: G矩阵 ~4 MB    → 可放入 Infinity Cache ✓
│      │    N=129: G矩阵 ~33 MB  → 可放入 Infinity Cache ✓
│      │    N=257: G矩阵 ~266 MB → 超出 Infinity Cache ✗
│      │
│      └─ 2b. G-S 方程求解（P-EFIT 核心创新）
│           Step 1: 特征分解 Ψ' = Q^T × Ψ        (矩阵乘)
│           Step 2: 转置 Ψ'                       (内存操作)
│           Step 3: 求解 M 个独立三对角系统        (前缀和并行)
│           Step 4: 转置 X'                       (内存操作)
│           Step 5: 逆特征分解 X = Q × X'          (矩阵乘)
│
├─► 3. 磁面搜索 (steps_)
│      寻找 X 点、O 点、等离子体边界
│      计算量: O(N²), 占比 <5%
│
└─► 4. 收敛检查
       max|ψ_new - ψ_old| < ε (典型 ε = 1e-4 ~ 1e-5)

═══════════════════════════════════════════════════════════════
```

### 2.2 P-EFIT 原始性能基准（TITAN X, FP32）

| 环节 | 65×65 | 129×129 | 257×257 |
|------|-------|---------|---------|
| G-S 求解器（5步） | 0.016 ms | 0.027 ms | 超出 TITAN X 能力 |
| pflux_ 完整 | ~0.17 ms | ~0.31 ms | — |
| 单次迭代 (fit_) | ~0.3 ms | ~0.375 ms | — |
| 完整重建 (~10次迭代) | ~3 ms | ~4-7 ms | — |

### 2.3 CUDA → HIP 移植策略

HIP（Heterogeneous-Compute Interface for Portability）与 CUDA 语法高度相似。P-EFIT 的 CUDA 代码可通过以下步骤移植：

**第一步：机械转换（hipify 工具）**

| CUDA API | HIP 等价 | 说明 |
|----------|---------|------|
| `cudaMalloc` | `hipMalloc` | 设备内存分配 |
| `cudaMemcpy` | `hipMemcpy` | 主机-设备数据传输 |
| `__shared__` | `__shared__` | **完全相同** |
| `__syncthreads()` | `__syncthreads()` | **完全相同** |
| `<<<grid, block>>>` | `<<<grid, block>>>` | **完全相同** |
| `cudaStream_t` | `hipStream_t` | 异步流 |

> P-EFIT 的 CUDA kernel 代码中，约 **90% 的语法可直接使用 AMD 的 `hipify-perl` 工具自动转换**，只需替换 API 前缀。

**第二步：架构适配（需要手动调优）**

这是关键，不能简单平移。RDNA 4 与 Pascal 的架构差异需要针对性优化：

```
需要手动调优的部分
═══════════════════════════════════════════════════════════════

1. Block/Grid 尺寸重新设计
   TITAN X: 28 SM, warp=32 → 典型 block 256 线程
   R9700:   64 CU, wave=32 → 可以使用更多更小的 block
   原因: CU 数量翻倍，每个 CU 的 LDS 更大，线程分配策略不同

2. 共享内存分块尺寸 (Tile Size)
   TITAN X: 96 KB shared (与 L1 共享) → tile 通常受限
   R9700:   128 KB LDS (专用) → 可以用更大的 tile
   影响: Steps 1 & 5 的矩阵乘法分块

3. 寄存器压力管理
   TITAN X: 256 KB VGPR/SM
   R9700:   768 KB VGPR/CU → 3× 更多寄存器
   机会: 可以在寄存器中保留更多中间变量，减少 LDS 溢出

4. 内存合并访问模式
   Wave32 的合并规则与 Warp32 基本相同
   但 Infinity Cache 的存在改变了非合并访问的惩罚程度
   可以放宽某些合并优化，换取更简洁的代码

═══════════════════════════════════════════════════════════════
```

### 2.4 HIP Kernel 核心设计

#### 2.4.1 G-S 求解器 Step 1 & 5：特征分解矩阵乘法

```cpp
// HIP Kernel: 分块矩阵乘法（特征分解/逆特征分解）
// 针对 RDNA 4 的 128KB LDS 优化

#define TILE_SIZE 32  // RDNA 4: 可从 TITAN X 的 16 提升到 32

__global__ void eigen_decomp_kernel(
    const float* __restrict__ Q,      // 特征矩阵 [M×M], 预计算
    const float* __restrict__ Psi,    // 输入磁通  [M×M]
    float* __restrict__ Psi_prime,    // 输出 Q^T × Psi [M×M]
    int M)                            // 网格维度 (63, 127, 255, 511)
{
    // 每个 Block 计算结果矩阵的一个 TILE_SIZE × TILE_SIZE 子块
    __shared__ float tile_Q[TILE_SIZE][TILE_SIZE + 1];   // +1 避免 bank conflict
    __shared__ float tile_Psi[TILE_SIZE][TILE_SIZE + 1];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    // 沿 K 维度分块累加
    for (int t = 0; t < (M + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 协作加载 Q^T 和 Psi 的子块到 LDS
        int k_Q = t * TILE_SIZE + threadIdx.x;
        int k_Psi = t * TILE_SIZE + threadIdx.y;

        tile_Q[threadIdx.y][threadIdx.x] =
            (row < M && k_Q < M) ? Q[k_Q * M + row] : 0.0f;  // Q^T
        tile_Psi[threadIdx.y][threadIdx.x] =
            (k_Psi < M && col < M) ? Psi[k_Psi * M + col] : 0.0f;

        __syncthreads();

        // 在 LDS 中完成子块乘法
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tile_Q[threadIdx.y][k] * tile_Psi[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < M) {
        Psi_prime[row * M + col] = sum;
    }
}

// 启动配置 (以 129×129 为例, M=127)
// grid: (4, 4) blocks  |  block: (32, 32) = 1024 threads
// LDS 使用: 2 × 32 × 33 × 4 = 8,448 bytes << 128 KB
// 每个 CU 可同时调度多个 block → 高占用率
```

#### 2.4.2 G-S 求解器 Step 3：并行三对角求解（前缀和）

```cpp
// HIP Kernel: 并行前缀和三对角求解器
// M 个独立三对角系统 × M 个方程 → 粗+细双层并行

__global__ void tridiag_solve_kernel(
    const float* __restrict__ a_coeff,  // 预计算的消元系数
    const float* __restrict__ m_coeff,  // 预计算的乘法因子
    const float* __restrict__ rhs,      // 右端项 [M×M]
    float* __restrict__ x,              // 解向量 [M×M]
    int M)
{
    // 粗粒度: blockIdx.x 对应第 j 个独立三对角系统
    int j = blockIdx.x;
    if (j >= M) return;

    // 细粒度: block 内线程用前缀和并行消元
    extern __shared__ float sdata[];
    float* s_d = sdata;           // 修改后的右端项
    float* s_x = sdata + M;      // 解

    int tid = threadIdx.x;

    // 加载右端项到 LDS
    if (tid < M) {
        s_d[tid] = rhs[tid * M + j];  // 注意转置后的访问模式
    }
    __syncthreads();

    // 前缀和消元: O(M) 串行 → O(log₂M) 并行
    // 上扫阶段 (up-sweep / reduce)
    for (int stride = 1; stride < M; stride <<= 1) {
        int idx = tid;
        if (idx >= stride && idx < M) {
            s_d[idx] += m_coeff[idx * M + (idx - stride)] * s_d[idx - stride];
        }
        __syncthreads();
    }

    // 下扫阶段 (down-sweep / distribute)
    // 最后一个元素已经是正确的
    if (tid == M - 1) {
        s_x[tid] = s_d[tid] * a_coeff[tid];
    }
    __syncthreads();

    for (int stride = M >> 1; stride >= 1; stride >>= 1) {
        int idx = tid;
        if (idx < M - stride) {
            s_x[idx] = a_coeff[idx] * (s_d[idx] - s_x[idx + stride]);
        }
        __syncthreads();
    }

    // 写回全局内存
    if (tid < M) {
        x[tid * M + j] = s_x[tid];
    }
}

// 启动配置 (以 129×129 为例, M=127)
// grid: (127) blocks — 每个 block 解一个三对角系统
// block: (128) threads (≥ M, 向上取 wave32 的倍数)
// LDS: 2 × 128 × 4 = 1,024 bytes
// 64 CU 可同时运行全部 127 个 block → 一轮完成!
// (TITAN X 28 SM 需要 ~5 轮)
```

#### 2.4.3 Green 函数边界条件计算

```cpp
// HIP Kernel: Green 函数矩阵-向量乘法 (边界磁通)
// ψ_boundary[i] = Σ_j G[i][j] × J_plasma[j]
// i = 边界点 (~4M), j = 内部网格点 (M²)
// 这是 O(N³) 操作，是 EFIT 最耗时的环节之一

__global__ void green_boundary_kernel(
    const float* __restrict__ G_matrix,   // Green 函数矩阵 [N_bnd × N_inner]
    const float* __restrict__ J_plasma,   // 等离子体电流密度向量
    float* __restrict__ psi_boundary,     // 输出: 边界磁通
    int N_bnd,                            // 边界点数
    int N_inner)                          // 内部点数
{
    // 每个线程计算一个边界点的磁通
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_bnd) return;

    float sum = 0.0f;
    const float* G_row = G_matrix + i * N_inner;

    // 内循环: 累加所有内部点的贡献
    // R9700 的 Infinity Cache 会缓存 J_plasma 向量（被所有线程共享）
    // 以及 G_matrix 的热点行（被相邻线程访问）
    for (int j = 0; j < N_inner; j += 4) {
        // 手动展开，利用指令级并行
        sum += G_row[j]     * J_plasma[j];
        sum += G_row[j + 1] * J_plasma[j + 1];
        sum += G_row[j + 2] * J_plasma[j + 2];
        sum += G_row[j + 3] * J_plasma[j + 3];
    }

    psi_boundary[i] = sum;
}

// 启动配置 (以 129×129 为例)
// N_bnd = 512, N_inner = 127² = 16129
// grid: (16), block: (32) — 512 个线程
// G_matrix 大小: 512 × 16129 × 4 = 33 MB → 恰好放入 64 MB Infinity Cache!
// J_plasma 大小: 16129 × 4 = 63 KB → 完全在 Infinity Cache 中
// TITAN X: 无 L3, 33 MB G_matrix 每次都走 GDDR → 带宽瓶颈
// R9700:   Infinity Cache 命中后 → 有效带宽 ~2-4 TB/s → 巨大提升
```

---

## 3. RDNA 4 架构特定优化设计

### 3.1 Infinity Cache：改变 EFIT 性能格局的关键

```
Infinity Cache 对 EFIT 各网格分辨率的影响
═══════════════════════════════════════════════════════════════

网格       Green 函数矩阵    能否放入 64MB       对 pflux_ 的影响
分辨率     (边界×内部)       Infinity Cache?
──────    ─────────────     ──────────────      ────────────────
65×65     ~4 MB              ✓ 完全缓存         有效带宽 ↑ 5-10×
                                                pflux_ 加速 3-5×

129×129   ~33 MB             ✓ 大部分缓存       有效带宽 ↑ 3-5×
                                                pflux_ 加速 2-4×

257×257   ~266 MB            ✗ 超出缓存         回退到 GDDR
                             但 J_vector (260KB)  仍有缓存收益 1.3×
                             仍可缓存

513×513   ~4.2 GB            ✗ 远超缓存         主要依赖 GDDR 带宽
                                                640 vs 480 GB/s = 1.33×

═══════════════════════════════════════════════════════════════

★ 对 ≤129×129 网格, Infinity Cache 是 R9700 相对 TITAN X 的决定性优势
  Green 函数矩阵在 TITAN X 上每次访问都走 GDDR5X (480 GB/s)
  在 R9700 上走 Infinity Cache (有效 2-4 TB/s)
  → 这比 FP32 算力的提升（4.4×）更重要

★ 对 257×257 网格, Infinity Cache 仍有帮助但不如小网格显著
  需要结合其他优化手段（分块、流水线）
```

### 3.2 LDS 128 KB 专用 —— 更大的矩阵分块

```
TITAN X vs R9700 矩阵乘法分块对比
═══════════════════════════════════════════════════════════════

TITAN X (96 KB Shared, 与 L1 共享):
  实际可用 ~48 KB (另一半给 L1)
  tile_A[16][17] + tile_B[16][17] = 2 × 16 × 17 × 4 = 2,176 bytes
  → Tile Size = 16
  → 每个 tile 乘法: 16³ = 4,096 FMA
  → K 维度需要 M/16 轮

R9700 (128 KB LDS, 专用, L1 独立):
  全部 128 KB 可用于 LDS
  tile_A[32][33] + tile_B[32][33] = 2 × 32 × 33 × 4 = 8,448 bytes
  → Tile Size = 32
  → 每个 tile 乘法: 32³ = 32,768 FMA (8× more work per tile)
  → K 维度需要 M/32 轮 (一半的轮数)
  → 同步开销减半，计算密度翻倍

═══════════════════════════════════════════════════════════════
```

### 3.3 768 KB VGPR —— 占用率与每线程工作量

RDNA 4 每个 CU 有 768 KB 向量通用寄存器（是 Pascal 的 3 倍），这允许：

| 策略 | 说明 | 对 EFIT 的影响 |
|------|------|--------------|
| **更高占用率** | 每个 CU 可同时运行更多 wavefront | 三对角求解器：127 个 block 在 64 CU 上全并行 |
| **寄存器密集 kernel** | 每线程可用更多寄存器，避免溢出到 LDS | G-S 求解中间变量全部驻留寄存器 |
| **循环展开** | 编译器可以更激进地展开循环 | Green 函数内循环 4× 展开无寄存器压力 |

### 3.4 Wave32 + CU 数量优势

```
三对角求解的并行调度对比
═══════════════════════════════════════════════════════════════

129×129 网格: M=127, 需要 127 个独立三对角系统同时求解

TITAN X (28 SM):
  127 blocks / 28 SM = 4.5 轮
  每轮需要等待最慢的 SM 完成 → 不均衡
  实际约 5 轮 × 单 block 延迟

R9700 (64 CU):
  127 blocks / 64 CU = 1.98 轮
  几乎 2 轮即可完成
  加速比: ~2.5× (仅从调度层面)

257×257 网格: M=255, 需要 255 个独立三对角系统
TITAN X: 255 / 28 = 9.1 轮 → 远超能力
R9700:   255 / 64 = 3.98 轮 → 完全可行!

513×513 网格: M=511
R9700: 511 / 64 = 8.0 轮 → 仍然合理

═══════════════════════════════════════════════════════════════
```

---

## 4. 各网格分辨率性能估算

### 4.1 估算方法

性能估算基于三个因素的叠加：
1. **计算加速**：FP32 算力比 × 算法中计算密集部分的占比
2. **带宽加速**：有效带宽比（含 Infinity Cache 效应）× 带宽密集部分占比
3. **调度加速**：CU 数量比 × 并行调度效率

### 4.2 G-S 求解器性能估算

| 网格 | TITAN X 实测 | R9700 估算 | 加速比 | 主要加速来源 |
|------|------------|-----------|:------:|------------|
| 65×65 | 0.016 ms | **~0.006 ms** | **~2.7×** | LDS 分块 + CU 调度 |
| 129×129 | 0.027 ms | **~0.008 ms** | **~3.4×** | LDS 分块 + CU 调度 + FP32 |
| 257×257 | 不可用 | **~0.04 ms** | **从不可能到可行** | 64 CU 支撑更大网格 |
| 513×513 | 不可用 | **~0.25 ms** | **从不可能到可行** | 4096 SP 全并行 |

> 注：65×65 的加速受限于 kernel 启动开销（~3-5 µs），已接近硬件延迟下限。

### 4.3 pflux_ 完整环节性能估算

pflux_ 包含 Green 函数边界计算 + G-S 求解。Green 函数部分受 Infinity Cache 影响最大：

| 网格 | TITAN X pflux_ | R9700 pflux_ 估算 | 加速比 | 关键因素 |
|------|--------------|-----------------|:------:|---------|
| 65×65 | ~0.17 ms | **~0.04 ms** | **~4.3×** | Green 矩阵 4MB ⊂ Infinity Cache |
| 129×129 | ~0.31 ms | **~0.08 ms** | **~3.9×** | Green 矩阵 33MB ⊂ Infinity Cache |
| 257×257 | 不可用 | **~0.8 ms** | **新能力** | Green 矩阵超出缓存，靠 FP32+带宽 |
| 513×513 | 不可用 | **~5 ms** | **新能力** | 完全带宽受限 |

### 4.4 完整平衡重建性能估算

| 网格 | TITAN X 单次迭代 | R9700 单次迭代 | 收敛次数 | **R9700 完整重建** | 可否实时？ |
|------|-------|-------|:---:|--------|:---:|
| 65×65 | 0.3 ms | **~0.08 ms** | ~10 | **~0.8 ms** | ✓ **亚毫秒** |
| 129×129 | 0.375 ms | **~0.12 ms** | ~10-15 | **~1.2-1.8 ms** | ✓ **< 2 ms** |
| 257×257 | 不可用 | **~1.2 ms** | ~15-20 | **~18-24 ms** | △ 近实时 |
| 513×513 | 不可用 | **~7 ms** | ~20-30 | **~140-210 ms** | ✗ 离线/炮间分析 |

### 4.5 性能对比总图

```
P-EFIT 完整平衡重建时间对比 (对数刻度)
═══════════════════════════════════════════════════════════════

延迟 (ms)   CPU 单核      TITAN X (CUDA)    R9700 (HIP)
            (基准)        (P-EFIT 原始)      (本方案)
──────────────────────────────────────────────────────────────

1000  │ ■■■■■■■■■■■■    ─────────────     ─────────────
      │ 513×513:1150ms
 100  │ ■■■■■■■■         ─────────────     ■■ 513×513:~175ms
      │ 257×257:170ms
  10  │ ■■■ 129:24ms     ─────────────     ■ 257×257:~21ms
      │
   1  │ ■ 65:4ms         ■■ 129:~5.5ms     ■ 129×129:~1.5ms
      │                  ■ 65:~3ms          65×65: ~0.8ms ★
  0.1 │                                    
      └──────────────────────────────────────────────────────

实时控制线 (1 ms): ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─

★ R9700 首次实现 65×65 亚毫秒完整平衡重建
★ R9700 使 129×129 进入实时控制范围 (<2 ms)
★ R9700 使 257×257 成为可能 (近实时 ~21 ms)
★ R9700 使 513×513 从 CPU 的 19 分钟（多时间片）降到秒级

═══════════════════════════════════════════════════════════════
```

---

## 5. 混合精度与 AI 加速器利用

### 5.1 R9700 的 AI 硬件能力

R9700 配备 **128 个二代 AI 加速器**，FP16 矩阵性能达 **191 TFLOPS**（是 FP32 的 4 倍）。这为 EFIT 提供了额外加速路径。

### 5.2 混合精度 EFIT 策略

```
混合精度 P-EFIT 设计
═══════════════════════════════════════════════════════════════

关键原则:
• 迭代过程中间计算可以容忍 FP16 精度（收敛过程会纠正误差）
• 最终收敛判断和结果输出使用 FP32
• 三对角求解必须 FP32（前缀和对精度敏感）

                    精度      算力           用于 EFIT 的哪些环节
─────────────────  ──────    ─────────      ───────────────────
FP16 Matrix        半精度    191 TFLOPS     ★ Step 1&5: 特征分解矩阵乘
                                            ★ Green 函数矩阵-向量乘
                                            （前 80% 迭代）

FP32               单精度    ~48 TFLOPS     • Step 3: 三对角求解（全程）
                                            • Step 1&5: 最后 20% 迭代
                                            • 最终收敛检查

混合精度加速估算:
  Step 1&5 (矩阵乘): FP16 → 4× 加速
  Green 函数 (矩阵-向量乘): FP16 → 4× 加速
  Step 3 (三对角): 不变 (FP32)

  pflux_ 中 Step 1&5 + Green 函数占比 ~70-80%
  → 混合精度总加速: ~2.5-3×（在纯 FP32 基础上再叠加）

═══════════════════════════════════════════════════════════════
```

### 5.3 混合精度后的修正性能估算

| 网格 | R9700 纯 FP32 | R9700 混合精度 | 额外加速 |
|------|-------------|-------------|:------:|
| 65×65 完整重建 | ~0.8 ms | **~0.35 ms** | ~2.3× |
| 129×129 完整重建 | ~1.5 ms | **~0.6 ms** | ~2.5× |
| 257×257 完整重建 | ~21 ms | **~9 ms** | ~2.3× |
| 513×513 完整重建 | ~175 ms | **~75 ms** | ~2.3× |

> **257×257 混合精度 ~9 ms** 已经接近 PCS 实时控制的可用范围。这意味着 BEST 可以在 9ms 内获得比 EAST 当前方案精度高 4 倍的平衡重建结果。

### 5.4 AI 加速器的进阶应用

128 个 AI 加速器（1,531 TOPS INT4）不仅用于混合精度，还可服务于以下场景：

| 应用 | 技术路线 | 效果 |
|------|---------|------|
| **ML 剖面重建** | 类似 DIII-D 的 RTCAKENN，用神经网络从诊断信号推断 ne/Te 剖面 | 替代传统剖面拟合，<1 ms |
| **EFIT 初始化预测** | 训练 NN 从上一时间片预测当前平衡 → 作为 Picard 迭代初值 | 迭代次数从 10-15 降至 3-5 |
| **平衡有效性检查** | 轻量 CNN 检测重建结果是否物理合理 | 实时质量控制 |

```
AI 辅助减少迭代次数的原理
═══════════════════════════════════════════════════════════════

传统 EFIT:
  初始猜测(较远) → 10-15 次迭代收敛 → 结果

AI 辅助 EFIT:
  NN 预测(很近) → 3-5 次迭代收敛 → 结果
  ↑
  用历史放电数据训练: 输入 = 诊断信号, 输出 = 平衡磁通

  NN 推理时间: ~0.1 ms (R9700 AI 加速器, INT8 量化)
  减少的迭代: ~7-10 次 × 0.12 ms/次 ≈ 节省 0.8-1.2 ms

  129×129 总时间: 0.1 ms (NN) + 5 × 0.12 ms (迭代) = ~0.7 ms ★

═══════════════════════════════════════════════════════════════
```

---

## 6. 与 BEST PCS 集成方案

### 6.1 系统架构

```
R9700 GPU-EFIT 在 BEST 控制系统中的位置
═══════════════════════════════════════════════════════════════

   诊断系统                PCS 服务器 (R9700 GPU-EFIT)        ECRH 系统
 ┌──────────┐      ┌─────────────────────────────────┐    ┌───────────┐
 │ Mirnov   │      │  CPU (AMD Ryzen / EPYC)          │    │ ECRH-MC   │
 │ ECE      │─────→│    ↓ 诊断数据预处理               │    │           │
 │ Thomson  │ RFM  │    ↓ (可选) NN 初始化预测         │    │ 接收:     │
 │ MSE      │      │  ┌─────────────────────────┐     │    │ • ρ_target│
 │ 磁探针    │      │  │ AMD Radeon AI PRO R9700  │     │    │ • ne,Te,B │
 └──────────┘      │  │                         │     │    │           │
                   │  │ GPU-EFIT (HIP kernels)  │     │───→│ GPU 射线  │
                   │  │ • Green 函数边界计算     │     │RFM │ 追踪      │
                   │  │ • G-S 求解 (5步)        │     │    │ (可选:    │
                   │  │ • 磁面搜索/q 剖面       │     │    │ 同一 GPU  │
                   │  │ • 输出: ψ, ρ, q(ρ)     │     │    │ 或独立GPU)│
                   │  └─────────────────────────┘     │    └───────────┘
                   │                                   │
                   │  输出 → ρ_target, ne(ρ), Te(ρ), B(R,Z) │
                   └─────────────────────────────────┘

   数据流延迟:
   诊断→PCS: ~0.2 ms (RFM)
   GPU-EFIT:  ~0.6-1.5 ms (129×129, 混合精度+AI初始化)
   PCS→ECRH:  ~0.1 ms (RFM)
   ──────────────────────────
   总计:       ~0.9-1.8 ms (PCS 环节)

═══════════════════════════════════════════════════════════════
```

### 6.2 GPU-EFIT 与 GPU 射线追踪的协同

R9700 的 32GB 显存足以同时运行 GPU-EFIT 和简化版射线追踪。两个计算可以流水线化：

```
同一 GPU 上的流水线执行
═══════════════════════════════════════════════════════════════

时间 →  0        0.6ms      1.0ms      1.5ms
        │         │          │          │
Stream 1 (EFIT):
        [  GPU-EFIT 129×129  ]
        │← 混合精度 + AI 初始化 →│

Stream 2 (Ray Tracing):
                  [  射线追踪（用 EFIT 中间结果）  ]
                  │← FP32, 12束×100角度 →│

合并输出:                              [最优角度 → ECRH]

总延迟: ~1.5 ms (两个计算部分重叠)

说明:
• 射线追踪不需要等 EFIT 完全收敛
  → 用第 5 次迭代的中间结果即可开始（精度已足够）
• 两个 kernel 使用不同的 HIP Stream, GPU 硬件自动调度
• 32 GB 显存: EFIT 占 ~200 MB + 射线追踪占 ~100 MB = 远未用满

═══════════════════════════════════════════════════════════════
```

> **重要发现**：如果在 R9700 上同时运行 GPU-EFIT 和 GPU 射线追踪，NTM 控制全链条的 PCS+ECRH 计算部分可以从 ~25 ms 压缩到 **~1.5-2 ms**。这是之前分析中"只优化 ECRH 内部只能达到 ~10ms"的解决方案——**不是分别优化 PCS 和 ECRH，而是在同一个 GPU 上流水线化整个计算链**。

### 6.3 显存使用估算

| 数据 | 65×65 | 129×129 | 257×257 |
|------|-------|---------|---------|
| Green 函数矩阵 | ~4 MB | ~33 MB | ~266 MB |
| 特征矩阵 Q, Q^T | ~0.1 MB | ~0.5 MB | ~2 MB |
| 三对角系数 | ~0.05 MB | ~0.2 MB | ~1 MB |
| 工作缓冲区 | ~0.5 MB | ~2 MB | ~8 MB |
| 射线追踪数据 | ~50 MB | ~100 MB | ~100 MB |
| **总计** | **~55 MB** | **~136 MB** | **~377 MB** |
| **R9700 剩余** | **31.9 GB** | **31.8 GB** | **31.6 GB** |

显存完全不是瓶颈。剩余空间可用于 AI 模型权重、历史数据缓存等。

---

## 7. 实施路线与风险分析

### 7.1 开发路线

| 阶段 | 时间 | 工作内容 | 交付物 |
|------|------|---------|--------|
| **Phase 0: 环境搭建** | 1 个月 | ROCm 7.1 + HIP 开发环境; R9700 硬件到位; 获取 P-EFIT 源码或基于论文重新实现 | 开发环境就绪 |
| **Phase 1: 核心移植** | 3 个月 | G-S 求解器 5 步算法的 HIP kernel 实现; 使用 hipify 转换 + 手动 RDNA 4 调优; 65×65 和 129×129 验证 | GPU-EFIT 核心库 (libGPU-EFIT) |
| **Phase 2: 完整 EFIT** | 3 个月 | 补全 current_, steps_ 等模块; 实现完整 Picard 迭代链; 257×257 网格支持; 与 EAST 离线数据对比验证 | 完整 GPU-EFIT 可执行程序 |
| **Phase 3: 混合精度** | 2 个月 | FP16 矩阵乘 kernel; 混合精度策略实现; 收敛性和精度验证 | 混合精度 GPU-EFIT |
| **Phase 4: AI 辅助** | 3 个月 | 训练 NN 初始化模型 (用 EAST/模拟数据); 集成到 GPU-EFIT 迭代流程; 推理优化 (INT8 量化) | AI-GPU-EFIT |
| **Phase 5: PCS 集成** | 3 个月 | 与 BEST PCS 数据接口; RFM 数据传输; 与 GPU 射线追踪流水线; 端到端延迟测试 | BEST 实时 GPU-EFIT |

**总计：约 15 个月**

### 7.2 风险分析

| 风险 | 等级 | 应对措施 |
|------|:----:|---------|
| **P-EFIT 源码不可获取** | 高 | 基于三篇公开论文（2016/2017/2018/2020）重新实现。算法细节已完全公开，5 步 G-S 求解器有伪代码 |
| **RDNA 4 的 ROCm 支持成熟度** | 中 | R9700 已确认支持 ROCm 7.1+, gfx1201。但可能遇到驱动 bug → 与 AMD 专业支持合作 |
| **FP64 不足** | 低 | P-EFIT 原始论文明确使用 FP32 单精度。EFIT 实时控制不需要 FP64 |
| **混合精度收敛性** | 中 | 先验证纯 FP32 正确性后再引入 FP16。保留 FP32 回退路径 |
| **257×257 网格实时性不足** | 中 | 混合精度 + AI 初始化 → 目标 <10 ms。若不够，降级到 129×129 仍优于现有方案 |
| **R9700 非数据中心级产品** | 低 | 工作站级可靠性对 PCS 应用足够。配备 ECC 内存保护。关键安全功能不依赖 GPU |

### 7.3 成本对比

| 方案 | 硬件成本 | 计算能力 (FP32) | EFIT 129×129 预估 | 部署位置 |
|------|---------|----------------|------------------|---------|
| **R9700 × 1** | **~$1,500** | 48 TFLOPS | **~0.6-1.5 ms** | 控制室工作站 |
| Instinct MI250X × 1 | ~$15,000 | 47.9 TFLOPS | ~0.6-1.5 ms | 专用机房 |
| Instinct MI300A × 1 | ~$15,000+ | 61.3 TFLOPS | ~0.5-1.2 ms | 专用机房 |
| NVIDIA A100 × 1 | ~$10,000 | 19.5 TFLOPS | ~2-4 ms | 专用机房 |
| NVIDIA H100 × 1 | ~$25,000+ | 51 TFLOPS | ~0.5-1.2 ms | 专用机房 |

> **R9700 的 FP32 算力与 MI250X 持平，但价格仅为其 1/10。** 对于 FP32 为主的 EFIT 平衡重建，R9700 是性价比最优的选择。其工作站级的形态也使其可以直接部署在 BEST 控制室内，无需专用计算基础设施。

---

## 8. 核心结论

```
本方案的关键数字
═══════════════════════════════════════════════════════════════

P-EFIT 在 R9700 上的预期性能（vs 原始 TITAN X 实现）:

  65×65 完整重建:   3 ms → ~0.35 ms    (8.6×, 混合精度)
  129×129 完整重建: 5.5 ms → ~0.6 ms   (9.2×, 混合精度)
  257×257 完整重建: 不可能 → ~9 ms      (新能力)
  513×513 完整重建: 不可能 → ~75 ms     (新能力)

  vs CPU 单核基准:
  129×129: 24 ms → 0.6 ms              (40×)
  257×257: 170 ms → 9 ms               (19×)

NTM 控制全链条影响（GPU-EFIT + GPU 射线追踪在同一 R9700 上）:
  当前业界最佳:  ~25 ms (CPU EFIT 5-10ms + CPU 射线追踪 15-20ms)
  R9700 方案:    ~1.5-2 ms (GPU-EFIT + GPU 射线追踪 流水线)
  加速比:        ~13-17×
  成本:          ~$1,500 (一块 R9700 GPU 卡)

═══════════════════════════════════════════════════════════════
```

---

## 参考文献

1. Huang Y., Xiao B.J., Luo Z.P., "Implementation of GPU parallel equilibrium reconstruction for plasma control in EAST," *Fusion Engineering and Design*, Vol. 112, 2016. — P-EFIT 首次发表。
2. Huang Y., Xiao B.J., Luo Z.P., "Fast parallel Grad–Shafranov solver for real-time equilibrium reconstruction in EAST tokamak using graphic processing unit," *Chinese Physics B*, 26(8), 2017. — G-S 求解器五步算法详细设计。
3. Huang Y., Xiao B.J., Luo Z.P., "Improvement of GPU parallel real-time equilibrium reconstruction for plasma control," *Fusion Engineering and Design*, Vol. 128, 2018. — P-EFIT 改进版。
4. Huang Y., Luo Z.P., Xiao B.J., Lao L.L., et al., "GPU-optimized fast plasma equilibrium reconstruction in fine grids for real-time control and data analysis," *Nuclear Fusion*, 60, 2020. — 高分辨率扩展。
5. Antepara O., Williams S., Kruger S., et al., "Performance-Portable GPU Acceleration of the EFIT Tokamak Plasma Equilibrium Reconstruction Code," SC '23 Workshops, 2023. — OpenMP 跨平台 GPU-EFIT，AMD MI250X 验证。
6. AMD, "Radeon AI PRO R9700 Quick Reference Guide," 2025. — R9700 硬件规格。
7. AMD, "RDNA 4 Instruction Set Architecture," 2025. — RDNA 4 架构文档。
8. ROCm Documentation, "GPU Hardware Specifications - gfx1201," 2025. — R9700 ROCm 技术参数。

---

*本文档设计了基于 AMD Radeon AI PRO R9700 GPU 的 P-EFIT 平衡重建移植和优化方案。通过 HIP 内核代码实现，结合 RDNA 4 架构的 Infinity Cache、128KB 专用 LDS 和 191 TFLOPS FP16 矩阵算力，预计可实现 129×129 网格亚毫秒级完整平衡重建——比原始 P-EFIT (TITAN X) 快 9 倍，比 CPU 快 40 倍。更重要的是，通过在同一块 R9700 上流水线化 GPU-EFIT 和 GPU 射线追踪，可以将 NTM 控制全链条计算延迟从 ~25ms 压缩到 ~1.5-2ms，以 ~$1,500 的硬件成本实现数据中心级 GPU 的同等效果。*
