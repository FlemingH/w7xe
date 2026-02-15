# GPU 加速平衡重建与射线追踪：研究综述

---

**编制日期**: 2026年2月15日  
**文档版本**: V1.0  
**关联文件**: AMD R9700 实时等离子体计算平台设计 V1.0

---

## 1. P-EFIT 公开论文成果

P-EFIT（Parallel EFIT）由中科院等离子体物理研究所（ASIPP）的黄曜、肖炳甲、罗正平等人开发，是全球首个在托克马克装置上实际部署的 **GPU 并行平衡重建代码**。其成果通过四篇核心论文公开发表：

### 1.1 论文一：首次实现（2016）

> Huang Y., Xiao B.J., Luo Z.P., et al., "Implementation of GPU parallel equilibrium reconstruction for plasma control in EAST," *Fusion Engineering and Design*, Vol. 112, pp. 40-48, 2016.

**核心贡献**：

| 内容 | 详情 |
|------|------|
| 目标 | 用 CUDA/GPU 替代 EAST 的 RT-EFIT (CPU) 实时平衡重建 |
| 硬件 | NVIDIA GeForce TITAN X (Maxwell 架构, 2015) |
| 网格 | 65×65 |
| 单次迭代 | **~300 μs** (含数据传输) |
| 完整重建 | ~3 ms (约 10 次迭代) |
| 算法 | Grad-Shafranov 方程求解 + Green 函数边界条件 |
| 验证 | 2014 EAST 放电, ISOFLUX 等离子体形状反馈控制 |
| 数据接口 | RFM (Reflective Memory) 连接 PCS |

**关键创新**：
- 首次将完整 EFIT 算法（不是简化版）以 GPU 并行形式实现
- 提出了 **五步 G-S 求解器**：特征分解将块三对角系统解耦为 M 个独立三对角系统
- 用前缀和（prefix sum）并行算法替代传统串行 Thomas 算法求解三对角系统
- 实际部署到 EAST PCS，不是纯学术仿真

### 1.2 论文二：算法详解（2017）

> Huang Y., Xiao B.J., Luo Z.P., "Fast parallel Grad–Shafranov solver for real-time equilibrium reconstruction in EAST tokamak using graphic processing unit," *Chinese Physics B*, 26(8), 085204, 2017.

**核心贡献**：

详细公开了 P-EFIT 的 **五步 G-S 求解器算法**：

```
Step 1: Ψ' = Q^T × Ψ       特征分解（矩阵乘法）
Step 2: 转置 Ψ'             内存操作
Step 3: 求解 M 个三对角系统   前缀和并行
Step 4: 转置 X'             内存操作
Step 5: X = Q × X'          逆特征分解（矩阵乘法）
```

- 公开了 CUDA kernel 的设计思路（分块矩阵乘、共享内存利用、前缀和实现）
- 分析了 GPU 各环节的占时比：pflux_（G-S 求解 + Green 函数）占总时间的 **57-92%**
- 给出了与 CPU 版本的逐环节对比

### 1.3 论文三：改进版（2018）

> Huang Y., Xiao B.J., Luo Z.P., et al., "Improvement of GPU parallel real-time equilibrium reconstruction for plasma control," *Fusion Engineering and Design*, Vol. 128, pp. 6-14, 2018.

**核心贡献**：

| 改进 | 详情 |
|------|------|
| 网格扩展 | 65×65 → **129×129** |
| 129×129 单次迭代 | **~375 μs** (含数据传输) |
| 内部电流剖面 | 新增定制化内部电流剖面重建模块 |
| 信号扩展 | 支持更多诊断信号输入 (磁探针 + MSE) |
| 数据传输优化 | PCS↔GPU 数据通路优化，降低通信延迟 |

### 1.4 论文四：高分辨率扩展（2020）

> Huang Y., Luo Z.P., Xiao B.J., Lao L.L., et al., "GPU-optimized fast plasma equilibrium reconstruction in fine grids for real-time control and data analysis," *Nuclear Fusion*, 60, 076021, 2020.

**核心贡献**：

| 内容 | 详情 |
|------|------|
| 硬件 | 升级到 NVIDIA TITAN X (Pascal 架构, 2016) |
| 新网格 | 支持 **257×257**（用于炮间精细分析） |
| 双模式运行 | 实时模式 (65/129) + 炮间分析模式 (257) |
| 与 EFIT 离线版对比 | 129×129 结果与标准 EFIT 离线重建一致 |
| 应用 | 准雪花偏滤器位形控制 |

**已公开的性能基准（TITAN X, FP32）**：

| 网格 | G-S 求解器 | pflux_ 完整 | 单次迭代 (fit_) | 完整重建 (~10次) |
|------|-----------|-----------|----------------|----------------|
| 65×65 | 0.016 ms | ~0.17 ms | ~0.3 ms | **~3 ms** |
| 129×129 | 0.027 ms | ~0.31 ms | ~0.375 ms | **~4-7 ms** |
| 257×257 | — | — | — | 仅用于离线分析 |

### 1.5 P-EFIT 成果总结

```
P-EFIT 的历史地位
═══════════════════════════════════════════════════════════

全球首个:
  ✓ 在实际托克马克 PCS 上部署的 GPU 平衡重建代码
  ✓ 实现 129×129 网格亚毫秒级单次迭代
  ✓ 算法完全公开（含五步 G-S 求解器伪代码）

已验证:
  ✓ 2014-2020 年 EAST 多轮实验验证
  ✓ ISOFLUX 形状控制、雪花偏滤器等先进应用
  ✓ FP32 精度对实时控制足够

公开程度:
  ✓ 算法结构和设计思路完全公开
  ✓ CUDA kernel 设计策略有详细描述
  ✗ 源代码未开源（需基于论文重新实现）
```

---

## 2. 全球 GPU/ML 加速平衡重建最新进展（2023-2026）

P-EFIT 之后，全球出现了多条并行的技术路线。按方法分类：

### 2.1 路线一：GPU 指令级移植（EFIT-AI / SC'23）

> Antepara O., Williams S., Kruger S., et al., "Performance-Portable GPU Acceleration of the EFIT Tokamak Plasma Equilibrium Reconstruction Code," SC '23 Workshops (P3HPC), 2023.

| 内容 | 详情 |
|------|------|
| 目标 | 用 **OpenMP/OpenACC 指令**（而非 CUDA）GPU 加速 EFIT-AI |
| 意义 | 实现 **跨平台可移植**——同一代码在 NVIDIA/AMD/Intel GPU 上运行 |
| 测试平台 | NVIDIA A100 (Perlmutter), **AMD MI250X (Frontier)**, Intel PVC (Sunspot) |
| 加速范围 | 仅加速 G-S 逆求解器（最耗时的单函数） |
| 网格 | 多分辨率测试 (65×65 至更高) |
| 编程模型 | OpenMP target offload + OpenACC |

**与 P-EFIT 的关键差异**：

| 对比 | P-EFIT (ASIPP) | EFIT-AI GPU (LBNL/GA) |
|------|:---:|:---:|
| 编程模型 | CUDA（手写 kernel） | OpenMP/OpenACC（指令级） |
| 优化深度 | 深度手动优化（LDS, 前缀和, 分块） | 编译器自动优化为主 |
| 可移植性 | 仅 NVIDIA | **NVIDIA + AMD + Intel** |
| 性能天花板 | 更高（手动调优极致） | 较低（指令级抽象有开销） |
| 已验证平台 | TITAN X (Maxwell/Pascal) | A100, MI250X, PVC |
| 实时部署 | ✓ (EAST PCS) | ✗ (离线 HPC 工作流) |

**核心结论**：这是首次在 **AMD MI250X** 上成功运行 EFIT 代码，证明了 AMD GPU 执行平衡重建的可行性。但因为使用指令级方法而非手写 kernel，性能不如 P-EFIT 的 CUDA 手动优化版。

### 2.2 路线二：EFIT-AI 项目（DOE 资助，2021-2024）

> Kruger S., et al., "EFIT-AI: Machine Learning and Artificial Intelligence Assisted Equilibrium Reconstruction for Tokamak Experiments and Burning Plasmas," DOE Final Report, 2024.

美国能源部（DOE）资助的三年期项目，由 GA/LBNL/PPPL 联合执行，2024年12月结题。

**核心成果**：

| 组件 | 方法 | 特点 |
|------|------|------|
| **EFIT-MORNN** | 物理约束神经网络 (PINN) + 模型降阶 (MOR) | 用 NN 近似 G-S 求解器，精度优于 RT-EFIT |
| **贝叶斯框架** | ML 辅助不确定性量化 | 用 Bayesian 推断替代传统最小二乘拟合 |
| **3D 扰动重建** | MOR 扩展到 3D 平衡 | 支持含 RMP 的扰动平衡 |
| **EFIT-AI 数据库** | 2500+ DIII-D 放电, IMAS 格式 | 三种约束级别 (磁+MSE+动理学) |

**意义**：EFIT-AI 代表了美国路线——**不追求极致的 GPU 手写 kernel 性能，而是用 ML 替代传统数值求解器**。如果 NN 推理足够快（<1ms），则无需深度 GPU 优化。

**局限**：
- NN 模型的训练数据来自 DIII-D，迁移到新装置需要重新训练
- EFIT-MORNN 尚未进行实时部署验证
- 缺乏公开的推理延迟基准（仅称"real-time capable"，无具体 ms 数字）

### 2.3 路线三：HL-3 的 EFITNN（2024）

> "Real-time equilibrium reconstruction by neural network based on HL-3 tokamak," arXiv:2405.11221, 2024.

| 内容 | 详情 |
|------|------|
| 装置 | HL-3 托克马克 (中国成都) |
| 方法 | 多任务学习神经网络，端到端推理 |
| 输入 | 68 通道磁诊断信号 |
| 输出 | 8 个等离子体参数 + 129×129 极向磁通/电流密度 |
| 推理速度 | **0.08-0.45 ms** |
| 精度 | R² = 0.941 (参数), 0.997 (磁通), 0.959 (电流密度) |
| 训练数据 | 1159 次 HL-3 放电 |

**意义**：目前公开报道中**推理速度最快**的平衡重建方案（0.08ms 比 P-EFIT 的 0.3ms 还快）。

**局限**：
- 纯 NN 方案，**不保证物理一致性**（ψ 不一定满足 G-S 方程）
- 对训练数据外的新运行模式泛化能力未知
- 未在 PCS 中实时部署验证

### 2.4 路线四：GPEC（IPP Garching, 2015）

> Rampp M., Preuss R., Fischer R., et al., "GPEC, a real-time capable Tokamak equilibrium code," arXiv:1511.04203, 2015.

| 内容 | 详情 |
|------|------|
| 装置 | ASDEX Upgrade |
| 方法 | 经典 G-S 迭代 + 混合并行（共享内存 + 分布式） |
| 硬件 | **标准服务器 CPU**（非 GPU） |
| 网格 | 32×64 |
| 性能 | < 1 ms (含 90 个诊断信号, 4 次迭代, q 计算) |
| 基函数 | 最多 15 个 |

**意义**：证明在 CPU 上通过算法优化也能达到 1ms，但代价是低分辨率（32×64 vs P-EFIT 的 129×129）。

### 2.5 路线五：GS-DeepNet / FBE-Net / PINN（学术前沿, 2023-2025）

| 方法 | 论文 | 特点 | 精度 |
|------|------|------|------|
| GS-DeepNet | *Scientific Reports*, 2023 | 无监督学习，双网络互约束 | 不依赖训练数据 |
| FBE-Net | *IEEE TPS*, 2024 | PINN 约束自由边界平衡 | 加速数值求解 |
| Multi-stage PINN | arXiv, 2025 | 多阶段训练，高精度 G-S 求解 | 误差 O(10⁻⁸) |

这些方法尚在学术探索阶段，距离实时部署仍有距离，但代表了"**用物理约束 NN 替代传统迭代器**"的长期趋势。

### 2.6 全球进展汇总

```
全球 GPU/ML 平衡重建方案对比
═══════════════════════════════════════════════════════════════════

方案          机构        方法          网格       延迟      实时部署
──────────  ──────────  ────────────  ────────  ────────  ────────
P-EFIT       ASIPP       CUDA kernel   129×129   0.375 ms  ✓ EAST
EFIT-AI GPU  LBNL/GA     OpenMP 指令   多网格    未公开     ✗
EFIT-MORNN   GA/PPPL     PINN+MOR      65×65+    未公开     ✗
EFITNN       HL-3/SWIP   纯 NN         129×129   0.08 ms   ✗
GPEC         IPP         CPU 并行      32×64     <1 ms      ✓ AUG
GS-DeepNet   学术        无监督 NN     —         —          ✗

═══════════════════════════════════════════════════════════════════

★ 唯一在实际 PCS 中部署并验证的 GPU 方案仍然是 P-EFIT
★ NN 方案速度更快但缺乏物理保证和实时部署验证
★ 跨平台 GPU (含 AMD) 可行性已被 SC'23 证明
```

---

## 3. 目前还有哪些没做到

### 3.1 P-EFIT 的未解决问题

| 未解决 | 现状 | 困难 |
|--------|------|------|
| **257×257 实时** | 仅用于离线分析，未尝试实时 | TITAN X 的 FP32 算力和显存带宽不足 |
| **AMD GPU 移植** | 仅 CUDA，未做 HIP/ROCm | 需要手动重写 kernel（hipify 可半自动） |
| **混合精度** | 全程 FP32 | 2016 年 GPU 无 FP16 矩阵硬件 |
| **AI 辅助** | 无 | 2016 年 NN 在聚变中的应用刚起步 |
| **与射线追踪集成** | 无 | P-EFIT 仅服务 PCS，不涉及 ECRH |
| **非 EAST 装置验证** | 仅 EAST | 需适配不同装置的 Green 函数和诊断布局 |
| **源码开源** | 未开源 | 体制/知识产权限制 |

### 3.2 全领域的未解决问题

| 问题 | 详情 | 当前最好水平 |
|------|------|------------|
| **GPU 平衡重建 + GPU 射线追踪联合计算** | 无人尝试在同一 GPU 上同时运行两者 | 各自独立（见第 4 章分析） |
| **GPU 射线追踪的实时部署** | 等离子体射线追踪的 GPU 加速仅有学术原型 | CPU rt-TORBEAM 15-20 ms（ASDEX/DIII-D） |
| **端到端控制链 GPU 化** | PCS→ECRH 全链条从未在 GPU 上完整实现 | 各环节分散在不同硬件/代码中 |
| **FP32 射线追踪精度验证** | 射线追踪传统用 FP64，FP32 在实时场景下的精度验证不充分 | 无公开论文 |
| **NN 平衡重建的物理一致性保证** | 纯 NN 输出的 ψ 不一定满足 G-S 方程 | PINN 部分解决，但精度/速度有折衷 |
| **新装置适配** | 每换一个装置就需要重新计算 Green 函数、调整诊断映射 | 手动流程，无自动化框架 |
| **GPU 容错与实时确定性** | GPU 计算在实时控制中的确定性延迟保证 | 无成熟的工程框架 |

### 3.3 关键差距可视化

```
从 P-EFIT (2016) 到完整实时计算平台的差距
═══════════════════════════════════════════════════════════════════

已完成 ██████████                     未完成 ░░░░░░░░░░
────────────────────────────────────────────────────────────────

GPU G-S 求解器        ██████████████████░░░░  (完成 ~80%, 缺 AMD 移植)
GPU Green 函数        ██████████████████░░░░  (完成 ~80%, 缺 Infinity Cache 优化)
GPU 三对角求解        ██████████████████████  (完成 100%, 前缀和算法成熟)
混合精度              ░░░░░░░░░░░░░░░░░░░░░  (未开始)
AI 初始化预测         ░░░░░░░░░░░░░░░░░░░░░  (未开始, HL-3 有 NN 原型)
GPU 射线追踪          ░░░░░░░░░░░░░░░░░░░░░  (无实时部署案例)
EFIT→射线追踪流水线   ░░░░░░░░░░░░░░░░░░░░░  (无人尝试)
AMD GPU 移植          ██░░░░░░░░░░░░░░░░░░░  (SC'23 指令级证明可行)
实时部署验证          ██████████░░░░░░░░░░░  (P-EFIT EAST 部署, 其他未验证)

═══════════════════════════════════════════════════════════════════
```

---

## 4. EFIT 平衡重建与射线追踪的联合研究

### 4.1 当前状态：完全独立的两个领域

经过全面调研，截至 2026 年初，**没有发现将 EFIT 平衡重建和 ECRH 射线追踪在同一 GPU 上合并计算的公开研究**。

这两个计算在当前聚变研究中是 **完全独立的**：

```
当前业界架构：两个计算各自为政
═══════════════════════════════════════════════════════════════════

PCS 服务器 (CPU 集群)              ECRH 控制机 (独立 CPU)
┌───────────────────┐             ┌───────────────────┐
│ RT-EFIT / GPEC    │             │ rt-TORBEAM         │
│ (CPU 实时平衡重建)  │── 网络 ──→│ (CPU 实时射线追踪)  │
│                   │  RFM/MDS   │                    │
│ 输出: ψ, ne, Te, B│  延迟:     │ 输入: ψ, ne, Te, B │
│ 延迟: 1-10 ms     │  0.1-1 ms  │ 延迟: 15-20 ms     │
└───────────────────┘             └───────────────────┘

★ 两个代码由不同团队开发
★ 运行在不同的硬件上
★ 通过网络传输中间数据
★ 各自有独立的实时化路线
```

### 4.2 为什么没人合并？

| 原因 | 详情 |
|------|------|
| **组织分割** | EFIT 由 PCS 团队开发（如 GA 的 Lao/Ferron, ASIPP 的黄曜/肖炳甲），射线追踪由 ECRH 团队开发（如 IPP 的 Poli, IFP-CNR 的 Farina）。两个团队隶属不同部门/机构 |
| **硬件分割** | PCS 和 ECRH 在物理上是独立的控制系统，各有自己的计算硬件。没有共享 GPU 的需求驱动 |
| **编程语言/框架不同** | EFIT 传统上是 Fortran，射线追踪 (TORBEAM/GRAY) 也是 Fortran，但两者的数值库和数据结构完全不同 |
| **GPU 化各自刚起步** | P-EFIT 是 CUDA，rt-TORBEAM 还在 CPU 上。当一方都没 GPU 化时，不可能讨论合并 |
| **时间尺度不匹配** | 历史上 EFIT 用于形状控制（~10ms 周期即可），射线追踪用于 ECCD 优化（秒级即可）。只有 NTM 快速控制才需要两者同时亚毫秒 |

### 4.3 最接近"合并"的案例

虽然没有真正的合并实现，但有一些工作**隐含地将两者关联**：

**案例 1：DIII-D 并行实时物理代码（2025）**

> "Parallelized Real-time Physics Codes for Plasma Control on DIII-D," arXiv:2511.11964, 2025.

DIII-D 开发了一个实时安全多线程库，将 rt-TORBEAM 和 STRIDE（稳定性代码）并行化运行在 PCS 上。虽然 EFIT 也在同一 PCS 中运行，但三个代码的并行化是 **分别实现** 的，没有共享 GPU，也没有流水线化。

这篇论文的意义在于：它表明 **在同一 PCS 中同时运行多个实时物理代码** 已经是 DIII-D 的工程需求，缺的只是 GPU 的引入。

**案例 2：反向射线追踪需要平衡输入（2025）**

> "Fast physics-based launcher optimization for electron cyclotron current drive," *Plasma Phys. Control. Fusion*, 2025.

这篇论文提出的反向射线追踪方法需要精确的平衡重建结果作为输入（磁面结构、q 剖面）。虽然论文本身不涉及 GPU 或实时运行，但其计算链条天然需要 EFIT→射线追踪的紧密耦合。

**案例 3：EFIT-AI 的 IMAS 数据框架（2024）**

EFIT-AI 项目将重建结果存储为 ITER IMAS 格式，这是一个标准化接口。如果射线追踪代码也支持 IMAS 输入，则可通过标准接口实现松耦合。但这是 **数据格式级** 的集成，不是计算级。

### 4.4 合并的技术可行性分析

从计算角度看，EFIT 和射线追踪合并到同一 GPU 不仅可行，而且有天然互补性：

| 计算特征 | GPU-EFIT | GPU 射线追踪 | 互补性 |
|---------|----------|------------|--------|
| 计算类型 | 矩阵运算（GEMM + 三对角） | ODE 积分（大规模独立射线） | **不同 kernel 类型** |
| 精度需求 | FP32 足够 | FP32/FP64 | 可共用 FP32 管线 |
| 带宽需求 | 高（Green 矩阵访问） | 中（等离子体剖面查询） | 时间错开 |
| 显存需求 | ~136 MB (129×129) | ~100 MB (12束×1000角度) | 总计 <1% of 32GB |
| 数据依赖 | — | **依赖 EFIT 输出** (ψ, ne, Te, B) | GPU 内零拷贝传递 |

**关键优势：消除数据传输延迟**

```
当前: EFIT (CPU) → 网络传输 (0.1-1 ms) → 射线追踪 (CPU)
合并: EFIT (GPU) → GPU 显存直接读取 (0 ms) → 射线追踪 (GPU)
```

EFIT 的输出 ψ(R,Z) 正是射线追踪的输入。如果两者在同一 GPU 上，这些数据无需出 GPU 显存，传输延迟从 ~0.1-1 ms 降到 **接近零**。

### 4.5 结论：一个空白的研究领域

```
GPU 平衡重建 + 射线追踪联合计算的研究现状
═══════════════════════════════════════════════════════════════════

已有研究:
  ✓ GPU-EFIT (P-EFIT, 2016-2020)          — ASIPP, CUDA
  ✓ GPU-EFIT 跨平台 (SC'23, 2023)         — LBNL/GA, OpenMP
  ✓ NN 平衡重建 (EFITNN, 2024)             — HL-3, PyTorch
  ✓ CPU rt-TORBEAM (2015-)                 — IPP, Fortran+MPI
  ✓ 反向射线追踪理论 (2025)                 — 离线优化

空白领域（无公开研究）:
  ✗ GPU 射线追踪实时部署
  ✗ EFIT + 射线追踪同一 GPU 联合计算
  ✗ EFIT → 射线追踪 GPU 流水线
  ✗ HIP/ROCm 平衡重建（手写 kernel 级）
  ✗ AMD GPU 上的实时等离子体计算

═══════════════════════════════════════════════════════════════════

★ "在一块 AMD GPU 上同时运行 GPU-EFIT 和 GPU 射线追踪"
  是一个没有公开先例的方案
  技术上完全可行，且有显著的延迟优势
  这正是 AMD R9700 实时计算平台设计方案的核心创新点
```

---

## 5. 对 AMD R9700 方案的启示

| 启示 | 来自 | 对 R9700 方案的影响 |
|------|------|-------------------|
| FP32 足够用于实时 EFIT | P-EFIT 四篇论文 | R9700 的 48 TFLOPS FP32 完全适配 |
| 五步 G-S 求解器算法公开 | 2017 CPB 论文 | 可基于论文重新实现，无需源码 |
| AMD GPU 可运行 EFIT | SC'23 论文 | AMD 平台已被验证，R9700 属同一生态 |
| NN 可加速初始化 | EFITNN (HL-3) | R9700 的 AI 加速器可运行 NN 预测 |
| 射线追踪 GPU 化是空白 | 全球调研 | **首创机会**——业界首个 GPU 射线追踪实时部署 |
| EFIT+射线追踪合并是空白 | 全球调研 | **首创机会**——业界首个单 GPU 联合计算平台 |
| CPU rt-TORBEAM 15-20ms 是天花板 | ASDEX/DIII-D | GPU 加速有 50-100× 提升空间 |

---

## 参考文献

### P-EFIT 系列

1. Huang Y., Xiao B.J., Luo Z.P., et al., "Implementation of GPU parallel equilibrium reconstruction for plasma control in EAST," *Fusion Eng. Des.*, 112, 40-48, 2016.
2. Huang Y., Xiao B.J., Luo Z.P., "Fast parallel Grad–Shafranov solver for real-time equilibrium reconstruction in EAST tokamak using GPU," *Chinese Physics B*, 26(8), 085204, 2017.
3. Huang Y., Xiao B.J., Luo Z.P., et al., "Improvement of GPU parallel real-time equilibrium reconstruction for plasma control," *Fusion Eng. Des.*, 128, 6-14, 2018.
4. Huang Y., Luo Z.P., Xiao B.J., Lao L.L., et al., "GPU-optimized fast plasma equilibrium reconstruction in fine grids for real-time control and data analysis," *Nuclear Fusion*, 60, 076021, 2020.

### GPU/ML 加速 EFIT

5. Antepara O., Williams S., Kruger S., et al., "Performance-Portable GPU Acceleration of the EFIT Tokamak Plasma Equilibrium Reconstruction Code," SC '23 Workshops, 2023.
6. Kruger S., et al., "EFIT-AI: ML and AI Assisted Equilibrium Reconstruction for Tokamak Experiments and Burning Plasmas," DOE Final Report, Dec 2024.
7. "Real-time equilibrium reconstruction by neural network based on HL-3 tokamak," arXiv:2405.11221, 2024.
8. Rampp M., Preuss R., Fischer R., et al., "GPEC, a real-time capable Tokamak equilibrium code," arXiv:1511.04203, 2015.
9. "GS-DeepNet: mastering tokamak plasma equilibria with deep neural networks," *Scientific Reports*, 2023.

### 实时射线追踪

10. Poli E., et al., "Real-time beam tracing for control of EC wave deposition," *Fusion Eng. Des.*, 2015.
11. "Parallelized Real-time Physics Codes for Plasma Control on DIII-D," arXiv:2511.11964, 2025.
12. "Fast physics-based launcher optimization for ECCD," *Plasma Phys. Control. Fusion*, 2025.

---

*本文综述了 P-EFIT 四篇公开论文的核心成果（全球首个 GPU 实时平衡重建部署），梳理了 2023-2026 年全球五条并行技术路线的最新进展，识别了当前领域的关键空白：GPU 射线追踪实时部署和 EFIT+射线追踪联合计算均无公开先例。这一空白恰好是 AMD R9700 实时计算平台设计方案的切入点——在一块 GPU 上流水线化两个计算，实现端到端 ~1.5-2 ms 的控制链延迟。*
