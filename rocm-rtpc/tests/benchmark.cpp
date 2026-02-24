#include "gpu_efit/gpu_efit.h"
#include "gpu_efit/efit_kernels.h"
#include "ray_tracing/ray_tracing.h"
#include "ray_tracing/rt_kernels.h"
#include "common/plasma_profiles.h"
#include "common/timer.h"
#include "common/hip_check.h"
#include "distributed/rfm_transport.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cfloat>

using namespace rocm_rtpc;

// ═══════════════════════════════════════════════════════════════════
// Utility: hipEvent-based timer pair for GPU kernel measurement
// ═══════════════════════════════════════════════════════════════════
struct GpuTimer {
    hipEvent_t start_evt, stop_evt;

    GpuTimer() {
        HIP_CHECK(hipEventCreate(&start_evt));
        HIP_CHECK(hipEventCreate(&stop_evt));
    }
    ~GpuTimer() {
        hipEventDestroy(start_evt);
        hipEventDestroy(stop_evt);
    }
    void start(hipStream_t stream = nullptr) {
        HIP_CHECK(hipEventRecord(start_evt, stream));
    }
    void stop(hipStream_t stream = nullptr) {
        HIP_CHECK(hipEventRecord(stop_evt, stream));
    }
    float elapsed_ms() {
        HIP_CHECK(hipEventSynchronize(stop_evt));
        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, start_evt, stop_evt));
        return ms;
    }
};

// ═══════════════════════════════════════════════════════════════════
// Utility: statistics from a vector of measurements
// ═══════════════════════════════════════════════════════════════════
struct Stats {
    double min_ms, max_ms, mean_ms, median_ms, stddev_ms;
    int    n;
};

Stats compute_stats(std::vector<double>& samples) {
    Stats s{};
    s.n = (int)samples.size();
    if (s.n == 0) return s;

    std::sort(samples.begin(), samples.end());
    s.min_ms = samples.front();
    s.max_ms = samples.back();
    s.median_ms = (s.n % 2 == 0)
        ? (samples[s.n/2 - 1] + samples[s.n/2]) / 2.0
        : samples[s.n/2];

    double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    s.mean_ms = sum / s.n;

    double sq_sum = 0.0;
    for (auto v : samples) sq_sum += (v - s.mean_ms) * (v - s.mean_ms);
    s.stddev_ms = (s.n > 1) ? std::sqrt(sq_sum / (s.n - 1)) : 0.0;
    return s;
}

void print_stats(const char* label, const Stats& s) {
    printf("  %-28s  %8.3f  %8.3f  %8.3f  %8.3f  %8.3f  (%d runs)\n",
           label, s.min_ms, s.median_ms, s.mean_ms, s.max_ms, s.stddev_ms, s.n);
}

// ═══════════════════════════════════════════════════════════════════
// Design targets from AMD_R9700_实时等离子体计算平台设计.md
// ═══════════════════════════════════════════════════════════════════
struct DesignTarget {
    const char* name;
    int grid;
    int beams;
    double target_ms;
    double cpu_baseline_ms;
};

static const DesignTarget efit_targets[] = {
    {"EFIT  65x65  10iter",   65, 0,  0.80,   4.0},
    {"EFIT 129x129 10iter",  129, 0,  1.50,  24.0},
    {"EFIT 257x257 10iter",  257, 0, 21.00, 170.0},
};

static const DesignTarget rt_targets[] = {
    {"RayTrace  1 beam",   129,  1, 0.50, 20.0},
    {"RayTrace  4 beams",  129,  4, 0.50, 80.0},
    {"RayTrace  8 beams",  129,  8, 0.50,160.0},
    {"RayTrace 12 beams",  129, 12, 0.50,240.0},
};

static const DesignTarget pipeline_target = {
    "E2E Pipeline 129x129 4beam", 129, 4, 1.40, 25.0
};

// ═══════════════════════════════════════════════════════════════════
// Helper: generate synthetic plasma current density
// ═══════════════════════════════════════════════════════════════════
float* make_J_plasma(int grid) {
    int M = grid - 2;
    int N_inner = M * M;
    auto* h_J = new float[N_inner];
    for (int i = 0; i < N_inner; i++) {
        int ix = i % M;
        int iy = i / M;
        float x = (float)(ix + 1) / (grid - 1) - 0.5f;
        float y = (float)(iy + 1) / (grid - 1) - 0.5f;
        float r2 = x * x + y * y;
        h_J[i] = (r2 < 0.2f) ? 1.0f - r2 / 0.2f : 0.0f;
    }
    return h_J;
}

// ═══════════════════════════════════════════════════════════════════
// BENCHMARK 1: GPU-EFIT across grid sizes
// ═══════════════════════════════════════════════════════════════════
void bench_efit(int grid, int iterations, int warmup, int repeats) {
    printf("\n  [EFIT %d×%d, %d iterations]\n", grid, grid, iterations);

    auto* h_J = make_J_plasma(grid);
    GpuEfit efit(grid);
    efit.initialize();

    // Warmup
    for (int w = 0; w < warmup; w++) {
        EquilibriumData eq{};
        efit.reconstruct(h_J, eq, iterations, 0.0f);
        PlasmaProfileGenerator::free_profiles(eq);
    }

    // Benchmark
    std::vector<double> times;
    Timer timer;
    for (int r = 0; r < repeats; r++) {
        EquilibriumData eq{};
        HIP_CHECK(hipDeviceSynchronize());
        timer.start();
        efit.reconstruct(h_J, eq, iterations, 0.0f);
        HIP_CHECK(hipDeviceSynchronize());
        times.push_back(timer.elapsed_ms());
        PlasmaProfileGenerator::free_profiles(eq);
    }

    Stats s = compute_stats(times);
    print_stats("reconstruct()", s);

    delete[] h_J;
}

// ═══════════════════════════════════════════════════════════════════
// BENCHMARK 2: Individual EFIT kernels (hipEvent timing)
// ═══════════════════════════════════════════════════════════════════
void bench_efit_kernels(int grid, int repeats) {
    printf("\n  [EFIT Kernel-level %d×%d]\n", grid, grid);

    int M = grid - 2;
    size_t mm = (size_t)M * M;
    int N_bnd = 4 * (grid - 1);

    // Allocate device memory
    float *d_A, *d_B, *d_C, *d_G, *d_J, *d_bnd;
    float *d_a_coeff, *d_m_coeff;
    HIP_CHECK(hipMalloc(&d_A, mm * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_B, mm * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_C, mm * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_G, (size_t)N_bnd * mm * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_J, mm * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_bnd, N_bnd * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_a_coeff, M * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_m_coeff, mm * sizeof(float)));

    // Fill with random-ish data
    auto* h_buf = new float[mm];
    for (size_t i = 0; i < mm; i++) h_buf[i] = 0.01f * (i % 100);
    HIP_CHECK(hipMemcpy(d_A, h_buf, mm * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_buf, mm * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_J, h_buf, mm * sizeof(float), hipMemcpyHostToDevice));

    for (int i = 0; i < M; i++) h_buf[i] = 1.0f / (i + 1.0f);
    HIP_CHECK(hipMemcpy(d_a_coeff, h_buf, M * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_m_coeff, d_A, mm * sizeof(float), hipMemcpyDeviceToDevice));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    GpuTimer gtimer;

    dim3 block_tile(TILE_SIZE, TILE_SIZE);
    dim3 grid_tile((M + TILE_SIZE - 1) / TILE_SIZE,
                   (M + TILE_SIZE - 1) / TILE_SIZE);

    // ── eigen_decomp_kernel (MatMul) ──
    {
        std::vector<double> times;
        for (int r = 0; r < repeats + 3; r++) {
            gtimer.start(stream);
            eigen_decomp_kernel<<<grid_tile, block_tile, 0, stream>>>(
                d_A, d_B, d_C, M);
            gtimer.stop(stream);
            float ms = gtimer.elapsed_ms();
            if (r >= 3) times.push_back(ms);
        }
        Stats s = compute_stats(times);
        print_stats("eigen_decomp_kernel", s);
    }

    // ── matrix_transpose_kernel ──
    {
        std::vector<double> times;
        for (int r = 0; r < repeats + 3; r++) {
            gtimer.start(stream);
            matrix_transpose_kernel<<<grid_tile, block_tile, 0, stream>>>(
                d_A, d_C, M);
            gtimer.stop(stream);
            float ms = gtimer.elapsed_ms();
            if (r >= 3) times.push_back(ms);
        }
        Stats s = compute_stats(times);
        print_stats("matrix_transpose_kernel", s);
    }

    // ── tridiag_solve_kernel ──
    {
        int threads = ((M + 31) / 32) * 32;
        size_t smem = 2 * M * sizeof(float);
        std::vector<double> times;
        for (int r = 0; r < repeats + 3; r++) {
            gtimer.start(stream);
            tridiag_solve_kernel<<<M, threads, smem, stream>>>(
                d_a_coeff, d_m_coeff, d_B, d_C, M);
            gtimer.stop(stream);
            float ms = gtimer.elapsed_ms();
            if (r >= 3) times.push_back(ms);
        }
        Stats s = compute_stats(times);
        print_stats("tridiag_solve_kernel", s);
    }

    // ── green_boundary_kernel ──
    {
        int threads = 256;
        int blocks = (N_bnd + threads - 1) / threads;
        std::vector<double> times;
        for (int r = 0; r < repeats + 3; r++) {
            gtimer.start(stream);
            green_boundary_kernel<<<blocks, threads, 0, stream>>>(
                d_G, d_J, d_bnd, N_bnd, (int)mm);
            gtimer.stop(stream);
            float ms = gtimer.elapsed_ms();
            if (r >= 3) times.push_back(ms);
        }
        Stats s = compute_stats(times);
        print_stats("green_boundary_kernel", s);
    }

    // ── convergence_kernel ──
    {
        int total = grid * grid;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        float* d_max;
        HIP_CHECK(hipMalloc(&d_max, sizeof(float)));

        std::vector<double> times;
        for (int r = 0; r < repeats + 3; r++) {
            HIP_CHECK(hipMemsetAsync(d_max, 0, sizeof(float), stream));
            gtimer.start(stream);
            convergence_kernel<<<blocks, threads, threads * sizeof(float), stream>>>(
                d_A, d_B, d_max, total);
            gtimer.stop(stream);
            float ms = gtimer.elapsed_ms();
            if (r >= 3) times.push_back(ms);
        }
        Stats s = compute_stats(times);
        print_stats("convergence_kernel", s);
        hipFree(d_max);
    }

    // ── profiles_from_psi_kernel ──
    {
        float *d_ne, *d_Te, *d_Bphi;
        int nn = grid * grid;
        HIP_CHECK(hipMalloc(&d_ne, nn * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_Te, nn * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_Bphi, nn * sizeof(float)));

        dim3 blk(16, 16);
        dim3 grd((grid + 15) / 16, (grid + 15) / 16);

        std::vector<double> times;
        for (int r = 0; r < repeats + 3; r++) {
            gtimer.start(stream);
            profiles_from_psi_kernel<<<grd, blk, 0, stream>>>(
                d_A, d_ne, d_Te, d_Bphi,
                grid, grid, 1.25f, 0.01f, -0.6f, 0.01f,
                1.85f, 2.0f, 6.0e19f, 5.0e3f, 0.0f, 1.0f);
            gtimer.stop(stream);
            float ms = gtimer.elapsed_ms();
            if (r >= 3) times.push_back(ms);
        }
        Stats s = compute_stats(times);
        print_stats("profiles_from_psi_kernel", s);

        hipFree(d_ne); hipFree(d_Te); hipFree(d_Bphi);
    }

    hipStreamDestroy(stream);
    hipFree(d_A); hipFree(d_B); hipFree(d_C);
    hipFree(d_G); hipFree(d_J); hipFree(d_bnd);
    hipFree(d_a_coeff); hipFree(d_m_coeff);
    delete[] h_buf;
}

// ═══════════════════════════════════════════════════════════════════
// BENCHMARK 2b: Green Boundary Kernel Comparison (3 variants)
// ═══════════════════════════════════════════════════════════════════
void bench_green_comparison(int grid, int repeats) {
    printf("\n  [Green Kernel Comparison %d×%d]\n", grid, grid);

    int M = grid - 2;
    size_t mm = (size_t)M * M;
    int N_bnd = 4 * (grid - 1);
    int N_inner = (int)mm;
    size_t G_size = (size_t)N_bnd * N_inner;

    // ── Prepare Green matrix on host (same model as GpuEfit) ──
    float h_step = 1.0f / (grid - 1);
    float reg = h_step * 0.5f;
    auto* h_G = new float[G_size];

    for (int i = 0; i < N_bnd; i++) {
        float xb, yb;
        int side = i / (grid - 1);
        int pos  = i % (grid - 1);
        float t  = pos * h_step;
        switch (side) {
            case 0: xb = t;   yb = 0.0f; break;
            case 1: xb = 1.0f; yb = t;   break;
            case 2: xb = 1.0f - t; yb = 1.0f; break;
            default: xb = 0.0f; yb = 1.0f - t; break;
        }
        for (int j = 0; j < N_inner; j++) {
            int ix = j % M;
            int iy = j / M;
            float xi = (ix + 1) * h_step;
            float yi = (iy + 1) * h_step;
            float dx = xb - xi, dy = yb - yi;
            float dist = std::sqrt(dx*dx + dy*dy + reg*reg);
            h_G[(size_t)i * N_inner + j] = -1.0f / (2.0f * PI * dist);
        }
    }

    // ── Convert to FP16 ──
    auto* h_G_fp16 = new __half[G_size];
    for (size_t i = 0; i < G_size; i++) {
        h_G_fp16[i] = __float2half(h_G[i]);
    }

    // ── Build CSR sparse (threshold = 1% of max |G|) ──
    float global_max = 0.0f;
    for (size_t i = 0; i < G_size; i++) {
        float av = std::fabs(h_G[i]);
        if (av > global_max) global_max = av;
    }
    float threshold = global_max * GREEN_SPARSE_THRESHOLD;

    std::vector<int> row_ptr(N_bnd + 1, 0);
    int nnz = 0;
    for (int i = 0; i < N_bnd; i++) {
        for (int j = 0; j < N_inner; j++) {
            if (std::fabs(h_G[(size_t)i * N_inner + j]) > threshold) nnz++;
        }
        row_ptr[i + 1] = nnz;
    }
    float sparsity = 1.0f - (float)nnz / (float)G_size;

    auto* h_sv = new __half[nnz];
    auto* h_sc = new int[nnz];
    int idx = 0;
    for (int i = 0; i < N_bnd; i++) {
        for (int j = 0; j < N_inner; j++) {
            float val = h_G[(size_t)i * N_inner + j];
            if (std::fabs(val) > threshold) {
                h_sv[idx] = __float2half(val);
                h_sc[idx] = j;
                idx++;
            }
        }
    }

    printf("    Green %d×%d: %.1f%% sparse, nnz=%d/%zu\n",
           N_bnd, N_inner, sparsity * 100.0f, nnz, G_size);
    printf("    FP32 dense: %.1f MB, FP16 dense: %.1f MB, Sparse CSR: %.1f MB\n",
           G_size * 4.0f / (1024*1024), G_size * 2.0f / (1024*1024),
           (nnz * (2 + 4) + (N_bnd + 1) * 4) / (1024.0 * 1024.0));

    // ── Upload to device ──
    float *d_G, *d_J, *d_bnd;
    __half *d_G_fp16, *d_G_sv;
    int *d_G_sc, *d_G_rp;

    HIP_CHECK(hipMalloc(&d_G,  G_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_J,  mm * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_bnd, N_bnd * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_G_fp16, G_size * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_G_sv, nnz * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_G_sc, nnz * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_G_rp, (N_bnd + 1) * sizeof(int)));

    HIP_CHECK(hipMemcpy(d_G, h_G, G_size * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_G_fp16, h_G_fp16, G_size * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_G_sv, h_sv, nnz * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_G_sc, h_sc, nnz * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_G_rp, row_ptr.data(), (N_bnd + 1) * sizeof(int), hipMemcpyHostToDevice));

    // Fill J_plasma with peaked profile
    auto* h_J_buf = new float[mm];
    for (size_t i = 0; i < mm; i++) {
        int ix = i % M, iy = (int)(i / M);
        float x = (float)(ix + 1) / (grid - 1) - 0.5f;
        float y = (float)(iy + 1) / (grid - 1) - 0.5f;
        float r2 = x*x + y*y;
        h_J_buf[i] = (r2 < 0.2f) ? 1.0f - r2 / 0.2f : 0.0f;
    }
    HIP_CHECK(hipMemcpy(d_J, h_J_buf, mm * sizeof(float), hipMemcpyHostToDevice));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    GpuTimer gtimer;

    int threads = 256;
    int blocks = (N_bnd + threads - 1) / threads;

    // ── Variant 1: Original FP32 dense ──
    {
        std::vector<double> times;
        for (int r = 0; r < repeats + 5; r++) {
            gtimer.start(stream);
            green_boundary_kernel<<<blocks, threads, 0, stream>>>(
                d_G, d_J, d_bnd, N_bnd, N_inner);
            gtimer.stop(stream);
            float ms = gtimer.elapsed_ms();
            if (r >= 5) times.push_back(ms);
        }
        Stats s = compute_stats(times);
        print_stats("[V1] FP32 dense (original)", s);
    }

    // ── Variant 2: FP16 dense + LDS tiling ──
    {
        size_t smem = threads * sizeof(float);
        std::vector<double> times;
        for (int r = 0; r < repeats + 5; r++) {
            gtimer.start(stream);
            green_boundary_fp16_tiled_kernel<<<blocks, threads, smem, stream>>>(
                d_G_fp16, d_J, d_bnd, N_bnd, N_inner);
            gtimer.stop(stream);
            float ms = gtimer.elapsed_ms();
            if (r >= 5) times.push_back(ms);
        }
        Stats s = compute_stats(times);
        print_stats("[V2] FP16 dense + LDS tiled", s);
    }

    // ── Variant 3: Sparse CSR + FP16 ──
    {
        std::vector<double> times;
        for (int r = 0; r < repeats + 5; r++) {
            gtimer.start(stream);
            green_boundary_sparse_kernel<<<blocks, threads, 0, stream>>>(
                d_G_sv, d_G_sc, d_G_rp, d_J, d_bnd, N_bnd);
            gtimer.stop(stream);
            float ms = gtimer.elapsed_ms();
            if (r >= 5) times.push_back(ms);
        }
        Stats s = compute_stats(times);
        print_stats("[V3] Sparse CSR + FP16", s);
    }

    // Cleanup
    hipStreamDestroy(stream);
    hipFree(d_G); hipFree(d_J); hipFree(d_bnd);
    hipFree(d_G_fp16); hipFree(d_G_sv); hipFree(d_G_sc); hipFree(d_G_rp);
    delete[] h_G; delete[] h_G_fp16; delete[] h_sv; delete[] h_sc; delete[] h_J_buf;
}

// ═══════════════════════════════════════════════════════════════════
// BENCHMARK 3: Ray Tracing across beam counts
// ═══════════════════════════════════════════════════════════════════
void bench_ray_tracing(int grid, int num_beams, int warmup, int repeats) {
    printf("\n  [Ray Tracing %d beams, grid %d×%d]\n", num_beams, grid, grid);

    // Generate equilibrium for ray tracing input
    PlasmaProfileGenerator gen;
    EquilibriumData eq{};
    gen.generate(eq, grid, grid);

    GpuRayTracing rt;
    rt.upload_equilibrium(eq);

    ECRHTarget target{};
    target.num_beams = num_beams;
    for (int b = 0; b < num_beams; b++) {
        target.rho_target[b] = 0.3f + 0.4f * b / (num_beams > 1 ? num_beams - 1 : 1);
        target.P_request[b] = 1.0f;
    }
    BeamResult results[MAX_BEAMS];

    // Warmup
    for (int w = 0; w < warmup; w++) {
        rt.compute_optimal_angles(target, results);
    }

    // Benchmark
    std::vector<double> times;
    Timer timer;
    for (int r = 0; r < repeats; r++) {
        HIP_CHECK(hipDeviceSynchronize());
        timer.start();
        rt.compute_optimal_angles(target, results);
        HIP_CHECK(hipDeviceSynchronize());
        times.push_back(timer.elapsed_ms());
    }

    Stats s = compute_stats(times);
    print_stats("compute_optimal_angles()", s);

    // Per-beam average
    printf("    → per beam: %.3f ms\n", s.median_ms / num_beams);

    PlasmaProfileGenerator::free_profiles(eq);
}

// ═══════════════════════════════════════════════════════════════════
// BENCHMARK 4: Ray tracing kernel directly (hipEvent)
// ═══════════════════════════════════════════════════════════════════
void bench_rt_kernel(int grid, int n_angles, int repeats) {
    printf("\n  [ray_trace_kernel direct: %d angles, grid %d×%d]\n",
           n_angles, grid, grid);

    PlasmaProfileGenerator gen;
    EquilibriumData eq{};
    gen.generate(eq, grid, grid);

    int nn = grid * grid;
    float *d_psi, *d_ne, *d_Te, *d_Bphi;
    HIP_CHECK(hipMalloc(&d_psi,  nn * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_ne,   nn * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_Te,   nn * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_Bphi, nn * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_psi,  eq.psi,  nn * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_ne,   eq.ne,   nn * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_Te,   eq.Te,   nn * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_Bphi, eq.Bphi, nn * sizeof(float), hipMemcpyHostToDevice));

    float *d_theta, *d_phi, *d_rho, *d_eta;
    HIP_CHECK(hipMalloc(&d_theta, n_angles * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_phi,   n_angles * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_rho,   n_angles * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_eta,   n_angles * sizeof(float)));

    auto* h_theta = new float[n_angles];
    auto* h_phi   = new float[n_angles];
    for (int i = 0; i < n_angles; i++) {
        h_theta[i] = -0.6f + 0.9f * i / (n_angles - 1);
        h_phi[i]   = -0.3f + 0.6f * i / (n_angles - 1);
    }
    HIP_CHECK(hipMemcpy(d_theta, h_theta, n_angles * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_phi,   h_phi,   n_angles * sizeof(float), hipMemcpyHostToDevice));

    float dR = (eq.R_max - eq.R_min) / (grid - 1);
    float dZ = (eq.Z_max - eq.Z_min) / (grid - 1);

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    GpuTimer gtimer;

    int threads = 256;
    int blocks = (n_angles + threads - 1) / threads;

    std::vector<double> times;
    for (int r = 0; r < repeats + 3; r++) {
        gtimer.start(stream);
        ray_trace_kernel<<<blocks, threads, 0, stream>>>(
            d_psi, d_ne, d_Te, d_Bphi,
            grid, grid,
            eq.R_min, dR, eq.Z_min, dZ,
            eq.psi_axis, eq.psi_boundary,
            d_theta, d_phi, n_angles,
            2.2f, 0.8f, FREQ_GHZ,
            d_rho, d_eta, ODE_STEPS);
        gtimer.stop(stream);
        float ms = gtimer.elapsed_ms();
        if (r >= 3) times.push_back(ms);
    }

    Stats s = compute_stats(times);
    print_stats("ray_trace_kernel", s);

    // Throughput
    double rays_per_sec = n_angles / (s.median_ms * 1e-3);
    printf("    → throughput: %.0f rays/sec\n", rays_per_sec);
    printf("    → per ray: %.3f μs\n", s.median_ms * 1e3 / n_angles);

    hipStreamDestroy(stream);
    hipFree(d_psi); hipFree(d_ne); hipFree(d_Te); hipFree(d_Bphi);
    hipFree(d_theta); hipFree(d_phi); hipFree(d_rho); hipFree(d_eta);
    delete[] h_theta; delete[] h_phi;
    PlasmaProfileGenerator::free_profiles(eq);
}

// ═══════════════════════════════════════════════════════════════════
// BENCHMARK 5: End-to-end pipeline
// ═══════════════════════════════════════════════════════════════════
void bench_pipeline(int grid, int beams, int efit_iter, int warmup, int repeats) {
    printf("\n  [Pipeline E2E: grid %d, %d beams, %d iter]\n",
           grid, beams, efit_iter);

    auto* h_J = make_J_plasma(grid);

    GpuEfit efit(grid);
    efit.initialize();
    GpuRayTracing rt;

    ECRHTarget target{};
    target.num_beams = beams;
    for (int b = 0; b < beams; b++) {
        target.rho_target[b] = 0.3f + 0.15f * b;
        target.P_request[b] = 1.0f;
    }
    BeamResult results[MAX_BEAMS];

    // Warmup
    for (int w = 0; w < warmup; w++) {
        EquilibriumData eq{};
        efit.reconstruct(h_J, eq, efit_iter, 0.0f);
        rt.upload_equilibrium(eq);
        rt.compute_optimal_angles(target, results);
        PlasmaProfileGenerator::free_profiles(eq);
    }

    // Benchmark with breakdown
    std::vector<double> t_efit, t_xfer, t_rt, t_total;
    Timer timer, seg_timer;

    for (int r = 0; r < repeats; r++) {
        EquilibriumData eq_efit{}, eq_rt{};
        HIP_CHECK(hipDeviceSynchronize());

        timer.start();

        // Phase 1: EFIT
        seg_timer.start();
        efit.reconstruct(h_J, eq_efit, efit_iter, 0.0f);
        HIP_CHECK(hipDeviceSynchronize());
        t_efit.push_back(seg_timer.elapsed_ms());

        // Phase 2: Transfer
        seg_timer.start();
        RfmTransport::local_transfer(eq_efit, eq_rt);
        t_xfer.push_back(seg_timer.elapsed_ms());

        // Phase 3: Ray tracing
        seg_timer.start();
        rt.upload_equilibrium(eq_rt);
        rt.compute_optimal_angles(target, results);
        HIP_CHECK(hipDeviceSynchronize());
        t_rt.push_back(seg_timer.elapsed_ms());

        t_total.push_back(timer.elapsed_ms());

        PlasmaProfileGenerator::free_profiles(eq_efit);
        PlasmaProfileGenerator::free_profiles(eq_rt);
    }

    Stats s_efit  = compute_stats(t_efit);
    Stats s_xfer  = compute_stats(t_xfer);
    Stats s_rt    = compute_stats(t_rt);
    Stats s_total = compute_stats(t_total);

    print_stats("EFIT phase", s_efit);
    print_stats("RFM transfer phase", s_xfer);
    print_stats("Ray tracing phase", s_rt);
    print_stats("Pipeline total", s_total);

    delete[] h_J;
}

// ═══════════════════════════════════════════════════════════════════
// BENCHMARK 6: Memory bandwidth test
// ═══════════════════════════════════════════════════════════════════
void bench_memory_bandwidth() {
    printf("\n  [Memory Bandwidth]\n");

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    GpuTimer gtimer;

    for (size_t size_mb : {1, 4, 16, 64, 128, 256}) {
        size_t bytes = size_mb * 1024 * 1024;
        float *d_src, *d_dst;
        float* h_src = new float[bytes / sizeof(float)];

        HIP_CHECK(hipMalloc(&d_src, bytes));
        HIP_CHECK(hipMalloc(&d_dst, bytes));

        // H2D
        std::vector<double> times;
        for (int r = 0; r < 13; r++) {
            gtimer.start(stream);
            HIP_CHECK(hipMemcpyAsync(d_src, h_src, bytes,
                                      hipMemcpyHostToDevice, stream));
            gtimer.stop(stream);
            float ms = gtimer.elapsed_ms();
            if (r >= 3) times.push_back(ms);
        }
        Stats s = compute_stats(times);
        double bw_h2d = bytes / (s.median_ms * 1e-3) / 1e9;

        // D2D
        times.clear();
        for (int r = 0; r < 13; r++) {
            gtimer.start(stream);
            HIP_CHECK(hipMemcpyAsync(d_dst, d_src, bytes,
                                      hipMemcpyDeviceToDevice, stream));
            gtimer.stop(stream);
            float ms = gtimer.elapsed_ms();
            if (r >= 3) times.push_back(ms);
        }
        Stats s2 = compute_stats(times);
        double bw_d2d = bytes / (s2.median_ms * 1e-3) / 1e9;

        // D2H
        times.clear();
        for (int r = 0; r < 13; r++) {
            gtimer.start(stream);
            HIP_CHECK(hipMemcpyAsync(h_src, d_src, bytes,
                                      hipMemcpyDeviceToHost, stream));
            gtimer.stop(stream);
            float ms = gtimer.elapsed_ms();
            if (r >= 3) times.push_back(ms);
        }
        Stats s3 = compute_stats(times);
        double bw_d2h = bytes / (s3.median_ms * 1e-3) / 1e9;

        printf("  %4zu MB  H2D: %7.1f GB/s (%6.3f ms)  D2D: %7.1f GB/s (%6.3f ms)"
               "  D2H: %7.1f GB/s (%6.3f ms)\n",
               size_mb, bw_h2d, s.median_ms, bw_d2d, s2.median_ms,
               bw_d2h, s3.median_ms);

        hipFree(d_src); hipFree(d_dst);
        delete[] h_src;
    }

    hipStreamDestroy(stream);
}

// ═══════════════════════════════════════════════════════════════════
// BENCHMARK 7: VRAM usage analysis
// ═══════════════════════════════════════════════════════════════════
void bench_vram_usage() {
    printf("\n  [VRAM Usage Analysis]\n");

    for (int grid : {65, 129, 257}) {
        size_t free_before, total;
        HIP_CHECK(hipMemGetInfo(&free_before, &total));

        {
            GpuEfit efit(grid);
            efit.initialize();

            size_t free_after_efit;
            HIP_CHECK(hipMemGetInfo(&free_after_efit, &total));
            double efit_mb = (double)(free_before - free_after_efit) / (1024 * 1024);

            PlasmaProfileGenerator gen;
            EquilibriumData eq{};
            gen.generate(eq, grid, grid);

            GpuRayTracing rt;
            rt.upload_equilibrium(eq);

            size_t free_after_rt;
            HIP_CHECK(hipMemGetInfo(&free_after_rt, &total));
            double rt_mb = (double)(free_after_efit - free_after_rt) / (1024 * 1024);
            double total_mb = (double)(free_before - free_after_rt) / (1024 * 1024);

            printf("  Grid %3d×%-3d  EFIT: %8.1f MB  RT: %8.1f MB  Total: %8.1f MB"
                   "  (VRAM free: %.0f / %.0f MB)\n",
                   grid, grid, efit_mb, rt_mb, total_mb,
                   free_after_rt / (1024.0 * 1024.0),
                   total / (1024.0 * 1024.0));

            PlasmaProfileGenerator::free_profiles(eq);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// MAIN: Run all benchmarks
// ═══════════════════════════════════════════════════════════════════
int main(int argc, char** argv) {
    int warmup  = 3;
    int repeats = 20;
    bool run_all = true;
    bool run_efit = false, run_rt = false, run_pipe = false;
    bool run_kernel = false, run_mem = false, run_vram = false;

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc)
            warmup = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--repeats") == 0 && i + 1 < argc)
            repeats = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--efit") == 0)
            { run_efit = true; run_all = false; }
        else if (std::strcmp(argv[i], "--rt") == 0)
            { run_rt = true; run_all = false; }
        else if (std::strcmp(argv[i], "--pipeline") == 0)
            { run_pipe = true; run_all = false; }
        else if (std::strcmp(argv[i], "--kernel") == 0)
            { run_kernel = true; run_all = false; }
        else if (std::strcmp(argv[i], "--mem") == 0)
            { run_mem = true; run_all = false; }
        else if (std::strcmp(argv[i], "--vram") == 0)
            { run_vram = true; run_all = false; }
    }

    // ── Header ──
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║   ROCM-RTPC Performance Benchmark                               ║\n");
    printf("║   Real-Time Plasma Computation Platform                          ║\n");
    printf("╠═══════════════════════════════════════════════════════════════════╣\n");
    printf("║   Warmup: %d    Repeats: %d                                      ║\n",
           warmup, repeats);
    printf("╚═══════════════════════════════════════════════════════════════════╝\n");

    // ── GPU Info ──
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("\n");
    printf("  GPU:             %s\n", prop.name);
    printf("  Compute Units:   %d\n", prop.multiProcessorCount);
    printf("  Clock:           %d MHz\n", prop.clockRate / 1000);
    printf("  VRAM:            %.1f GB\n", prop.totalGlobalMem / 1e9);
    printf("  L2 Cache:        %.0f KB\n", prop.l2CacheSize / 1024.0);
    printf("  Max Threads/CU:  %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  Warp Size:       %d\n", prop.warpSize);
    printf("  GCN Arch:        %s\n", prop.gcnArchName);

    printf("\n  Stats format:    [label]  min  median  mean  max  stddev  (N)\n");

    // ══════════════════════════════════════════════════════════════
    // Section 1: GPU-EFIT Reconstruction
    // ══════════════════════════════════════════════════════════════
    if (run_all || run_efit) {
        printf("\n");
        printf("══════════════════════════════════════════════════════════════════\n");
        printf("  SECTION 1: GPU-EFIT Reconstruction\n");
        printf("══════════════════════════════════════════════════════════════════\n");

        for (int grid : {65, 129, 257}) {
            bench_efit(grid, 10, warmup, repeats);
        }
    }

    // ══════════════════════════════════════════════════════════════
    // Section 2: EFIT Kernel-Level Profiling
    // ══════════════════════════════════════════════════════════════
    if (run_all || run_kernel) {
        printf("\n");
        printf("══════════════════════════════════════════════════════════════════\n");
        printf("  SECTION 2: EFIT Kernel-Level Profiling\n");
        printf("══════════════════════════════════════════════════════════════════\n");

        for (int grid : {65, 129, 257}) {
            bench_efit_kernels(grid, repeats);
        }

        printf("\n");
        printf("──────────────────────────────────────────────────────────────────\n");
        printf("  SECTION 2b: Green Kernel Optimization Comparison\n");
        printf("  (FP32 Dense vs FP16+LDS Tiled vs Sparse CSR+FP16)\n");
        printf("──────────────────────────────────────────────────────────────────\n");

        for (int grid : {65, 129, 257}) {
            bench_green_comparison(grid, repeats);
        }
    }

    // ══════════════════════════════════════════════════════════════
    // Section 3: GPU Ray Tracing
    // ══════════════════════════════════════════════════════════════
    if (run_all || run_rt) {
        printf("\n");
        printf("══════════════════════════════════════════════════════════════════\n");
        printf("  SECTION 3: GPU Ray Tracing\n");
        printf("══════════════════════════════════════════════════════════════════\n");

        for (int beams : {1, 4, 8, 12}) {
            bench_ray_tracing(129, beams, warmup, repeats);
        }

        printf("\n  [ray_trace_kernel scaling with thread count]\n");
        for (int n_angles : {100, 500, 1000, 5000, 10000}) {
            bench_rt_kernel(129, n_angles, repeats);
        }
    }

    // ══════════════════════════════════════════════════════════════
    // Section 4: End-to-End Pipeline
    // ══════════════════════════════════════════════════════════════
    if (run_all || run_pipe) {
        printf("\n");
        printf("══════════════════════════════════════════════════════════════════\n");
        printf("  SECTION 4: End-to-End Pipeline\n");
        printf("══════════════════════════════════════════════════════════════════\n");

        bench_pipeline(129, 4, 10, warmup, repeats);
    }

    // ══════════════════════════════════════════════════════════════
    // Section 5: Memory Bandwidth
    // ══════════════════════════════════════════════════════════════
    if (run_all || run_mem) {
        printf("\n");
        printf("══════════════════════════════════════════════════════════════════\n");
        printf("  SECTION 5: Memory Bandwidth\n");
        printf("══════════════════════════════════════════════════════════════════\n");

        bench_memory_bandwidth();
    }

    // ══════════════════════════════════════════════════════════════
    // Section 6: VRAM Usage
    // ══════════════════════════════════════════════════════════════
    if (run_all || run_vram) {
        printf("\n");
        printf("══════════════════════════════════════════════════════════════════\n");
        printf("  SECTION 6: VRAM Usage\n");
        printf("══════════════════════════════════════════════════════════════════\n");

        bench_vram_usage();
    }

    // ══════════════════════════════════════════════════════════════
    // Section 7: Design Target Comparison
    // ══════════════════════════════════════════════════════════════
    printf("\n");
    printf("══════════════════════════════════════════════════════════════════\n");
    printf("  SECTION 7: Design Target Comparison\n");
    printf("══════════════════════════════════════════════════════════════════\n");

    printf("\n  Running final comparison tests (median of %d runs)...\n\n", repeats);

    printf("  ┌────────────────────────────────┬──────────┬──────────┬──────────┬────────┬────────┐\n");
    printf("  │ Test Case                       │ Measured │ Target   │ CPU Base │ vs Tgt │vs CPU  │\n");
    printf("  │                                 │ (ms)     │ (ms)     │ (ms)     │        │        │\n");
    printf("  ├────────────────────────────────┼──────────┼──────────┼──────────┼────────┼────────┤\n");

    // EFIT targets
    for (auto& t : efit_targets) {
        auto* h_J = make_J_plasma(t.grid);
        GpuEfit efit(t.grid);
        efit.initialize();

        // warmup
        for (int w = 0; w < warmup; w++) {
            EquilibriumData eq{};
            efit.reconstruct(h_J, eq, 10, 0.0f);
            PlasmaProfileGenerator::free_profiles(eq);
        }

        std::vector<double> times;
        Timer timer;
        for (int r = 0; r < repeats; r++) {
            EquilibriumData eq{};
            HIP_CHECK(hipDeviceSynchronize());
            timer.start();
            efit.reconstruct(h_J, eq, 10, 0.0f);
            HIP_CHECK(hipDeviceSynchronize());
            times.push_back(timer.elapsed_ms());
            PlasmaProfileGenerator::free_profiles(eq);
        }
        Stats s = compute_stats(times);
        const char* status = (s.median_ms <= t.target_ms) ? " PASS " : " FAIL ";
        double speedup = t.cpu_baseline_ms / s.median_ms;
        printf("  │ %-30s │ %8.3f │ %8.2f │ %8.1f │%s│ %5.1fx │\n",
               t.name, s.median_ms, t.target_ms, t.cpu_baseline_ms,
               status, speedup);
        delete[] h_J;
    }

    // RT targets
    for (auto& t : rt_targets) {
        PlasmaProfileGenerator gen;
        EquilibriumData eq{};
        gen.generate(eq, t.grid, t.grid);

        GpuRayTracing rt;
        rt.upload_equilibrium(eq);

        ECRHTarget target{};
        target.num_beams = t.beams;
        for (int b = 0; b < t.beams; b++) {
            target.rho_target[b] = 0.3f + 0.4f * b / (t.beams > 1 ? t.beams - 1 : 1);
            target.P_request[b] = 1.0f;
        }
        BeamResult results[MAX_BEAMS];

        for (int w = 0; w < warmup; w++)
            rt.compute_optimal_angles(target, results);

        std::vector<double> times;
        Timer timer;
        for (int r = 0; r < repeats; r++) {
            HIP_CHECK(hipDeviceSynchronize());
            timer.start();
            rt.compute_optimal_angles(target, results);
            HIP_CHECK(hipDeviceSynchronize());
            times.push_back(timer.elapsed_ms());
        }
        Stats s = compute_stats(times);
        const char* status = (s.median_ms <= t.target_ms) ? " PASS " : " FAIL ";
        double speedup = t.cpu_baseline_ms / s.median_ms;
        printf("  │ %-30s │ %8.3f │ %8.2f │ %8.1f │%s│ %5.1fx │\n",
               t.name, s.median_ms, t.target_ms, t.cpu_baseline_ms,
               status, speedup);
        PlasmaProfileGenerator::free_profiles(eq);
    }

    // Pipeline target
    {
        auto& t = pipeline_target;
        auto* h_J = make_J_plasma(t.grid);
        GpuEfit efit(t.grid);
        efit.initialize();
        GpuRayTracing grt;

        ECRHTarget target{};
        target.num_beams = t.beams;
        for (int b = 0; b < t.beams; b++) {
            target.rho_target[b] = 0.3f + 0.15f * b;
            target.P_request[b] = 1.0f;
        }
        BeamResult results[MAX_BEAMS];

        for (int w = 0; w < warmup; w++) {
            EquilibriumData eq{}, eq2{};
            efit.reconstruct(h_J, eq, 10, 0.0f);
            RfmTransport::local_transfer(eq, eq2);
            grt.upload_equilibrium(eq2);
            grt.compute_optimal_angles(target, results);
            PlasmaProfileGenerator::free_profiles(eq);
            PlasmaProfileGenerator::free_profiles(eq2);
        }

        std::vector<double> times;
        Timer timer;
        for (int r = 0; r < repeats; r++) {
            EquilibriumData eq{}, eq2{};
            HIP_CHECK(hipDeviceSynchronize());
            timer.start();
            efit.reconstruct(h_J, eq, 10, 0.0f);
            RfmTransport::local_transfer(eq, eq2);
            grt.upload_equilibrium(eq2);
            grt.compute_optimal_angles(target, results);
            HIP_CHECK(hipDeviceSynchronize());
            times.push_back(timer.elapsed_ms());
            PlasmaProfileGenerator::free_profiles(eq);
            PlasmaProfileGenerator::free_profiles(eq2);
        }

        Stats s = compute_stats(times);
        const char* status = (s.median_ms <= t.target_ms) ? " PASS " : " FAIL ";
        double speedup = t.cpu_baseline_ms / s.median_ms;
        printf("  │ %-30s │ %8.3f │ %8.2f │ %8.1f │%s│ %5.1fx │\n",
               t.name, s.median_ms, t.target_ms, t.cpu_baseline_ms,
               status, speedup);
        delete[] h_J;
    }

    printf("  └────────────────────────────────┴──────────┴──────────┴──────────┴────────┴────────┘\n");

    printf("\n  Legend: vs Tgt = measured vs design target, vs CPU = speedup over CPU baseline\n");
    printf("          PASS = measured ≤ target,  FAIL = measured > target\n");

    printf("\nBenchmark complete.\n");
    return 0;
}
