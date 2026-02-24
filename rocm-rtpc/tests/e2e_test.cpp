#include "gpu_efit/gpu_efit.h"
#include "ray_tracing/ray_tracing.h"
#include "common/plasma_profiles.h"
#include "common/timer.h"
#include "common/hip_check.h"
#include "distributed/rfm_transport.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdlib>

using namespace rocm_rtpc;

// Single-process end-to-end test:
// GPU-EFIT → local transfer → GPU Ray Tracing
// Simulates the full distributed pipeline on one GPU.
int main(int argc, char** argv) {
    int grid = 129;
    int efit_iter = 10;
    int num_beams = 4;

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--grid") == 0 && i + 1 < argc)
            grid = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--iter") == 0 && i + 1 < argc)
            efit_iter = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--beams") == 0 && i + 1 < argc)
            num_beams = std::atoi(argv[++i]);
    }

    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║  ROCM-RTPC Real-Time Plasma Computation: E2E Test   ║\n");
    printf("╠══════════════════════════════════════════════════════╣\n");
    printf("║  Grid: %3d×%-3d  EFIT iter: %2d  Beams: %2d            ║\n",
           grid, grid, efit_iter, num_beams);
    printf("╚══════════════════════════════════════════════════════╝\n\n");

    // Print GPU info
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("  Compute Units: %d\n", prop.multiProcessorCount);
    printf("  Clock: %d MHz\n", prop.clockRate / 1000);
    printf("  VRAM: %.1f GB\n", prop.totalGlobalMem / 1e9);
    printf("\n");

    Timer timer, total_timer;
    TimingInfo timing{};

    total_timer.start();

    // ════════ Phase 1: GPU-EFIT Equilibrium Reconstruction ════════
    printf("── Phase 1: GPU-EFIT ─────────────────────────────────\n");

    GpuEfit efit(grid);
    efit.initialize();

    // Generate synthetic current density
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

    EquilibriumData eq_efit{};
    timer.start();
    efit.reconstruct(h_J, eq_efit, efit_iter);
    timing.efit_ms = timer.elapsed_ms();
    printf("  EFIT reconstruction: %.3f ms\n", timing.efit_ms);

    // ════════ Phase 2: RFM Data Transfer (simulated local) ════════
    printf("── Phase 2: RFM Transfer (local simulation) ──────────\n");

    EquilibriumData eq_rt{};
    timer.start();
    RfmTransport::local_transfer(eq_efit, eq_rt);
    timing.transfer_ms = timer.elapsed_ms();
    printf("  Local transfer: %.3f ms\n", timing.transfer_ms);

    // ════════ Phase 3: GPU Ray Tracing ════════════════════════════
    printf("── Phase 3: GPU Ray Tracing ──────────────────────────\n");

    GpuRayTracing rt;
    rt.upload_equilibrium(eq_rt);

    ECRHTarget target{};
    target.num_beams = num_beams;
    for (int b = 0; b < num_beams; b++) {
        target.rho_target[b] = 0.3f + 0.15f * b;
        target.P_request[b] = 1.0f;
    }

    BeamResult results[MAX_BEAMS];
    timer.start();
    rt.compute_optimal_angles(target, results);
    timing.raytrace_ms = timer.elapsed_ms();

    timing.total_ms = total_timer.elapsed_ms();

    printf("  Ray tracing: %.3f ms\n", timing.raytrace_ms);
    printf("\n");

    // ════════ Results ═════════════════════════════════════════════
    printf("── Results ───────────────────────────────────────────\n");
    for (int b = 0; b < target.num_beams; b++) {
        printf("  Beam %2d: target ρ=%.3f → θ=%+.4f, φ=%+.4f, ρ_dep=%.4f",
               b, target.rho_target[b],
               results[b].theta_opt, results[b].phi_opt,
               results[b].rho_dep);
        float err = std::fabs(results[b].rho_dep - target.rho_target[b]);
        printf("  Δρ=%.4f %s\n", err, (err < 0.1f ? "✓" : "✗"));
    }

    // ════════ Timing Summary ═════════════════════════════════════
    printf("\n");
    printf("── Timing Summary ────────────────────────────────────\n");
    printf("  GPU-EFIT:       %8.3f ms\n", timing.efit_ms);
    printf("  RFM transfer:   %8.3f ms\n", timing.transfer_ms);
    printf("  Ray tracing:    %8.3f ms\n", timing.raytrace_ms);
    printf("  ──────────────────────────────\n");
    printf("  Pipeline total: %8.3f ms\n",
           timing.efit_ms + timing.transfer_ms + timing.raytrace_ms);
    printf("  Wall-clock:     %8.3f ms\n", timing.total_ms);
    printf("\n");

    float cpu_baseline = 25.0f;
    float speedup = cpu_baseline /
        (timing.efit_ms + timing.transfer_ms + timing.raytrace_ms);
    printf("  CPU baseline:   ~%.0f ms\n", cpu_baseline);
    printf("  Speedup:        ~%.1f×\n", speedup);

    // Cleanup
    PlasmaProfileGenerator::free_profiles(eq_efit);
    PlasmaProfileGenerator::free_profiles(eq_rt);
    delete[] h_J;

    printf("\nDone.\n");
    return 0;
}
