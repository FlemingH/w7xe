#include "gpu_efit/gpu_efit.h"
#include "common/plasma_profiles.h"
#include "common/timer.h"
#include "common/hip_check.h"
#include "distributed/rfm_transport.h"
#include <cstdio>
#include <cstring>

using namespace rocm_rtpc;

int main(int argc, char** argv) {
    int grid = 129;
    int port = 50051;
    int iterations = 10;

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--grid") == 0 && i + 1 < argc)
            grid = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--port") == 0 && i + 1 < argc)
            port = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--iter") == 0 && i + 1 < argc)
            iterations = std::atoi(argv[++i]);
    }

    printf("=== ROCM-RTPC GPU-EFIT Server ===\n");
    printf("Grid: %d×%d  Iterations: %d  RFM port: %d\n",
           grid, grid, iterations, port);

    // Initialize GPU-EFIT
    GpuEfit efit(grid);
    efit.initialize();
    printf("GPU-EFIT initialized.\n");

    // Generate synthetic current density (simulating diagnostic input)
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

    // Initialize RFM sender
    RfmTransport rfm;
    rfm.init_sender(port);
    printf("RFM transport ready on port %d.\n", port);

    // Main reconstruction loop (simulates continuous operation)
    printf("\nRunning equilibrium reconstruction...\n");

    Timer timer;
    EquilibriumData eq{};

    timer.start();
    efit.reconstruct(h_J, eq, iterations);
    double efit_ms = timer.elapsed_ms();

    printf("EFIT reconstruction: %.3f ms\n", efit_ms);
    printf("  ψ range: [%.4f, %.4f]\n", eq.psi_axis, eq.psi_boundary);
    printf("  Axis: (R=%.3f, Z=%.3f)\n", eq.R_axis, eq.Z_axis);

    // Send equilibrium data via RFM
    timer.start();
    rfm.send_equilibrium(eq);
    double send_ms = timer.elapsed_ms();
    printf("RFM send: %.3f ms\n", send_ms);

    // Cleanup
    PlasmaProfileGenerator::free_profiles(eq);
    delete[] h_J;

    printf("\nGPU-EFIT server done. Total: %.3f ms\n", efit_ms + send_ms);
    return 0;
}
