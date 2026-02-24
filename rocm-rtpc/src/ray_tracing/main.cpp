#include "ray_tracing/ray_tracing.h"
#include "common/types.h"
#include "common/timer.h"
#include "common/hip_check.h"
#include "common/plasma_profiles.h"
#include "distributed/rfm_transport.h"
#include <cstdio>
#include <cstring>

using namespace rocm_rtpc;

int main(int argc, char** argv) {
    int port = 50051;
    int num_beams = 4;

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--port") == 0 && i + 1 < argc)
            port = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--beams") == 0 && i + 1 < argc)
            num_beams = std::atoi(argv[++i]);
    }

    printf("=== ROCM-RTPC GPU Ray Tracing Server ===\n");
    printf("Beams: %d  RFM port: %d\n", num_beams, port);

    // Initialize ray tracer
    GpuRayTracing rt;

    // Initialize RFM receiver
    RfmTransport rfm;
    rfm.init_receiver(port);
    printf("Waiting for equilibrium data on port %d...\n", port);

    // Receive equilibrium from GPU-EFIT via RFM
    EquilibriumData eq{};
    Timer timer;

    timer.start();
    rfm.receive_equilibrium(eq);
    double recv_ms = timer.elapsed_ms();
    printf("RFM receive: %.3f ms (grid %d×%d)\n", recv_ms, eq.nr, eq.nz);

    // Upload to GPU
    timer.start();
    rt.upload_equilibrium(eq);
    double upload_ms = timer.elapsed_ms();
    printf("GPU upload: %.3f ms\n", upload_ms);

    // Set up NTM suppression targets
    ECRHTarget target{};
    target.num_beams = num_beams;
    for (int b = 0; b < num_beams; b++) {
        target.rho_target[b] = 0.4f + 0.1f * b;  // typical NTM island locations
        target.P_request[b] = 1.0f;  // 1 MW per beam
    }

    // Compute optimal angles
    BeamResult results[MAX_BEAMS];

    timer.start();
    rt.compute_optimal_angles(target, results);
    double rt_ms = timer.elapsed_ms();

    printf("\nRay tracing results (%.3f ms):\n", rt_ms);
    for (int b = 0; b < target.num_beams; b++) {
        printf("  Beam %d: target ρ=%.3f → θ=%.4f rad, φ=%.4f rad, "
               "ρ_dep=%.4f\n",
               b, target.rho_target[b],
               results[b].theta_opt, results[b].phi_opt,
               results[b].rho_dep);
    }

    // Cleanup
    PlasmaProfileGenerator::free_profiles(eq);

    printf("\nTotal: RFM %.3f + upload %.3f + raytrace %.3f = %.3f ms\n",
           recv_ms, upload_ms, rt_ms, recv_ms + upload_ms + rt_ms);
    return 0;
}
