#include "ray_tracing/ray_tracing.h"
#include "ray_tracing/rt_kernels.h"
#include "common/hip_check.h"
#include <cstdio>
#include <cmath>

namespace best_rtpc {

GpuRayTracing::GpuRayTracing() {
    HIP_CHECK(hipStreamCreate(&stream_));

    // Pre-allocate for multi-beam parallel: MAX_BEAMS × max_angles_per_stage
    int max_angles = MAX_ANGLES_COARSE;  // coarse and fine use same grid size
    int total_slots = MAX_BEAMS * max_angles;

    HIP_CHECK(hipMalloc(&d_rho_dep_,     total_slots * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_eta_cd_,      total_slots * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_theta_grid_,  max_angles * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_phi_grid_,    max_angles * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_best_theta_,  MAX_BEAMS * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_best_phi_,    MAX_BEAMS * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_best_rho_,    MAX_BEAMS * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_rho_targets_, MAX_BEAMS * sizeof(float)));

    d_psi_ = d_ne_ = d_Te_ = d_Bphi_ = nullptr;
}

GpuRayTracing::~GpuRayTracing() {
    hipFree(d_rho_dep_);   hipFree(d_eta_cd_);
    hipFree(d_theta_grid_);hipFree(d_phi_grid_);
    hipFree(d_best_theta_);hipFree(d_best_phi_);
    hipFree(d_best_rho_);  hipFree(d_rho_targets_);
    if (d_psi_)  hipFree(d_psi_);
    if (d_ne_)   hipFree(d_ne_);
    if (d_Te_)   hipFree(d_Te_);
    if (d_Bphi_) hipFree(d_Bphi_);
    hipStreamDestroy(stream_);
}

void GpuRayTracing::upload_equilibrium(const EquilibriumData& eq) {
    eq_nr_ = eq.nr;  eq_nz_ = eq.nz;
    eq_R_min_ = eq.R_min;  eq_R_max_ = eq.R_max;
    eq_Z_min_ = eq.Z_min;  eq_Z_max_ = eq.Z_max;
    psi_axis_ = eq.psi_axis;  psi_bnd_ = eq.psi_boundary;
    R_axis_ = eq.R_axis;  Z_axis_ = eq.Z_axis;

    size_t n = (size_t)eq.nr * eq.nz * sizeof(float);

    if (d_psi_) hipFree(d_psi_);
    if (d_ne_)  hipFree(d_ne_);
    if (d_Te_)  hipFree(d_Te_);
    if (d_Bphi_) hipFree(d_Bphi_);

    HIP_CHECK(hipMalloc(&d_psi_,  n));
    HIP_CHECK(hipMalloc(&d_ne_,   n));
    HIP_CHECK(hipMalloc(&d_Te_,   n));
    HIP_CHECK(hipMalloc(&d_Bphi_, n));

    HIP_CHECK(hipMemcpyAsync(d_psi_,  eq.psi,  n, hipMemcpyHostToDevice, stream_));
    HIP_CHECK(hipMemcpyAsync(d_ne_,   eq.ne,   n, hipMemcpyHostToDevice, stream_));
    HIP_CHECK(hipMemcpyAsync(d_Te_,   eq.Te,   n, hipMemcpyHostToDevice, stream_));
    HIP_CHECK(hipMemcpyAsync(d_Bphi_, eq.Bphi, n, hipMemcpyHostToDevice, stream_));
    HIP_CHECK(hipStreamSynchronize(stream_));
}

void GpuRayTracing::generate_angle_grid(
    float theta_min, float theta_max,
    float phi_min, float phi_max,
    int n_theta, int n_phi)
{
    int n_total = n_theta * n_phi;
    auto* h_theta = new float[n_total];
    auto* h_phi   = new float[n_total];

    float dtheta = (n_theta > 1) ? (theta_max - theta_min) / (n_theta - 1) : 0.0f;
    float dphi   = (n_phi > 1)   ? (phi_max - phi_min)     / (n_phi - 1)   : 0.0f;

    for (int it = 0; it < n_theta; it++) {
        for (int ip = 0; ip < n_phi; ip++) {
            int idx = it * n_phi + ip;
            h_theta[idx] = theta_min + it * dtheta;
            h_phi[idx]   = phi_min   + ip * dphi;
        }
    }

    HIP_CHECK(hipMemcpyAsync(d_theta_grid_, h_theta,
                              n_total * sizeof(float),
                              hipMemcpyHostToDevice, stream_));
    HIP_CHECK(hipMemcpyAsync(d_phi_grid_, h_phi,
                              n_total * sizeof(float),
                              hipMemcpyHostToDevice, stream_));

    delete[] h_theta;
    delete[] h_phi;
}

// Multi-beam parallel search: all beams × all angles in a single kernel launch
void GpuRayTracing::multibeam_search(
    int num_beams, int n_angles,
    float theta_lo, float theta_hi,
    float phi_lo, float phi_hi,
    int n_theta, int n_phi)
{
    float dR = (eq_R_max_ - eq_R_min_) / (eq_nr_ - 1);
    float dZ = (eq_Z_max_ - eq_Z_min_) / (eq_nz_ - 1);

    // Generate angle grid (shared across all beams)
    generate_angle_grid(theta_lo, theta_hi, phi_lo, phi_hi, n_theta, n_phi);

    // Launch: 2D grid (angle_blocks × num_beams), each thread traces one ray
    int threads = 256;
    dim3 grid_dim((n_angles + threads - 1) / threads, num_beams);

    ray_trace_multibeam_kernel<<<grid_dim, threads, 0, stream_>>>(
        d_psi_, d_ne_, d_Te_, d_Bphi_,
        eq_nr_, eq_nz_,
        eq_R_min_, dR, eq_Z_min_, dZ,
        psi_axis_, psi_bnd_,
        d_theta_grid_, d_phi_grid_, n_angles,
        LAUNCHER_R, LAUNCHER_Z, FREQ_GHZ,
        d_rho_dep_, d_eta_cd_, ODE_STEPS,
        num_beams);

    // One block per beam for reduction
    int opt_threads = 256;
    size_t opt_smem = opt_threads * (sizeof(float) + sizeof(int));

    angle_optimize_multibeam_kernel<<<num_beams, opt_threads, opt_smem, stream_>>>(
        d_rho_dep_, d_eta_cd_, d_theta_grid_, d_phi_grid_,
        n_angles, d_rho_targets_,
        d_best_theta_, d_best_phi_, d_best_rho_,
        num_beams);
}

void GpuRayTracing::compute_optimal_angles(
    const ECRHTarget& target, BeamResult* results)
{
    int num_beams = target.num_beams;

    // Upload all target ρ values at once
    HIP_CHECK(hipMemcpyAsync(d_rho_targets_, target.rho_target,
                              num_beams * sizeof(float),
                              hipMemcpyHostToDevice, stream_));

    // ──── Stage 1: Coarse search (all beams in parallel) ────
    int n_theta_c = 10, n_phi_c = 10;
    int n_coarse = n_theta_c * n_phi_c;

    multibeam_search(num_beams, n_coarse,
                     THETA_MIN, THETA_MAX, PHI_MIN, PHI_MAX,
                     n_theta_c, n_phi_c);

    // Read coarse best for all beams
    float h_best_theta[MAX_BEAMS], h_best_phi[MAX_BEAMS];
    HIP_CHECK(hipMemcpyAsync(h_best_theta, d_best_theta_,
                              num_beams * sizeof(float),
                              hipMemcpyDeviceToHost, stream_));
    HIP_CHECK(hipMemcpyAsync(h_best_phi, d_best_phi_,
                              num_beams * sizeof(float),
                              hipMemcpyDeviceToHost, stream_));
    HIP_CHECK(hipStreamSynchronize(stream_));

    // ──── Stage 2: Fine search around best coarse angles ────
    // Use the center of all beams' best angles for the fine grid.
    // This is a simplification; a per-beam fine grid would need separate launches.
    float dtheta = (THETA_MAX - THETA_MIN) / (n_theta_c - 1);
    float dphi   = (PHI_MAX - PHI_MIN) / (n_phi_c - 1);

    // Find the bounding box of all beams' best angles
    float t_lo = h_best_theta[0] - dtheta, t_hi = h_best_theta[0] + dtheta;
    float p_lo = h_best_phi[0] - dphi,     p_hi = h_best_phi[0] + dphi;
    for (int b = 1; b < num_beams; b++) {
        t_lo = std::min(t_lo, h_best_theta[b] - dtheta);
        t_hi = std::max(t_hi, h_best_theta[b] + dtheta);
        p_lo = std::min(p_lo, h_best_phi[b] - dphi);
        p_hi = std::max(p_hi, h_best_phi[b] + dphi);
    }

    int n_theta_f = 10, n_phi_f = 10;
    int n_fine = n_theta_f * n_phi_f;

    multibeam_search(num_beams, n_fine,
                     t_lo, t_hi, p_lo, p_hi,
                     n_theta_f, n_phi_f);

    // Download all results in one batch
    float h_rho[MAX_BEAMS];
    HIP_CHECK(hipMemcpyAsync(h_best_theta, d_best_theta_,
                              num_beams * sizeof(float),
                              hipMemcpyDeviceToHost, stream_));
    HIP_CHECK(hipMemcpyAsync(h_best_phi, d_best_phi_,
                              num_beams * sizeof(float),
                              hipMemcpyDeviceToHost, stream_));
    HIP_CHECK(hipMemcpyAsync(h_rho, d_best_rho_,
                              num_beams * sizeof(float),
                              hipMemcpyDeviceToHost, stream_));
    HIP_CHECK(hipStreamSynchronize(stream_));

    for (int b = 0; b < num_beams; b++) {
        results[b].theta_opt = h_best_theta[b];
        results[b].phi_opt   = h_best_phi[b];
        results[b].rho_dep   = h_rho[b];
        results[b].delta_rho = 0.05f;
        results[b].eta_cd    = 0.0f;
    }
}

}  // namespace best_rtpc
