#pragma once

#include "common/types.h"
#include <hip/hip_runtime.h>

namespace best_rtpc {

// GPU-EFIT: Grad-Shafranov equation solver using the P-EFIT 5-step algorithm.
// Reference: Huang et al., Nuclear Fusion 60 (2020) 076021
class GpuEfit {
public:
    explicit GpuEfit(int grid_size = 129);
    ~GpuEfit();

    // Initialize with precomputed eigenmatrix Q and Green function matrix G.
    // For testing, generates synthetic data internally.
    void initialize();

    // Run one complete equilibrium reconstruction.
    // Takes diagnostic input (J_plasma current density) and iterates Picard loop.
    // Stores result in eq_out (host memory).
    void reconstruct(const float* h_J_plasma, EquilibriumData& eq_out,
                     int max_iterations = 10, float tol = 1e-4f);

    // Access device-side equilibrium data for zero-copy to ray tracing
    float* device_psi()  const { return d_psi_; }
    float* device_ne()   const { return d_ne_; }
    float* device_Te()   const { return d_Te_; }
    float* device_B()    const { return d_Bphi_; }

    int grid_size() const { return N_; }

private:
    int N_;       // grid dimension (65, 129, 257)
    int M_;       // interior dimension = N - 2

    // Device arrays
    float* d_psi_;        // current ψ solution [N×N]
    float* d_psi_new_;    // next iteration ψ
    float* d_psi_rhs_;    // right-hand side for G-S equation
    float* d_Q_;          // eigenmatrix Q [M×M], precomputed
    float* d_Qt_;         // Q^T [M×M]
    float* d_G_matrix_;   // Green function matrix [N_bnd × N_inner]
    float* d_J_plasma_;   // plasma current density [M×M]
    float* d_psi_bnd_;    // boundary ψ values [4*M]
    float* d_work1_;      // workspace [M×M]
    float* d_work2_;      // workspace [M×M]
    float* d_a_coeff_;    // tridiagonal solve coefficients [M]
    float* d_m_coeff_;    // tridiagonal multiplier matrix [M×M]

    // Profile arrays
    float* d_ne_;
    float* d_Te_;
    float* d_BR_;
    float* d_BZ_;
    float* d_Bphi_;

    // Pre-allocated convergence check buffer (avoids per-iteration hipMalloc)
    float* d_conv_max_;
    float  h_conv_max_;   // pinned host staging for async readback

    hipStream_t stream_;

    int N_bnd_;           // boundary points count = 4*(N-1)
    int N_inner_;         // interior points count = M*M

    // G-S solver steps
    void gs_step1_eigen_decomp(float* d_out, const float* d_Q, const float* d_in);
    void gs_step2_transpose(float* d_out, const float* d_in);
    void gs_step3_tridiag_solve(float* d_out, const float* d_rhs);
    void gs_step5_inv_eigen(float* d_out, const float* d_Q, const float* d_in);

    void compute_green_boundary(float* d_psi_bnd, const float* d_J);

    void compute_profiles_from_psi();

    float check_convergence();

    void precompute_eigen_matrix();
    void precompute_green_matrix();
    void precompute_tridiag_coefficients();
};

}  // namespace best_rtpc
