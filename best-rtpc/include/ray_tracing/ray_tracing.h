#pragma once

#include "common/types.h"
#include <hip/hip_runtime.h>

namespace best_rtpc {

// GPU Ray Tracing: Hamilton ray equation solver for ECRH launcher optimization.
// Solves the inverse problem: given ρ_target, find optimal (θ, φ).
class GpuRayTracing {
public:
    GpuRayTracing();
    ~GpuRayTracing();

    // Upload equilibrium data from GPU-EFIT (or received via RFM)
    void upload_equilibrium(const EquilibriumData& eq);

    // Compute optimal launcher angles for all active beams.
    // Uses two-stage adaptive search: coarse (100) → fine (100).
    void compute_optimal_angles(const ECRHTarget& target,
                                BeamResult* results);

private:
    // Device equilibrium data
    float* d_psi_;
    float* d_ne_;
    float* d_Te_;
    float* d_Bphi_;
    int    eq_nr_, eq_nz_;
    float  eq_R_min_, eq_R_max_, eq_Z_min_, eq_Z_max_;
    float  psi_axis_, psi_bnd_;
    float  R_axis_, Z_axis_;

    // Device workspace for ray tracing (multi-beam: MAX_BEAMS × n_angles)
    float* d_rho_dep_;       // deposition ρ [MAX_BEAMS × max_angles]
    float* d_eta_cd_;        // ECCD efficiency
    float* d_theta_grid_;    // candidate θ angles [max_angles]
    float* d_phi_grid_;      // candidate φ angles [max_angles]
    float* d_best_theta_;    // best θ per beam [MAX_BEAMS]
    float* d_best_phi_;      // best φ per beam [MAX_BEAMS]
    float* d_best_rho_;      // best ρ_dep per beam [MAX_BEAMS]
    float* d_rho_targets_;   // target ρ per beam [MAX_BEAMS]

    hipStream_t stream_;

    // Launcher geometry (BEST upper-port)
    static constexpr float LAUNCHER_R  = 2.2f;   // radial position [m]
    static constexpr float LAUNCHER_Z  = 0.8f;   // vertical position [m]
    static constexpr float THETA_MIN   = -0.6f;  // poloidal angle range [rad]
    static constexpr float THETA_MAX   =  0.3f;
    static constexpr float PHI_MIN     = -0.3f;  // toroidal angle range [rad]
    static constexpr float PHI_MAX     =  0.3f;

    void generate_angle_grid(float theta_min, float theta_max,
                             float phi_min, float phi_max,
                             int n_theta, int n_phi);

    // Multi-beam parallel search (all beams in one kernel launch)
    void multibeam_search(int num_beams, int n_angles,
                          float theta_lo, float theta_hi,
                          float phi_lo, float phi_hi,
                          int n_theta, int n_phi);
};

}  // namespace best_rtpc
