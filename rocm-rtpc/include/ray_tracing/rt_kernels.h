#pragma once

#include <hip/hip_runtime.h>
#include "common/types.h"

namespace rocm_rtpc {

// Main ray tracing kernel: each thread traces one ray (one beam × one angle).
// Solves Hamilton ray equations via RK4 integration.
// Output: deposition location ρ_dep and ECCD efficiency η_cd.
__global__ void ray_trace_kernel(
    const float* __restrict__ psi,
    const float* __restrict__ ne,
    const float* __restrict__ Te,
    const float* __restrict__ Bphi,
    int nr, int nz,
    float R_min, float dR, float Z_min, float dZ,
    float psi_axis, float psi_bnd,
    const float* __restrict__ theta_grid,
    const float* __restrict__ phi_grid,
    int n_angles,
    float launcher_R, float launcher_Z,
    float freq_ghz,
    float* __restrict__ rho_dep,
    float* __restrict__ eta_cd,
    int ode_steps);

// Parallel reduction kernel: find the angle with min |ρ_dep - ρ_target|
__global__ void angle_optimize_kernel(
    const float* __restrict__ rho_dep,
    const float* __restrict__ eta_cd,
    const float* __restrict__ theta_grid,
    const float* __restrict__ phi_grid,
    int n_angles,
    float rho_target,
    float* __restrict__ opt_theta,
    float* __restrict__ opt_phi,
    float* __restrict__ opt_rho);

// Multi-beam ray tracing: blockIdx.y = beam, threadIdx.x + blockIdx.x = angle
// All beams × all angles in a single kernel launch.
__global__ void ray_trace_multibeam_kernel(
    const float* __restrict__ psi,
    const float* __restrict__ ne,
    const float* __restrict__ Te,
    const float* __restrict__ Bphi,
    int nr, int nz,
    float R_min, float dR, float Z_min, float dZ,
    float psi_axis, float psi_bnd,
    const float* __restrict__ theta_grid,
    const float* __restrict__ phi_grid,
    int n_angles,
    float launcher_R, float launcher_Z,
    float freq_ghz,
    float* __restrict__ rho_dep,   // [num_beams × n_angles]
    float* __restrict__ eta_cd,    // [num_beams × n_angles]
    int ode_steps,
    int num_beams);

// Multi-beam angle optimization: one block per beam, reduction within block.
__global__ void angle_optimize_multibeam_kernel(
    const float* __restrict__ rho_dep,     // [num_beams × n_angles]
    const float* __restrict__ eta_cd,
    const float* __restrict__ theta_grid,
    const float* __restrict__ phi_grid,
    int n_angles,
    const float* __restrict__ rho_targets, // [num_beams]
    float* __restrict__ opt_theta,        // [num_beams]
    float* __restrict__ opt_phi,          // [num_beams]
    float* __restrict__ opt_rho,          // [num_beams]
    int num_beams);

}  // namespace rocm_rtpc
