#pragma once

#include <hip/hip_runtime.h>
#include "common/types.h"

namespace best_rtpc {

// Step 1 & 5: Tiled matrix multiplication Ψ' = Q^T × Ψ (or Q × X')
// Optimized for RDNA 4: TILE_SIZE=32, 128KB LDS, +1 padding to avoid bank conflicts
__global__ void eigen_decomp_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ Psi,
    float* __restrict__ Psi_prime,
    int M);

// Step 2 & 4: Matrix transpose
__global__ void matrix_transpose_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    int M);

// Step 3: Parallel prefix-sum tridiagonal solver
// Each block solves one independent tridiagonal system
__global__ void tridiag_solve_kernel(
    const float* __restrict__ a_coeff,
    const float* __restrict__ m_coeff,
    const float* __restrict__ rhs,
    float* __restrict__ x,
    int M);

// Green function boundary condition: ψ_bnd[i] = Σ_j G[i][j] * J[j]
__global__ void green_boundary_kernel(
    const float* __restrict__ G_matrix,
    const float* __restrict__ J_plasma,
    float* __restrict__ psi_boundary,
    int N_bnd,
    int N_inner);

// Convergence check: compute max |ψ_new - ψ_old|
__global__ void convergence_kernel(
    const float* __restrict__ psi_new,
    const float* __restrict__ psi_old,
    float* __restrict__ max_diff,
    int N);

// Compute plasma profiles (ne, Te) from ψ using analytic model
__global__ void profiles_from_psi_kernel(
    const float* __restrict__ psi,
    float* __restrict__ ne,
    float* __restrict__ Te,
    float* __restrict__ Bphi,
    int nr, int nz,
    float R_min, float dR, float Z_min, float dZ,
    float R0, float B0, float ne0, float Te0,
    float psi_axis, float psi_bnd);

}  // namespace best_rtpc
