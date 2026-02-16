#include "gpu_efit/efit_kernels.h"
#include <float.h>

namespace best_rtpc {

// ═══════════════════════════════════════════════════════════════
// Kernel 1: Tiled matrix multiplication (Steps 1 & 5)
// Computes C = A × B where A and B are M×M matrices.
// Uses shared memory tiling with TILE_SIZE=32 and +1 padding
// to avoid LDS bank conflicts on RDNA 4 (128 KB LDS per CU).
// ═══════════════════════════════════════════════════════════════
__global__ void eigen_decomp_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ Psi,
    float* __restrict__ Psi_prime,
    int M)
{
    __shared__ float tile_Q[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float tile_Psi[TILE_SIZE][TILE_SIZE + 1];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    int num_tiles = (M + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        int k_Q   = t * TILE_SIZE + threadIdx.x;
        int k_Psi = t * TILE_SIZE + threadIdx.y;

        // Load Q^T tile: tile_Q[ty][tx] = Q^T[row][k_Q] = Q[k_Q][row]
        tile_Q[threadIdx.y][threadIdx.x] =
            (row < M && k_Q < M) ? Q[k_Q * M + row] : 0.0f;

        // Load Psi tile: tile_Psi[ty][tx] = Psi[k_Psi][col]
        tile_Psi[threadIdx.y][threadIdx.x] =
            (k_Psi < M && col < M) ? Psi[k_Psi * M + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tile_Q[threadIdx.y][k] * tile_Psi[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < M) {
        Psi_prime[row * M + col] = sum;
    }
}

// ═══════════════════════════════════════════════════════════════
// Kernel 2: Matrix transpose (Steps 2 & 4)
// Uses shared memory for coalesced reads and writes.
// ═══════════════════════════════════════════════════════════════
__global__ void matrix_transpose_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    int M)
{
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];

    int x_in = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y_in = blockIdx.y * TILE_SIZE + threadIdx.y;

    if (x_in < M && y_in < M) {
        tile[threadIdx.y][threadIdx.x] = in[y_in * M + x_in];
    }
    __syncthreads();

    int x_out = blockIdx.y * TILE_SIZE + threadIdx.x;
    int y_out = blockIdx.x * TILE_SIZE + threadIdx.y;

    if (x_out < M && y_out < M) {
        out[y_out * M + x_out] = tile[threadIdx.x][threadIdx.y];
    }
}

// ═══════════════════════════════════════════════════════════════
// Kernel 3: Parallel prefix-sum tridiagonal solver (Step 3)
// Each block solves one of M independent tridiagonal systems.
// Uses parallel prefix sum: O(M) serial → O(log₂M) parallel.
// On R9700: 127 blocks / 64 CU = ~2 rounds (TITAN X needs ~5).
// ═══════════════════════════════════════════════════════════════
__global__ void tridiag_solve_kernel(
    const float* __restrict__ a_coeff,
    const float* __restrict__ m_coeff,
    const float* __restrict__ rhs,
    float* __restrict__ x,
    int M)
{
    int j = blockIdx.x;
    if (j >= M) return;

    extern __shared__ float sdata[];
    float* s_d = sdata;
    float* s_x = sdata + M;

    int tid = threadIdx.x;

    // Load right-hand side into shared memory (transposed access pattern)
    if (tid < M) {
        s_d[tid] = rhs[tid * M + j];
    }
    __syncthreads();

    // Forward sweep: parallel prefix-sum elimination
    for (int stride = 1; stride < M; stride <<= 1) {
        float val = 0.0f;
        if (tid >= stride && tid < M) {
            val = m_coeff[tid * M + (tid - stride)] * s_d[tid - stride];
        }
        __syncthreads();
        if (tid >= stride && tid < M) {
            s_d[tid] += val;
        }
        __syncthreads();
    }

    // Back substitution
    if (tid == M - 1) {
        s_x[tid] = s_d[tid] * a_coeff[tid];
    }
    __syncthreads();

    for (int stride = M >> 1; stride >= 1; stride >>= 1) {
        float val = 0.0f;
        if (tid < M - stride) {
            val = a_coeff[tid] * (s_d[tid] - s_x[tid + stride]);
        }
        __syncthreads();
        if (tid < M - stride) {
            s_x[tid] = val;
        }
        __syncthreads();
    }

    // Write back to global memory
    if (tid < M) {
        x[tid * M + j] = s_x[tid];
    }
}

// ═══════════════════════════════════════════════════════════════
// Kernel 4: Green function boundary condition
// ψ_boundary[i] = Σ_j G_matrix[i][j] × J_plasma[j]
// O(N³) operation; benefits from Infinity Cache when G < 64MB.
// 4× loop unrolling exploits R9700's 768KB VGPR per CU.
// ═══════════════════════════════════════════════════════════════
__global__ void green_boundary_kernel(
    const float* __restrict__ G_matrix,
    const float* __restrict__ J_plasma,
    float* __restrict__ psi_boundary,
    int N_bnd,
    int N_inner)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_bnd) return;

    float sum = 0.0f;
    const float* G_row = G_matrix + (size_t)i * N_inner;

    // Process groups of 4 for instruction-level parallelism
    int j = 0;
    int N_inner_4 = (N_inner / 4) * 4;
    for (; j < N_inner_4; j += 4) {
        sum += G_row[j]     * J_plasma[j];
        sum += G_row[j + 1] * J_plasma[j + 1];
        sum += G_row[j + 2] * J_plasma[j + 2];
        sum += G_row[j + 3] * J_plasma[j + 3];
    }
    // Remainder
    for (; j < N_inner; j++) {
        sum += G_row[j] * J_plasma[j];
    }

    psi_boundary[i] = sum;
}

// ═══════════════════════════════════════════════════════════════
// Kernel 5: Convergence check
// Parallel reduction to find max |ψ_new - ψ_old|
// ═══════════════════════════════════════════════════════════════
__global__ void convergence_kernel(
    const float* __restrict__ psi_new,
    const float* __restrict__ psi_old,
    float* __restrict__ max_diff,
    int N)
{
    extern __shared__ float smax[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    float local_max = 0.0f;
    if (gid < N) {
        local_max = fabsf(psi_new[gid] - psi_old[gid]);
    }
    smax[tid] = local_max;
    __syncthreads();

    // Tree reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && smax[tid + s] > smax[tid]) {
            smax[tid] = smax[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax((int*)max_diff, __float_as_int(smax[0]));
    }
}

// ═══════════════════════════════════════════════════════════════
// Kernel 6: Compute plasma profiles from ψ
// ═══════════════════════════════════════════════════════════════
__global__ void profiles_from_psi_kernel(
    const float* __restrict__ psi,
    float* __restrict__ ne,
    float* __restrict__ Te,
    float* __restrict__ Bphi,
    int nr, int nz,
    float R_min, float dR, float Z_min, float dZ,
    float R0, float B0, float ne0, float Te0,
    float psi_axis, float psi_bnd)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nr || j >= nz) return;

    int idx = j * nr + i;
    float R = R_min + i * dR;

    // Normalized poloidal flux
    float psi_range = psi_bnd - psi_axis;
    float psi_norm = (psi_range != 0.0f) ?
        (psi[idx] - psi_axis) / psi_range : 0.0f;
    psi_norm = fmaxf(0.0f, fminf(psi_norm, 1.0f));
    float rho = sqrtf(psi_norm);

    // Density: ne(ρ) = ne0 * (1 - ρ²)^0.5
    float val = 1.0f - rho * rho;
    ne[idx] = ne0 * (val > 0.0f ? sqrtf(val) : 0.0f);

    // Temperature: Te(ρ) = Te0 * (1 - ρ²)²
    Te[idx] = Te0 * (val > 0.0f ? val * val : 0.0f);

    // Toroidal field: Bφ = B0 * R0 / R
    Bphi[idx] = (R > 0.1f) ? B0 * R0 / R : B0;
}

}  // namespace best_rtpc
