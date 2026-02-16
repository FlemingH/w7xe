#pragma once

#include <cstdint>
#include <cmath>

namespace best_rtpc {

// Grid resolution presets
enum class GridSize : int {
    G65  = 65,
    G129 = 129,
    G257 = 257
};

// Interior grid dimension (boundary excluded)
inline constexpr int interior_dim(GridSize g) {
    return static_cast<int>(g) - 2;
}

// EFIT equilibrium result: output of GPU-EFIT, input to ray tracing
struct EquilibriumData {
    int nr;              // R grid points
    int nz;              // Z grid points
    float R_min, R_max;  // radial domain [m]
    float Z_min, Z_max;  // vertical domain [m]

    float* psi;          // poloidal flux ψ(R,Z) [nr × nz]
    float* ne;           // electron density ne(R,Z) [nr × nz]
    float* Te;           // electron temperature Te(R,Z) [nr × nz]
    float* BR;           // magnetic field B_R(R,Z) [nr × nz]
    float* BZ;           // magnetic field B_Z(R,Z) [nr × nz]
    float* Bphi;         // toroidal magnetic field Bφ(R,Z) [nr × nz]

    float psi_axis;      // ψ at magnetic axis
    float psi_boundary;  // ψ at plasma boundary
    float R_axis, Z_axis;// magnetic axis position

    // Normalized flux coordinate: ρ = sqrt((ψ - ψ_axis) / (ψ_boundary - ψ_axis))
};

// Ray tracing result per beam
struct BeamResult {
    float theta_opt;     // optimal poloidal launcher angle [rad]
    float phi_opt;       // optimal toroidal launcher angle [rad]
    float rho_dep;       // deposition location (normalized flux)
    float delta_rho;     // deposition width
    float eta_cd;        // ECCD drive efficiency
};

// Target command from PCS to ECRH
struct ECRHTarget {
    int num_beams;       // number of active beams (≤12)
    float rho_target[12];// target deposition ρ per beam
    float P_request[12]; // requested power per beam [MW]
};

// Timing measurement
struct TimingInfo {
    double efit_ms;
    double transfer_ms;
    double raytrace_ms;
    double total_ms;
};

// Constants
constexpr int    MAX_BEAMS          = 12;
constexpr int    MAX_ANGLES_COARSE  = 100;
constexpr int    MAX_ANGLES_FINE    = 100;
constexpr int    ODE_STEPS          = 10000;
constexpr float  FREQ_GHZ           = 140.0f;  // microwave frequency
constexpr float  PI                 = 3.14159265358979323846f;
constexpr int    TILE_SIZE          = 32;

}  // namespace best_rtpc
