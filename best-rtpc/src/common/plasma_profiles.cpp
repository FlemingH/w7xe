#include "common/plasma_profiles.h"
#include <cmath>
#include <cstdlib>

namespace best_rtpc {

PlasmaProfileGenerator::PlasmaProfileGenerator(float R0, float a, float B0)
    : R0_(R0), a_(a), B0_(B0) {}

float PlasmaProfileGenerator::solovev_psi(float R, float Z) const {
    // Simplified Solovev equilibrium: ψ ∝ (R² - R0²)² / R0² + Z²
    float x = (R - R0_) / a_;
    float z = Z / a_;
    return (x * x + z * z);  // normalized: 0 at axis, 1 at boundary
}

float PlasmaProfileGenerator::solovev_ne(float rho) const {
    // Peaked density profile: ne(ρ) = ne0 * (1 - ρ²)^0.5
    float ne0 = 6.0e19f;  // central density [m^-3]
    float val = 1.0f - rho * rho;
    return ne0 * (val > 0.0f ? std::sqrt(val) : 0.0f);
}

float PlasmaProfileGenerator::solovev_Te(float rho) const {
    // Peaked temperature profile: Te(ρ) = Te0 * (1 - ρ²)^2
    float Te0 = 5.0e3f;  // central Te [eV]
    float val = 1.0f - rho * rho;
    return Te0 * (val > 0.0f ? val * val : 0.0f);
}

void PlasmaProfileGenerator::generate(EquilibriumData& eq, int nr, int nz) {
    eq.nr = nr;
    eq.nz = nz;
    eq.R_min = R0_ - 1.2f * a_;
    eq.R_max = R0_ + 1.2f * a_;
    eq.Z_min = -1.2f * a_;
    eq.Z_max =  1.2f * a_;

    int n = nr * nz;
    eq.psi  = new float[n];
    eq.ne   = new float[n];
    eq.Te   = new float[n];
    eq.BR   = new float[n];
    eq.BZ   = new float[n];
    eq.Bphi = new float[n];

    float dR = (eq.R_max - eq.R_min) / (nr - 1);
    float dZ = (eq.Z_max - eq.Z_min) / (nz - 1);

    eq.psi_axis = 0.0f;
    eq.psi_boundary = 1.0f;
    eq.R_axis = R0_;
    eq.Z_axis = 0.0f;

    for (int j = 0; j < nz; j++) {
        float Z = eq.Z_min + j * dZ;
        for (int i = 0; i < nr; i++) {
            float R = eq.R_min + i * dR;
            int idx = j * nr + i;

            float psi_norm = solovev_psi(R, Z);
            eq.psi[idx] = psi_norm;

            float rho = std::sqrt(std::min(psi_norm, 1.0f));
            eq.ne[idx]  = solovev_ne(rho);
            eq.Te[idx]  = solovev_Te(rho);

            // Simplified magnetic field
            eq.Bphi[idx] = B0_ * R0_ / R;  // 1/R toroidal field

            // Poloidal field from ψ gradient (finite difference, interior only)
            eq.BR[idx] = 0.0f;
            eq.BZ[idx] = 0.0f;
        }
    }

    // Compute poloidal field from ψ gradient (interior points)
    for (int j = 1; j < nz - 1; j++) {
        for (int i = 1; i < nr - 1; i++) {
            int idx = j * nr + i;
            float R = eq.R_min + i * dR;
            // B_R = -(1/R) ∂ψ/∂Z,  B_Z = (1/R) ∂ψ/∂R
            eq.BR[idx] = -(eq.psi[(j+1)*nr+i] - eq.psi[(j-1)*nr+i]) / (2.0f * dZ * R);
            eq.BZ[idx] =  (eq.psi[j*nr+i+1]   - eq.psi[j*nr+i-1])   / (2.0f * dR * R);
        }
    }
}

void PlasmaProfileGenerator::free_profiles(EquilibriumData& eq) {
    delete[] eq.psi;  eq.psi  = nullptr;
    delete[] eq.ne;   eq.ne   = nullptr;
    delete[] eq.Te;   eq.Te   = nullptr;
    delete[] eq.BR;   eq.BR   = nullptr;
    delete[] eq.BZ;   eq.BZ   = nullptr;
    delete[] eq.Bphi; eq.Bphi = nullptr;
}

}  // namespace best_rtpc
