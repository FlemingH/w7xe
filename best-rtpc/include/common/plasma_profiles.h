#pragma once

#include "common/types.h"

namespace best_rtpc {

// Generate synthetic plasma profiles for testing.
// Models a typical BEST H-mode equilibrium with:
//   - Parabolic pressure profile
//   - Peaked current density
//   - Solovev-like analytic equilibrium
class PlasmaProfileGenerator {
public:
    // Major radius [m], minor radius [m], toroidal field on axis [T]
    PlasmaProfileGenerator(float R0 = 1.85f, float a = 0.5f, float B0 = 2.0f);

    // Fill an EquilibriumData struct with analytic profiles on host memory.
    // Caller owns the pointers; call free_profiles() to release.
    void generate(EquilibriumData& eq, int nr, int nz);

    // Free host arrays inside EquilibriumData
    static void free_profiles(EquilibriumData& eq);

private:
    float R0_, a_, B0_;

    // Solovev equilibrium: ψ(R,Z) analytic formula
    float solovev_psi(float R, float Z) const;
    float solovev_ne(float rho_norm) const;
    float solovev_Te(float rho_norm) const;
};

}  // namespace best_rtpc
