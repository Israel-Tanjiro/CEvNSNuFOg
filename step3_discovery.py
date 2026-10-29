#!/usr/bin/env python3
"""Step 3: 3-sigma discovery cross section sigma(m_chi) in the presence of the neutrino background."""
import os
import time
import numpy as np
import config as C
import nufog_core as core
from likelihood import AsimovQ0

# Order of sources with their flux normalizations [cm^-2 s^-1] and relative uncertainties
SOURCES = ["B8", "hep", "atm", "dsnb3", "O15", "N13", "F17", "pp", "Be7384", "Be7861", "pep"]
NORMS = np.array([5.46e6, 7.98e3, 10.5, 86.0, 2.05e8, 2.78e8, 5.29e6, 5.98e10, 4.84e8, 4.35e9, 1.44e8])
REL_SIGMA = np.array([0.12, 0.30, 0.20, 0.50, 0.17, 0.15, 0.20, 0.006, 0.06, 0.06, 0.01])


def main():
    t0 = time.time()
    edges = core.energy_edges()
    keep = edges[:-1] >= C.E_TH_KEV * (1.0 - 1e-9)
    print(f"  threshold {C.E_TH_KEV*1e3:g} eV -> {keep.sum()} of {C.NBINS} bins "
          f"(first edge {edges[:-1][keep][0]*1e3:.4g} eV)")

    A = np.array([np.loadtxt(os.path.join(C.NEUTRINO_DIR, f"{C.MATERIAL}_{s}Events.dat"))
                  for s in SOURCES]) * NORMS[:, None] * C.EXPOSURE_KG_YR
    tag = C.dm_tag()
    masses = np.atleast_1d(np.loadtxt(os.path.join(C.OUTPUT_DIR, f"DM_Masses_{tag}.dat")))
    dm = np.loadtxt(os.path.join(C.OUTPUT_DIR, f"DM_Events_Matrix_{tag}.dat")) * C.EXPOSURE_KG_YR

    if C.RESOLUTION_FRAC > 0:                       # smear the full spectra, then apply the cut
        sig_keV = C.RESOLUTION_FRAC * C.E_TH_KEV
        R = core.resolution_matrix(edges, sig_keV)
        A, dm = A @ R, dm @ R
        print(f"  energy resolution: sigma = {sig_keV*1e6:.3g} meV (= {C.RESOLUTION_FRAC:g} E_th)")
    else:
        print("  energy resolution: perfect")
    A, dm = A[:, keep], dm[:, keep]
    print(f"  neutrino events above threshold: {A.sum():.4e}  ({C.EXPOSURE_KG_YR/1000:g} t yr)")

    lo, hi = np.log10(C.SIGMA_SEARCH_CM2[0]), np.log10(C.SIGMA_SEARCH_CM2[1])
    sigma = np.full(len(masses), np.nan)
    for i in range(len(masses)):
        sigma[i] = AsimovQ0(A, dm[i], REL_SIGMA).sigma_at(C.DISCOVERY_Q0, lo, hi, tol_dex=C.SIGMA_TOL_DEX)

    run = C.run_tag()
    out = os.path.join(C.OUTPUT_DIR, f"discovery_{run}.csv")
    np.savetxt(out, np.column_stack([masses / 1000.0, sigma, sigma * 1e36]), delimiter=",",
               header="m_chi_GeV,sigma_cm2,sigma_pb", comments='', fmt='%.6e')
    ok = np.isfinite(sigma)
    print(f"  {ok.sum()} / {len(masses)} masses reach q0 = {C.DISCOVERY_Q0:g}; "
          f"lightest: {masses[ok].min():.3g} MeV" if ok.any() else "  no mass reaches q0")
    print(f"  saved discovery_{run}.csv  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
