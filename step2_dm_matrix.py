#!/usr/bin/env python3
"""Step 2: DM event spectra on the mass grid (events / kg / yr per bin, sigma_n = 1 cm^2)."""
import os
import time
import numpy as np
import config as C
import nufog_core as core


def main():
    t0 = time.time()
    resp = core.lattice_response(C.N_MAX)
    masses = np.logspace(np.log10(C.M_CHI_MIN_MEV), np.log10(C.M_CHI_MAX_MEV), C.N_MASSES)
    edges = core.energy_edges()
    w_keV = resp['w'] / 1e3
    matrix = np.zeros((len(masses), C.NBINS))
    chunk = 20
    for i in range(0, len(masses), chunk):
        rates = core.dm_rate_spectra(resp, masses[i:i + chunk])
        for k, r in enumerate(rates):
            matrix[i + k] = core.integrate_into_bins(w_keV, r, edges)
        print(f"  masses {i + 1}-{min(i + chunk, len(masses))} / {len(masses)}", end='\r', flush=True)
    tag = C.dm_tag()
    np.savetxt(os.path.join(C.OUTPUT_DIR, f"DM_Events_Matrix_{tag}.dat"), matrix, fmt='%.10e')
    np.savetxt(os.path.join(C.OUTPUT_DIR, f"DM_Masses_{tag}.dat"), masses, fmt='%.10e')
    print(f"\n  saved DM_Events_Matrix_{tag}.dat  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
