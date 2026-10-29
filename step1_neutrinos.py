#!/usr/bin/env python3
"""Step 1: CEvNS event spectra (events / kg / yr per bin) for every neutrino source."""
import os
import time
import numpy as np
import config as C
import nufog_core as core

# name: (file, E_max [MeV], flux normalization [cm^-2 s^-1], n_E, included in total)
CONTINUOUS = {
    'B8':    ('test1P.dat',          16.360, 5.46e6,     250, True),
    'hep':   ('hep.dat',             18.784, 7.98e3,     250, True),
    'atm':   ('Atm-10002.dat',      981.75,  10.5,       500, True),
    'dsnb3': ('DSNB-10002.dat',      91.200, 86.0,       350, True),
    'O15':   ('O15.dat',              1.732, 2.05e8,     200, True),
    'N13':   ('N13.dat',              1.199, 2.78e8,     200, True),
    'F17':   ('F17.dat',              1.740, 5.29e6,     200, True),
    'pp':    ('pp.dat',               0.423, 5.98e10,    150, True),
    'dsnb5': ('Pruebadsnb5.dat',     57.0,   1.0592e-12, 250, False),
    'dsnb8': ('normalizeddsnb8.dat', 83.0,   1.0592e-12, 250, False),
}
LINES = {   # name: (E_nu [MeV], flux [cm^-2 s^-1], included in total)
    'Be7384': (0.3843, 4.84e8, True),
    'Be7861': (0.8613, 4.35e9, True),
    'pep':    (1.440,  1.44e8, True),
}


def load_flux(fname):
    d = np.loadtxt(os.path.join(C.DATA_DIR, fname))
    o = np.argsort(d[:, 0])
    E, s = d[o, 0], d[o, 1]
    return (lambda x: np.interp(x, E, s, left=0.0, right=0.0)), max(1e-4, E[0])


def main():
    t0 = time.time()
    resp = core.lattice_response(C.N_MAX_NEUTRINO)
    S_nu = core.neutrino_S(resp)
    w_keV = resp['w'] / 1e3
    edges = core.energy_edges()
    total = np.zeros(C.NBINS)
    out = C.NEUTRINO_DIR
    for name, (fname, Emax, norm, nE, in_total) in CONTINUOUS.items():
        path = os.path.join(C.DATA_DIR, fname)
        if not os.path.exists(path):
            print(f"  skipping {name}: {fname} not found")
            continue
        flux, Emin = load_flux(fname)
        ev = core.integrate_into_bins(w_keV, core.cevns_rate_continuous(resp, S_nu, flux, Emin, Emax, nE), edges)
        np.savetxt(os.path.join(out, f"{C.MATERIAL}_{name}Events.dat"), ev, fmt='%.10e')
        if in_total:
            total += ev * norm
        print(f"  {name:7s}: {np.sum(ev) * norm:.4e} events / kg / yr")
    for name, (E, norm, in_total) in LINES.items():
        ev = core.integrate_into_bins(w_keV, core.cevns_rate_line(resp, S_nu, E), edges)
        np.savetxt(os.path.join(out, f"{C.MATERIAL}_{name}Events.dat"), ev, fmt='%.10e')
        if in_total:
            total += ev * norm
        print(f"  {name:7s}: {np.sum(ev) * norm:.4e} events / kg / yr")
    np.savetxt(os.path.join(out, f"{C.MATERIAL}_TotalEvents.dat"), total, fmt='%.10e')
    print(f"  TOTAL  : {total.sum():.4e} events / kg / yr  (spectra in {out}, {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
