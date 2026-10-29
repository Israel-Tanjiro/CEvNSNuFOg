#!/usr/bin/env python3
"""Validation of directional.py: theta_E-averaged rate vs the isotropic pipeline (q-avg S),
and the Si null test (cubic crystal -> no anisotropy from the DOS tensor)."""
import os, numpy as np
import config as C, nufog_core as core
_trapz = getattr(np, "trapezoid", None) or np.trapz      # NumPy >= 2 renamed trapz
d = np.load(os.path.join(C.OUTPUT_DIR, f"directional_{C.MATERIAL}_n{C.N_MAX}.npz"))
m, tE, ev, edges = d['masses_MeV'], d['theta_E_deg'], d['events'], d['edges_keV']
# isotropic: average over theta_E with weight sin(theta_E) (= average over v_E directions)
x = np.cos(np.radians(tE)); avg = -_trapz(ev, x, axis=1)        # int_0^1 dcos
resp = core.lattice_response(C.N_MAX, qdir='avg', verbose=False, use_cache=False)
iso = core.dm_rate_spectra(resp, m)
w = core.omega_grid_eV() / 1e3
for k, mm in enumerate(m):
    b = core.integrate_into_bins(w, iso[k], edges)
    keep = edges[:-1] >= 3e-5
    print(f"  m = {mm:7g} MeV  total(>30 meV): <dir>_vE / iso = {avg[k][keep].sum()/max(b[keep].sum(),1e-300):.4f}")
