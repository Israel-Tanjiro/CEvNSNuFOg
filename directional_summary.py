#!/usr/bin/env python3
"""Collapse directional_<MAT>_n<N>.npz into events above threshold R(m, theta_E)
(resolution sigma = 0.2 E_th) for every material -> outputs/directional_summary.json."""
import os, json, numpy as np
import config as C, nufog_core as core
MATS = ['CaWO4', 'Al2O3', 'TeO2', 'LiNbO3', 'Si']
THRESHOLDS_KEV = [3e-5, 1e-4]
RES = 0.2
out = {}
for mat in MATS:
    f = os.path.join(C.OUTPUT_DIR, f"directional_{mat}_n{C.N_MAX}.npz")
    if not os.path.exists(f):
        continue
    d = np.load(f); ev = d['events']; e = d['edges_keV']
    out[mat] = {}
    for Eth in THRESHOLDS_KEV:
        Rm = core.resolution_matrix(e, RES * Eth); keep = e[:-1] >= Eth * (1 - 1e-9)
        tot = np.array([[(ev[i, j] @ Rm)[keep].sum() for j in range(ev.shape[1])] for i in range(ev.shape[0])])
        out[mat][f"{Eth:g}"] = tot.tolist()
    out['m_MeV'] = d['masses_MeV'].tolist(); out['theta_E_deg'] = d['theta_E_deg'].tolist()
path = os.path.join(C.OUTPUT_DIR, 'directional_summary.json')
json.dump(out, open(path, 'w'))
print('wrote', path)
