#!/usr/bin/env python3
"""
Exposure scan of the 3-sigma discovery cross section: is the reach limited by exposure,
by the neutrino background, or by the neutrino-flux systematics (the 'fog')?

For each mass, sigma_disc(X) is computed for exposures X from 1 g yr to 1000 t yr, together
with the fog index n = -d ln(sigma) / d ln(X):
   n ~ 1   background-free  (sigma ~ 1/X: a few signal events suffice)
   n ~ 1/2 Poisson background-limited (sigma ~ 1/sqrt(X))
   n < 1/2 systematics-limited: inside the neutrino fog
Also reported: neutrino events above threshold and DM signal events at the discovery point.

Usage: NUFOG_MATERIAL=Al2O3 python exposure_scan.py [E_th_eV] [res_frac]
"""
import os, sys, numpy as np
import config as C
import nufog_core as core
from likelihood import AsimovQ0
from step3_discovery import SOURCES, NORMS, REL_SIGMA

E_TH_EV = float(sys.argv[1]) if len(sys.argv) > 1 else 0.03
RES = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
C.E_TH_KEV, C.RESOLUTION_FRAC = E_TH_EV * 1e-3, RES
C.NEUTRINO_DIR = os.path.join(C.OUTPUT_DIR, f"neutrinos_{C.neutrino_tag()}")
EXPOSURES_TYR = np.logspace(-6, 3, 19)
M_SHOW_MEV = [1, 5, 20, 100, 500, 2000]

edges = core.energy_edges()
keep = edges[:-1] >= C.E_TH_KEV * (1 - 1e-9)
A1 = np.array([np.loadtxt(os.path.join(C.NEUTRINO_DIR, f"{C.MATERIAL}_{s}Events.dat")) for s in SOURCES]) * NORMS[:, None]
tag = C.dm_tag()
masses = np.atleast_1d(np.loadtxt(os.path.join(C.OUTPUT_DIR, f"DM_Masses_{tag}.dat")))
dm1 = np.loadtxt(os.path.join(C.OUTPUT_DIR, f"DM_Events_Matrix_{tag}.dat"))
if RES > 0:
    Rm = core.resolution_matrix(edges, RES * C.E_TH_KEV)
    A1, dm1 = A1 @ Rm, dm1 @ Rm
A1, dm1 = A1[:, keep] * 1e3, dm1[:, keep] * 1e3                     # per t yr
idx = [int(np.argmin(np.abs(masses - m))) for m in M_SHOW_MEV]
lo, hi = np.log10(C.SIGMA_SEARCH_CM2[0]), np.log10(C.SIGMA_SEARCH_CM2[1])

print(f"{C.MATERIAL}, E_th = {E_TH_EV*1e3:g} meV, sigma_E = {RES:g} E_th")
print(f"neutrino events above threshold per t yr: {A1.sum():.3e}  "
      f"(pp {A1[SOURCES.index('pp')].sum():.2e}, 7Be {A1[SOURCES.index('Be7861')].sum()+A1[SOURCES.index('Be7384')].sum():.2e}, "
      f"8B {A1[SOURCES.index('B8')].sum():.2e})")
rows = []
for X in EXPOSURES_TYR:
    row = []
    for i in idx:
        s = AsimovQ0(A1 * X, dm1[i] * X, REL_SIGMA).sigma_at(C.DISCOVERY_Q0, lo, hi, tol_dex=2e-3)
        row.append(s)
    rows.append(row)
S = np.array(rows)
out = os.path.join(C.OUTPUT_DIR, f"exposure_scan_{C.MATERIAL}_Eth{E_TH_EV*1e3:g}meV_res{RES:g}.csv")
np.savetxt(out, np.column_stack([EXPOSURES_TYR, S]), delimiter=',', fmt='%.5e',
           header='exposure_tyr,' + ','.join(f"sigma_cm2_m{masses[i]:.3g}MeV" for i in idx), comments='')
print("\n exposure    " + "".join(f"  m={masses[i]:6.3g} MeV       " for i in idx))
print(" [t yr]      " + "".join("  sigma[cm2]  n   N_DM  " for _ in idx))
for k, X in enumerate(EXPOSURES_TYR):
    line = f" {X:8.1e}  "
    for j, i in enumerate(idx):
        s = S[k, j]
        n = -(np.log(S[k + 1, j]) - np.log(S[k - 1, j])) / (np.log(EXPOSURES_TYR[k + 1]) - np.log(EXPOSURES_TYR[k - 1])) if 0 < k < len(EXPOSURES_TYR) - 1 else np.nan
        Ndm = s * dm1[i].sum() * X
        line += f"  {s:9.2e} {n:4.2f} {Ndm:6.1e}"
    print(line)
print(f"saved {out}")
