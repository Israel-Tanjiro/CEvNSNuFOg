#!/usr/bin/env python3
"""
Daily modulation of the DM rate from crystal anisotropy (uses outputs of directional.py).

Geometry: the DM wind comes from the direction of the Earth's motion, v_E toward Cygnus
(alpha_E ~ 21h12m, delta_E ~ +48 deg). For a crystal c-axis with equatorial coordinates
(alpha_c, delta_c), cos(theta_E) = sin d_c sin d_E + cos d_c cos d_E cos(alpha_c - alpha_E).
A lab-fixed axis rotates with the local sidereal time (LST):
  c vertical   : delta_c = latitude, alpha_c = LST
  c horizontal pointing east : delta_c = 0, alpha_c = LST + 6h
At LNGS (42.45 N) a vertical c-axis sweeps theta_E from ~6 deg to ~90 deg each sidereal day,
i.e. almost the whole independent range [0, 90] deg.

Writes outputs/daily_modulation_<MAT>.csv:
  m_MeV, E_th_keV, orientation, R_min, R_max, A = (Rmax-Rmin)/(Rmax+Rmin), then R(LST) on 48 points
(events above threshold per kg yr for sigma_n = 1 cm^2; resolution sigma = RES * E_th applied).
"""
import os, sys, numpy as np
import config as C, nufog_core as core

ALPHA_E, DELTA_E = np.radians(318.0), np.radians(48.0)
LAT = np.radians(42.45)                 # LNGS
LST = np.linspace(0, 24, 48, endpoint=False)
THRESHOLDS_KEV = [3e-5, 1e-4]
RES = 0.2


def theta_E(orientation):
    a = np.radians(LST * 15.0)
    if orientation == 'c_vertical':
        dc, ac = LAT, a
    elif orientation == 'c_east':
        dc, ac = 0.0, a + np.pi / 2
    ct = np.sin(dc) * np.sin(DELTA_E) + np.cos(dc) * np.cos(DELTA_E) * np.cos(ac - ALPHA_E)
    return np.degrees(np.arccos(np.abs(np.clip(ct, -1, 1))))    # fold to [0, 90]


def main(material):
    d = np.load(os.path.join(C.OUTPUT_DIR, f"directional_{material}_n{C.N_MAX}.npz"))
    m, tE, ev, edges = d['masses_MeV'], d['theta_E_deg'], d['events'], d['edges_keV']
    centers = np.sqrt(edges[1:] * edges[:-1])
    rows = []
    for Eth in THRESHOLDS_KEV:
        Rm = core.resolution_matrix(edges, RES * Eth) if RES > 0 else None
        keep = edges[:-1] >= Eth * (1 - 1e-9)
        tot = np.zeros(ev.shape[:2])
        for i in range(ev.shape[0]):
            for j in range(ev.shape[1]):
                s = ev[i, j] if Rm is None else ev[i, j] @ Rm
                tot[i, j] = s[keep].sum()
        for orient in ('c_vertical', 'c_east'):
            th = theta_E(orient)
            for i, mm in enumerate(m):
                R = np.interp(th, tE, tot[i])
                if R.max() <= 0:
                    continue
                A = (R.max() - R.min()) / (R.max() + R.min())
                rows.append([mm, Eth, orient, R.min(), R.max(), A] + list(R))
    out = os.path.join(C.OUTPUT_DIR, f"daily_modulation_{material}.csv")
    with open(out, 'w') as f:
        f.write("m_MeV,E_th_keV,orientation,R_min,R_max,A," + ",".join(f"LST{t:.2f}" for t in LST) + "\n")
        for r in rows:
            f.write(",".join([f"{r[0]:g}", f"{r[1]:g}", r[2]] + [f"{v:.6e}" for v in r[3:]]) + "\n")
    print(f"{material}: wrote {out}")
    for r in rows:
        if r[2] == 'c_vertical':
            print(f"   Eth={r[1]*1e6:5.0f} meV  m={r[0]:7g} MeV  A={100*r[5]:6.2f} %")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else C.MATERIAL)
