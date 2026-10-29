#!/usr/bin/env python3
"""
Benchmark of the directional calculation against Stratman & Lin, arXiv:2411.03433 (Al2O3,
massive scalar mediator = contact interaction, as in this pipeline).
Their conventions: v0 = 220, vE = 240, vesc = 500 km/s; no energy resolution;
crystal z (= c) axis parallel to v_E at t = 0, theta_e = 42 deg, so
cos theta_E(t) = cos^2 theta_e + sin^2 theta_e cos(2 pi t / 24h)   (theta_E: 0 -> 84 deg);
A_mod = |R(t=0) - <R>_t| / <R>_t, R = events above E_th.
"""
import os, sys, numpy as np
os.environ.setdefault('NUFOG_MATERIAL', 'Al2O3')     # benchmark material of the paper
import config as C
C.V0_KMS, C.VE_KMS, C.VESC_KMS = 220.0, 240.0, 500.0
import nufog_core as core
import directional as D
D.W_GRID = np.linspace(0.0, C.VESC_KMS + C.VE_KMS, 1200)
D.MASSES_MEV = np.array([1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 70, 100])
D.THETA_E_DEG = np.linspace(0.0, 90.0, 19)
tag = f"directional_{C.MATERIAL}_n{C.N_MAX}"
out = os.path.join(C.OUTPUT_DIR, f"{tag}_SL.npz")
if not os.path.exists(out):
    std = os.path.join(C.OUTPUT_DIR, f"{tag}.npz"); bak = std + ".bak"
    if os.path.exists(std):
        os.replace(std, bak)                     # keep the standard-halo result
    try:
        D.main()
        os.replace(std, out)
    finally:
        if os.path.exists(bak):
            os.replace(bak, std)
d = np.load(out); ev, e, tE, m = d['events'], d['edges_keV'], d['theta_E_deg'], d['masses_MeV']
th_e = np.radians(42.0); t = np.linspace(0, 24, 241)
cE = np.cos(th_e)**2 + np.sin(th_e)**2 * np.cos(2 * np.pi * t / 24)
x = np.cos(np.radians(tE))**2; o = np.argsort(x)
print("E_th[meV]  " + "  ".join(f"{mm:6g}" for mm in m) + "   (m_chi [MeV]; A_mod in %)")
for Eth in (0.12e-3, 0.2e-3, 0.3e-3):
    keep = e[:-1] >= Eth * (1 - 1e-9)
    row = []
    for i in range(len(m)):
        R = ev[i][:, keep].sum(axis=1)
        Rt = np.interp(cE**2, x[o], R[o])
        row.append(100 * abs(Rt[0] - Rt.mean()) / Rt.mean() if Rt.mean() > 0 else np.nan)
    print(f"{Eth*1e6:7.0f}    " + "  ".join(f"{a:6.2f}" for a in row))
