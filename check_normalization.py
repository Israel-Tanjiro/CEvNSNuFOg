#!/usr/bin/env python3
"""
Benchmark the absolute normalization of the pipeline against analytic free-nucleus rates
(Helm form factor, standard SHM, Q_W = N - (1 - 4 sin^2 theta_W) Z). Above ~1 eV the
lattice only broadens the recoil peak, so the integrated rates must agree to ~1-2%.
Run after steps 1 and 2.
"""
import os
import numpy as np
import config as C
import nufog_core as core

E_CHECK_KEV = 1e-3                                         # integrate above 1 eV
AMU = 0.93149410242                                        # GeV
from dynamic_structure_factor import MATERIAL_DATABASE
_atoms = MATERIAL_DATABASE[C.MATERIAL]['atoms']
SPECIES = {a['symbol']: (a['Z'], a['A'], a['mass_amu']) for a in _atoms}
COUNT = {a['symbol']: a['n'] for a in _atoms}                # per cell (overall factor cancels)
M_FORM = sum(COUNT[s] * SPECIES[s][2] for s in SPECIES)
N_PER_KG = 1.0 / (M_FORM * 1.66053907e-27)
YR, HC2 = 3.15576e7, 0.389379e-27                          # s, cm^2 GeV^2
GF = 1.1663787e-5                                          # GeV^-2


def helm2(q_GeV, A):
    return core.helm_ff2(np.atleast_1d(q_GeV * 1e9), A)


def analytic_8B(E_th_GeV):
    d = np.loadtxt(os.path.join(C.DATA_DIR, 'test1P.dat')); d = d[np.argsort(d[:, 0])]
    E = np.linspace(d[0, 0], d[-1, 0], 3000) * 1e-3
    phi = np.interp(E * 1e3, d[:, 0], d[:, 1]) * 1e3 * 5.46e6          # cm^-2 s^-1 GeV^-1
    total = 0.0
    for s, (Z, A, m) in SPECIES.items():
        M = m * AMU; QW = core.weak_charge(Z, A)
        sig = np.zeros_like(E)
        for k, En in enumerate(E):
            Tmax = 2 * En**2 / (M + 2 * En)
            if Tmax > E_th_GeV:
                T = np.logspace(np.log10(E_th_GeV), np.log10(Tmax), 400)
                ds = GF**2 * M / (4 * np.pi) * QW**2 * (1 - M * T / (2 * En**2)) * helm2(np.sqrt(2 * M * T), A) * HC2
                sig[k] = np.trapezoid(ds, T)
        total += np.trapezoid(phi * sig, E) * COUNT[s] * N_PER_KG * YR
    return total


def analytic_dm(m_GeV, E_th_GeV):
    c = 299792.458
    mu_n = m_GeV * AMU / (m_GeV + AMU)
    rho = C.RHO_CHI_GEV_CM3
    total = 0.0
    for s, (Z, A, mm) in SPECIES.items():
        M = mm * AMU; mu = m_GeV * M / (m_GeV + M)
        Emax = 2 * mu**2 * ((C.VESC_KMS + C.VE_KMS) / c)**2 / M
        if Emax <= E_th_GeV:
            continue
        ER = np.logspace(np.log10(E_th_GeV), np.log10(Emax), 3000)
        vmin = np.sqrt(M * ER / (2 * mu**2)) * c
        dR = rho / m_GeV * M * A**2 / (2 * mu_n**2) * core.eta_shm(vmin) * c * helm2(np.sqrt(2 * M * ER), A)
        total += np.trapezoid(dR, ER) * c * 1e5 * COUNT[s] * N_PER_KG * YR      # sigma_n = 1 cm^2
    return total


def main():
    edges = core.energy_edges()
    above = edges[:-1] >= E_CHECK_KEV * (1 - 1e-9)
    print(f"{C.MATERIAL}: benchmark above {E_CHECK_KEV*1e3:g} eV (pipeline / analytic free nucleus; expect 1.00 +- 0.02)")
    f8 = os.path.join(C.NEUTRINO_DIR, f"{C.MATERIAL}_B8Events.dat")
    if os.path.exists(f8):
        pipe = np.loadtxt(f8)[above].sum() * 5.46e6
        ana = analytic_8B(E_CHECK_KEV * 1e-6)
        print(f"  8B CEvNS   : pipeline {pipe:.4e}  analytic {ana:.4e}  /kg/yr   ratio {pipe/ana:.4f}")
    tag = C.dm_tag()
    fm = os.path.join(C.OUTPUT_DIR, f"DM_Events_Matrix_{tag}.dat")
    if os.path.exists(fm):
        M = np.loadtxt(fm); ms = np.loadtxt(os.path.join(C.OUTPUT_DIR, f"DM_Masses_{tag}.dat"))
        for target in (1000.0, 5000.0):
            i = np.argmin(abs(ms - target))
            pipe = M[i, above].sum(); ana = analytic_dm(ms[i] * 1e-3, E_CHECK_KEV * 1e-6)
            print(f"  DM {ms[i]:6.0f} MeV: pipeline {pipe:.4e}  analytic {ana:.4e}  /kg/yr/cm2  ratio {pipe/ana:.4f}")


if __name__ == "__main__":
    main()
