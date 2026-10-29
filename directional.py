#!/usr/bin/env python3
"""
Directional (daily-modulation) DM signal in anisotropic crystals.

The lattice response depends on the direction of q through the projected phonon DOS,
D_qhat = qhat . D . qhat. The DM velocity distribution seen from the lab is anisotropic
(the "DM wind" along -v_E), so the rate depends on the angle theta_E between v_E and the
crystal c-axis. That angle changes over a sidereal day, giving a daily modulation.

Rate (generalizing the isotropic formula, eta -> 2 g):
  dR/domega = pref * Int q dq Int dOmega_q/(4 pi) 2 g(v*, qhat) S(q, omega; qhat),
  v* = q/(2 m_chi) + omega/q,
  g(w, qhat) = Int d^3v f(v) delta(v.qhat - w)          (Radon transform of the SHM)
             = [exp(-(w + qhat.v_E)^2/v0^2) - exp(-v_esc^2/v0^2)] / (N_esc sqrt(pi) v0)
Averaging 2g over directions gives the isotropic eta(v_min) exactly (checked at start).

S depends on qhat only through the polar angle theta_q from c (uniaxial and cubic crystals;
small azimuthal noise in a tensor is averaged over PHI_S). S(theta) = S(pi - theta) because
D_qhat is quadratic in qhat. Gauss-Legendre nodes in cos(theta_q) are used.

Caveat: the incoherent multiphonon approximation captures anisotropy only through the
per-atom DOS tensors (Debye-Waller factor and multiphonon convolutions), not through phonon
dispersion; at q below a few keV coherent single-phonon calculations predict larger effects.

Output: outputs/directional_<MATERIAL>_n<N_MAX>.npz with binned DM spectra per (mass, theta_E).
"""
import os
import time
import numpy as np
from scipy.special import erf
import config as C
import nufog_core as core

N_COS = 24                                   # GL nodes on [-1, 1] (12 independent S tables)
N_SUB = 48                                   # sub-samples of g inside each node's cos-cell
PHI_S_DEG = [0.0, 30.0, 60.0, 90.0]          # azimuths averaged for S (symmetry noise)
N_PHI_G = 64                                 # azimuths for the velocity-distribution average
THETA_E_DEG = np.linspace(0.0, 90.0, 10)     # rate(theta_E) = rate(180 - theta_E)
MASSES_MEV = np.array([0.5, 1, 2, 5, 10, 20, 35, 50, 70, 100, 150, 250, 500, 1000, 3000])
W_GRID = np.linspace(0.0, C.VESC_KMS + C.VE_KMS, 1200)     # v_min grid [km/s]


def radon_shm(u_kms):
    v0, vesc = C.V0_KMS, C.VESC_KMS
    z = vesc / v0
    Nesc = erf(z) - 2.0 / np.sqrt(np.pi) * z * np.exp(-z**2)
    g = (np.exp(-(u_kms / v0)**2) - np.exp(-z**2)) / (Nesc * np.sqrt(np.pi) * v0)
    return np.where(np.abs(u_kms) < vesc, np.clip(g, 0.0, None), 0.0)          # [s/km]


def g_phi_avg(cos_q, theta_E_deg):
    """Azimuthal average of g(w, qhat) over phi_q, on W_GRID, for v_E at angle theta_E from c."""
    tE = np.radians(theta_E_deg)
    phi = (np.arange(N_PHI_G) + 0.5) * 2.0 * np.pi / N_PHI_G
    sin_q = np.sqrt(max(0.0, 1.0 - cos_q**2))
    proj = C.VE_KMS * (cos_q * np.cos(tE) + sin_q * np.sin(tE) * np.cos(phi))   # qhat . v_E
    return radon_shm(W_GRID[:, None] + proj[None, :]).mean(axis=1)


def g_cell(i, theta_E_deg):
    """Integral of the phi-averaged g over the cos-cell of GL node i (product quadrature:
    S is treated as constant over the cell, g is integrated finely, because g has a kink at
    the escape velocity that GL nodes alone resolve poorly)."""
    x, wgl = np.polynomial.legendre.leggauss(N_COS)
    edges = np.concatenate([[-1.0], -1.0 + np.cumsum(wgl)])
    a, b = edges[i], edges[i + 1]
    xs = a + (np.arange(N_SUB) + 0.5) * (b - a) / N_SUB
    return sum(g_phi_avg(c, theta_E_deg) for c in xs) * (b - a) / N_SUB


def check_isotropic_limit():
    """Direction average of 2 g must equal eta(v_min) for any theta_E."""
    worst = 0.0
    for tE in (0.0, 45.0, 90.0):
        avg = sum(g_cell(i, tE) for i in range(N_COS))
        eta = core.eta_shm(W_GRID)
        m = eta > 1e-3 * eta.max()
        worst = max(worst, np.max(np.abs(avg[m] / eta[m] - 1.0)))
    print(f"  isotropic limit check: max |<2g>/eta - 1| = {worst:.2e}")
    return worst


def S_dm_at(cos_q):
    """DM-weighted S(q, omega) at polar angle theta_q, averaged over PHI_S_DEG."""
    th = np.arccos(cos_q)
    acc = None
    for ph in np.radians(PHI_S_DEG):
        qhat = (np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th))
        r = core.lattice_response(C.N_MAX, qdir=qhat, verbose=False, use_cache=False)
        S = np.tensordot(r['A']**2, r['S'], axes=1)
        acc = S if acc is None else acc + S
    return r, acc / len(PHI_S_DEG)


def main():
    t0 = time.time()
    print(f"{C.MATERIAL}: directional DM spectra, n_max = {C.N_MAX}")
    check_isotropic_limit()
    x, wgl = np.polynomial.legendre.leggauss(N_COS)
    pos = np.where(x > 0)[0]                      # pair (c, -c): S identical
    edges = core.energy_edges()
    q = core.q_grid_eV(); dq = np.gradient(q); w = core.omega_grid_eV()
    rhoT = None
    rate = np.zeros((len(MASSES_MEV), len(THETA_E_DEG), len(w)))
    v_max = (C.VESC_KMS + C.VE_KMS) / core.C_KMS
    for i in pos:
        c = x[i]; j = N_COS - 1 - i                   # mirror node, x[j] = -c
        r, S = S_dm_at(c)
        if rhoT is None:
            rhoT = core.rho_T_eV4(r['density'])
        Gp = {tE: g_cell(i, tE) + g_cell(j, tE) for tE in THETA_E_DEG}   # = 2 * (1/2) * cell integrals
        for im, m_MeV in enumerate(MASSES_MEV):
            m = m_MeV * 1e6
            rows = q <= 2.0 * m * v_max * 1.001
            qr = q[rows]
            vstar_kms = (qr[:, None] / (2.0 * m) + w[None, :] / qr[:, None]) * core.C_KMS
            QS = (qr * dq[rows])[:, None] * S[rows]
            for it, tE in enumerate(THETA_E_DEG):
                G = np.interp(vstar_kms, W_GRID, Gp[tE], right=0.0) * core.C_KMS          # dimensionless
                rate[im, it] += np.einsum('qw,qw->w', QS, G)
        print(f"  cos(theta_q) = {c:.3f} done ({time.time() - t0:.0f}s)")
    rho_chi = C.RHO_CHI_GEV_CM3 * 1e9 / core.EV_PER_INVCM**3
    for im, m_MeV in enumerate(MASSES_MEV):
        m = m_MeV * 1e6; mu = m * core.M_NUCLEON_EV / (m + core.M_NUCLEON_EV)
        pref = (1.0 / rhoT) * (rho_chi / m) * core.EV_PER_INVCM**2 / (4.0 * np.pi * mu**2)
        rate[im] *= pref * 1e3 * core.CONVERSION_FACTOR                                   # / (kg yr keV)
    binned = np.array([[core.integrate_into_bins(w / 1e3, rate[im, it], edges)
                        for it in range(len(THETA_E_DEG))] for im in range(len(MASSES_MEV))])
    out = os.path.join(C.OUTPUT_DIR, f"directional_{C.MATERIAL}_n{C.N_MAX}.npz")
    np.savez(out, masses_MeV=MASSES_MEV, theta_E_deg=THETA_E_DEG, edges_keV=edges, events=binned)
    print(f"  saved {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
