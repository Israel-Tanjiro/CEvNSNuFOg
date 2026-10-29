"""
NuFog v2 core: vectorized lattice response S(q, omega), DM and CEvNS rates.

Physics is identical to v1 (incoherent multiphonon expansion below a per-atom
momentum threshold, impulse approximation above it, Helm form factor) except for
two normalization fixes, both verified against analytic free-nucleus rates:
  * CEvNS: prefactor G_F^2 / (8 pi^2) with Q_W = N - (1 - 4 sin^2 theta_W) Z
           (v1 used G_F^2/(4 pi) with Q_W/2 -> rates pi/2 too large)
  * DM:    standard SHM eta(v_min)  (v1 was low by sqrt(pi)/2)
"""
import os
import hashlib
import numpy as np
from scipy.special import erf, gammaln

import config as C
from dynamic_structure_factor import MATERIAL_DATABASE

# ---------------- constants (same values as v1) ----------------
AMU_TO_EV = 931.49410242e6
HBARC_EV_A = 1973.269804
HC_EV_FM = 197326980.4
EV_PER_INVCM = 50677.309               # 1 cm = 50677.309 eV^-1
CONVERSION_FACTOR = 2.6891e58          # natural units -> events/(kg yr)
FLUX_CONV = 2.5624e-25                 # cm^-2 s^-1 MeV^-1 -> natural units
GF_EV2 = 1.1663787e-23
SIN2_THETA_W = 0.23857
M_NUCLEON_EV = 931.49410242e6
C_KMS = 299792.458

OMEGA_OUT_MAX_EV = 3.0                 # multiphonon T_n tabulated up to here (v1: 0.6 eV)


# ---------------- grids ----------------
def energy_edges():
    return np.logspace(np.log10(C.ER_MIN_KEV), np.log10(C.ER_MAX_KEV), C.NBINS + 1)


def omega_grid_eV():
    ph = np.linspace(C.ER_MIN_KEV * 1e6, C.OMEGA_PHONON_MAX_MEV, C.N_OMEGA_PHONON, endpoint=False)
    im = np.logspace(np.log10(C.OMEGA_PHONON_MAX_MEV), np.log10(C.ER_MAX_KEV * 1e6), C.N_OMEGA_IMPULSE)
    return np.concatenate([ph, im]) / 1000.0          # meV -> eV


def q_grid_eV():
    n = int(round(np.log10(C.Q_MAX_KEV / C.Q_MIN_KEV) * C.Q_PER_DECADE)) + 1
    return np.logspace(np.log10(C.Q_MIN_KEV * 1e3), np.log10(C.Q_MAX_KEV * 1e3), n)


# ---------------- material ----------------
def _qvec(qdir):
    if isinstance(qdir, str):
        if qdir == 'avg':
            return None
        return {'x': np.array([1., 0., 0.]), 'y': np.array([0., 1., 0.]), 'z': np.array([0., 0., 1.])}[qdir]
    v = np.asarray(qdir, float)
    return v / np.linalg.norm(v)


def load_material(material, qdir):
    """Atoms, projected phonon DOS D_q(omega) and symmetry groups of identical atoms."""
    info = MATERIAL_DATABASE[material]
    data = np.load(os.path.join(C.TENSOR_DIR, C.TENSOR_FILES[material]), allow_pickle=True)
    omega = np.asarray(data['omega'], float)
    ids = sorted(int(k.split('_')[1]) for k in data.files
                 if k.startswith('atom_') and k.split('_')[1].isdigit())
    D = np.array([data[f'atom_{i}'] for i in ids])                     # (N, 3, 3, n_w)
    v = _qvec(qdir)
    Dq = np.trace(D, axis1=1, axis2=2) / 3.0 if v is None else np.einsum('i,aijk,j->ak', v, D, v)

    sym, mass, A, Z = [], [], [], []
    for at in info['atoms']:                                           # same order as v1
        for _ in range(at['n']):
            sym.append(at['symbol']); mass.append(at['mass_amu'] * AMU_TO_EV)
            A.append(at['A']); Z.append(at['Z'])
    sym, mass, A, Z = sym[:len(ids)], np.array(mass[:len(ids)]), np.array(A[:len(ids)]), np.array(Z[:len(ids)])

    # group atoms with identical species and identical projected DOS (crystal symmetry)
    groups = []
    for a in range(len(ids)):
        for g in groups:
            b = g['rep']
            if sym[a] == sym[b] and np.allclose(Dq[a], Dq[b], rtol=1e-6,
                                                atol=1e-9 * np.max(np.abs(Dq[b]))):
                g['count'] += 1
                break
        else:
            groups.append({'rep': a, 'count': 1})
    out = []
    for g in groups:
        a = g['rep']
        out.append(dict(symbol=sym[a], mass_eV=mass[a], A=A[a], Z=Z[a], count=g['count'], Dq=Dq[a]))
    norm_factor = (2.0 * np.pi) / info['V_prim_A3'] * HBARC_EV_A**3
    return dict(info=info, omega=omega, groups=out, norm_factor=norm_factor)


def helm_ff2(q_eV, A):
    q_fm = q_eV / HC_EV_FM
    s, a = 0.9, 0.52
    c = 1.23 * A**(1.0 / 3.0) - 0.60
    Rn = np.sqrt(c**2 + (7.0 / 3.0) * np.pi**2 * a**2 - 5.0 * s**2)
    x = q_fm * Rn
    out = np.ones_like(q_eV)
    m = (q_eV >= 1e-6) & (x >= 1e-4)
    xm = x[m]
    j1 = (np.sin(xm) - xm * np.cos(xm)) / xm**2
    out[m] = (3.0 * j1 / xm * np.exp(-0.5 * (q_fm[m] * s)**2))**2
    return out


def regularize_ir(omega, Dq, cutoff_eV):
    """
    Replace the Gaussian-smeared IR tail (finite D at omega -> 0, which makes D/omega
    diverge) by the acoustic Debye form D ~ omega^2 below cutoff_eV, matched continuously,
    keeping the normalization of D unchanged.
    """
    D = Dq.copy()
    if cutoff_eV <= 0:
        return D
    norm = np.trapezoid(Dq, omega)
    Dc = np.interp(cutoff_eV, omega, Dq)
    low = omega < cutoff_eV
    D[low] = Dc * (omega[low] / cutoff_eV)**2
    return D * norm / np.trapezoid(D, omega)


def multiphonon_Tn(omega, Dq, n_max):
    """
    T_1 = D/omega and T_n = T_1 * T_{n-1} by FFT convolution on a uniform grid with the
    tensor's own resolution, from 0 to OMEGA_OUT_MAX_EV. <1/omega> (Debye-Waller) and
    <omega> are computed from the same T_1, so sum_n S_n obeys the f-sum rule.
    """
    dw = omega[1] - omega[0]
    w_f = np.arange(0.0, OMEGA_OUT_MAX_EV + dw, dw)
    D = regularize_ir(omega, Dq, C.DOS_IR_CUTOFF_MEV * 1e-3)
    D_f = np.interp(w_f, omega, D, left=0.0, right=0.0)
    D_f[w_f < omega[0]] = 0.0
    T1 = np.zeros_like(w_f)
    T1[1:] = D_f[1:] / w_f[1:]
    N = len(w_f)
    L = 1 << int(np.ceil(np.log2(2 * N)))
    F1 = np.fft.rfft(T1, L)
    Tn = np.zeros((n_max, N))
    Tn[0] = T1
    for n in range(1, n_max):
        Tn[n] = np.clip(np.fft.irfft(F1 * np.fft.rfft(Tn[n - 1], L), L)[:N] * dw, 0.0, None)
    omega_inv = np.trapezoid(T1, w_f)                             # <1/omega>
    w_bar = np.trapezoid(w_f * D_f, w_f)                          # <omega>
    return w_f, Tn, omega_inv, w_bar


# ---------------- S(q, omega) per atom group ----------------
def _grid_hash(*arrays):
    h = hashlib.sha1()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:10]


def lattice_response(n_max, qdir=None, material=None, verbose=True, use_cache=True):
    """
    Unweighted structure factors S_s(q, omega) [eV^-2] summed over the atoms of each species,
    already multiplied by Helm form factor and 2 pi / V.
    The coupling (A^2 for DM, Q_W^2 for CEvNS) is applied later.
    """
    qdir = C.Q_DIR if qdir is None else qdir
    material = C.MATERIAL if material is None else material
    q, w = q_grid_eV(), omega_grid_eV()
    tensor_path = os.path.join(C.TENSOR_DIR, C.TENSOR_FILES[material])
    with open(tensor_path, 'rb') as fh:                       # cache is tied to the tensor contents
        tensor_id = hashlib.sha1(fh.read()).hexdigest()[:8]
    key = (f"S_{material}_{C.dir_tag(qdir)}_n{n_max}_ir{C.DOS_IR_CUTOFF_MEV:g}_T{tensor_id}"
           f"_{_grid_hash(q, w, np.array([OMEGA_OUT_MAX_EV]))}.npz")
    path = os.path.join(C.CACHE_DIR, key)
    if use_cache and os.path.exists(path):
        d = np.load(path)
        if verbose:
            print(f"  loaded cached S(q,w): {key}")
        return dict(q=d['q'], w=d['w'], S=d['S'], A=d['A'], Z=d['Z'],
                    symbols=list(d['symbols']), density=cell_density_g_cm3(material))

    mat = load_material(material, qdir)
    if verbose:
        desc = ", ".join(f"{g['symbol']}x{g['count']}" for g in mat['groups'])
        print(f"  building S(q,w): {len(q)} q x {len(w)} w, groups [{desc}], n_max={n_max}")
    dq = np.gradient(q)
    dw = np.gradient(w)
    mp_mask = w <= OMEGA_OUT_MAX_EV
    n = np.arange(1, n_max + 1)
    species = []
    for g in mat['groups']:
        if g['symbol'] not in species:
            species.append(g['symbol'])
    S_all = np.zeros((len(species), len(q), len(w)))
    sp_A = np.zeros(len(species)); sp_Z = np.zeros(len(species))

    for g in mat['groups']:
        ig = species.index(g['symbol'])
        sp_A[ig], sp_Z[ig] = g['A'], g['Z']
        m = g['mass_eV']
        w_out, Tn, omega_inv, w_bar = multiphonon_Tn(mat['omega'], g['Dq'], n_max)
        q_thr = 4.0 * np.sqrt(2.0 * m * (1.0 / omega_inv if omega_inv > 0 else 0.0))
        S = np.zeros((len(q), len(w)))

        imp = q > q_thr                                    # impulse approximation
        if np.any(imp):
            qi, dqi = q[imp], dq[imp]
            center = qi**2 / (2.0 * m)
            var = (qi**2 * w_bar / (2.0 * m) + (0.5 * qi * dqi / m)**2)[:, None] + (0.5 * dw)[None, :]**2
            S[imp] = np.exp(-0.5 * (w[None, :] - center[:, None])**2 / var) / np.sqrt(2.0 * np.pi * var)

        mp = ~imp                                          # multiphonon expansion
        if np.any(mp):
            qm = q[mp]
            scale = qm**2 / (2.0 * m)
            DW = np.exp(-2.0 * (qm**2 / (4.0 * m)) * omega_inv)
            pref = np.exp(np.outer(np.log(scale), n) - gammaln(n + 1.0)[None, :])
            pref[~(np.isfinite(pref) & (pref > 1e-30))] = 0.0
            Tn_w = np.zeros((n_max, len(w)))
            for k in range(n_max):
                Tn_w[k, mp_mask] = np.interp(w[mp_mask], w_out, Tn[k], left=0.0, right=0.0)
            S[mp] = DW[:, None] * (pref @ Tn_w)

        S_all[ig] += S * (g['count'] * mat['norm_factor']) * helm_ff2(q, g['A'])[:, None]

    out = dict(q=q, w=w, S=S_all, A=sp_A, Z=sp_Z, symbols=species,
               density=cell_density_g_cm3(material))
    if use_cache:
        np.savez(path, q=q, w=w, S=S_all, A=out['A'], Z=out['Z'],
                 symbols=np.array(out['symbols']), density=out['density'])
    return out


def cell_density_g_cm3(material):
    """
    Density implied by the cell used in S(q, omega): (sum of atomic masses) / V_cell.
    The rate per unit mass contains (2 pi / V_cell) / rho_T = 2 pi / M_cell, so using this
    value makes the result independent of V_cell and of any tabulated density.
    """
    info = MATERIAL_DATABASE[material]
    m_amu = sum(a['n'] * a['mass_amu'] for a in info['atoms'])
    return m_amu * 1.66053907e-24 / (info['V_prim_A3'] * 1e-24)


def rho_T_eV4(density_g_cm3):
    return density_g_cm3 * 5.60958e32 / EV_PER_INVCM**3


# ---------------- binning ----------------
def integrate_into_bins(w_keV, dR_dw, edges, n_sub=40):
    """Integrate dR/domega [per keV] over each recoil bin (same scheme as v1, vectorized)."""
    a = np.maximum(edges[:-1], w_keV[0])
    b = np.minimum(edges[1:], w_keV[-1])
    ok = b > a
    out = np.zeros(len(edges) - 1)
    if not np.any(ok):
        return out
    t = np.linspace(0.0, 1.0, n_sub)
    E = np.exp(np.log(a[ok])[:, None] + t[None, :] * (np.log(b[ok]) - np.log(a[ok]))[:, None])
    r = np.interp(E.ravel(), w_keV, dR_dw, left=0.0, right=0.0).reshape(E.shape)
    out[ok] = np.trapezoid(r, E, axis=1)
    return out


def resolution_matrix(edges, sigma_keV, n_sub=8):
    """
    Gaussian energy resolution as a bin-migration matrix: R[i, j] is the probability that an
    event deposited in true bin i is reconstructed in bin j (constant sigma, as CRESST's
    sigma_0 at zero energy). Each true bin is represented by n_sub log-spaced points.
    """
    lo, hi = edges[:-1], edges[1:]
    t = (np.arange(n_sub) + 0.5) / n_sub
    E = np.exp(np.log(lo)[:, None] + t[None, :] * np.log(hi / lo)[:, None])        # (B, n_sub)
    s = np.sqrt(2.0) * sigma_keV
    up = erf((hi[None, None, :] - E[:, :, None]) / s)
    dn = erf((lo[None, None, :] - E[:, :, None]) / s)
    return 0.5 * (up - dn).mean(axis=1)                                          # (B, B)


# ---------------- dark matter ----------------
def eta_shm(vmin_kms):
    """Standard Halo Model mean inverse speed eta(v_min) [s/km]."""
    v0, ve, vesc = C.V0_KMS, C.VE_KMS, C.VESC_KMS
    z = vesc / v0
    Nesc = erf(z) - 2.0 / np.sqrt(np.pi) * z * np.exp(-z**2)
    pre = 1.0 / (2.0 * Nesc * ve)
    ez = np.exp(-z**2)
    x = np.asarray(vmin_kms, float)
    out = np.zeros_like(x)
    m1 = x < (vesc - ve)
    m2 = (x >= vesc - ve) & (x < vesc + ve)
    out[m1] = pre * (erf((x[m1] + ve) / v0) - erf((x[m1] - ve) / v0) - 4.0 * ve / (np.sqrt(np.pi) * v0) * ez)
    out[m2] = pre * (erf(z) - erf((x[m2] - ve) / v0) - 2.0 * (vesc + ve - x[m2]) / (np.sqrt(np.pi) * v0) * ez)
    return np.clip(out, 0.0, None)


def dm_rate_spectra(resp, masses_MeV):
    """dR/domega [events / (kg yr keV)] for sigma_n = 1 cm^2, one row per mass."""
    q, w = resp['q'], resp['w']
    dq = np.gradient(q)
    S_dm = np.tensordot(resp['A']**2, resp['S'], axes=1)            # coherent A^2
    rhoT = rho_T_eV4(resp['density'])
    rho_chi = C.RHO_CHI_GEV_CM3 * 1e9 / EV_PER_INVCM**3
    sigma_eV2 = 1.0 * EV_PER_INVCM**2
    v_max = (C.VESC_KMS + C.VE_KMS) / C_KMS
    out = np.zeros((len(masses_MeV), len(w)))
    for i, m_MeV in enumerate(masses_MeV):
        m = m_MeV * 1e6
        mu = m * M_NUCLEON_EV / (m + M_NUCLEON_EV)
        rows = q <= 2.0 * m * v_max * 1.001
        if not np.any(rows):
            continue
        qr = q[rows]
        vstar = qr[:, None] / (2.0 * m) + w[None, :] / qr[:, None]
        eta_c = eta_shm(vstar * C_KMS) * C_KMS
        integral = np.einsum('q,qw->w', qr * dq[rows], eta_c * S_dm[rows])
        pref = (1.0 / rhoT) * (rho_chi / m) * sigma_eV2 / (4.0 * np.pi * mu**2)
        out[i] = pref * integral * 1e3 * CONVERSION_FACTOR
    return out


# ---------------- CEvNS ----------------
def weak_charge(Z, A):
    return (A - Z) - (1.0 - 4.0 * SIN2_THETA_W) * Z


def neutrino_S(resp):
    S = np.tensordot(weak_charge(resp['Z'], resp['A'])**2, resp['S'], axes=1)
    S = S.copy()
    S[resp['q'][:, None] < resp['w'][None, :]] = 0.0               # omega <= q (as v1)
    return S


def cevns_prefactor(resp):
    return (FLUX_CONV / rho_T_eV4(resp['density'])) * GF_EV2**2 / (8.0 * np.pi**2) * 1e3 * CONVERSION_FACTOR


def cevns_rate_continuous(resp, S_nu, flux_interp, E_min_MeV, E_max_MeV, n_E):
    """dR/domega [events/(kg yr keV)] per unit flux normalization."""
    q = resp['q']; dq = np.gradient(q)
    E = np.logspace(np.log10(E_min_MeV), np.log10(E_max_MeV), n_E)
    phi_dE = flux_interp(E) * np.gradient(E)
    E_eV = E * 1e6
    kin = 1.0 - q[None, :]**2 / (4.0 * E_eV[:, None]**2)
    kin[q[None, :] > 2.0 * E_eV[:, None]] = 0.0
    Wq = q * dq * (phi_dE @ kin)
    return cevns_prefactor(resp) * (Wq @ S_nu)


def cevns_rate_line(resp, S_nu, E_line_MeV):
    q = resp['q']; dq = np.gradient(q)
    E_eV = E_line_MeV * 1e6
    kin = np.where(q <= 2.0 * E_eV, 1.0 - q**2 / (4.0 * E_eV**2), 0.0)
    return cevns_prefactor(resp) * ((q * dq * kin) @ S_nu)
