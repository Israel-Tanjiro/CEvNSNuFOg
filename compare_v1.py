#!/usr/bin/env python3
"""
Overlay v1 discovery curves (from ../outputs lambda matrices, q0 = 9 crossing) with the
v2 curves, to document the effect of the normalization and multiphonon fixes.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import config as C

V1 = os.path.join(C.BASE_DIR, '..', 'outputs')
PAIRS = [   # (v1 tag, v2 E_th [keV], v2 n_max, label, color)
    ('CaWO4_Eth0.03eV_1tyr_Emin0.001eV_m0.5-10000MeV_n15', 3e-5, 60, '30 meV, full', '#1f78b4'),
    ('CaWO4_Eth0.03eV_1tyr_Emin0.001eV_m0.5-10000MeV_n1',  3e-5, 1,  '30 meV, n = 1', '#e6550d'),
    ('CaWO4_Eth0.1eV_1tyr_Emin0.1eV_m2-10000MeV_n15',     1e-4, 60, '0.1 eV, full', '#33a02c'),
    ('CaWO4_Eth1eV_1tyr_Emin0.1eV_m2-10000MeV_n15',       1e-3, 60, '1 eV, full', '#6a3d9a'),
]


def v1_curve(tag):
    Z = np.loadtxt(os.path.join(V1, f'lambda_matrix_{tag}.csv'), delimiter=',')
    m = np.loadtxt(os.path.join(V1, f'mass_grid_GeV_{tag}.csv'))
    s = np.loadtxt(os.path.join(V1, f'sigma_grid_cm2_{tag}.csv'))
    out = np.full(len(m), np.nan)
    for i, z in enumerate(Z):
        j = np.argmax(z >= C.DISCOVERY_Q0)
        if j > 0 and z[j] >= C.DISCOVERY_Q0:
            f = (C.DISCOVERY_Q0 - z[j - 1]) / (z[j] - z[j - 1])
            out[i] = 10**(np.log10(s[j - 1]) + f * np.log10(s[j] / s[j - 1]))
    return m, out


def main():
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(9, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    for tag, e_th, n_max, label, color in PAIRS:
        f2 = os.path.join(C.OUTPUT_DIR, f"discovery_{C.run_tag(e_th_keV=e_th, n_max=n_max, qdir='z', res_frac=0.0)}.csv")
        if not (os.path.exists(f2) and os.path.exists(os.path.join(V1, f'lambda_matrix_{tag}.csv'))):
            print(f"  skipping {label}")
            continue
        m1, s1 = v1_curve(tag)
        d2 = np.loadtxt(f2, delimiter=',', skiprows=1)
        ax.plot(m1, s1 * 1e36, color=color, ls='--', lw=1.8)
        ax.plot(d2[:, 0], d2[:, 2], color=color, lw=2.2, label=label)
        s2 = np.interp(np.log(m1), np.log(d2[:, 0]), np.log(d2[:, 1]))
        axr.plot(m1, np.exp(s2) / s1, color=color, lw=2)
    ax.plot([], [], 'k--', label='v1 (dashed)'); ax.plot([], [], 'k-', label='v2 (solid)')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(1e-10, 1e4)
    ax.set_ylabel(r'$\sigma_{n\chi}$ [pb]', fontsize=16); ax.legend(fontsize=11)
    axr.set_xscale('log'); axr.set_ylim(0, 1.5); axr.axhline(1, color='gray', lw=1)
    axr.set_ylabel('v2 / v1', fontsize=14); axr.set_xlabel(r'$m_\chi$ [GeV]', fontsize=16)
    fig.savefig(os.path.join(C.OUTPUT_DIR, 'compare_v1_v2.png'), dpi=300, bbox_inches='tight')
    print("  saved compare_v1_v2.png")


if __name__ == "__main__":
    main()
