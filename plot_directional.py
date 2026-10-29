#!/usr/bin/env python3
"""
Plot the daily modulation of the DM rate from crystal anisotropy.
Input: outputs/directional_summary.json (directional.py -> directional_summary.py).

(a) angle theta_E between the crystal c-axis and the DM wind over a sidereal day at LNGS
(b) signed asymmetry  A = [R(theta_E=0) - R(90 deg)] / [R(0) + R(90 deg)]  vs DM mass
(c) R(LST)/<R> - 1 for a c-axis mounted vertically at LNGS, at a fixed mass

Rates are events above threshold (resolution sigma = E_th/5). theta_E is folded into
[0, 90] deg because R(theta_E) = R(180 - theta_E).
"""
import os, json, sys
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'outputs', 'directional_summary.json')
OUT = os.path.join(HERE, 'outputs', 'directional_modulation.png')

ALPHA_E_H, DELTA_E = 21.2, np.radians(48.0)          # Earth velocity -> Cygnus (equatorial)
LAT = np.radians(42.45)                               # LNGS
M_SHOW = 1.0                                          # MeV, panel (c)
ETH_SHOW = '0.0001'                                   # keV, panel (c)

STYLE = {'CaWO4': '#2a78d6', 'Al2O3': '#eb6834', 'TeO2': '#1baf7a', 'LiNbO3': '#e87ba4', 'Si': '#8a8a85'}
LABEL = {'CaWO4': r'CaWO$_4$', 'Al2O3': r'Al$_2$O$_3$', 'TeO2': r'TeO$_2$', 'LiNbO3': r'LiNbO$_3$', 'Si': 'Si'}
ETH_LS = {'3e-05': ('-', '30 meV'), '0.0001': ('--', '100 meV')}


def theta_E(lst_h, orientation):
    a = np.radians(15.0 * lst_h)
    if orientation == 'vertical':
        dc, ac = LAT, a
    else:                                             # horizontal, pointing east
        dc, ac = 0.0, a + np.pi / 2
    ct = np.sin(dc) * np.sin(DELTA_E) + np.cos(dc) * np.cos(DELTA_E) * np.cos(ac - np.radians(15 * ALPHA_E_H))
    return np.degrees(np.arccos(np.abs(np.clip(ct, -1, 1))))


def R_of_theta(R_grid, tE_grid, tE):
    """Interpolate linearly in cos^2(theta_E): the projected DOS is quadratic in qhat."""
    x = np.cos(np.radians(tE_grid))**2
    o = np.argsort(x)
    return np.interp(np.cos(np.radians(tE))**2, x[o], np.asarray(R_grid)[o])


def main():
    d = json.load(open(SUMMARY))
    m = np.array(d['m_MeV']); tE = np.array(d['theta_E_deg'])
    mats = [k for k in STYLE if k in d]
    lst = np.linspace(0, 24, 289)

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # (a) geometry
    for orient, ls, lab in (('vertical', '-', 'c-axis vertical'), ('east', '--', 'c-axis horizontal (east)')):
        ax[0].plot(lst, theta_E(lst, orient), color='#2f2f2d', ls=ls, lw=2, label=lab)
    ax[0].set(xlabel='local sidereal time [h]', ylabel=r'$\theta_E$ (c-axis vs DM wind) [deg]',
              xlim=(0, 24), ylim=(0, 92), xticks=range(0, 25, 4))
    ax[0].set_title('(a) DM wind direction at LNGS', fontsize=11, loc='left')
    ax[0].legend(frameon=False, fontsize=9, loc='lower left')

    # (b) signed asymmetry vs mass
    for mat in mats:
        for eth, (ls, _) in ETH_LS.items():
            R = np.array(d[mat][eth])
            A = 100 * (R[:, 0] - R[:, -1]) / (R[:, 0] + R[:, -1])
            ax[1].plot(m, A, color=STYLE[mat], ls=ls, lw=2, marker='o' if ls == '-' else None, ms=4,
                       label=LABEL[mat] if ls == '-' else None)
    ax[1].axhline(0, color='#bdbdb8', lw=0.8, zorder=0)
    ax[1].set_xscale('log')
    ax[1].set(xlabel=r'$m_\chi$ [MeV]', ylabel=r'$[R(0^\circ)-R(90^\circ)]/[R(0^\circ)+R(90^\circ)]$ [%]')
    ax[1].set_title(r'(b) anisotropy of the rate above $E_{\rm th}$', fontsize=11, loc='left')
    h1 = ax[1].legend(frameon=False, fontsize=9, loc='lower right')
    ax[1].add_artist(h1)
    from matplotlib.lines import Line2D
    ax[1].legend([Line2D([], [], color='#2f2f2d', ls=ls, lw=2) for ls, _ in ETH_LS.values()],
                 [r'$E_{\rm th}$ = ' + t for _, t in ETH_LS.values()], frameon=False, fontsize=9, loc='upper right')

    # (c) daily modulation
    im = int(np.argmin(np.abs(m - M_SHOW)))
    th = theta_E(lst, 'vertical')
    for mat in mats:
        R = R_of_theta(d[mat][ETH_SHOW][im], tE, th)
        ax[2].plot(lst, 100 * (R / R.mean() - 1), color=STYLE[mat], lw=2, label=LABEL[mat])
    ax[2].axhline(0, color='#bdbdb8', lw=0.8, zorder=0)
    ax[2].set(xlabel='local sidereal time [h]', ylabel=r'$R/\langle R\rangle - 1$ [%]', xlim=(0, 24),
              xticks=range(0, 25, 4))
    ax[2].set_title(rf'(c) daily modulation, $m_\chi$ = {m[im]:g} MeV, $E_{{\rm th}}$ = '
                    f'{float(ETH_SHOW)*1e6:.0f} meV, c vertical', fontsize=11, loc='left')
    ax[2].legend(frameon=False, fontsize=9, ncol=2, loc='upper center')

    for a in ax:
        a.tick_params(direction='in', which='both', top=True, right=True)
        for s in a.spines.values():
            s.set_linewidth(0.8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=200)
    print('saved', OUT)


if __name__ == '__main__':
    main()
