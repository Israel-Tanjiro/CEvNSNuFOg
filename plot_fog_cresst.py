#!/usr/bin/env python3
"""
Thesis figures: neutrino fog for all materials, and comparison with CRESST.

Figure 1 (outputs/fog_all_materials.png)
    3 sigma discovery limits (1 t yr) for the five materials at E_th = 30 meV, 100 meV
    (sigma = E_th/5) and 1 eV.
Figure 2 (outputs/fog_vs_cresst.png)
    (a) Landscape: CRESST-III limits and the CRESST upgrade projection against the
        phonon-threshold neutrino fog of this work (E_th = 30 meV).
    (b) Validation in the CRESST regime: CaWO4, 1 t yr, E_th = 1 eV, this work vs
        Bento et al. (2024), plus the other materials at the same threshold.

Literature curves (literature/):
    CRESST-III_2019_DetA_OHare.txt  published data (github.com/cajohare/NeutrinoFog)
    all others                      digitized from CRESST, Commun. Phys. (2025) Figs. 6-7
                                    with literature/digitize_cresst.py (~10% in sigma)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, 'outputs')
LIT = os.path.join(HERE, 'literature')

MATS = ['CaWO4', 'Al2O3', 'TeO2', 'LiNbO3', 'Si']
COL = {'CaWO4': '#2a78d6', 'Al2O3': '#eb6834', 'TeO2': '#1baf7a', 'LiNbO3': '#e87ba4', 'Si': '#8a8a85'}
LAB = {'CaWO4': r'CaWO$_4$', 'Al2O3': r'Al$_2$O$_3$', 'TeO2': r'TeO$_2$', 'LiNbO3': r'LiNbO$_3$', 'Si': 'Si'}
INK = '#2f2f2d'
PB = 1e36                     # cm^2 -> pb; curves are plotted in pb (left axis)
YL_PB = r'$\sigma_{\chi n}$ [pb]'
YL_CM = r'$\sigma_{\chi n}$ [cm$^2$]'
CASES = [('0.03', True, '30 meV'), ('0.1', True, '100 meV'), ('1', False, '1 eV')]


def discovery(mat, eth, res):
    tag = f"Eth{eth}eV" + ("_res0.2Eth" if res else "")
    f = os.path.join(OUTD, f"discovery_{tag}_1tyr_{mat}_Emin0.001eV_qc_m0.5-10000MeV_n60.csv")
    d = np.genfromtxt(f, delimiter=',', names=True)
    ok = np.isfinite(d['sigma_cm2'])
    return d['m_chi_GeV'][ok], d['sigma_cm2'][ok] * PB


def lit(name):
    d = np.loadtxt(os.path.join(LIT, name + '.txt'))
    return d[:, 0], d[:, 1] * PB


def style(ax, right_label=True):
    """Log axes; left axis in pb, right axis in cm^2 (as in the CRESST figures)."""
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.tick_params(direction='in', which='both', top=True, right=False)
    ax.grid(which='major', color='#e6e6e3', lw=0.6); ax.set_axisbelow(True)
    sec = ax.secondary_yaxis('right', functions=(lambda y: y / PB, lambda y: y * PB))
    sec.tick_params(direction='in', which='both')
    if right_label:
        sec.set_ylabel(YL_CM)
    else:
        sec.set_yticklabels([], minor=False); sec.set_yticklabels([], minor=True)
    return sec


def fig_all_materials():
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.8), sharey=True)
    for ax, (eth, res, lab) in zip(axs, CASES):
        for mat in MATS:
            m, s = discovery(mat, eth, res)
            ax.plot(m, s, color=COL[mat], lw=2, label=LAB[mat])
        style(ax, right_label=(ax is axs[-1]))
        ax.set(xlim=(5e-4, 10), ylim=(1e-47 * PB, 1e-40 * PB), xlabel=r'$m_\chi$ [GeV]')
        ttl = rf'$E_{{\rm th}}$ = {lab}' + (r', $\sigma=E_{\rm th}/5$' if res else '')
        ax.set_title(ttl, fontsize=11, loc='left')
    axs[0].set_ylabel(YL_PB + r'  (3$\sigma$ discovery, 1 t$\cdot$yr)')
    axs[0].legend(frameon=False, fontsize=9, loc='upper right')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTD, 'fog_all_materials.png'), dpi=200)
    plt.close(fig)


def cresst_lines(ax):
    m, s = lit('CRESST-III_2019_DetA_OHare'); ax.plot(m, s, color=INK, lw=2)
    m, s = lit('CRESST-III_2024_SOS'); ax.plot(m, s, color=INK, lw=2, ls='-.')
    m, s = lit('CRESST_upgrade_projection_100xLEE'); ax.plot(m, s, color=INK, lw=1.6, ls='--')
    m, s = lit('Bento2024_CaWO4_fog_boundary')
    ax.fill_between(m, 1e-60 * PB, s, color='#d9d9d6', lw=0, zorder=0)


def fig_cresst():
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.4))
    # (a) landscape
    ax = axs[0]
    cresst_lines(ax)
    for mat in MATS:
        m, s = discovery(mat, '0.03', True)
        ax.plot(m, s, color=COL[mat], lw=2)
    style(ax)
    ax.set(xlim=(5e-4, 5), ylim=(1e-47 * PB, 1e-28 * PB), xlabel=r'$m_\chi$ [GeV]', ylabel=YL_PB)
    ax.set_title(r'(a) CRESST vs. the phonon-threshold fog ($E_{\rm th}$ = 30 meV, 1 t$\cdot$yr)',
                 fontsize=11, loc='left')
    ax.text(0.9, 2e-35 * PB, 'CRESST-III\nexcluded', fontsize=9, color=INK, ha='center')
    ax.text(0.13, 3e-44 * PB, 'CaWO$_4$ fog\n(Bento+ 2024)', fontsize=8.5, color='#6b6b67', ha='center')
    h_exp = [Line2D([], [], color=INK, lw=2), Line2D([], [], color=INK, lw=2, ls='-.'),
             Line2D([], [], color=INK, lw=1.6, ls='--')]
    l_exp = ['CRESST-III 2019 (Det. A)', 'CRESST-III 2024 (SOS)', 'CRESST upgrade (100$\\times$ less LEE)']
    leg = ax.legend(h_exp, l_exp, frameon=False, fontsize=8.5, loc='upper right')
    ax.add_artist(leg)
    ax.legend([Line2D([], [], color=COL[k], lw=2) for k in MATS], [LAB[k] for k in MATS],
              frameon=False, fontsize=8.5, loc='center left', ncol=1, title='this work', title_fontsize=8.5)

    # (b) CRESST regime: E_th = 1 eV, 1 t yr
    ax = axs[1]
    cresst_lines(ax)
    m, s = lit('Bento2024_CaWO4_1tyr_1eV')
    ax.plot(m, s, color=COL['CaWO4'], lw=2.4, ls=':', label=r'CaWO$_4$, Bento+ 2024')
    for mat in MATS:
        m, s = discovery(mat, '1', False)
        ax.plot(m, s, color=COL[mat], lw=2, label=LAB[mat] + (', this work' if mat == 'CaWO4' else ''))
    style(ax)
    ax.set(xlim=(0.05, 5), ylim=(1e-47 * PB, 1e-40 * PB), xlabel=r'$m_\chi$ [GeV]', ylabel=YL_PB)
    ax.set_title(r'(b) CRESST regime: $E_{\rm th}$ = 1 eV, 1 t$\cdot$yr', fontsize=11, loc='left')
    ax.legend(frameon=False, fontsize=8.5, loc='lower center', ncol=3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTD, 'fog_vs_cresst.png'), dpi=200)
    plt.close(fig)


def print_ratio():
    mb, sb = lit('Bento2024_CaWO4_1tyr_1eV')
    m, s = discovery('CaWO4', '1', False)
    print('CaWO4, 1 eV, 1 t yr: this work / Bento+ 2024 (digitized)')
    for mm in (0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 4):
        r = np.exp(np.interp(np.log(mm), np.log(m), np.log(s)) - np.interp(np.log(mm), np.log(mb), np.log(sb)))
        print(f'  m = {mm:5g} GeV  ratio = {r:.2f}')


if __name__ == '__main__':
    fig_all_materials()
    fig_cresst()
    print_ratio()
    print('saved outputs/fog_all_materials.png, outputs/fog_vs_cresst.png')
