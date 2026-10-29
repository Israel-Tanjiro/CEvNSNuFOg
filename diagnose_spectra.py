#!/usr/bin/env python3
"""
Diagnostic for the bump of the single-phonon (n = 1) discovery curve near 70 MeV.

Top: event spectra at a few masses. The neutrino background (total, pp, 7Be) is compared
with the DM signal from the n = 1 run and from the full expansion, each scaled to its own
3-sigma discovery cross section.
Bottom: n = 1 / full rate above threshold, and the mean deposited energy, versus mass.

Note: the "n = 1" run keeps the impulse branch (q > 4 sqrt(2 m omega_0)). Above ~40 MeV
its signal is therefore single-phonon plus free-nucleus-like recoils, with n >= 2 missing.
"""
import os
import numpy as np
import config as C
import nufog_core as core
from step3_discovery import SOURCES, NORMS

E_TH_KEV = 3e-5
MASSES_MEV = [5.0, 20.0, 70.0, 150.0]
OMEGA_MAX_EV = 0.11                 # top of the CaWO4 one-phonon band
NPZ = os.path.join(C.OUTPUT_DIR, 'diagnostic_spectra.npz')


def compute():
    edges = core.energy_edges()
    keep = edges[:-1] >= E_TH_KEV * (1 - 1e-9)
    ndir = os.path.join(C.OUTPUT_DIR, f"neutrinos_{C.neutrino_tag()}")
    A = np.array([np.loadtxt(os.path.join(ndir, f"{C.MATERIAL}_{s}Events.dat")) for s in SOURCES])
    A = A * NORMS[:, None] * C.EXPOSURE_KG_YR
    out = dict(lo=edges[:-1], hi=edges[1:], keep=keep, A=A, sources=np.array(SOURCES))
    for n in (1, 60):
        tag = C.dm_tag(n_max=n)
        out[f'dm{n}'] = np.loadtxt(os.path.join(C.OUTPUT_DIR, f"DM_Events_Matrix_{tag}.dat")) * C.EXPOSURE_KG_YR
        out['ms'] = np.loadtxt(os.path.join(C.OUTPUT_DIR, f"DM_Masses_{tag}.dat"))
        d = np.loadtxt(os.path.join(C.OUTPUT_DIR, f"discovery_{C.run_tag(e_th_keV=E_TH_KEV, n_max=n, res_frac=0.0)}.csv"),
                       delimiter=',', skiprows=1)
        out[f'sig{n}'] = d[:, 1]
    np.savez_compressed(NPZ, **out)
    return NPZ


def plot(path=NPZ):
    import matplotlib.pyplot as plt
    d = np.load(path)
    k = d['keep']; E = np.sqrt(d['lo'] * d['hi'])[k] * 1e3          # bin centres [eV]
    A = d['A'][:, k]; src = list(d['sources']); bkg = A.sum(0); ms = d['ms']
    dm = {n: d[f'dm{n}'][:, k] for n in (1, 60)}; sig = {n: d[f'sig{n}'] for n in (1, 60)}
    style = {1: ('#e6550d', '--', 'n = 1 (+ impulse)'), 60: ('#1f78b4', '-', 'full multiphonon')}

    fig = plt.figure(figsize=(12, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.9], hspace=0.34, wspace=0.22, top=0.88)
    for j, m in enumerate(MASSES_MEV):
        ax = fig.add_subplot(gs[j // 2, j % 2])
        i = np.argmin(abs(ms - m))
        ax.axvspan(E.min(), OMEGA_MAX_EV, color='#fdd0a2', alpha=0.35, lw=0)
        ax.step(E, bkg, where='mid', color='k', lw=1.8, label=r'$\nu$ total')
        ax.step(E, A[src.index('pp')], where='mid', color='0.45', lw=1, ls='--', label='pp')
        ax.step(E, A[src.index('Be7861')], where='mid', color='0.45', lw=1, ls=':', label=r'$^7$Be 861')
        for n in (1, 60):
            c, ls, lab = style[n]
            ax.step(E, dm[n][i] * sig[n][i], where='mid', color=c, ls=ls, lw=2, label=f'DM, {lab}')
        ax.text(0.97, 0.05, '\n'.join(rf'$\sigma_{{3\sigma}}$ ({style[n][2]}) = {sig[n][i]:.2e} cm$^2$'
                                      for n in (1, 60)), transform=ax.transAxes, ha='right', va='bottom',
                fontsize=8.5, bbox=dict(boxstyle='round', fc='white', ec='0.8'))
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(E.min(), 1e3); ax.set_ylim(1e-4, 1e4)
        ax.set_title(rf'$m_\chi$ = {ms[i]:.0f} MeV', fontsize=13)
        ax.set_xlabel(r'$\omega$ [eV]'); ax.set_ylabel(r'events / bin / (t$\cdot$yr)')
        if j == 0:
            fig.legend(*ax.get_legend_handles_labels(), loc='upper center', ncol=5, fontsize=10,
                       bbox_to_anchor=(0.5, 0.94), frameon=False)
    ax = fig.add_subplot(gs[2, :])
    R = {n: dm[n].sum(1) for n in (1, 60)}
    ax.plot(ms, R[1] / R[60], color='#e6550d', lw=2.2, label=r'rate$_{n=1}$ / rate$_{\rm full}$ (left)')
    ax.set_xscale('log'); ax.set_ylim(0, 1.05); ax.set_xlim(ms.min(), 2e3)
    ax.set_xlabel(r'$m_\chi$ [MeV]', fontsize=13); ax.set_ylabel('rate ratio', fontsize=12)
    ax2 = ax.twinx()
    for n in (1, 60):
        c, ls, lab = style[n]
        ax2.plot(ms, (dm[n] * E).sum(1) / dm[n].sum(1), color=c, ls=ls, lw=1.4, alpha=0.8,
                 label=rf'$\langle\omega\rangle$ {lab} (right)')
    ax2.axhline(OMEGA_MAX_EV, color='#fd8d3c', lw=1, ls=':')
    ax2.set_yscale('log'); ax2.set_ylabel(r'$\langle\omega\rangle$ above threshold [eV]', fontsize=12)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, loc='center left')
    fig.suptitle(rf'{C.MATERIAL}, $E_{{th}}$ = {E_TH_KEV*1e6:g} meV, 1 t$\cdot$yr  (shaded: one-phonon band $\omega<$ {OMEGA_MAX_EV*1e3:.0f} meV)',
                 fontsize=13, y=0.975)
    out = os.path.join(os.path.dirname(path), 'diagnostic_spectra.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f"  saved {out}")


if __name__ == "__main__":
    import sys
    p = compute()
    if '--no-plot' not in sys.argv:
        plot(p)
