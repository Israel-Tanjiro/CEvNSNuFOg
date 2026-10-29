#!/usr/bin/env python3
"""Fog index n = -dln(sigma)/dln(exposure) and sigma_disc vs exposure (outputs of exposure_scan.py)."""
import sys, os, numpy as np, matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'outputs')
MATS = [('Al2O3', r'Al$_2$O$_3$'), ('CaWO4', r'CaWO$_4$')]
COLS = ['#2a78d6', '#eb6834', '#1baf7a', '#e87ba4', '#8a8a85', '#2f2f2d']
fig, axs = plt.subplots(2, 2, figsize=(12, 8.2), sharex=True)
for j, (mat, lab) in enumerate(MATS):
    f = os.path.join(D, f'exposure_scan_{mat}_Eth30meV_res0.2.csv')
    hdr = open(f).readline().strip().split(',')
    d = np.loadtxt(f, delimiter=',', skiprows=1)
    X = d[:, 0]
    for k in range(1, d.shape[1]):
        m = hdr[k].split('_m')[1].replace('MeV', '')
        s = d[:, k]
        axs[0, j].plot(X, s * 1e36, color=COLS[k - 1], lw=2, label=rf'$m_\chi$ = {float(m):.3g} MeV')
        n = -np.gradient(np.log(s), np.log(X))
        axs[1, j].plot(X[1:-1], n[1:-1], color=COLS[k - 1], lw=2)
    axs[0, j].set(xscale='log', yscale='log', ylabel=r'$\sigma_{\rm disc}$ [pb]')
    axs[0, j].set_title(rf'{lab}, $E_{{\rm th}}$ = 30 meV, $\sigma_E=E_{{\rm th}}/5$', fontsize=11, loc='left')
    axs[1, j].axhspan(0, 0.5, color='#d9d9d6', lw=0, zorder=0)
    for y, t in ((1.0, 'background-free'), (0.5, 'Poisson-limited')):
        axs[1, j].axhline(y, color='#6b6b67', lw=0.8, ls='--')
        axs[1, j].text(1.5e-6, y + 0.02, t, fontsize=8.5, color='#6b6b67')
    axs[1, j].text(1.5e-6, 0.12, r'fog: $n<1/2$', fontsize=8.5, color='#6b6b67')
    axs[1, j].set(xscale='log', ylim=(0, 1.05), xlabel=r'exposure $X$ [t$\cdot$yr]',
                  ylabel=r'$n=-d\ln\sigma_{\rm disc}/d\ln X$')
    for a in axs[:, j]:
        a.tick_params(direction='in', which='both', top=True, right=True)
        a.grid(which='major', color='#e6e6e3', lw=0.6)
axs[0, 0].legend(frameon=False, fontsize=8.5, loc='lower left')
fig.tight_layout()
out = os.path.join(D, 'exposure_scan.png')
fig.savefig(out, dpi=200)
print('saved', out)
