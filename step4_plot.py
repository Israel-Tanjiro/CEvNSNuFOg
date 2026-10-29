#!/usr/bin/env python3
"""Step 4: overlay the 3-sigma discovery curves listed in config.PLOT_CASES."""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import config as C

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Nimbus Sans", "Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams['mathtext.fontset'] = 'dejavusans'


def free_nucleus_cutoff_GeV(e_th_keV, A_mass_amu=15.9994):
    """Lightest DM mass able to give a free recoil above threshold on the lightest nucleus (O)."""
    v_max = (C.VESC_KMS + C.VE_KMS) / 299792.458
    mN = A_mass_amu * 0.93149410242
    mu = np.sqrt(e_th_keV * 1e-6 * mN / (2.0 * v_max**2))
    return mu * mN / (mN - mu)


def main():
    fig, ax = plt.subplots(figsize=(10, 8))
    handles, m_min, cut_color = [], [], {}
    for e_th, n_max, qdir, res, label, color, ls in C.PLOT_CASES:
        run = C.run_tag(e_th_keV=e_th, n_max=n_max, qdir=qdir, res_frac=res)
        f = os.path.join(C.OUTPUT_DIR, f"discovery_{run}.csv")
        if not os.path.exists(f):
            print(f"  skipping (not computed yet): {run}")
            continue
        d = np.loadtxt(f, delimiter=",", skiprows=1)
        ok = np.isfinite(d[:, 2])
        ax.plot(d[ok, 0], d[ok, 2], color=color, ls=ls, lw=2.5)
        handles.append(Line2D([0], [0], color=color, ls=ls, lw=2.5, label=label))
        m_min.append(d[ok, 0].min())
        cut_color.setdefault(e_th, color)

    for path in (os.path.join(C.BASE_DIR, 'RefExp1_2.dat'), os.path.join(C.BASE_DIR, '..', 'RefExp1_2.dat')):
        if os.path.exists(path):
            r = np.loadtxt(path)
            ax.plot(r[:, 0], r[:, 1], color='tab:red', ls='-.')
            handles.append(Line2D([0], [0], color='tab:red', ls='-.', label=r'Reference, $E_{th}$ = 0.1 eV'))
            break

    # free-nucleus kinematic cutoff (lightest nucleus, O), one per threshold, in the curve colour
    for e_th, color in sorted(cut_color.items()):
        m_cut = free_nucleus_cutoff_GeV(e_th)
        ax.axvline(m_cut, color=color, ls=':', lw=1.8, alpha=0.9)
        ax.text(m_cut * 1.07, 1e-1, f"{m_cut*1e3:.1f} MeV", rotation=90, color=color,
                fontsize=11, va='center')
    if cut_color:
        handles.append(Line2D([0], [0], color='0.3', ls=':', lw=1.8,
                              label='free-nucleus cutoff (O), same colour'))

    for x, lab in [(1.5e-1, 'pp'), (5.2e-1, r'$^{7}$Be'), (5.8, r'$^{8}$B')]:
        ax.axvline(x, color='0.85', ls='--', lw=1.5, zorder=0)
        ax.text(x * 0.9, 2e-10, lab, rotation=90, color='0.55', size=13, va='bottom', ha='right')

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(max(1e-4, 0.8 * min(m_min)) if m_min else 1e-3, 1.3e1)
    ax.set_ylim(1e-10, 1e4)
    ax.set_xlabel(r'$m_\chi$ [GeV]', fontsize=22)
    ax.set_ylabel(r'$\sigma_{n\chi}$ [pb]', fontsize=22)
    sec = ax.secondary_yaxis('right', functions=(lambda pb: pb * 1e-36, lambda cm2: cm2 * 1e36))
    sec.set_ylabel(r'$\sigma_{n\chi}$ [cm$^2$]', fontsize=18)
    sec.tick_params(axis='y', which='major', direction='in', length=10, width=1.5, labelsize=15)
    sec.tick_params(axis='y', which='minor', direction='in', length=5, width=1)
    for s_ in ax.spines.values():
        s_.set_linewidth(2)
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10), numticks=20))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.tick_params(axis='both', which='major', direction='in', length=10, width=1.5, labelsize=20)
    ax.tick_params(axis='both', which='minor', direction='in', length=5, width=1)
    ax.legend(handles=handles, fontsize=10.5, frameon=True, loc='upper right',
              title=rf'{C.MATERIAL}, {C.EXPOSURE_KG_YR/1000:g} t$\cdot$yr, 3$\sigma$ discovery',
              title_fontsize=13)

    name = f"NeutrinoFog_{C.MATERIAL}_{C.binning_tag()}_compare.png"
    fig.savefig(os.path.join(C.OUTPUT_DIR, name), dpi=300, bbox_inches='tight')
    print(f"  saved {name}")


if __name__ == "__main__":
    main()
