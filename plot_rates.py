#!/usr/bin/env python3
"""
Differential rates dR/domega for the neutrino background (CEvNS, per source) and for DM,
for every material, with the full lattice response (n_max = 60, q || c, no resolution).

  python plot_rates.py collect   -> outputs/rates_summary.npz   (needs only numpy)
  python plot_rates.py plot      -> outputs/rates_dRdw.png       (needs matplotlib)
  python plot_rates.py           -> both

Neutrinos: events / (kg yr eV) with the flux normalizations of step3_discovery.
DM:        events / (kg yr eV) for sigma_n = SIGMA_REF_CM2 at the masses in M_SHOW_MEV.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, 'outputs')
NPZ = os.path.join(OUTD, 'rates_summary.npz')
MATS = ['CaWO4', 'Al2O3', 'TeO2', 'LiNbO3', 'Si']
SOURCES = ["B8", "hep", "atm", "dsnb3", "O15", "N13", "F17", "pp", "Be7384", "Be7861", "pep"]
NORMS = np.array([5.46e6, 7.98e3, 10.5, 86.0, 2.05e8, 2.78e8, 5.29e6, 5.98e10, 4.84e8, 4.35e9, 1.44e8])
GROUPS = {'pp': ['pp'], '$^7$Be': ['Be7384', 'Be7861'], 'CNO': ['N13', 'O15', 'F17'], 'pep': ['pep'],
          '$^8$B': ['B8'], 'hep + atm + DSNB': ['hep', 'atm', 'dsnb3']}
M_SHOW_MEV = [1.0, 10.0, 100.0, 1000.0]
SIGMA_REF_CM2 = 1e-40
NBINS, ER_MIN_KEV, ER_MAX_KEV = 200, 1e-6, 100.0       # as in config.py
TAG = "Emin0.001eV_qc"


def collect():
    edges_eV = np.logspace(np.log10(ER_MIN_KEV), np.log10(ER_MAX_KEV), NBINS + 1) * 1e3
    width = np.diff(edges_eV)
    out = {'edges_eV': edges_eV, 'masses_MeV': np.array(M_SHOW_MEV), 'sigma_ref': SIGMA_REF_CM2}
    for mat in MATS:
        ndir = os.path.join(OUTD, f"neutrinos_{mat}_{TAG}_n60")
        ev = {s: np.loadtxt(os.path.join(ndir, f"{mat}_{s}Events.dat")) * n for s, n in zip(SOURCES, NORMS)}
        for g, members in GROUPS.items():
            out[f"{mat}|nu|{g}"] = sum(ev[s] for s in members) / width
        out[f"{mat}|nu|total"] = sum(ev.values()) / width
        tag = f"{mat}_{TAG}_m0.5-10000MeV_n60"
        masses = np.loadtxt(os.path.join(OUTD, f"DM_Masses_{tag}.dat"))
        dm = np.loadtxt(os.path.join(OUTD, f"DM_Events_Matrix_{tag}.dat"))
        for m in M_SHOW_MEV:
            i = int(np.argmin(np.abs(masses - m)))
            out[f"{mat}|dm|{m:g}"] = dm[i] * SIGMA_REF_CM2 / width
            out[f"{mat}|dm_mass|{m:g}"] = masses[i]
    np.savez(NPZ, **out)
    print('saved', NPZ)


def plot():
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    d = np.load(NPZ)
    e = d['edges_eV']; x = np.sqrt(e[1:] * e[:-1])
    LAB = {'CaWO4': r'CaWO$_4$', 'Al2O3': r'Al$_2$O$_3$', 'TeO2': r'TeO$_2$', 'LiNbO3': r'LiNbO$_3$', 'Si': 'Si'}
    NU_COL = {'pp': '#2a78d6', '$^7$Be': '#eb6834', 'CNO': '#1baf7a', 'pep': '#e87ba4',
              '$^8$B': '#eda100', 'hep + atm + DSNB': '#008300'}
    DM_COL = ['#b7d3f6', '#6da7ec', '#2a78d6', '#104281']          # light -> dark with mass
    INK = '#2f2f2d'
    fig, axs = plt.subplots(2, 3, figsize=(16, 9.6), sharex=True, sharey=True)
    axs = axs.ravel()
    for ax, mat in zip(axs, MATS):
        for g, c in NU_COL.items():
            y = d[f"{mat}|nu|{g}"]; ok = y > 0
            ax.plot(x[ok], y[ok], color=c, lw=1.6)
        y = d[f"{mat}|nu|total"]; ok = y > 0
        ax.plot(x[ok], y[ok], color=INK, lw=2.4)
        for m, c in zip(d['masses_MeV'], DM_COL):
            y = d[f"{mat}|dm|{m:g}"]; ok = y > 0
            ax.plot(x[ok], y[ok], color=c, lw=2, ls='--')
        for eth in (0.03, 0.1, 1.0):
            ax.axvline(eth, color='#bdbdb8', lw=0.8, ls=':', zorder=0)
        ax.set_title(LAB[mat], fontsize=12, loc='left')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(1e-3, 1e4); ax.set_ylim(1e-8, 1e8)
        ax.tick_params(direction='in', which='both', top=True, right=True)
        ax.grid(which='major', color='#e6e6e3', lw=0.6); ax.set_axisbelow(True)
    for ax in axs[3:5]:
        ax.set_xlabel(r'deposited energy $\omega$ [eV]')
    for ax in (axs[0], axs[3]):
        ax.set_ylabel(r'$dR/d\omega$ [events kg$^{-1}$ yr$^{-1}$ eV$^{-1}$]')
    # legend panel
    axl = axs[5]; axl.axis('off')
    h = [Line2D([], [], color=INK, lw=2.4)] + [Line2D([], [], color=c, lw=1.6) for c in NU_COL.values()]
    l = ['CEvNS total'] + [f'CEvNS {g}' for g in NU_COL]
    leg1 = axl.legend(h, l, loc='upper left', frameon=False, fontsize=10, title='neutrino background',
                      title_fontsize=10)
    axl.add_artist(leg1)
    h2 = [Line2D([], [], color=c, lw=2, ls='--') for c in DM_COL]
    l2 = [rf'$m_\chi$ = {m:g} MeV' for m in d['masses_MeV']]
    s_exp = int(np.log10(float(d['sigma_ref'])))
    axl.legend(h2, l2, loc='center left', bbox_to_anchor=(0.0, 0.36), frameon=False, fontsize=10,
               title=rf'DM signal, $\sigma_n=10^{{{s_exp}}}$ cm$^2$', title_fontsize=10)
    axl.text(0.0, 0.06, 'Full multiphonon response ($n_{\\max}=60$), $\\hat q\\parallel c$, no energy\n'
             'resolution. Dotted vertical lines: thresholds of 30 meV,\n100 meV and 1 eV used in the discovery analysis.',
             transform=axl.transAxes, fontsize=9, color='#6b6b67', va='bottom')
    fig.tight_layout()
    out = os.path.join(OUTD, 'rates_dRdw.png')
    fig.savefig(out, dpi=200)
    print('saved', out)


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what in ('collect', 'all'):
        collect()
    if what in ('plot', 'all'):
        plot()
