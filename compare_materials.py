#!/usr/bin/env python3
"""
Compare target materials: 3-sigma discovery curves (1 t yr, q || c) at one threshold per
figure, with the ratio to CaWO4 in a lower panel. Materials not yet computed are skipped.
Run the pipeline for each material first, e.g.  NUFOG_MATERIAL=Al2O3 python3 run_all.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import config as C

# material: (colour, label, caveat shown in the legend)
MATERIALS = {
    'CaWO4':  ('#1f78b4', r'CaWO$_4$', ''),
    'Al2O3':  ('#e31a1c', r'Al$_2$O$_3$', ''),
    'LiNbO3': ('#33a02c', r'LiNbO$_3$', ''),
    'TeO2':   ('#ff7f00', r'TeO$_2$', ' (no FC check)'),
    'Si':     ('#6a3d9a', 'Si', ''),
}
THRESHOLDS = [(3e-5, '30 meV'), (1e-4, '0.1 eV'), (1e-3, '1 eV')]
REFERENCE = 'CaWO4'
RES_FRAC = 0.2          # sigma_E = E_th / 5 ; 0.0 for perfect resolution
N_MAX = 60


def curve(material, e_th):
    res = f"_res{RES_FRAC:g}Eth" if RES_FRAC > 0 else ""
    tag = (f"Eth{e_th*1e3:g}eV{res}_{C.EXPOSURE_KG_YR/1000:g}tyr_{material}_{C.binning_tag()}_"
           f"{C.dir_tag()}_m{C.M_CHI_MIN_MEV:g}-{C.M_CHI_MAX_MEV:g}MeV_n{N_MAX}")
    f = os.path.join(C.OUTPUT_DIR, f"discovery_{tag}.csv")
    return np.loadtxt(f, delimiter=',', skiprows=1) if os.path.exists(f) else None


def figure(e_th, tlab):
    d = {m: curve(m, e_th) for m in MATERIALS}
    have = [m for m in MATERIALS if d[m] is not None]
    if not have:
        print(f"  nothing computed for E_th = {tlab}"); return
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(10, 10), sharex=True,
                                  gridspec_kw={'height_ratios': [3, 1.2], 'hspace': 0.06})
    for m in have:
        col, lab, note = MATERIALS[m]
        ok = np.isfinite(d[m][:, 2])
        ax.plot(d[m][ok, 0], d[m][ok, 2], color=col, lw=2.3, label=lab + note)
        if REFERENCE in have and m != REFERENCE:
            axr.plot(d[m][:, 0], d[m][:, 1] / d[REFERENCE][:, 1], color=col, lw=2)
    for x, lab in [(1.5e-1, 'pp'), (5.2e-1, r'$^{7}$Be'), (5.8, r'$^{8}$B')]:
        for a in (ax, axr):
            a.axvline(x, color='0.85', ls='--', lw=1.2, zorder=0)
        ax.text(x * 0.9, 2e-11, lab, rotation=90, color='0.55', size=12, va='bottom', ha='right')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(1e-11, 1e2); ax.set_xlim(4e-4, 1.3e1)
    ax.set_ylabel(r'$\sigma_{n\chi}$ [pb]', fontsize=18)
    res_txt = r', $\sigma_E = E_{th}/5$' if RES_FRAC > 0 else ', perfect resolution'
    ax.legend(fontsize=12, loc='upper right',
              title=rf'$E_{{th}}$ = {tlab}{res_txt}, 1 t$\cdot$yr, 3$\sigma$ discovery', title_fontsize=11)
    axr.set_yscale('log'); axr.set_ylim(3e-3, 30); axr.axhline(1, color=MATERIALS[REFERENCE][0], lw=1.5)
    axr.set_ylabel(rf'$\sigma / \sigma_{{\rm {REFERENCE}}}$'.replace('CaWO4', r'CaWO_4'), fontsize=14)
    axr.set_xlabel(r'$m_\chi$ [GeV]', fontsize=18)
    axr.text(0.99, 0.06, f'< 1: more sensitive than {REFERENCE}', transform=axr.transAxes,
             ha='right', fontsize=10, color='0.3')
    for a in (ax, axr):
        a.tick_params(which='both', direction='in', labelsize=13)
    out = os.path.join(C.OUTPUT_DIR, f"compare_materials_Eth{e_th*1e3:g}eV{'_res' if RES_FRAC > 0 else ''}.png")
    fig.savefig(out, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f"  saved {out}  ({', '.join(have)})")


if __name__ == "__main__":
    for e_th, tlab in THRESHOLDS:
        figure(e_th, tlab)
