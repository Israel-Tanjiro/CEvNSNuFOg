#!/usr/bin/env python3
"""
Run every case listed in config.PLOT_CASES (steps 1-3, reusing cached S(q,w) and
neutrino spectra whenever possible), then plot them together (step 4).
"""
import os
import time
import config as C
import step1_neutrinos, step2_dm_matrix, step3_discovery


def set_case(e_th, n_max, qdir, res=None):
    C.E_TH_KEV, C.N_MAX, C.Q_DIR = e_th, n_max, qdir
    if res is not None:
        C.RESOLUTION_FRAC = res
    C.NEUTRINO_DIR = os.path.join(C.OUTPUT_DIR, f"neutrinos_{C.neutrino_tag()}")
    os.makedirs(C.NEUTRINO_DIR, exist_ok=True)


def main():
    """Steps 1-2 are recomputed in every run (seconds), so outputs never go stale when the
    tensor or the physics changes; S(q, omega) is still reused from the cache when valid."""
    t0 = time.time()
    done_nu, done_dm = set(), set()
    for e_th, n_max, qdir, res, label, *_ in C.PLOT_CASES:
        print(f"\n=== E_th = {e_th*1e3:g} eV, n_max = {n_max}, q direction = {qdir}, sigma_E/E_th = {res:g} ===")
        set_case(e_th, n_max, qdir, res)
        if C.NEUTRINO_DIR not in done_nu:
            print("[1] neutrino spectra"); step1_neutrinos.main(); done_nu.add(C.NEUTRINO_DIR)
        if C.dm_tag() not in done_dm:
            print("[2] DM matrix"); step2_dm_matrix.main(); done_dm.add(C.dm_tag())
        print("[3] discovery curve"); step3_discovery.main()
    print("\n[4] plot")
    import step4_plot
    step4_plot.main()
    print(f"\nall done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
