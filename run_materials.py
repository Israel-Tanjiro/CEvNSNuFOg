#!/usr/bin/env python3
"""
Reproduce every discovery curve used by the figures, for all materials.

  python3 run_materials.py                 # all materials
  python3 run_materials.py Al2O3 Si        # selected materials

For each material: neutrino spectra (step 1), DM matrices for n_max = 60 and n_max = 1
(step 2), and discovery curves (step 3) for the cases below. Each material runs in its own
process because config.py reads NUFOG_MATERIAL at import time.
"""
import os, sys, subprocess, time

MATERIALS = ['CaWO4', 'Al2O3', 'TeO2', 'LiNbO3', 'Si']
# (E_th [keV], n_max, resolution fraction sigma_E / E_th)
CASES = [(3e-5, 60, 0.0), (3e-5, 60, 0.2), (3e-5, 1, 0.2),
         (1e-4, 60, 0.0), (1e-4, 60, 0.2),
         (1e-3, 60, 0.0), (1e-3, 60, 0.2)]


def run_one():
    import config as C
    import step1_neutrinos, step2_dm_matrix, step3_discovery
    from run_all import set_case
    done_dm = set()
    set_case(CASES[0][0], 60, 'z', 0.0)
    print(f"[{C.MATERIAL}] step 1: neutrino spectra"); step1_neutrinos.main()
    for e_th, n_max, res in CASES:
        set_case(e_th, n_max, 'z', res)
        if C.dm_tag() not in done_dm:
            print(f"[{C.MATERIAL}] step 2: DM matrix, n_max = {n_max}"); step2_dm_matrix.main()
            done_dm.add(C.dm_tag())
        print(f"[{C.MATERIAL}] step 3: E_th = {e_th*1e6:g} meV, n_max = {n_max}, sigma_E/E_th = {res:g}")
        step3_discovery.main()


if __name__ == '__main__':
    if os.environ.get('NUFOG_RUN_ONE'):
        run_one()
        sys.exit(0)
    mats = sys.argv[1:] or MATERIALS
    t0 = time.time()
    for m in mats:
        env = dict(os.environ, NUFOG_MATERIAL=m, NUFOG_RUN_ONE='1')
        subprocess.run([sys.executable, os.path.abspath(__file__)], env=env, check=True)
    print(f"all materials done in {time.time() - t0:.0f} s")
