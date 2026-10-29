# NuFog v2 — neutrino fog for CaWO4 with the lattice (multiphonon) response

## Run
```bash
python3 run_all.py              # every case in config.PLOT_CASES, then the plot
python3 check_normalization.py  # absolute-normalization benchmark (expect ratios 1.00 +- 0.02)
python3 compare_v1.py           # v1 vs v2 curves and their ratio
```
Single steps: `step1_neutrinos.py`, `step2_dm_matrix.py`, `step3_discovery.py`, `step4_plot.py`.
All settings live in `config.py`. Typical runtime: under 30 s for all four default cases.

## Outputs (`outputs/`)
| File | Content |
|---|---|
| `neutrinos_<tag>/CaWO4_<source>Events.dat` | CEvNS events / kg / yr per bin, per unit flux normalization |
| `DM_Events_Matrix_<tag>.dat`, `DM_Masses_<tag>.dat` | DM events / kg / yr / cm^2 per bin, one row per mass [MeV] |
| `discovery_<run tag>.csv` | m_chi [GeV], sigma [cm^2], sigma [pb] at Asimov q0 = 9 (3 sigma) |
| `cache/S_*.npz` | S(q, omega) per species; reused whenever material, n_max, q direction and grids match |

Changing the threshold, exposure or mass range reuses the cache and the neutrino spectra.

## Changes with respect to v1
1. **CEvNS normalization.** v1 combined G_F^2/(4 pi) with Q_W/2 (g_V convention) and was pi/2
   too large. v2 uses G_F^2/(8 pi^2) with Q_W = N - (1 - 4 sin^2 theta_W) Z, as in the thesis.
2. **SHM velocity integral.** v1's eta(v_min) was low by sqrt(pi)/2. v2 uses the standard
   truncated-Maxwellian expression.
3. **Multiphonon / Debye-Waller consistency.** The tensor DOS is Gaussian-smeared
   (sigma = 0.5 meV), so D(0) is finite and D/omega diverges at omega -> 0. In v1, <1/omega>
   in the Debye-Waller factor picked up this spike, but the T_n convolutions did not. That
   mismatch removed up to ~70% of the spectral weight just below the impulse switch
   (q = 4 sqrt(2 m omega_0), where <n> = 16). v2 replaces D below `DOS_IR_CUTOFF_MEV` = 2 meV
   with the acoustic form D ~ omega^2, computes <1/omega>, <omega> and all T_n from the same
   T_1, and raises n_max to 60 and the T_n range to 3 eV.
   Result: the f-sum rule int S d omega = sum_d A_d^2 F^2 (1 - e^{-2W_d}) holds to ~1% at
   all q (v1: 27-72% between 3 and 230 keV).
4. **Speed.** S(q, omega) is built once on a global grid (vectorized multiphonon sum, FFT
   convolutions) and cached. Each DM mass is one array operation. The discovery cross
   section is found by bisection in log sigma, using an exact Newton profile of the
   nuisance parameters (convex likelihood). Timings: step 1 ~4 s, step 2 ~6 s,
   step 3 < 1 s (v1: minutes each).
5. **Crystal orientation.** `Q_DIR` = 'z' (c-axis, v1 default), 'x' (a-axis), 'avg' (Tr D / 3)
   or any vector. Runs with different directions are labelled and cached separately.

6. **Material density.** The rate per unit mass contains (2 pi / V_cell) / rho_T = 2 pi / M_cell.
   v1 took rho_T from a tabulated value inconsistent with V_cell. The error was +1% for CaWO4,
   +9% for Al2O3 (V_cell = 77.7 A^3 in the database; sapphire's primitive cell is 84.9 A^3) and
   +8% for TeO2. v2 uses rho = M_cell / V_cell, so V_cell cancels exactly.
7. **Energy resolution.** `RESOLUTION_FRAC` applies a Gaussian of sigma = RESOLUTION_FRAC * E_th
   (CRESST: E_th / 5) to the DM and neutrino spectra before the threshold cut, as a bin-migration
   matrix. Runs with and without resolution are labelled separately (`_res0.2Eth`).

## Validation
* Newton profile vs scipy L-BFGS-B: |delta q0| < 2e-3, and Newton always reaches the lower minimum.
* Free-nucleus benchmark above 1 eV: 8B CEvNS 0.9998, DM at 1 GeV 0.9988, DM at 5 GeV 0.9999
  (v1: 1.585, 0.897, 0.899).
* S(q, omega) in the single-phonon and impulse regimes matches v1 to machine precision when
  the same inputs are used.

## Open issue: the phonon tensor file
Symmetry-equivalent atoms must have identical Tr D(omega). In
`pDoS_tensor_v4_mesh100_CaWO4.npz` they do not. Mean phonon energies <omega> per atom are:
atoms 0-3 about 27 meV; atoms 4-11 between 26 and 48 meV with no clear grouping; atoms
12-23 between 52 and 57 meV. With the code's assumed order (Ca x4, W x4, O x16), some "O"
atoms
would have <omega> = 26 meV, which is implausible for the lightest atom. Either the atom
order in the file differs from Ca4 W4 O16, or the force constants break the crystal
symmetry. Check the order against the phonopy.yaml / POSCAR used to build the tensor.
W carries A^2 = 33856, so this matters for the low-mass DM rates.

## Effect of energy resolution (sigma = E_th / 5, CaWO4, 1 t yr, q || c)
* E_th = 30 meV: the reach improves by ~6% at m < 20 MeV. More events fluctuate above
  threshold (+12%), and per-event discrimination drops by ~10%. The phonon-DOS peaks (45, 100 meV)
  survive 6 meV smearing.
* E_th = 0.1 eV: the reach worsens by up to 1.4x at m < 5 MeV. The threshold coincides with the top
  optical peak (~110 meV); smearing turns that sharp edge into a shoulder (q0 per event / 3.4).
* E_th = 1 eV: below ~35 MeV the curve is set by the Gaussian tail of the resolution (upward
  fluctuations of sub-threshold multiphonon events). That relies on a Gaussian many sigma out,
  so do not quote it. Above 40 MeV the change is less than 2x.
* Above ~150 MeV resolution has no effect.

## Phonon tensors (validated against phonopy force constants)
* **CaWO4:** `pDoS_tensor_v5_mesh32x32x16_CaWO4.npz` was rebuilt with `make_tensor.py` from
  phonopy.yaml + FORCE_SETS (full mesh, no mesh symmetry). It replaces the v4 file, which broke
  the crystal symmetry: <omega> of W was 28-38 meV instead of 21.8 meV. Zero-point <u_z^2>
  values (W 1.4e-3, O 3.8e-3, Ca 4.8e-3 A^2) are plausible. No BORN file was available, so
  the file has no LO-TO correction (a <1% effect, judging by Al2O3). The 2x2x1 supercell gives
  small imaginary acoustic modes (> -1.8 meV), which are excluded.
* **Al2O3:** `pDoS_tensor_v4_mesh100.npz` agrees with a rebuild from its own force constants
  + BORN (non-analytic correction) to 0.1% in <omega>, <1/omega> and 0.3-0.5% in shape. Band
  edge 107 meV. The Debye sound speed is 6.6-6.7 km/s vs ~7.1 km/s experimental, so the
  acoustic DOS is ~20% high (DFT softening). c-axis vs in-plane anisotropy is 5-8%.
* The S(q, omega) cache is keyed by the tensor's contents, and `run_all.py` always
  recomputes steps 1-2, so swapping a tensor cannot reuse stale results.

## CaWO4 vs Al2O3 (1 t yr, q || c, sigma_E = E_th / 5)
`NUFOG_MATERIAL=Al2O3 python3 run_all.py` runs sapphire; `compare_materials.py` overlays both.
* m < ~50 MeV: Al2O3 is 8-16x more sensitive at 30 meV and ~10x at 0.1 eV. Tungsten's
  phonon weight lies below a 30 meV threshold, the CEvNS background is 7.6x lower in Al2O3
  (N^2), and the discrimination per DM event is ~20x better.
* m > ~100 MeV: CaWO4 is 1.3-1.8x more sensitive (coherent A^2 of W in the impulse regime).
* Single-phonon gap (30 meV): n=1 / full reaches 38x at 49 MeV in CaWO4, but only 2.9x at
  69 MeV in Al2O3.
* Corrected vs old CaWO4 tensor: low-mass reach is 1.5-1.65x weaker; unchanged above ~150 MeV.

## Five-material comparison (1 t yr, q || c, sigma_E = E_th / 5) - all pass the benchmark (0.1%)
* E_th = 30 meV, m <= 5 MeV: Al2O3 best (6.7e-10 pb at 5 MeV); Si 1.3x, LiNbO3 3.6x, TeO2 8x,
  CaWO4 16x weaker. 20-50 MeV: Si ~ Al2O3. m >= 100 MeV: heavy targets (TeO2, CaWO4) best by 1.3-1.8x.
* Multiphonon gap (max n=1/full at 30 meV) tracks the heaviest, softest atom:
  CaWO4 38x (W) > TeO2 16x (Te) > LiNbO3 8x (Nb) > Si 5.6x > Al2O3 2.9x.
* Reliability: Al2O3 and LiNbO3 force constants validated (acoustic slopes). CaWO4 (2x2x1) and
  Si (16 atoms) have acoustic slopes 20-40% low -> systematic in the W / Si response.
  TeO2 has no force constants to check. LiNbO3 in the files was labelled LiNbO2.

## Directionality: daily modulation from crystal anisotropy

Scripts (run in this order, any material via NUFOG_MATERIAL):
1. `directional.py`: S(q,omega; theta_q) on 24 Gauss-Legendre nodes in cos(theta_q), using
   S(theta) = S(pi - theta) and an average over 4 azimuths. The SHM velocity distribution enters
   through its Radon transform g(w, qhat). The rate R(m, theta_E) is computed for 10 angles between
   the c-axis and v_E, binned like the main pipeline. Takes 30-60 s per material.
2. `check_directional.py`: averaging over v_E directions must reproduce the isotropic pipeline.
   Si gives 1.0000 at all masses; CaWO4 is within 0.3%, as expected because the response is
   nonlinear in D (<S(D)> != S(<D>)).
3. `daily_modulation.py MAT`: maps sidereal time to theta_E at LNGS (v_E toward Cygnus) and writes a
   CSV per material.
4. `directional_summary.py`, then `plot_directional.py` -> outputs/directional_modulation.png.

Results (resolution sigma = E_th/5, c-axis vertical at LNGS, where theta_E sweeps 6-90 deg per day):
- The effect is limited to m < ~50 MeV. At higher masses the rate comes from the impulse regime,
  where the response is isotropic.
- At E_th = 30 meV: LiNbO3 and TeO2 ~1.7%, Al2O3 ~0.7%, CaWO4 < 0.2% (rising to 1.4% at 20 MeV),
  Si 0 (cubic).
- At E_th = 100 meV: LiNbO3 ~4.5%, TeO2 ~4%, Al2O3 ~2.5%, CaWO4 ~1.2%.
- For several materials the sign reverses between 30 and 100 meV. Above the highest optical
  phonon, the direction with harder phonons gives more events; near threshold, the softer
  direction does.
- Caveat: the incoherent approximation keeps only the per-atom DOS-tensor anisotropy. Coherent
  single-phonon dispersion is not included, so these numbers are a conservative lower bound for
  the low-q regime.
