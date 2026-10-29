# NuFog: neutrino fog with the lattice (multiphonon) response — work log and conclusions

**Date:** 29 October 2026  
**Folder:** `~/Documents/CEvNS/NuFog` (original v1 programs) and `~/Documents/CEvNS/NuFog/NuFog_v2` (new pipeline)  
**Scope:** the work starts from the original programs: `NeutrinoEvents2.py` → `DM_Rate_Matrix_MassRange2.py` → `Neutrino_Floor_phonon2.py` → `PlotFog.py`.  
**Companion document:** `NuFog_report.pdf` contains the full derivations.

---

## 1. Starting point: the v1 pipeline

The original sequence computes a 3σ discovery curve ("neutrino fog") for spin-independent DM–nucleus scattering in CaWO₄, including the lattice response:

| Step | Program | What it does |
|---|---|---|
| 1 | `NeutrinoEvents2.py` | CEvNS spectra for each solar, atmospheric and DSNB source |
| 2 | `DM_Rate_Matrix_MassRange2.py` | DM spectra on a mass grid |
| 3 | `Neutrino_Floor_phonon2.py` | Asimov discovery cross section σ(m_χ) with flux nuisance parameters |
| 4 | `PlotFog.py` | Plot |

The physics modules are `dynamic_structure_factor_Impulse2.py` and `dynamic_structure_factor_neutrinos2.py`. They contain the per-atom incoherent multiphonon expansion, the impulse approximation, and the Helm form factor. Settings live in `config_nufog.py`.

The run with these programs reproduced the 0.1 eV curve of Fig. 7 of the CRESST perspective paper (Commun. Phys. 2025).

### 1.1 First modifications to v1 (in place, in `NuFog/`)

- **Paths.** The hard-coded directories were replaced, so the four scripts read and write consistently. Missing-file errors in `PlotFog.py` are fixed.
- **Low masses.** The mass range was extended down to 0.5 MeV and the recoil binning down to E_R,min = 1 meV, so the multiphonon reach is visible.
- **Single phonon vs full.** A switch was added to compare n = 1 (single phonon) with the full multiphonon expansion.
- **Originals preserved.** The unmodified scripts are in `originals_v2/`; the state before v2 is in `originals_v3/`.

### 1.2 Theory check: thesis CEvNS expression vs code

The thesis CEvNS rate used the prefactor G_F²/(8π² ρ_T) together with Q_W ≈ −N/2. These belong to different conventions:

- G_F²/(8π²) goes with Q_W = N − (1 − 4 sin²θ_W) Z ≈ N.
- G_F²/(2π²) goes with Q_W ≈ −N/2.

Mixing the two underestimates the rate by a factor of 4. The free-nucleus limit fixes the convention, since it must reproduce dσ/dE_R = (G_F² M / 4π) Q_W² (1 − M E_R / 2E_ν²) F². The thesis text should use one convention consistently.

---

## 2. Errors found in v1 and fixed in v2

Each fix was validated against an analytic limit or a sum rule.

| # | Problem in v1 | Size of the effect | Fix in v2 |
|---|---|---|---|
| 1 | **CEvNS normalization.** G_F²/(4π) was combined with Q_W/2. | Rates π/2 too large (×1.585 in the ⁸B benchmark) | G_F²/(8π²) with Q_W = N − (1 − 4 sin²θ_W) Z |
| 2 | **SHM velocity integral η(v_min).** | Low by √π/2 (DM ×0.897) | Standard truncated-Maxwellian η |
| 3 | **Threshold.** E_th was not applied in the likelihood. | Curves did not depend on E_th | Only bins with lower edge ≥ E_th are used |
| 4 | **Mass grid.** Masses were duplicated. | Redundant work, plotting artefacts | Single log grid, 200 masses from 0.5 MeV to 10 GeV |
| 5 | **Multiphonon / Debye–Waller inconsistency.** The tensor DOS is Gaussian-smeared (σ = 0.5 meV), so D(0) is finite and D/ω diverges as ω → 0. ⟨1/ω⟩ in e^{−2W} picked up this spike, but the T_n convolutions did not. n_max = 15 truncated the series. | Up to 70% of spectral weight lost below the impulse switch; the f-sum rule failed by 27–72% for q = 3–230 keV | Debye form D ∝ ω² below 2 meV (normalization kept). ⟨1/ω⟩, ⟨ω⟩ and all T_n computed from the same T₁. n_max = 60, T_n tabulated to 3 eV. The f-sum rule now holds to about 1% at all q. |
| 6 | **Density.** A tabulated ρ_T was inconsistent with the cell volume V_cell used in S(q,ω). | CaWO₄ +1%, Al₂O₃ +9%, TeO₂ +8% | ρ = M_cell/V_cell, so (2π/V_cell)/ρ_T = 2π/M_cell and V_cell cancels |
| 7 | **Stale outputs.** Outputs and cache did not depend on the tensor file. | Swapping a tensor could reuse old results | Cache key includes a SHA-1 hash of the tensor. Steps 1–2 always rerun (seconds). |
| 8 | **Speed.** | Minutes per step | Vectorized S(q,ω) built once and cached, FFT convolutions, exact Newton profiling of nuisances, bisection in log σ. Step 1 ≈ 4 s, step 2 ≈ 6 s, step 3 < 1 s. |

**Absolute benchmark** (`check_normalization.py`, free nucleus above 1 eV):

| Benchmark | v1 | v2 |
|---|---|---|
| ⁸B CEvNS | 1.585 | 0.9998 |
| DM at 1 GeV | 0.897 | 0.9988 |
| DM at 5 GeV | 0.899 | 0.9999 |

All five materials pass this benchmark to 0.1%.

---

## 3. The new pipeline: `NuFog_v2/`

### 3.1 Core pipeline

| File | Role |
|---|---|
| `config.py` | All settings: material (`NUFOG_MATERIAL` environment variable), grids, n_max, threshold, exposure, resolution, halo, tensor files, file tags |
| `nufog_core.py` | Grids; material loading and projected DOS; IR regularization; multiphonon T_n; S(q,ω) with caching; Helm form factor; η_SHM; DM and CEvNS rates; binning; resolution matrix |
| `likelihood.py` | `AsimovQ0`: Poisson likelihood with 11 Gaussian-constrained flux nuisances, convex Newton profile, q₀(σ), bisection for σ at q₀ = 9 |
| `step1_neutrinos.py` | CEvNS spectra per source, including lines (⁷Be, pep) |
| `step2_dm_matrix.py` | DM spectra (events / kg / yr / cm²), in chunks of 20 masses |
| `step3_discovery.py` | Resolution smearing, then threshold cut, then σ_disc(m) → `discovery_<tag>.csv` |
| `step4_plot.py` | Overlay of the cases in `PLOT_CASES`, with colour-matched free-nucleus cut-offs and a secondary cm² axis |
| `run_all.py` | Runs steps 1–3 for each case, then step 4 |
| `dynamic_structure_factor.py` | Copy of the v1 module (`MATERIAL_DATABASE`), extended with Si and LiNbO₃ |

### 3.2 Validation and diagnostic programs

| File | Role |
|---|---|
| `check_normalization.py` | Free-nucleus benchmark for any material |
| `compare_v1.py` | v1 vs v2 curves and their ratio |
| `diagnose_spectra.py` | DM and neutrino spectra at 5, 20, 70 and 150 MeV, n = 1 vs full |
| `compare_materials.py` | Five materials, one figure per threshold, ratio to CaWO₄ |
| `exposure_scan.py` | σ_disc vs exposure and the fog index (Sec. 8) |

### 3.3 Phonon tensors

The tensor builders `make_tensor.py` (CaWO₄) and `make_tensor_any.py` (generic) were run in the cloud workspace. They build the per-atom DOS tensor from phonopy force constants on the full mesh, without mesh symmetry, with the non-analytic (LO–TO) correction when a BORN file exists.

| Material | File | Status |
|---|---|---|
| CaWO₄ | `pDoS_tensor_v5_mesh32x32x16_CaWO4.npz` | Rebuilt from `phonopy.yaml` + `FORCE_SETS`. The old v4 file broke the crystal symmetry: ⟨ω⟩ of W was 28–38 meV instead of 21.8 meV. |
| Al₂O₃ | `pDoS_tensor_v4_mesh100.npz` | Agrees with a rebuild from its own force constants + BORN to 0.1% in ⟨ω⟩ and ⟨1/ω⟩ |
| TeO₂ | `pDoS_tensor_TeO2_reordered.npz` | Atom order fixed (the file stored O×8, Te×4 under Te-first labels) |
| Si | `pDoS_tensor_Si_phonondb.npz` | Togo phonon database, 64-atom supercell. Supersedes the 16-atom version, whose TA branch was 40% soft. |
| LiNbO₃ | `pDoS_tensor_LiNbO3.npz` | New; the files were labelled LiNbO₂ |

**Acoustic-branch checks against experimental sound speeds:**

| Material | Result |
|---|---|
| Al₂O₃ | Within about 5% (DFT softening makes the acoustic DOS about 20% high) |
| LiNbO₃ | Good along c; small in-plane imaginary pocket |
| CaWO₄ | One in-plane imaginary branch; slopes 20–40% low (2×2×1 supercell). 87% of W's ⟨1/ω⟩ comes from 2–10 meV, so this is a systematic on the low-mass CaWO₄ reach. **Deferred:** a converged 2×2×2 supercell. |
| Si | TA 4.86 vs 5.84 km/s (about 15% soft) |
| TeO₂ | No force constants available to check |

---

## 4. Physics summary

Full derivations are in `NuFog_report.pdf`.

**Lattice response, incoherent approximation:**

S(q,ω) = (2π/V) Σ_d A_d² F_d²(q) e^{−2W_d(q)} Σ_{n≥1} (1/n!) (q²/2m_d)^n T_{d,n}(ω)

with:
- T₁ = D_d(ω)/ω and T_n = T₁ * T_{n−1} (n-fold convolution);
- the Debye–Waller exponent W_d = (q²/4m_d) ⟨1/ω⟩_d.

**Impulse approximation.** Above q_thr = 4√(2 m_d ω̄₀), where ⟨n⟩ = 16, the response becomes a Gaussian centred at q²/2m_d with width Δ² = q² ⟨ω⟩/2m_d. This connects continuously to free-nucleus recoil.

**DM rate:**

dR/dω = (1/ρ_T)(ρ_χ/m_χ)(σ_n / 4πμ²) ∫ q dq η(v*) S(q,ω),  with v* = q/2m_χ + ω/q.

**CEvNS rate:**

dR/dω = (1/ρ_T)(G_F²/8π²) ∫ dE_ν (dΦ/dE_ν) ∫ q dq (1 − q²/4E_ν²) S_ν(q,ω),

where S_ν uses Q_W² instead of A².

**Discovery test (Asimov).** q₀ = 2[min_θ(−ln L₀) − (−ln L₁)] = 9 for 3σ. There are 11 flux nuisances θ_k with Gaussian priors of relative width r_k, for the sources B8, hep, atm, DSNB, O15, N13, F17, pp, Be7(384), Be7(861) and pep.

**Energy resolution.** A Gaussian with σ = E_th/5 (CRESST convention) is applied as a bin-migration matrix to the full spectra, before the threshold cut.

---

## 5. Results for CaWO₄ (1 t·yr, q ∥ c)

### 5.1 Single phonon vs full multiphonon

- The full expansion extends the low-mass reach. The largest n = 1 / full ratio at 30 meV is **38× for CaWO₄** (at 49 MeV, driven by W).
- The n = 1 "bump" near 70 MeV is mainly a collapse of the single-phonon rate, plus a shape penalty in the impulse regime. It is not a spectral degeneracy with neutrinos.

### 5.2 Energy resolution (σ = E_th/5)

| Threshold | Effect | Reason |
|---|---|---|
| 30 meV | Reach about 6% better below 20 MeV | More events fluctuate above threshold (+12%); per-event discrimination drops about 10%. The DOS peaks at 45 and 100 meV survive 6 meV smearing. |
| 0.1 eV | Up to 1.4× worse below 5 MeV | The threshold sits on the top optical peak (about 110 meV); smearing turns the sharp edge into a shoulder |
| 1 eV | Do not quote below about 35 MeV | The curve is set by the Gaussian tail many σ out. Above 40 MeV the change is less than 2×. |
| Any | No effect above about 150 MeV | — |

### 5.3 Corrected vs old CaWO₄ tensor

The low-mass reach is **1.5–1.65× weaker** with the corrected tensor, and unchanged above about 150 MeV.

---

## 6. Five-material comparison (1 t·yr, q ∥ c, σ_E = E_th/5)

**Low mass (E_th = 30 meV, m ≤ 5 MeV).** Al₂O₃ is best (6.7×10⁻¹⁰ pb at 5 MeV). Relative to Al₂O₃:

| Material | Reach |
|---|---|
| Si | 1.3× weaker |
| LiNbO₃ | 3.6× weaker |
| TeO₂ | 8× weaker |
| CaWO₄ | 16× weaker |

**Intermediate and high mass.** At 20–50 MeV, Si ≈ Al₂O₃. At m ≥ 100 MeV the heavy targets (TeO₂, CaWO₄) are best, by 1.3–1.8×, from the coherent A² in the impulse regime.

**Multiphonon gap** (largest n = 1 / full ratio at 30 meV). It follows the heaviest, softest atom:

| Material | Gap | Driving atom |
|---|---|---|
| CaWO₄ | 38× | W |
| TeO₂ | 16× | Te |
| LiNbO₃ | 8× | Nb |
| Si | 5.6× | — |
| Al₂O₃ | 2.9× | — |

**Why CaWO₄ is weakest at low mass:**
- Tungsten's phonon weight lies below a 30 meV threshold.
- The CEvNS background is 7.6× larger in CaWO₄ than in Al₂O₃ (∝ N²).
- The discrimination per DM event is about 20× worse.

**Figure:** `outputs/fog_all_materials.png` (pb on the left axis, cm² on the right). It has three panels: 30 meV, 100 meV (both with resolution) and 1 eV.

- Above about 0.3 GeV all materials converge, because both DM and neutrinos scatter off quasi-free nuclei.
- Below about 0.1 GeV they separate by up to an order of magnitude.

---

## 7. Directionality: daily modulation from crystal anisotropy

### 7.1 Method (new programs)

| File | Role |
|---|---|
| `directional.py` | S(q,ω;θ_q) on 24 Gauss–Legendre nodes in cos θ_q, using S(θ) = S(π − θ) and an average over 4 azimuths. The SHM enters through its Radon transform g(w, q̂), with ⟨2g⟩_Ω = η. Rate R(m, θ_E) for 10 angles between the c-axis and v_E. 30–60 s per material. |
| `check_directional.py` | The v_E-averaged rate must equal the isotropic rate. Si: 1.0000 at all masses; CaWO₄: within 0.3% (expected, because ⟨S(D)⟩ ≠ S(⟨D⟩) at second order). Uses `np.trapezoid` for NumPy 2. |
| `daily_modulation.py` | Maps θ_E to local sidereal time at LNGS (v_E toward Cygnus, α ≈ 21ʰ12ᵐ, δ ≈ +48°). A vertical c-axis sweeps θ_E from 6° to 90° per day. |
| `directional_summary.py`, `plot_directional.py` | Produce `outputs/directional_modulation.png` |
| `compare_stratman_lin.py` | Benchmark against Stratman & Lin (arXiv:2411.03433) under their conventions. Defaults to Al₂O₃ and keeps the standard output files. |

`nufog_core.lattice_response` gained a `use_cache` option, so the directional tables are not written to disk.

Before computing rates, the species-summed ⟨1/ω⟩ tensor was checked for anisotropy:

| Material | Azimuthal (φ) variation | Polar anisotropy (c vs ab) |
|---|---|---|
| CaWO₄ | None | W +48%, Ca −4%, O +1.4% |
| LiNbO₃ | None | Li +25%, O +10%, Nb −8% |
| Al₂O₃ | 2–4% | Al +5%, O +8% |
| TeO₂ | 3–6% | Te −7%, O −16% |
| Si | — | Isotropic |

So S ≈ S(θ_q) is a good approximation.

### 7.2 Results (σ_E = E_th/5, c-axis vertical at LNGS)

Signed asymmetry [R(0°) − R(90°)] / [R(0°) + R(90°)] at m_χ = 1 MeV:

| Material | 30 meV | 100 meV |
|---|---|---|
| CaWO₄ | +0.1% | −1.2% |
| Al₂O₃ | +0.7% | −2.4% |
| TeO₂ | +1.6% | +4.0% |
| LiNbO₃ | +1.6% | −4.3% |
| Si | 0 | 0 |

- **Mass range.** The effect is confined to m_χ ≲ 50 MeV. At higher masses the impulse regime dominates and the response is isotropic.
- **Sign reversal with threshold** (LiNbO₃, Al₂O₃; CaWO₄ near 20 MeV). Near threshold, the direction with softer modes wins. Above the highest optical phonon, the stiffer direction wins. A daily phase that flips between energy windows is hard for a background to mimic.
- **Confirmation, not discovery.** A 3σ detection of a modulation of amplitude A needs N ≈ 18/A² events, about 10⁴ for 4%.
- **Lower bound.** The incoherent approximation omits the anisotropy of the coherent single-phonon dispersion, so these values underestimate the low-q modulation.

### 7.3 Benchmark against Stratman & Lin (Al₂O₃)

Their conventions: v₀ = 220, v_E = 240, v_esc = 500 km/s; no resolution; c ∥ v_E at t = 0 with θ_e = 42°; A_mod = |R(0) − ⟨R⟩| / ⟨R⟩.

| m_χ | 120 meV | 200 meV | 300 meV |
|---|---|---|---|
| 10 MeV | 2.7% | 5.3% | 7.4% |
| 3–5 MeV | ≈ 5% | 8–9% | 9.8% (peak) |
| ≥ 30 MeV | < 0.5% | < 0.5% | < 0.5% |

They report up to about 11% at 10 MeV and 300 meV. The magnitude, the growth with threshold and the cut-off around 30–40 MeV agree; our value at 10 MeV is about 30% lower. Plausible causes:
- independent DFT inputs;
- our azimuthal averaging;
- their impulse-approximation width taken along ẑ;
- their q_BZ cut-off.

Under the same conventions CaWO₄ gives 8.1% at 1 MeV and 300 meV, plus a second bump at 40–70 MeV from W.

**Thesis text:** `directional_section.tex`.

---

## 8. Neutrino fog vs CRESST

### 8.1 Literature curves (`NuFog_v2/literature/`)

| File | Source |
|---|---|
| `CRESST-III_2019_DetA_OHare.txt` | Published data, O'Hare NeutrinoFog repository |
| `CRESST-III_2024_SOS.txt` | Digitized from CRESST Commun. Phys. 2025, Fig. 6 |
| `CRESST_upgrade_projection_100xLEE.txt` | Digitized, Fig. 7 |
| `Bento2024_CaWO4_*.txt` | Digitized, Fig. 7 (discovery curves and grey fog region) |
| `digitize_cresst.py` | Digitizer. Its own digitization of the 2019 curve agrees with the published file to 0.03–0.08 dex. |

### 8.2 Figure `outputs/fog_vs_cresst.png` (`plot_fog_cresst.py`; pb left, cm² right)

**Panel (a), landscape:**
- At 1 GeV, CRESST-III is about six orders of magnitude above the fog, and the upgrade about two. CRESST is limited by the low-energy excess (LEE), not by neutrinos.
- CRESST reaches down to 73 MeV (SOS). A 30 meV phonon threshold extends the neutrino-limited region down to about 0.5 MeV.
- In that region CaWO₄, CRESST's baseline material, is the weakest target. Al₂O₃ and Si, which CRESST already operates, are better by more than 10×.

**Panel (b), validation at E_th = 1 eV, 1 t·yr.** Our CaWO₄ curve reproduces the shape of Bento et al. (2024), including the ⁸B feature at 0.5 GeV and the shoulder at 0.1 GeV. It is lower by a factor of 0.61–0.78, partly from digitization and partly plausibly from different flux inputs and low-energy treatment.

**Thesis text:** `fog_cresst_section.tex`.

---

## 9. Exposure scan: is it a fog or a background? (`exposure_scan.py`)

The discovery cross section was computed for exposures from 10⁻⁶ to 10³ t·yr. The fog index is n = −d ln σ / d ln X:

| n | Meaning |
|---|---|
| ≈ 1 | Background-free |
| ≈ 0.5 | Poisson background-limited |
| < 0.5 | Systematics-saturated (the fog) |

O'Hare's index is the inverse of this one, so his fog condition n > 2 corresponds to n < 0.5 here.

**Results (E_th = 30 meV, σ_E = E_th/5):**

| | Al₂O₃ | CaWO₄ |
|---|---|---|
| Neutrino events above threshold | 3.7×10³ / t·yr (pp 52%, ⁷Be 37%, ⁸B 3%) | 2.8×10⁴ / t·yr |
| Same, per kg·day | ≈ 0.01 | ≈ 0.08 |
| n at 1 t·yr, 1–20 MeV | 0.53–0.54 | 0.49–0.50 |
| n at ≥ 3 t·yr, ≈ 100 MeV | 0.43–0.47 | 0.43–0.46 |
| n at 0.5 GeV (0.1–10 t·yr) | 0.21–0.40 | 0.24–0.39 |
| σ_disc at 1 kg·yr, m = 1 / 20 MeV | 2.8×10⁻⁴³ / 2.5×10⁻⁴⁴ cm² | 2.5×10⁻⁴² / 1.2×10⁻⁴³ cm² |
| σ_disc at 1 t·yr, m = 1 / 20 MeV | 3.3×10⁻⁴⁵ / 3.4×10⁻⁴⁶ cm² | 5.4×10⁻⁴⁴ / 2.8×10⁻⁴⁵ cm² |

**Conclusion.**
- **Phonon regime (1–20 MeV).** The neutrino background behaves as a Poisson background (n → 0.50 and never below). The multiphonon DM spectrum differs from the pp and ⁷Be CEvNS spectra, so DM stays discoverable and the reach improves as X^{−1/2}.
- **Around 100 MeV.** A mild saturation (n ≈ 0.43–0.47) starts above a few t·yr.
- **Near 0.5 GeV.** A genuine fog (n ≈ 0.2–0.4) appears, because the DM spectrum mimics ⁸B.

The figure is `outputs/exposure_scan.png` (`plot_exposure_scan.py`).

**Detector requirements to reach this regime:**
1. **Threshold** ≈ 30 meV with σ_E ≈ 6 meV. Today's best CRESST threshold is 6.7 eV, so this is two to three orders of magnitude lower.
2. **Exposure.** About 0.1–1 t·yr for Al₂O₃ before neutrinos become the limit; a few kg·yr is already competitive.
3. **Non-neutrino backgrounds** above threshold must stay below about 0.01 events/(kg·day) for Al₂O₃. This is the hardest requirement: the LEE observed today is many orders of magnitude larger.

---

## 10. Why S(q,ω) must be known precisely

The full argument is in Sec. 9 of the report.

1. **Exponential sensitivity.** The low-mass rate carries e^{−2W}, with W = q²⟨1/ω⟩/4m. ⟨1/ω⟩ is dominated by the softest acoustic modes. An error δ on ⟨1/ω⟩ changes the rate by a factor e^{−2Wδ}, which is large exactly where multiphonon emission matters (W ~ 1–10). The CaWO₄ tensor correction alone moved the reach by 1.5–1.65×.
2. **Threshold position.** The multiphonon tail above E_th depends on the high-frequency edge and the shape of D(ω). At 0.1 eV, resolution plus the 110 meV optical peak changed the CaWO₄ reach by 1.4×.
3. **Discrimination needs the spectral shape.** DM is separated from neutrinos (n ≈ 0.5 instead of a fog) only because the shapes differ. DM and the low-energy CEvNS background both pass through the same S(q,ω), weighted differently in q. An uncertain S acts as a correlated nuisance and can turn a background into a fog.
4. **Target choice.** The material ranking (16× spread at low mass) and the multiphonon gap (2.9–38×) are properties of D(ω).
5. **Directionality** depends on the per-atom anisotropy of the DOS tensor, so it needs accurate eigenvectors, not only frequencies.
6. **Known systematics of the inputs:**
   - DFT softening (Al₂O₃ acoustic DOS about 20% high);
   - small supercells, giving soft TA branches and imaginary pockets (CaWO₄, Si);
   - missing LO–TO correction (CaWO₄);
   - harmonic, T = 0 approximation;
   - incoherent approximation (valid for q above a few keV).

---

## 11. Open items

- **CaWO₄ force constants** (deferred). Converged 2×2×2 supercell plus BORN, then rerun. This is the most valuable single improvement for the thesis material.
- **Directional mass grid.** Rerun on a finer grid (about 40 masses) for the final figure.
- **Solar-neutrino directionality.** Add the directional CEvNS term; it modulates with the solar day, not the sidereal day.
- **S(q,ω) uncertainty in the likelihood.** Treat it as a nuisance (for example, scale ⟨1/ω⟩ and the acoustic slopes) to quantify point 3 of Sec. 10.
- **Thesis edits:**
  - check the `\ref{sec:...}` labels in `directional_section.tex` and `fog_cresst_section.tex`;
  - move the BibTeX entries into the `.bib`;
  - add the DarkELF/Campbell-Deem comparison figure (`fig:lin_comparison`).

## 12. Literature

- CRESST Collaboration, *The CRESST experiment towards the next generation of sub-GeV direct dark matter detection*, Commun. Phys. (2025), doi:10.1038/s42005-025-02476-5
- A. Bento et al., *Solar neutrinos in cryogenic detectors*, Eur. Phys. J. C 84, 1118 (2024)
- B. Campbell-Deem, S. Knapen, T. Lin, E. Villarama, PRD 106, 036019 (2022), arXiv:2205.02250
- S. Knapen, J. Kozaczuk, T. Lin, *DarkELF*, PRD 105, 015014 (2022), arXiv:2104.12786
- C. Stratman, T. Lin, *Daily modulation of low-energy nuclear recoils from sub-GeV dark matter*, arXiv:2411.03433
- S. Griffin, S. Knapen, T. Lin, K. Zurek, PRD 98, 115034 (2018), arXiv:1807.10291
- C. A. J. O'Hare, *New definition of the neutrino floor for direct dark matter searches*, PRL 127, 251802 (2021); data from github.com/cajohare/NeutrinoFog
