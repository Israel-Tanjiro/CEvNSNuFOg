"""
NuFog v2 - central configuration.

Pipeline (run from this folder):
  python3 step1_neutrinos.py   -> CEvNS spectra for every neutrino source
  python3 step2_dm_matrix.py   -> DM spectra on the mass grid (events / kg / yr / cm^2)
  python3 step3_discovery.py   -> 3-sigma discovery curve sigma(m_chi)
  python3 step4_plot.py        -> overlay of the cases listed in PLOT_CASES
or simply:  python3 run_all.py
Sanity check of the absolute normalization:  python3 check_normalization.py

The lattice response S(q, omega) is computed once per (material, n_max, q direction,
grids) and cached in outputs/cache; changing the threshold, exposure or mass range
never recomputes it.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = BASE_DIR                 # neutrino flux tables (pp.dat, test1P.dat = 8B, ...)
TENSOR_DIR = BASE_DIR               # phonon tensor .npz files
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
CACHE_DIR = os.path.join(OUTPUT_DIR, 'cache')

# Target material; can also be chosen per run with:  NUFOG_MATERIAL=Al2O3 python3 run_all.py
MATERIAL = os.environ.get('NUFOG_MATERIAL', 'CaWO4')
TENSOR_FILES = {
    'Al2O3': 'pDoS_tensor_v4_mesh100.npz',
    # v5: rebuilt from phonopy.yaml + FORCE_SETS on a full mesh (make_tensor.py). The v4 file
    # broke the crystal symmetry (W <omega> off by 30-75%); kept only for comparison.
    'CaWO4': 'pDoS_tensor_v5_mesh32x32x16_CaWO4.npz',
    'CaWO4_v4': 'data/tensors_superseded/pDoS_tensor_v4_mesh100_CaWO4.npz',
    # TeO2: arrays reordered to Te x4, O x8 (the original file stored O x8, Te x4 under Te-first labels)
    'TeO2': 'pDoS_tensor_TeO2_reordered.npz',
    # Si: Togo phonon database (PBEsol, 64-atom supercell). TA ~15% soft, LA/optical within 1-6%.
    # (pDoS_tensor_Si.npz from the 16-atom supercell had TA ~40% soft; superseded.)
    'Si': 'pDoS_tensor_Si_phonondb.npz',
    'LiNbO3': 'pDoS_tensor_LiNbO3.npz',
}

# ---- Direction of the momentum transfer relative to the crystal axes ----
# 'z' = c-axis (what v1 used everywhere), 'x' = a-axis, 'avg' = isotropic Tr(D)/3,
# or an explicit vector such as (1, 0, 1).
Q_DIR = 'z'

# ---- Energy binning (shared by DM and neutrino spectra) ----
NBINS = 200
ER_MIN_KEV = 1e-6          # 1 meV
ER_MAX_KEV = 100.0

# ---- Numerical grids for S(q, omega) ----
Q_MIN_KEV = 1e-3           # 1 eV
Q_MAX_KEV = 2.0e5          # covers CEvNS of the most energetic sources and 10 GeV DM
Q_PER_DECADE = 300
OMEGA_PHONON_MAX_MEV = 300.0   # linear grid (multiphonon regime) up to here
N_OMEGA_PHONON = 500
N_OMEGA_IMPULSE = 700          # log grid from OMEGA_PHONON_MAX_MEV up to ER_MAX_KEV

# ---- Phonon expansion order ----
# The impulse approximation takes over at q = 4 sqrt(2 m omega_0), where the mean phonon
# number is <n> = 16 for every atom; converging the expansion there needs n up to ~16 + 6*4.
N_MAX = 60                 # DM signal: 1 = single phonon only, 60 = full multiphonon
N_MAX_NEUTRINO = 60        # neutrino background: always the full expansion

# Gaussian smearing of the phonon DOS leaves D(0) finite, so D/omega diverges at omega -> 0.
# Below this cutoff D is replaced by the acoustic Debye form D ~ omega^2 (set 0 to disable).
DOS_IR_CUTOFF_MEV = 2.0

# ---- Dark matter ----
M_CHI_MIN_MEV = 0.5
M_CHI_MAX_MEV = 10000.0
N_MASSES = 200
RHO_CHI_GEV_CM3 = 0.3
V0_KMS, VE_KMS, VESC_KMS = 220.0, 232.0, 544.0

# ---- Statistics ----
E_TH_KEV = 3e-5            # analysis threshold: bins with lower edge >= E_TH are used
EXPOSURE_KG_YR = 1000.0    # 1 tonne-year
# Detector energy resolution: Gaussian with sigma = RESOLUTION_FRAC * E_th
# (CRESST convention: sigma_0 = E_th / 5  ->  0.2). Set 0 for a perfect detector.
RESOLUTION_FRAC = 0.2
DISCOVERY_Q0 = 9.0         # Asimov q0 = 9  <->  3 sigma
SIGMA_SEARCH_CM2 = (1e-52, 1e-20)
SIGMA_TOL_DEX = 1e-3


# =================== labels (do not edit) ===================
def dir_tag(qdir=None):
    d = Q_DIR if qdir is None else qdir
    if isinstance(d, str):
        return {'z': 'qc', 'x': 'qa', 'y': 'qb', 'avg': 'qavg'}.get(d, f"q{d}")
    return "q" + "_".join(f"{v:g}" for v in d)


def binning_tag(er_min_keV=None):
    er = ER_MIN_KEV if er_min_keV is None else er_min_keV
    return f"Emin{er*1e3:g}eV"


def dm_tag(n_max=None, qdir=None, m_min=None, m_max=None):
    n = N_MAX if n_max is None else n_max
    lo = M_CHI_MIN_MEV if m_min is None else m_min
    hi = M_CHI_MAX_MEV if m_max is None else m_max
    return f"{MATERIAL}_{binning_tag()}_{dir_tag(qdir)}_m{lo:g}-{hi:g}MeV_n{n}"


def neutrino_tag(qdir=None):
    return f"{MATERIAL}_{binning_tag()}_{dir_tag(qdir)}_n{N_MAX_NEUTRINO}"


def run_tag(e_th_keV=None, n_max=None, qdir=None, exposure_kg_yr=None, res_frac=None):
    e = E_TH_KEV if e_th_keV is None else e_th_keV
    x = EXPOSURE_KG_YR if exposure_kg_yr is None else exposure_kg_yr
    r = RESOLUTION_FRAC if res_frac is None else res_frac
    res = f"_res{r:g}Eth" if r > 0 else ""
    return f"Eth{e*1e3:g}eV{res}_{x/1000.0:g}tyr_{dm_tag(n_max=n_max, qdir=qdir)}"


NEUTRINO_DIR = os.path.join(OUTPUT_DIR, f"neutrinos_{neutrino_tag()}")

# ---- Curves overlaid by step4_plot.py:
# (E_th [keV], n_max, Q_DIR, resolution sigma/E_th, label, color, linestyle)
# The free-nucleus cutoff of each threshold is drawn in the colour of its first curve.
PLOT_CASES = [
    (3e-5, 60, 'z', 0.0, r'$E_{th}$ = 30 meV, perfect resolution',          '#1f78b4', ':'),
    (3e-5, 60, 'z', 0.2, r'$E_{th}$ = 30 meV, $\sigma_E = E_{th}/5$',        '#1f78b4', '-'),
    (3e-5, 1,  'z', 0.2, r'$E_{th}$ = 30 meV, $\sigma_E = E_{th}/5$, $n=1$ (+ impulse)', '#1f78b4', '--'),
    (1e-4, 60, 'z', 0.2, r'$E_{th}$ = 0.1 eV, $\sigma_E = E_{th}/5$',        '#33a02c', '-'),
    (1e-3, 60, 'z', 0.2, r'$E_{th}$ = 1 eV, $\sigma_E = E_{th}/5$',          '#6a3d9a', '-'),
]

for _d in (OUTPUT_DIR, CACHE_DIR, NEUTRINO_DIR):
    os.makedirs(_d, exist_ok=True)
