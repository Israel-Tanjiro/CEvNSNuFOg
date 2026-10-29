#!/usr/bin/env python3
"""
UNIFIED DYNAMIC STRUCTURE FACTOR MODULE (DM & CEvNS, Multiphonon + Impulse)
Save this file as BOTH:
  - dynamic_structure_factor_Impulse.py
  - dynamic_structure_factor_neutrinos.py
"""

import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d
import math, pickle
from time import time

# ========== PHYSICAL CONSTANTS ==========
AMU_TO_EVC2 = 931.49410242e6      # eV/c^2
HBAR_C_EV_ANGSTROM = 1973.269804   # eV*A

# ========== MATERIAL DATABASE ==========
MATERIAL_DATABASE = {
    'Al2O3': {
        'name': 'Al₂O₃ (Sapphire)',
        'formula': 'Al₂O₃',
        'space_group': 'R-3c (No. 167)',
        'N_atoms': 10,
        'atoms': [
            {'symbol': 'Al', 'n': 4, 'mass_amu': 26.9815385, 'A': 27, 'Z': 13},
            {'symbol': 'O',  'n': 6, 'mass_amu': 15.9994,    'A': 16, 'Z': 8}
        ],
        'V_prim_A3': 233.0 / 3.0,
        'density_g_cm3': 3.99,
        'color': '#1b9e77',
    },
    'TeO2': {
        'name': 'TeO₂ (Paratellurite)',
        'formula': 'TeO₂',
        'space_group': 'P4₁2₁2 (No. 92)',
        'N_atoms': 12,
        'atoms': [
            {'symbol': 'Te', 'n': 4, 'mass_amu': 127.6,   'A': 128, 'Z': 52},
            {'symbol': 'O',  'n': 8, 'mass_amu': 15.9994, 'A': 16,  'Z': 8}
        ],
        'V_prim_A3': 4.82073203**2 * 7.44328478,
        'density_g_cm3': 5.67,
        'color': '#d95f02',
    },
    'Si': {
        'name': 'Si (diamond)',
        'formula': 'Si',
        'space_group': 'Fd-3m (No. 227)',
        'N_atoms': 2,
        'atoms': [
            {'symbol': 'Si', 'n': 2, 'mass_amu': 28.0855, 'A': 28, 'Z': 14}
        ],
        'V_prim_A3': 40.127,          # phonopy primitive cell (phonon database, PBEsol)
        'density_g_cm3': 2.33,
        'color': '#7570b3',
    },
    'LiNbO3': {
        'name': 'LiNbO3 (lithium niobate)',
        'formula': 'LiNbO3',
        'space_group': 'R3c (No. 161)',
        'N_atoms': 10,
        'atoms': [                     # order of the phonopy primitive cell: Li2 O6 Nb2
            {'symbol': 'Li', 'n': 2, 'mass_amu': 6.941,    'A': 7,  'Z': 3},
            {'symbol': 'O',  'n': 6, 'mass_amu': 15.9994,  'A': 16, 'Z': 8},
            {'symbol': 'Nb', 'n': 2, 'mass_amu': 92.90638, 'A': 93, 'Z': 41}
        ],
        'V_prim_A3': 106.793,
        'density_g_cm3': 4.64,
        'color': '#66a61e',
    },
    'CaWO4': {
        'name': 'CaWO₄ (Calcium Tungstate)',
        'formula': 'CaWO4',
        'space_group': 'I4₁/a (No. 88)',
        'N_atoms': 24,
        'atoms': [
            {'symbol': 'Ca', 'n': 4,  'mass_amu': 40.078,  'A': 40,  'Z': 20},
            {'symbol': 'W',  'n': 4,  'mass_amu': 183.84,  'A': 184, 'Z': 74},
            {'symbol': 'O',  'n': 16, 'mass_amu': 15.9994, 'A': 16,  'Z': 8}
        ],
        'V_prim_A3': 312.6,  # Consistent 24-atom cell volume for both DM and CEvNS
        'density_g_cm3': 6.06,
        'color': '#e7298a',
    },
}


class DynamicStructureFactor:

    def __init__(self, tensor_npz, material='auto', n_max=20, interaction='DM'):
        self.tensor_npz = tensor_npz
        self.n_max = n_max
        self.interaction = interaction

        data = np.load(tensor_npz, allow_pickle=True)
        self.omega_eV = data['omega']
        self.dw_eV = self.omega_eV[1] - self.omega_eV[0]
        self.omega_max_eV = self.omega_eV[-1]

        atom_ids = sorted([int(k.split('_')[1]) for k in data.files if k.startswith('atom_') and k.split('_')[1].isdigit()])
        self.N_atoms = len(atom_ids)

        if material == 'auto':
            material = self._detect_material()
        self.material = material

        if material not in MATERIAL_DATABASE:
            raise ValueError(f"Material '{material}' not found. Available: {list(MATERIAL_DATABASE.keys())}")

        info = MATERIAL_DATABASE[material]
        self.info = info
        self.V_prim_A3 = info['V_prim_A3']

        self.D_d_ij = np.zeros((self.N_atoms, 3, 3, len(self.omega_eV)))
        self.atom_masses_eV = np.zeros(self.N_atoms)
        self.atom_symbols = []
        self.atom_A = np.zeros(self.N_atoms)
        self.atom_Z = np.zeros(self.N_atoms)

        current_idx = 0
        for atom_type in info['atoms']:
            for _ in range(atom_type['n']):
                if current_idx < self.N_atoms:
                    self.atom_masses_eV[current_idx] = atom_type['mass_amu'] * AMU_TO_EVC2
                    self.atom_symbols.append(atom_type['symbol'])
                    self.atom_A[current_idx] = atom_type['A']
                    self.atom_Z[current_idx] = atom_type['Z']
                    current_idx += 1

        for idx, aid in enumerate(atom_ids):
            self.D_d_ij[idx] = data[f'atom_{aid}']

        self._set_form_factors()

        self.norm_factor = (2.0 * np.pi) / self.V_prim_A3 * (HBAR_C_EV_ANGSTROM)**3
        self.Tn_cache = {}
        self.w_bar_z = np.zeros(self.N_atoms)
        self.omega_inv_avg = {}
        self.q_threshold_d_eV = np.zeros(self.N_atoms)
        self._cache_loaded = False

        print(f"  {info['name']} ({self.interaction}): {self.N_atoms} atoms, V = {self.V_prim_A3:.1f} Å³")

    def _detect_material(self):
        if self.N_atoms == 10: return 'Al2O3'
        elif self.N_atoms == 12: return 'TeO2'
        elif self.N_atoms == 24: return 'CaWO4'
        return 'Al2O3'

    def _set_form_factors(self):
        """Set coupling factors for either Dark Matter (A^2) or CEvNS (Q_w^2)."""
        self.form_factors = np.zeros(self.N_atoms)
        if self.interaction == 'DM':
            for d in range(self.N_atoms):
                self.form_factors[d] = self.atom_A[d]**2
        elif self.interaction == 'CEvNS':
            SIN2_THETA_W = 0.23857
            gVp = 0.5 - 2.0 * SIN2_THETA_W
            gVn = -0.5
            for d in range(self.N_atoms):
                Z, A = self.atom_Z[d], self.atom_A[d]
                N = A - Z
                Q_w = Z * gVp + N * gVn
                self.form_factors[d] = Q_w**2
        else:
            raise ValueError(f"Unrecognized interaction '{self.interaction}'. Use 'DM' or 'CEvNS'.")

    def precompute_Tn(self, force=False):
        if self._cache_loaded and not force:
            return

        print("  Precomputing T_n and impulse parameters...", end=" ", flush=True)
        t0 = time()

        omega_out_max_eV = 0.6
        N_out = 2000
        self.omega_out = np.linspace(0, omega_out_max_eV, N_out)
        dirs = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}
        nonzero = self.omega_eV > 1e-12

        for d in range(self.N_atoms):
            for dir_name, q_hat_vec in dirs.items():
                q_hat = np.array(q_hat_vec, dtype=np.float64)
                D_d_q = np.einsum('i,ijk,j->k', q_hat, self.D_d_ij[d], q_hat)

                T1_raw = np.zeros_like(self.omega_eV)
                T1_raw[nonzero] = D_d_q[nonzero] / self.omega_eV[nonzero]
                omega_inv = integrate.trapezoid(T1_raw, self.omega_eV)
                self.omega_inv_avg[(d, dir_name)] = omega_inv

                if dir_name == 'z':
                    self.w_bar_z[d] = integrate.trapezoid(self.omega_eV * D_d_q, self.omega_eV)
                    omega_0_d = 1.0 / omega_inv if omega_inv > 0 else 0.0
                    # Per-atom impulse threshold so light atoms (O, Ca, Al) switch at their own scale
                    self.q_threshold_d_eV[d] = 4.0 * np.sqrt(2.0 * self.atom_masses_eV[d] * omega_0_d)

                T1_interp = interp1d(self.omega_eV, T1_raw, fill_value=0.0, bounds_error=False)
                self.Tn_cache[(d, dir_name)] = self._compute_Tn(T1_interp)

        self._cache_loaded = True
        print(f"✓ {time() - t0:.1f}s")

    def _compute_Tn(self, T1_interp):
        Tn = {1: T1_interp(self.omega_out)}
        T_prev = interp1d(self.omega_out, Tn[1], fill_value=0.0, bounds_error=False)
        T1_vals = T1_interp(self.omega_eV)

        for n in range(2, self.n_max + 1):
            Tn_n = np.zeros(len(self.omega_out))
            for i, W in enumerate(self.omega_out):
                if W <= n * self.omega_eV[-1]:
                    shifted = T_prev(W - self.omega_eV)
                    Tn_n[i] = integrate.trapezoid(T1_vals * shifted, self.omega_eV)
            Tn[n] = Tn_n
            T_prev = interp1d(self.omega_out, Tn_n, fill_value=0.0, bounds_error=False)
        return Tn

    def S_q_omega(self, q_keV, q_hat=np.array([0., 0., 1.]), n_points=500,
                  omega_grid_eV=None, dq_keV=0.0):
        """
        Compute S(q, omega) [in keV^-2].
        If omega_grid_eV is supplied, evaluates directly on omega_grid_eV and
        regularizes the Gaussian width with the (q, omega) grid spacing so narrow
        nuclear recoil peaks never fall between grid points.
        """
        if not self._cache_loaded:
            self.precompute_Tn()

        q_eV = q_keV * 1000.0
        dq_eV = dq_keV * 1000.0
        q_hat = np.asarray(q_hat, dtype=np.float64) / np.linalg.norm(q_hat)

        dirs = {'x': np.array([1., 0., 0.]), 'y': np.array([0., 1., 0.]), 'z': np.array([0., 0., 1.])}
        best_dir = min(dirs, key=lambda k: np.linalg.norm(q_hat - dirs[k]))

        if omega_grid_eV is not None:
            omega_grid = np.asarray(omega_grid_eV, dtype=np.float64)
        else:
            min_m_eV = np.min(self.atom_masses_eV)
            max_recoil_eV = q_eV**2 / (2.0 * min_m_eV)
            max_delta_eV = np.sqrt(max_recoil_eV * np.max(self.w_bar_z))
            grid_max_eV = max(0.6, max_recoil_eV + 5.0 * max_delta_eV)
            omega_grid = np.linspace(0.0, grid_max_eV, n_points)

        dw_grid = np.gradient(omega_grid)
        S_total = np.zeros_like(omega_grid, dtype=np.float64)

        # Evaluate multiphonon interpolation on omega_grid only where omega <= 0.6 eV
        mp_mask = omega_grid <= self.omega_out[-1]
        omega_mp = omega_grid[mp_mask]

        for d in range(self.N_atoms):
            m_d = self.atom_masses_eV[d]
            FF_sq = self._helm_form_factor_sq(q_eV, self.atom_A[d])
            weight_d = self.form_factors[d] * FF_sq

            if q_eV > self.q_threshold_d_eV[d]:
                # IMPULSE APPROXIMATION (per-atom regime switch)
                center = q_eV**2 / (2.0 * m_d)
                Delta_d_sq = (q_eV**2 * self.w_bar_z[d]) / (2.0 * m_d)

                # Include grid step resolution so narrow recoil peaks at high q are never missed
                d_center_dq = (q_eV * dq_eV) / m_d
                sigma_eff_sq = Delta_d_sq + (0.5 * d_center_dq)**2 + (0.5 * dw_grid)**2

                gaussian = (1.0 / np.sqrt(2.0 * np.pi * sigma_eff_sq)) * \
                           np.exp(-0.5 * ((omega_grid - center)**2) / sigma_eff_sq)
                S_total += weight_d * gaussian
            else:
                # MULTIPHONON EXPANSION
                if not np.any(mp_mask):
                    continue
                integral_DW = self.omega_inv_avg[(d, best_dir)]
                W_d = (q_eV**2 / (4.0 * m_d)) * integral_DW
                DW = np.exp(-2.0 * W_d)

                scale = q_eV**2 / (2.0 * m_d)
                if scale <= 0.0:
                    continue
                log_scale = np.log(scale)
                Tn_dict = self.Tn_cache[(d, best_dir)]

                S_mp_d = np.zeros_like(omega_mp)
                for n in range(1, self.n_max + 1):
                    log_prefactor = n * log_scale - math.lgamma(n + 1)
                    prefactor = np.exp(log_prefactor)
                    if np.isfinite(prefactor) and prefactor > 1e-30:
                        Tn_vals = np.interp(omega_mp, self.omega_out, Tn_dict[n], left=0.0, right=0.0)
                        S_mp_d += prefactor * Tn_vals

                S_total[mp_mask] += weight_d * DW * S_mp_d

        S_total *= self.norm_factor * 1e-6
        return S_total, omega_grid

    def S_q_integrated(self, q_keV, q_hat=np.array([0., 0., 1.])):
        S, omega_eV = self.S_q_omega(q_keV, q_hat)
        return integrate.trapezoid(S, omega_eV * 1e-3)

    def _helm_form_factor_sq(self, q_eV, A):
        if q_eV < 1e-6:
            return 1.0
        HC_EV_FM = 197326980.4
        q_fm = q_eV / HC_EV_FM
        s = 0.9
        a = 0.52
        c = 1.23 * (A**(1.0 / 3.0)) - 0.60
        R_n = np.sqrt(c**2 + (7.0 / 3.0) * (np.pi**2) * (a**2) - 5.0 * (s**2))
        qr = q_fm * R_n
        if qr < 1e-4:
            return 1.0
        j1 = (np.sin(qr) - qr * np.cos(qr)) / (qr**2)
        F_q = 3.0 * j1 / qr * np.exp(-0.5 * (q_fm * s)**2)
        return F_q**2
