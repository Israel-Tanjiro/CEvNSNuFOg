#!/usr/bin/env python3
"""
Projected phonon DOS tensor D_d^{ij}(omega) for any material from phonopy input, in the
NuFog npz format (omega + atom_<i> + metadata). Full Gamma-centred mesh WITHOUT mesh symmetry
(required for atom-projected quantities), Gaussian smearing, each D_ii normalized to 1.
Non-analytic (LO-TO) correction is applied when Born charges are available.

  python3 make_tensor_any.py --yaml phonopy_params.yaml --mesh 24 24 24 --out pDoS_tensor_LiNbO3.npz
  python3 make_tensor_any.py --yaml phonopy_disp.yaml --force-sets FORCE_SETS --mesh 40 40 40 --out pDoS_tensor_Si.npz
"""
import argparse, time, numpy as np, phonopy

THz_to_eV = 4.135667696e-3
ap = argparse.ArgumentParser()
ap.add_argument('--yaml', required=True)
ap.add_argument('--force-sets', default=None)
ap.add_argument('--born', default=None)
ap.add_argument('--primitive', default='auto', help="'auto' (primitive cell) or 'P' (cell as given)")
ap.add_argument('--mesh', type=int, nargs=3, default=[24, 24, 24])
ap.add_argument('--sigma-meV', type=float, default=0.5)
ap.add_argument('--n-omega', type=int, default=1000)
ap.add_argument('--out', required=True)
a = ap.parse_args()

kw = dict(primitive_matrix=a.primitive, produce_fc=True, log_level=0)
if a.force_sets: kw['force_sets_filename'] = a.force_sets
if a.born: kw.update(born_filename=a.born, is_nac=True)
ph = phonopy.load(a.yaml, **kw)
ph.symmetrize_force_constants()
sym = list(ph.primitive.symbols); N = len(sym)
t = time.time()
ph.run_mesh(a.mesh, is_mesh_symmetry=False, with_eigenvectors=True, is_gamma_center=True)
f = ph.mesh.frequencies * THz_to_eV; E = ph.mesh.eigenvectors; nq = len(f)
wmax = 1.08 * f.max()
w = np.linspace(0.0, wmax, a.n_omega); dw = w[1] - w[0]
fine = np.arange(0.0, wmax + 0.1 * dw, 0.1 * dw)
H = np.zeros((N, 3, 3, len(fine)))
for iq in range(nq):
    ok = f[iq] > 1e-5
    e = E[iq].T[ok].reshape(-1, N, 3)
    idx = np.clip(np.round(f[iq][ok] / (0.1 * dw)).astype(int), 0, len(fine) - 1)
    np.add.at(H, (slice(None), slice(None), slice(None), idx), np.einsum('mai,maj->aijm', e, e.conj()).real)
sig = a.sigma_meV * 1e-3
K = np.exp(-0.5 * ((w[:, None] - fine[None, :]) / sig)**2) / (np.sqrt(2 * np.pi) * sig)
D = np.einsum('aijf,wf->aijw', H, K) / nq
for i_a in range(N):
    for i in range(3):
        D[i_a, i, i] /= np.trapezoid(D[i_a, i, i], w)
out = {'omega': w, 'mesh': np.array(a.mesh), 'sigma': np.array(sig),
       'meta_atom_types': np.array(sym), 'meta_masses_amu': np.array(ph.primitive.masses),
       'meta_V_cell_A3': np.array(ph.primitive.volume), 'meta_nac': np.array(ph.nac_params is not None),
       'meta_space_group': np.array(ph.symmetry.get_international_table())}
out.update({f'atom_{i}': D[i] for i in range(N)})
np.savez(a.out, **out)
tr = np.trace(D, axis1=1, axis2=2); mw = 1e3 * np.trapezoid(w * tr, w) / np.trapezoid(tr, w)
print(f"{a.out}: {''.join(sym)} ({ph.symmetry.get_international_table()}), V = {ph.primitive.volume:.3f} A^3, "
      f"NAC = {ph.nac_params is not None}, {nq} q, freq {f.min()*1e3:.2f}..{f.max()*1e3:.1f} meV ({time.time()-t:.0f}s)")
for s in dict.fromkeys(sym):
    ids = [i for i in range(N) if sym[i] == s]
    print(f"   {s:2s} x{len(ids)}: <omega> = {mw[ids].mean():6.2f} meV (spread {np.ptp(mw[ids]):.3f})")
