"""
Corrected CaWO4 projected phonon DOS tensor, same npz format as pDoS_tensor_v4 (omega + atom_<i>).
24-atom conventional cell (Ca4 W4 O16, POSCAR order), full Gamma-centred mesh without mesh
symmetry (required for atom-projected DOS), Gaussian sigma = 0.5 meV, each D_ii normalized to 1.
"""
import sys, time, numpy as np, phonopy
THz_to_eV = 4.135667696e-3
mesh = [int(x) for x in sys.argv[1:4]]
ph = phonopy.load('phonopy.yaml', force_sets_filename='FORCE_SETS', primitive_matrix='P', produce_fc=True, log_level=0)
ph.symmetrize_force_constants()
N = len(ph.primitive.masses); sym = ph.primitive.symbols
assert sym == ['Ca']*4 + ['W']*4 + ['O']*16, sym
t = time.time()
ph.run_mesh(mesh, is_mesh_symmetry=False, with_eigenvectors=True, is_gamma_center=True)
f = ph.mesh.frequencies * THz_to_eV; E = ph.mesh.eigenvectors; nq = len(f)
print(f"mesh {mesh}: {nq} q-points diagonalized in {time.time()-t:.0f}s; freq range {f.min()*1e3:.2f} .. {f.max()*1e3:.1f} meV")
# histogram of e_ai e*_aj on a fine grid, then Gaussian smearing
w = np.linspace(1e-6, 0.118, 1000); dw = w[1] - w[0]
fine = np.arange(0.0, 0.118 + 0.1 * dw, 0.1 * dw)
H = np.zeros((N, 3, 3, len(fine)))
for iq in range(nq):
    ok = f[iq] > 1e-5
    e = E[iq].T[ok].reshape(-1, N, 3)                         # (modes, atom, cart)
    idx = np.clip(np.round(f[iq][ok] / (0.1 * dw)).astype(int), 0, len(fine) - 1)
    np.add.at(H, (slice(None), slice(None), slice(None), idx),
              np.einsum('mai,maj->aijm', e, e.conj()).real)
sig = 5e-4
K = np.exp(-0.5 * ((w[:, None] - fine[None, :]) / sig)**2) / (np.sqrt(2 * np.pi) * sig)   # (w, fine)
D = np.einsum('aijf,wf->aijw', H, K) / nq
for a in range(N):                                              # normalize each diagonal to 1
    for i in range(3):
        D[a, i, i] /= np.trapezoid(D[a, i, i], w)
out = {'omega': w, 'mesh': np.array(mesh), 'sigma': np.array(sig)}
out.update({f'atom_{a}': D[a] for a in range(N)})
name = f'pDoS_tensor_v5_mesh{mesh[0]}x{mesh[1]}x{mesh[2]}_CaWO4.npz'
np.savez(name, **out)
tr = np.trace(D, axis1=1, axis2=2)
mw = 1e3 * np.trapezoid(w * tr, w) / np.trapezoid(tr, w)
for s in ('Ca', 'W', 'O'):
    ids = [a for a in range(N) if sym[a] == s]
    zz = [1e3 * np.trapezoid(w * D[a, 2, 2], w) for a in ids]
    print(f"  {s:2s}: <omega>_Tr = {mw[ids].mean():6.2f} meV (spread {np.ptp(mw[ids]):.3f}),  <omega>_zz = {np.mean(zz):6.2f} (spread {np.ptp(zz):.3f})")
print(f"saved {name}  ({time.time()-t:.0f}s total)")
