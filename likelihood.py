"""
Asimov discovery test statistic with Gaussian-constrained neutrino flux nuisances.

  -ln L0(theta) = sum_b [ n_b - N_b ln n_b ] + sum_k (theta_k - 1)^2 / (2 r_k^2),   n = theta @ A
  q0 = 2 [ min_theta (-ln L0) - (-ln L1) ],   -ln L1 = sum_b [ N_b - N_b ln N_b ]  (theta = 1)

-ln L0 is convex in theta (Poisson with a linear model + Gaussian priors), so a projected
Newton iteration with the analytic Hessian converges to the global minimum. Validated
against scipy L-BFGS-B (v1): |dq0| < 2e-3, Newton always reaching the lower minimum.
Bins with zero background follow v1: the expectation is clipped at 1e-30.
"""
import numpy as np

N_CLIP = 1e-30


class AsimovQ0:
    def __init__(self, A, dm, rel_sigma):
        """A: (K sources, B bins) expected events per source; dm: (B,) DM events per unit sigma."""
        A = np.asarray(A, float)
        dm = np.asarray(dm, float)
        bkg = A.sum(axis=0)
        keep = (bkg > 0) | (dm > 0)
        self.zero_bkg = (bkg[keep] <= 0)
        self.A = A[:, keep][:, ~self.zero_bkg]
        self.bkg = bkg[keep][~self.zero_bkg]
        self.dm_pos = dm[keep][~self.zero_bkg]
        self.dm_zero = dm[keep][self.zero_bkg]
        self.inv_r2 = 1.0 / np.asarray(rel_sigma, float)**2
        self.theta = np.ones(A.shape[0])

    def _f(self, theta, N):
        n = theta @ self.A
        if np.any(n <= 0):
            return np.inf
        return np.sum(n - N * np.log(n)) + 0.5 * np.sum((theta - 1.0)**2 * self.inv_r2)

    def profile(self, N, theta0=None, tol=1e-12, max_iter=100):
        theta = np.ones(self.A.shape[0]) if theta0 is None else np.maximum(theta0.copy(), 0.0)
        f = self._f(theta, N)
        if not np.isfinite(f):
            theta = np.ones(self.A.shape[0]); f = self._f(theta, N)
        for _ in range(max_iter):
            n = theta @ self.A
            g = self.A @ (1.0 - N / n) + (theta - 1.0) * self.inv_r2
            H = (self.A * (N / n**2)) @ self.A.T + np.diag(self.inv_r2)
            step = -np.linalg.solve(H, g)
            dec = -g @ step                     # Newton decrement^2
            if dec < 2.0 * tol:
                break
            alpha = 1.0
            while alpha > 1e-12:
                t_new = np.maximum(theta + alpha * step, 0.0)
                f_new = self._f(t_new, N)
                if f_new <= f - 1e-4 * alpha * dec or (f_new <= f and alpha < 1e-6):
                    break
                alpha *= 0.5
            if f_new > f:
                break
            theta, f = t_new, f_new
        return f, theta

    def q0(self, sigma):
        N = self.bkg + sigma * self.dm_pos
        f0, self.theta = self.profile(N, self.theta)
        L1 = np.sum(N - N * np.log(N))
        q = 2.0 * (f0 - L1)
        if np.any(self.dm_zero > 0):            # zero-background bins (v1 convention)
            Nz = sigma * self.dm_zero
            Nz = Nz[Nz > 0]
            q += 2.0 * np.sum(Nz * np.log(Nz / N_CLIP) - Nz + N_CLIP)
        return max(q, 0.0)

    def sigma_at(self, q_target, log_lo, log_hi, tol_dex=1e-3, step_dex=0.5):
        """Smallest sigma with q0 >= q_target (q0 is monotonic in sigma); NaN if not reached."""
        total = self.dm_pos.sum() + self.dm_zero.sum()
        if total <= 0:
            return np.nan
        lo = max(log_lo, np.log10(1e-4 / total))
        self.theta = np.ones(self.A.shape[0])
        while self.q0(10**lo) >= q_target and lo > log_lo:
            lo -= 2.0
        hi = lo
        while True:
            hi = min(hi + step_dex, log_hi)
            if self.q0(10**hi) >= q_target:
                break
            if hi >= log_hi:
                return np.nan
            lo = hi
        self.theta = np.ones(self.A.shape[0])
        while hi - lo > tol_dex:
            mid = 0.5 * (lo + hi)
            if self.q0(10**mid) >= q_target:
                hi = mid
            else:
                lo = mid
        return 10**(0.5 * (lo + hi))
