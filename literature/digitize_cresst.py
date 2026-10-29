#!/usr/bin/env python3
"""
Digitize curves from the raster figures of CRESST Collaboration, Commun. Phys. (2025),
doi:10.1038/s42005-025-02476-5 (Figs. 6 and 7), extracted with `pdfimages -png`.
Axis calibration uses the plot frame and checked major grid lines (0.1 and 1 GeV/c^2).
Output: two-column text files, m [GeV], sigma [cm^2] (1 pb = 1e-36 cm^2).

Digitization accuracy is ~1-2 pixels, i.e. ~2-4% in m and ~5-15% in sigma.
The CRESST-III 2019 curve is also digitized as a check against the published data file.
"""
import sys, numpy as np
from PIL import Image

FIG6, FIG7 = sys.argv[1], sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else '.'


def calib(x0px, x1px, m0, m1, y0px, y1px, s0, s1):
    def to_m(c): return 10**(np.log10(m0) + (c - x0px) / (x1px - x0px) * np.log10(m1 / m0))
    def to_s(r): return 10**(np.log10(s0) + (r - y0px) / (y1px - y0px) * np.log10(s1 / s0)) * 1e-36
    def col(m): return x0px + np.log10(m / m0) / np.log10(m1 / m0) * (x1px - x0px)
    return to_m, to_s, col


def clusters(idx):
    """Split sorted row indices into contiguous clusters; return their centres."""
    if len(idx) == 0:
        return []
    out, start = [], idx[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a > 2:
            out.append(0.5 * (start + a)); start = b
    out.append(0.5 * (start + idx[-1]))
    return out


def track(mask, c_start, c_end, r_start, max_jump=25):
    """Follow a curve column by column, choosing the cluster closest to the previous row."""
    rows, cols, r = [], [], r_start
    for c in range(c_start, c_end):
        cl = clusters(np.where(mask[:, c])[0])
        if not cl:
            continue
        best = min(cl, key=lambda x: abs(x - r))
        if abs(best - r) <= max_jump:
            r = best; rows.append(r); cols.append(c)
    return np.array(cols), np.array(rows)


def save(name, m, s, note):
    o = np.argsort(m)
    np.savetxt(f"{OUT}/{name}.txt", np.c_[m[o], s[o]], fmt='%.5e',
               header=f"{note}\nm_GeV  sigma_cm2  (digitized)")
    print(f"  {name}: {len(m)} points, m = {m.min():.3g}-{m.max():.3g} GeV")


# ---------------- Fig. 7: discovery potentials (Bento et al. 2024) and upgrade projection
a = np.array(Image.open(FIG7).convert('RGB')).astype(int)
R, G, B = a[..., 0], a[..., 1], a[..., 2]
to_m, to_s, col = calib(87, 899, 0.05, 5.0, 797, 107, 1e-10, 1e-4)
c0, c1 = 89, 898
colors = {
    'Bento2024_CaWO4_10tyr_1eV': (B > 200) & (R < 60) & (G < 60),
    'Bento2024_CaWO4_1tyr_1eV': (G > 100) & (G < 160) & (R < 60) & (B < 60),
    'Bento2024_CaWO4_0p1tyr_1eV': (R > 100) & (R < 160) & (B > 100) & (B < 160) & (G < 50),
    'Bento2024_CaWO4_1tday_10eV': (R > 220) & (G < 50) & (B < 50),
}
for name, mk in colors.items():
    m, s = [], []
    for c in range(c0, c1):
        idx = np.where(mk[110:795, c])[0] + 110
        if len(idx):
            m.append(to_m(c)); s.append(to_s(np.median(idx)))
    save(name, np.array(m), np.array(s), "Bento et al., EPJC 84, 1118 (2024), as shown in CRESST Commun. Phys. 2025 Fig. 7; 3 sigma discovery, CaWO4")
# grey region (neutrino fog of Bento et al.): upper boundary
grey = (np.abs(R - G) < 4) & (np.abs(G - B) < 4) & (R >= 200) & (R <= 222)
m, s = [], []
for c in range(c0, c1):
    col_g = grey[110:795, c]
    # topmost grey pixel with >= 80% grey below it (rejects vertical grid lines in the white area)
    frac = np.cumsum(col_g[::-1])[::-1] / np.arange(len(col_g), 0, -1)
    idx = np.where(col_g & (frac >= 0.8))[0]
    if len(idx):
        m.append(to_m(c)); s.append(to_s(idx.min() + 110))
s = np.exp(np.array([np.median(np.log(s[max(0, i - 3):i + 4])) for i in range(len(s))]))
save('Bento2024_CaWO4_fog_boundary', np.array(m), np.array(s), "Upper boundary of the grey neutrino-fog region (Bento et al. 2024), CRESST Fig. 7")
# CRESST upgrade projection (100x less LEE): envelope (lowest) of the two orange lines
orange = (R > 240) & (G > 120) & (G < 170) & (B < 60)
low = np.full(c1, np.nan)
for c in range(c0, c1):
    idx = np.where(orange[110:795, c])[0] + 110
    if len(idx):
        low[c] = idx.max()
# dashed / dash-dotted lines: bridge the gaps with the lowest curve point within +-12 columns
cols = [c for c in range(c0, c1) if np.isfinite(low[max(c0, c - 12):c + 13]).any()]
rows = [np.nanmax(low[max(c0, c - 12):c + 13]) for c in cols]
m, s = to_m(np.array(cols)), to_s(np.array(rows))
save('CRESST_upgrade_projection_100xLEE', np.array(m), np.array(s), "CRESST upgrade projection, 100x less LEE, 515 kg day, envelope of 5 eV and 30.1 eV modules (median)")

# ---------------- Fig. 6: CRESST-III limits
b = np.array(Image.open(FIG6).convert('RGB')).astype(int)
R, G, B = b[..., 0], b[..., 1], b[..., 2]
to_m, to_s, col = calib(122, 1211, 0.07, 5.0, 839, 224, 1e-10, 1e7)
red = (R > 200) & (G < 60) & (B < 60)
red[:226] = False                                     # legend
c_s = next(c for c in range(int(col(0.0705)), int(col(0.09))) if red[:, c].any())
r_s = np.where(red[:, c_s])[0].min()
cc, rr = track(red, c_s, int(col(2.0)), r_s)
save('CRESST-III_2024_SOS', to_m(cc), to_s(rr), "CRESST-III 2024 (SOS detector) 90% CL limit, CRESST Commun. Phys. 2025 Fig. 6")
c_s = int(round(col(0.30)))
cl = clusters(np.where(red[:, c_s])[0])                # at 0.3 GeV: [2024, 2019, (text)] from top
cc, rr = track(red, c_s, int(col(5.0)) - 1, sorted(cl)[1])
save('CRESST-III_2019_DetA_digitized', to_m(cc), to_s(rr), "CRESST-III 2019 Detector A, digitized (m >= 0.3 GeV, check only)")
