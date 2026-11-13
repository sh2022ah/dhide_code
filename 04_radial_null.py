# 04_radial_null.py
# Radial-null control for the concentration ratio (Table 6, last rows).
# Extracted from the Colab notebook used for the paper; table numbers in the
# comments below are those of earlier drafts, see README for the mapping.

# ---- control: is the concentration explained by centre bias alone? ------
import numpy as np, cv2, jpegio, gc, json
ROOT, NB = "/content/data", 64
te = df[df.split == "test"].sample(50, random_state=42)
yy, xx = np.mgrid[0:NB, 0:NB]
radial = 1.0 / (1.0 + np.hypot(yy - NB/2 + .5, xx - NB/2 + .5))   # centre-only null

real, null, frac = [], [], []
for _, r in te.iterrows():
    jp, mp = f"{ROOT}/jpeg_qf75/{r.slice_id}.jpg", f"{ROOT}/masks/{r.slice_id}.png"
    j = jpegio.read(jp); Q = j.quant_tables[0].astype(float); del j; gc.collect()
    G = rho_G(net, cv2.imread(jp, cv2.IMREAD_GRAYSCALE), Q)
    gb = G.reshape(NB, 8, NB, 8).sum(axis=(1, 3))
    m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 127
    tum = m.reshape(NB, 8, NB, 8).transpose(0, 2, 1, 3).reshape(NB, NB, 64).any(2)
    if tum.sum() == 0: continue
    f = tum.mean(); frac.append(f)
    real.append(gb[tum].sum() / gb.sum())
    null.append(radial[tum].sum() / radial.sum())

real, null, frac = map(np.array, (real, null, frac))
cr_real = real.mean() / frac.mean()
cr_null = null.mean() / frac.mean()
from scipy.stats import wilcoxon
s, p = wilcoxon(real, null)
print("=" * 62)
print(f"diagnostic map  concentration = {cr_real:.2f}x")
print(f"radial null     concentration = {cr_null:.2f}x")
print(f"excess over the centre-bias null = {cr_real - cr_null:+.2f}x")
print(f"paired Wilcoxon, real vs null: p = {p:.2e}")
print(f"map exceeds the null on {100*(real > null).mean():.0f}% of images")
print("=" * 62)
json.dump(dict(real=cr_real, null=cr_null, p=float(p),
               win=float((real > null).mean())),
          open("/content/centre_control.json", "w"), indent=2)
