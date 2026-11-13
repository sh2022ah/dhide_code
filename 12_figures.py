# 12_figures.py
# Figures 3 and 4 (and the first version of Figure 2).
# Extracted from the Colab notebook used for the paper; table numbers in the
# comments below are those of earlier drafts, see README for the mapping.

# ============================================================
#  Figures 2, 3 and 4 — the ones that need the images and the model.
#  Writes PDF and PNG at 300 dpi, then copies them to Drive.
# ============================================================
import os, gc, warnings
import numpy as np, cv2, jpegio, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import embed_core as E
warnings.filterwarnings("ignore")

plt.rcParams.update({"font.family": "serif", "font.size": 8, "figure.dpi": 300,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
                     "axes.spines.top": False, "axes.spines.right": False})
INK, ACC, MUT = "#1a1a1a", "#0b6e99", "#999999"
OUT = "/content/figs"; os.makedirs(OUT, exist_ok=True)
ROOT, QF = "/content/data", 75
LBL = {1: "meningioma", 2: "glioma", 3: "pituitary"}

def load(sid):
    jpg = f"{ROOT}/jpeg_qf{QF}/{sid}.jpg"
    j = jpegio.read(jpg)
    C = np.array(j.coef_arrays[0], dtype=np.int64, copy=True)
    Q = j.quant_tables[0].astype(float); del j; gc.collect()
    img = cv2.imread(jpg, cv2.IMREAD_GRAYSCALE)
    m = cv2.imread(f"{ROOT}/masks/{sid}.png", cv2.IMREAD_GRAYSCALE) > 127
    return jpg, C, Q, img, m

def blockmap(a, nb=64):
    s = a.shape[0] // nb
    return a.reshape(nb, s, nb, s).sum(axis=(1, 3))

te = df[df.split == "test"]

# ---------------------------------------- Figure 2: the three cost maps
sid = te[te.label == 2].slice_id.iloc[0]
jpg, C, Q, img, mask = load(sid)
TAB = E.rate_table([jpg], per_img=400)
maps = [("cover", img, "gray"),
        ("distortion, UERD (5)", blockmap(E.uerd_cost(C, Q)), "magma"),
        ("rate (6)", blockmap(np.abs(E.rho_R(TAB, C.shape))), "magma"),
        ("diagnostic (9)", blockmap(rho_G(net, img, Q)), "magma")]
fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.0))
for ax, (title, m, cm) in zip(axes, maps):
    d = m if cm == "gray" else np.log1p(m / (m.max() + 1e-12) * 1e3)
    ax.imshow(d, cmap=cm, interpolation="nearest")
    ax.contour(cv2.resize(mask.astype(np.uint8), d.shape[::-1],
                          interpolation=cv2.INTER_NEAREST),
               levels=[0.5], colors="#39ff88", linewidths=0.7)
    ax.set_title(title, fontsize=7.5); ax.set_xticks([]); ax.set_yticks([])
fig.savefig(f"{OUT}/figure2_maps.pdf"); fig.savefig(f"{OUT}/figure2_maps.png")
plt.close(fig); print("figure 2 done", flush=True)

# ---------------------------------------- Figure 3: distortion against diagnostic
G = rho_G(net, img, Q); D = E.rho_D(Q, C.shape)
nz = np.argwhere(C != 0)
sel = nz[np.random.default_rng(0).choice(len(nz), min(4000, len(nz)), replace=False)]
d = D[sel[:, 0], sel[:, 1]]; g = G[sel[:, 0], sel[:, 1]]
rho, _ = spearmanr(d, g)
fig, ax = plt.subplots(figsize=(3.3, 3.0))
ax.scatter(d, g, s=2, alpha=0.25, color=ACC, edgecolors="none", rasterized=True)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"distortion cost  $Q_f^2$")
ax.set_ylabel(r"diagnostic cost  $\rho^G$")
ax.set_title(f"Spearman $\\rho$ = {rho:+.3f}", fontsize=8, loc="left")
fig.savefig(f"{OUT}/figure3_scatter.pdf"); fig.savefig(f"{OUT}/figure3_scatter.png")
plt.close(fig); print("figure 3 done", flush=True)

# ---------------------------------------- Figure 4: map against radiologist mask
fig, axes = plt.subplots(2, 3, figsize=(6.4, 4.3))
for k, lab in enumerate([1, 2, 3]):
    s = te[te.label == lab].slice_id.iloc[0]
    _, Ck, Qk, imk, mk = load(s)
    Gk = blockmap(rho_G(net, imk, Qk))
    axes[0, k].imshow(imk, cmap="gray")
    axes[0, k].contour(mk.astype(np.uint8), levels=[0.5], colors="#39ff88", linewidths=0.8)
    axes[0, k].set_title(LBL[lab], fontsize=8)
    axes[1, k].imshow(np.log1p(Gk / (Gk.max() + 1e-12) * 1e3), cmap="magma")
    axes[1, k].contour(cv2.resize(mk.astype(np.uint8), Gk.shape[::-1],
                                  interpolation=cv2.INTER_NEAREST),
                       levels=[0.5], colors="#39ff88", linewidths=0.8)
    for r in (0, 1):
        axes[r, k].set_xticks([]); axes[r, k].set_yticks([])
axes[0, 0].set_ylabel("cover", fontsize=8)
axes[1, 0].set_ylabel("diagnostic map", fontsize=8)
fig.savefig(f"{OUT}/figure4_overlay.pdf"); fig.savefig(f"{OUT}/figure4_overlay.png")
plt.close(fig); print("figure 4 done", flush=True)

import subprocess
subprocess.run(["cp", "-r", OUT, "/content/drive/MyDrive/dhide_ckpt/"], check=False)
for f in sorted(os.listdir(OUT)):
    print(f"  {f}  {os.path.getsize(OUT+'/'+f)/1024:.0f} KB")
print("\nfigures copied to Drive")
