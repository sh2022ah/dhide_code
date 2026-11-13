# 13_figure2_final.py
# Final Figure 2.
# Extracted from the Colab notebook used for the paper; table numbers in the
# comments below are those of earlier drafts, see README for the mapping.

# ---- Figure 2, final: mask background, report rate-table coverage -------
import os, gc, warnings, numpy as np, cv2, jpegio
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import embed_core as E
warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "serif", "font.size": 8, "figure.dpi": 300,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.03})
OUT, NB, GREEN = "/content/figs", 64, "#00c853"

sid = df[(df.split == "test") & (df.label == 2)].slice_id.iloc[0]
jpg = f"/content/data/jpeg_qf75/{sid}.jpg"
j = jpegio.read(jpg)
C = np.array(j.coef_arrays[0], dtype=np.int64, copy=True)
Q = j.quant_tables[0].astype(float); del j; gc.collect()
img = cv2.imread(jpg, cv2.IMREAD_GRAYSCALE)
mask = cv2.imread(f"/content/data/masks/{sid}.png", cv2.IMREAD_GRAYSCALE) > 127
pool = lambda a: a.reshape(NB, 8, NB, 8).sum(axis=(1, 3))
tissue = pool((img > 12).astype(float)) > 8
tum = mask.reshape(NB, 8, NB, 8).transpose(0, 2, 1, 3).reshape(NB, NB, 64).any(2)

def show(ax, m, title):
    m = np.ma.masked_where(~tissue, np.log10(m + 1e-12))     # hide background
    lo, hi = np.percentile(m.compressed(), [2, 98])
    cm = plt.get_cmap("magma").copy(); cm.set_bad("white")
    ax.imshow(m, cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
    ax.contour(tum.astype(np.uint8), levels=[0.5], colors=GREEN, linewidths=0.8)
    ax.set_title(title, fontsize=7.5); ax.set_xticks([]); ax.set_yticks([])

# rate table with an explicit coverage mask
TAB = np.zeros((8, 8)); CNT = np.zeros((8, 8))
import tempfile
base = os.path.getsize(jpg)
nz = np.argwhere(C != 0)
sel = nz[np.random.default_rng(0).choice(len(nz), min(1200, len(nz)), replace=False)]
for r, c in sel:
    jj = jpegio.read(jpg); jj.coef_arrays[0][r, c] += 1
    t = tempfile.mktemp(suffix=".jpg"); jpegio.write(jj, t)
    TAB[r % 8, c % 8] += os.path.getsize(t) - base; CNT[r % 8, c % 8] += 1
    os.remove(t)
COV = CNT > 0
TAB = np.divide(TAB, np.maximum(CNT, 1))
print(f"rate table coverage: {COV.sum()} of 64 frequencies sampled "
      f"({100*COV.mean():.0f}%), {int(CNT.sum())} probes")

fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.15),
                         gridspec_kw={"width_ratios": [1, 1, 1, 1], "wspace": 0.28})
axes[0].imshow(img, cmap="gray")
axes[0].contour(mask.astype(np.uint8), levels=[0.5], colors=GREEN, linewidths=0.8)
axes[0].set_title("(a) cover, glioma", fontsize=7.5)
axes[0].set_xticks([]); axes[0].set_yticks([])
show(axes[1], pool(E.uerd_cost(C, Q)), "(b) distortion, UERD (5)")
show(axes[2], pool(rho_G(net, img, Q)), "(c) diagnostic (9)")

ax = axes[3]
v = np.ma.masked_where(~COV, TAB)
cmv = plt.get_cmap("coolwarm").copy(); cmv.set_bad("#dcdcdc")
im = ax.imshow(v, cmap=cmv, vmin=-abs(TAB).max(), vmax=abs(TAB).max())
ax.set_title("(d) rate (6), per frequency", fontsize=7.5)
ax.set_xlabel("horizontal", fontsize=6.5); ax.set_ylabel("vertical", fontsize=6.5)
ax.set_xticks([0, 7]); ax.set_yticks([0, 7]); ax.tick_params(labelsize=6)
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.06)
cb.set_label("bytes", fontsize=6); cb.ax.tick_params(labelsize=5.5)
ax.text(3.5, 8.9, f"grey: {64-int(COV.sum())} frequencies never sampled",
        fontsize=5.5, ha="center", color="#666666")

fig.savefig(f"{OUT}/figure2_maps.pdf"); fig.savefig(f"{OUT}/figure2_maps.png")
plt.close(fig)
import subprocess
subprocess.run(["cp", "-r", OUT, "/content/drive/MyDrive/dhide_ckpt/"], check=False)
print("figure 2 final, copied to Drive")

for a in (axes[1], axes[2]):
    for s in a.spines.values(): s.set_visible(False)
ax.text(3.5, 9.6, f"grey: {64-int(COV.sum())} frequencies never sampled",
        fontsize=5.5, ha="center", color="#666666")
