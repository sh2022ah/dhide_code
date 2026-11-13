# 11_table8_statistics.py
# Table 8 statistics recomputed from the saved raw data.

# ============================================================
#  CELL D — verify Table 6 and the J-UNIWARD vs UERD statistics (reads Drive only)
#  No restore needed, runs in seconds.
# ============================================================
import json, numpy as np
from scipy.stats import wilcoxon
try:
    from google.colab import drive; drive.mount("/content/drive", force_remount=False)
except Exception: pass
D = "/content/drive/MyDrive/dhide_ckpt"
U = json.load(open(f"{D}/unified_raw.json"))["raw"]["resnet50"]   # [rate][arm] -> 5 x 120
J = json.load(open(f"{D}/juniward_results.json"))["raw"]           # [arm][rate] -> 5 x 120
RATES = ["0.05", "0.1"]
im_u = lambda r, a: np.array(U[r][a]).mean(axis=0)
im_j = lambda r, a: np.array(J[a][r]).mean(axis=0)

def cmp(a, b):
    s, p = wilcoxon(a, b); d = np.sign(a - b)
    return dict(a=a.mean(), b=b.mean(), a_med=np.median(a), b_med=np.median(b),
                red=100*(1-a.mean()/b.mean()), red_med=100*(1-np.median(a)/np.median(b)),
                p=p, cliff=(d < 0).mean() - (d > 0).mean(), lower=(a < b).mean())

def holm(rows):
    o = sorted(rows, key=lambda k: rows[k]["p"]); run = 0
    for j, k in enumerate(o):
        run = max(run, rows[k]["p"] * (len(o) - j)); rows[k]["holm"] = min(1.0, run)

print("consistency: UERD in the two runs (should be identical or nearly)")
for r in RATES:
    print(f"  rate {r}: unified {im_u(r,'UERD').mean()*1e4:.3f}e-4   J-run {im_j(r,'UERD').mean()*1e4:.3f}e-4")

T = {}
for r in RATES:
    T[f"UERD+0.25G@{r}"] = cmp(im_u(r, "UERD+0.25G"), im_u(r, "UERD"))
    T[f"UERD+0.50G@{r}"] = cmp(im_u(r, "UERD+0.50G"), im_u(r, "UERD"))
    T[f"JUNI+0.25G@{r}"] = cmp(im_j(r, "JUNI+0.25G"), im_j(r, "J-UNIWARD"))
holm(T)
four = {k: dict(v) for k, v in T.items() if "0.50" not in k}; holm(four)
print("\nTABLE 6 check (image level, n = 120)")
print(f"{'row':>18} {'mean e-4':>9} {'base e-4':>9} {'med e-5':>8} {'base e-5':>9} "
      f"{'red mean/med':>14} {'raw p':>10} {'Holm6':>9} {'Holm4':>9} {'Cliff':>7} {'lower':>6}")
for k, v in T.items():
    h4 = four[k]["holm"] if k in four else float("nan")
    print(f"{k:>18} {v['a']*1e4:>9.3f} {v['b']*1e4:>9.3f} {v['a_med']*1e5:>8.3f} {v['b_med']*1e5:>9.3f} "
          f"{v['red']:>6.1f}/{v['red_med']:>5.1f}% {v['p']:>10.2e} {v['holm']:>9.2e} {h4:>9.2e} "
          f"{v['cliff']:>+7.3f} {100*v['lower']:>5.1f}%")

print("\nJ-UNIWARD against UERD, no diagnostic term (J-run)")
for r in RATES:
    x, y = im_j(r, "J-UNIWARD"), im_j(r, "UERD")
    s, p = wilcoxon(x, y); d = np.sign(x - y)
    print(f"  rate {r}: J-UNIWARD {x.mean()*1e4:.3f} vs UERD {y.mean()*1e4:.3f} e-4, "
          f"J-UNIWARD higher by {100*(x.mean()/y.mean()-1):+.1f}%  p={p:.2e}  "
          f"Cliff={float((d<0).mean()-(d>0).mean()):+.3f}")
