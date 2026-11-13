# 10_ablation_factorial_tests.py
# Table 13: paired factorial tests from the saved ablation runs.

# ============================================================
#  CELL C — paired tests on the five-seed ablation (reads Drive only)
#  No restore needed, runs in seconds.
#  Adding G: 1->5, 2->6, 4->7.   Adding R: 1->4, 3->6, 5->7.
#  Image level (seed means within image), Wilcoxon, Holm within each family.
# ============================================================
import json, numpy as np, os
from scipy.stats import wilcoxon
try:
    from google.colab import drive; drive.mount("/content/drive", force_remount=False)
except Exception: pass
OUT = "/content/drive/MyDrive/dhide_ckpt/ablation5"
SEEDS = [42, 123, 456, 789, 1011]; RATES = [0.05, 0.10, 0.20, 0.30]
runs = [json.load(open(f"{OUT}/seed_{s}.json")) for s in SEEDS]
im = lambda key: np.nanmean(np.array([r["rec"][key]["dp"] for r in runs]), axis=0)

def family(title, pairs):
    res = []
    for a_cfg, b_cfg in pairs:
        for r in RATES:
            a, b = im(f"{b_cfg}@{r}"), im(f"{a_cfg}@{r}")      # with term vs without
            k = ~(np.isnan(a) | np.isnan(b)); a, b = a[k], b[k]
            s, p = wilcoxon(a, b); d = np.sign(a - b)
            res.append(dict(pair=f"{a_cfg}->{b_cfg}", rate=r, n=int(k.sum()),
                            without=float(b.mean()), with_=float(a.mean()),
                            change=float(100 * (a.mean() / b.mean() - 1)), p=float(p),
                            cliff=float((d < 0).mean() - (d > 0).mean()),
                            lower=float((a < b).mean())))
    o = sorted(range(len(res)), key=lambda i: res[i]["p"]); run = 0
    for j, i in enumerate(o):
        run = max(run, res[i]["p"] * (len(o) - j)); res[i]["holm"] = min(1.0, run)
    print("=" * 100); print(title, f"(Holm over {len(res)} tests)")
    print(f"{'pair':>6} {'rate':>5} {'without':>9} {'with':>9} {'change':>8} {'raw p':>10} "
          f"{'Holm p':>9} {'Cliff':>7} {'with lower on':>14}")
    for v in res:
        print(f"{v['pair']:>6} {v['rate']:>5.2f} {v['without']:>9.5f} {v['with_']:>9.5f} "
              f"{v['change']:>+7.1f}% {v['p']:>10.2e} {v['holm']:>9.4f} {v['cliff']:>+7.3f} "
              f"{100*v['lower']:>13.1f}%")
    return res

g = family("ADDING THE DIAGNOSTIC AXIS G", [("1", "5"), ("2", "6"), ("4", "7")])
r = family("ADDING THE RATE AXIS R", [("1", "4"), ("3", "6"), ("5", "7")])
json.dump(dict(add_G=g, add_R=r), open(f"{OUT}/ablation5_factorial.json", "w"), indent=2)
print("=" * 100); print("saved", f"{OUT}/ablation5_factorial.json")
