# 08_ablation_five_seeds.py
# Tables 12 and 13 (run first): cost-axis ablation, UERD base, five seeds.

# ============================================================
#  CELL A — Cost-axis ablation, five payload seeds, UERD as distortion term
#  Replaces Tables 10 and 11 (Reviewer B, point 6).
#
#  Changes from the original ablation:
#    * distortion axis = UERD (the recommended base), not the non-adaptive Q^2
#    * five payload seeds, identical payload bits across all 8 configurations
#    * seed outcomes averaged within each image before testing (n = 120)
#    * Wilcoxon signed-rank, config 7 vs config 4, Holm over the 4 rates
#
#  RUN FIRST: the RESTORE cell (the one that rebuilds CACHE with UERD).
#  Resumable: each finished seed is saved to Drive and skipped on rerun.
#  About 45 min per seed on a T4, about 4 h in total.
# ============================================================
import os, json, time, gc, warnings
import numpy as np, cv2, torch
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
import embed_core as E
warnings.filterwarnings("ignore")

for _v in ("paths", "sub", "CACHE", "net", "to_tensor"):
    assert _v in dir(), f"{_v} missing: run the RESTORE cell first"
assert len(paths) == 120 and "UERD" in CACHE[paths[0]], "CACHE incomplete: run RESTORE"

DRIVE = "/content/drive/MyDrive/dhide_ckpt"
OUT = f"{DRIVE}/ablation5"; os.makedirs(OUT, exist_ok=True)
SEEDS = [42, 123, 456, 789, 1011]
RATES = [0.05, 0.10, 0.20, 0.30]
CFG = [("1", 1, 0, 0), ("2", 0, 1, 0), ("3", 0, 0, 1), ("4", 1, 1, 0),
       ("5", 1, 0, 1), ("6", 0, 1, 1), ("7", 1, 1, 1), ("8", "rand", 0, 0)]
N = len(paths); ii = np.arange(N)
labels = np.asarray(sub.label.values) - 1

@torch.no_grad()
def probs(files):
    P = []
    for k in range(0, len(files), 32):
        xs = [to_tensor(cv2.imread(f, cv2.IMREAD_GRAYSCALE)) for f in files[k:k+32]]
        P.append(torch.softmax(net(torch.cat(xs)), 1).cpu().numpy())
    return np.vstack(P)

P0 = probs(paths); CLS0 = P0.argmax(1)
AUC0 = roc_auc_score(labels, P0, multi_class="ovr", average="macro", labels=[0, 1, 2])
ACC0 = float((CLS0 == labels).mean())
print(f"clean cover: accuracy {ACC0:.4f}, AUC {AUC0:.4f}   (expected 0.9083 / 0.9858)")
assert ACC0 > 0.85, "inference path is wrong"

def cost_map(c, wd, wr, wg, seed, i=0):
    if wd == "rand":   # random costs, per image, from a stream separate from the payload
        return np.random.default_rng(seed + 10**6 + i).random(c["UERD"].shape)
    return (wd * c["UERD"] + wr * c["R"] + wg * c["G"]) / (wd + wr + wg)

t0 = time.time()
for seed in SEEDS:
    fn = f"{OUT}/seed_{seed}.json"
    if os.path.exists(fn):
        print(f"seed {seed}: already on Drive, skipped"); continue
    rec = {}
    for name, wd, wr, wg in CFG:
        for rate in RATES:
            rng = np.random.default_rng(seed)        # same payload bits in every config
            outs, ok, sizes, fails = [], [], [], 0
            for i, p in enumerate(paths):
                c = CACHE[p]
                m = int(rate * c["n_ac"])
                bits = rng.integers(0, 2, m).astype(np.uint8)
                o = f"/tmp/a5_{i}.jpg"
                try:
                    st = E.embed_mode(p, o, bits, cost_map(c, wd, wr, wg, seed, i), mode="ac")
                except Exception:
                    fails += 1; continue
                outs.append(o); ok.append(i); sizes.append(st["size_pct"])
            P1 = probs(outs); ok = np.array(ok)
            dp = np.full(N, np.nan); fl = np.full(N, np.nan); sz = np.full(N, np.nan)
            dp[ok] = np.abs(P1[np.arange(len(ok)), CLS0[ok]] - P0[ok, CLS0[ok]])
            fl[ok] = (P1.argmax(1) != CLS0[ok]).astype(float)
            sz[ok] = sizes
            dauc = float(AUC0 - roc_auc_score(labels[ok], P1, multi_class="ovr",
                                              average="macro", labels=[0, 1, 2]))
            rec[f"{name}@{rate}"] = dict(dp=dp.tolist(), flip=fl.tolist(), size=sz.tolist(),
                                         dauc=dauc, fails=fails)
            for o in outs: os.remove(o)
            print(f"  seed {seed} cfg {name} rate {rate:.2f} | mean|dp| {np.nanmean(dp):.5f} "
                  f"| flips {100*np.nanmean(fl):.2f}% | size {np.nanmean(sz):+.2f}% "
                  f"| fails {fails}  [{time.time()-t0:.0f}s]", flush=True)
    # exact-recovery check on five images, configuration 7, highest rate
    ber = []
    for i in range(5):
        c = CACHE[paths[i]]; m = int(0.30 * c["n_ac"])
        bits = np.random.default_rng(seed).integers(0, 2, m).astype(np.uint8)
        E.embed_mode(paths[i], "/tmp/a5_chk.jpg", bits, cost_map(c, 1, 1, 1, seed), mode="ac")
        ber.append(float(np.mean(bits != E.extract_mode("/tmp/a5_chk.jpg", m, mode="ac"))))
    json.dump(dict(seed=seed, auc0=float(AUC0), acc0=ACC0, ber=ber, rec=rec), open(fn, "w"))
    print(f"seed {seed} saved to Drive, BER {max(ber):.4f}\n", flush=True)

# ---------------- analysis, image level ----------------
runs = [json.load(open(f"{OUT}/seed_{s}.json")) for s in SEEDS]
def im(key, field="dp"):   # mean over seeds within each image
    return np.nanmean(np.array([r["rec"][key][field] for r in runs]), axis=0)

print("=" * 96)
print("TABLE 10 (revised) — cost-axis ablation, UERD distortion term, "
      f"{len(SEEDS)} seeds averaged within image, n = {N}")
print(f"{'cfg':>4} {'D':>2} {'R':>2} {'G':>2} " + "".join(f"{'|dp| @'+str(r):>13}" for r in RATES)
      + f"{'flips%@0.30':>12} {'size%@0.30':>11} {'max dAUC':>9}")
for name, wd, wr, wg in CFG:
    row = [np.nanmean(im(f"{name}@{r}")) for r in RATES]
    fl = 100 * np.nanmean(im(f"{name}@0.3", "flip"))
    sz = np.nanmean(im(f"{name}@0.3", "size"))
    mx = max(r["rec"][f"{name}@{q}"]["dauc"] for r in runs for q in RATES)
    tag = lambda w: "rnd" if w == "rand" else ("x" if w else "-")
    print(f"{name:>4} {tag(wd):>2} {tag(wr):>2} {tag(wg):>2} "
          + "".join(f"{v:>13.5f}" for v in row) + f"{fl:>12.2f} {sz:>+11.2f} {mx:>+9.4f}")

print("\nTABLE 11 (revised) — config 7 (UERD+R+G) vs config 4 (UERD+R), paired by image")
tests = {}
for r in RATES:
    a, b = im(f"7@{r}"), im(f"4@{r}")
    k = ~(np.isnan(a) | np.isnan(b)); a, b = a[k], b[k]
    s, p = wilcoxon(a, b); d = np.sign(a - b)
    tests[r] = dict(n=int(k.sum()), cfg7=float(a.mean()), cfg4=float(b.mean()),
                    red=float(100 * (1 - a.mean() / b.mean())), p=float(p),
                    cliff=float((d < 0).mean() - (d > 0).mean()), lower=float((a < b).mean()))
order = sorted(tests, key=lambda r: tests[r]["p"])
for j, r in enumerate(order):
    tests[r]["holm"] = float(min(1.0, max(tests[q]["p"] * (len(order) - i)
                                          for i, q in enumerate(order[:j + 1]))))
print(f"{'rate':>5} {'n':>4} {'cfg7':>9} {'cfg4':>9} {'reduction':>10} {'raw p':>10} "
      f"{'Holm p':>9} {'Cliff':>7} {'lower on':>9}")
for r in RATES:
    v = tests[r]
    print(f"{r:>5.2f} {v['n']:>4} {v['cfg7']:>9.5f} {v['cfg4']:>9.5f} {v['red']:>+9.1f}% "
          f"{v['p']:>10.2e} {v['holm']:>9.4f} {v['cliff']:>+7.3f} {100*v['lower']:>8.1f}%")

# mean rank of the configurations that contain the diagnostic axis
withG = {"3", "5", "6", "7"}; ranks = {n: [] for n, *_ in CFG}
for r in RATES:
    vals = sorted((np.nanmean(im(f"{n}@{r}")), n) for n, *_ in CFG)
    for k, (_, n) in enumerate(vals, 1): ranks[n].append(k)
mg = np.mean([np.mean(ranks[n]) for n in withG])
mo = np.mean([np.mean(ranks[n]) for n in ranks if n not in withG])
print(f"\nmean rank over rates: with G {mg:.2f}, without G {mo:.2f}")
print(f"clean reference on these {N} images: AUC {AUC0:.4f}, accuracy {ACC0:.4f}")
print(f"max BER over all seeds: {max(max(r['ber']) for r in runs):.4f}")
json.dump(dict(tests={str(k): v for k, v in tests.items()}, mean_rank_G=float(mg),
               mean_rank_noG=float(mo), seeds=SEEDS, rates=RATES),
          open(f"{OUT}/ablation5_summary.json", "w"), indent=2)
print("summary saved to", f"{OUT}/ablation5_summary.json")
print("=" * 96)
