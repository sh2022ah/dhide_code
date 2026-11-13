# 05_baselines.py
# Table 7: matched-payload comparison with random, DC-only, non-adaptive, UERD and the equal-weight three-axis cost.
# Extracted from the Colab notebook used for the paper; table numbers in the
# comments below are those of earlier drafts, see README for the mapping.

# ============================================================
#  TABLE 6 — comparison against JPEG-domain baselines
#  UERD [12] re-implemented from Guo et al.; DC-only [23] is the manual
#  solution and uses a different carrier, hence a hard capacity ceiling.
#  All methods embed the SAME number of bits per image, sized from the AC
#  carrier, and are evaluated on the same frozen surrogate.
#  ~35 min on CPU.
# ============================================================
import os, gc, json, time, warnings
import numpy as np, cv2, jpegio, torch
from scipy.stats import wilcoxon
from skimage.metrics import structural_similarity as ssim_fn
import embed_core as E
warnings.filterwarnings("ignore")

ROOT, QF, N_IMG = "/content/data", 75, 120
RATES = [0.05, 0.10]
rng = np.random.default_rng(42)
te = df[df.split == "test"]
sub = te.sample(min(N_IMG, len(te)), random_state=42)
paths = [f"{ROOT}/jpeg_qf{QF}/{s}.jpg" for s in sub.slice_id]

print("caching cost maps and baselines (slowest step on CPU) ...", flush=True)
t0 = time.time()
TAB = E.rate_table(paths[:20], per_img=150)
CACHE = {}
for i, p in enumerate(paths):
    j = jpegio.read(p)
    C = np.array(j.coef_arrays[0], dtype=np.int64, copy=True)
    Q = j.quant_tables[0].astype(float); del j; gc.collect()
    img8 = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    D = E.ranknorm(E.rho_D(Q, C.shape))
    R = E.ranknorm(E.rho_R(TAB, C.shape))
    G = E.ranknorm(rho_G(net, img8, Q))
    CACHE[p] = dict(D=D, R=R, G=G, UERD=E.ranknorm(E.uerd_cost(C, Q)),
                    prop=(D + R + G) / 3, cover=img8,
                    n_ac=len(E.carrier(C)[0]), n_dc=len(E.carrier_dc(C)[0]))
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{len(paths)}  [{time.time()-t0:.0f}s]", flush=True)

dc_ceiling = float(np.median([CACHE[p]["n_dc"] / 2 / CACHE[p]["n_ac"] for p in paths]))
print(f"\nDC-only capacity ceiling: {dc_ceiling:.4f} bpnzAC-equivalent (median)\n")

@torch.no_grad()
def probs(files):
    P = []
    for k in range(0, len(files), 32):
        xs = [to_tensor(cv2.imread(f, cv2.IMREAD_GRAYSCALE)) for f in files[k:k+32]]
        P.append(torch.softmax(net(torch.cat(xs)), 1).cpu().numpy())
    return np.vstack(P)

P0 = probs(paths); CLS0 = P0.argmax(1)
METHODS = ["random", "UERD", "DC-only", "distortion", "proposed"]
rows, per_img = [], {}

for rate in RATES:
    for meth in METHODS:
        outs, sizes, psnr, ssim, ok_idx, ber, fails = [], [], [], [], [], [], 0
        for i, p in enumerate(paths):
            c = CACHE[p]
            m = int(rate * c["n_ac"])
            mode = "dc" if meth == "DC-only" else "ac"
            rho = {"random": None, "UERD": c["UERD"], "DC-only": c["D"],
                   "distortion": c["D"], "proposed": c["prop"]}[meth]
            if rho is None: rho = rng.random(c["D"].shape)
            bits = rng.integers(0, 2, m).astype(np.uint8)
            o = f"/tmp/b_{os.path.basename(p)}"
            try:
                st = E.embed_mode(p, o, bits, rho, mode=mode)
            except Exception:
                fails += 1; continue
            stg = cv2.imread(o, cv2.IMREAD_GRAYSCALE)
            mse = ((stg.astype(np.float64) - c["cover"].astype(np.float64)) ** 2).mean()
            psnr.append(10 * np.log10(255 ** 2 / max(mse, 1e-12)))
            ssim.append(ssim_fn(c["cover"], stg, data_range=255))
            sizes.append(st["size_pct"]); outs.append(o); ok_idx.append(i)
            if i < 5:
                ber.append(float(np.mean(bits != E.extract_mode(o, m, mode=mode))))
        if not outs:
            rows.append(dict(rate=rate, method=meth, feasible=False, fails=fails))
            print(f"  rate {rate:.2f} {meth:11s} INFEASIBLE on all {fails} images", flush=True)
            continue
        P1 = probs(outs)
        dp = np.abs(P1[np.arange(len(ok_idx)), CLS0[ok_idx]] - P0[ok_idx, CLS0[ok_idx]])
        flip = float((P1.argmax(1) != CLS0[ok_idx]).mean())
        rows.append(dict(rate=rate, method=meth, feasible=True, n=len(ok_idx), fails=fails,
                         mean_dp=float(dp.mean()), flip=flip, size=float(np.mean(sizes)),
                         psnr=float(np.mean(psnr)), ssim=float(np.mean(ssim)),
                         ber=float(np.mean(ber)) if ber else None))
        per_img[f"{meth}@{rate}"] = (ok_idx, dp.tolist())
        print(f"  rate {rate:.2f} {meth:11s} |dp| {dp.mean():.5f} | flip {100*flip:4.2f}% | "
              f"size {np.mean(sizes):+.2f}% | PSNR {np.mean(psnr):5.2f} | SSIM {np.mean(ssim):.5f}"
              f"{'' if not fails else f'  ({fails} failed)'}", flush=True)
        for o in outs: os.remove(o)
    print(f"  --- rate {rate} done [{time.time()-t0:.0f}s]\n", flush=True)

tests = {}
for rate in RATES:
    a = per_img.get(f"proposed@{rate}")
    if not a: continue
    for meth in ["random", "UERD", "DC-only", "distortion"]:
        b = per_img.get(f"{meth}@{rate}")
        if not b: continue
        ia, va = a; ib, vb = b
        shared = sorted(set(ia) & set(ib))
        x = np.array([va[ia.index(i)] for i in shared])
        y = np.array([vb[ib.index(i)] for i in shared])
        s, pv = wilcoxon(x, y); d = np.sign(x - y)
        tests[f"{meth}@{rate}"] = dict(n=len(shared), prop=float(x.mean()),
            base=float(y.mean()), reduction=float(100 * (1 - x.mean() / y.mean())),
            p=float(pv), cliff=float((d < 0).mean() - (d > 0).mean()))

json.dump(dict(rows=rows, tests=tests, dc_ceiling=dc_ceiling),
          open("/content/table6.json", "w"), indent=2)
print("=" * 84)
print(f"{'rate':>5} {'method':>11} {'mean|dp|':>10} {'flip%':>6} {'size%':>7} "
      f"{'PSNR':>7} {'SSIM':>8} {'BER':>5}")
for r in rows:
    if not r["feasible"]:
        print(f"{r['rate']:>5.2f} {r['method']:>11}   infeasible (carrier ceiling)"); continue
    print(f"{r['rate']:>5.2f} {r['method']:>11} {r['mean_dp']:>10.5f} {100*r['flip']:>6.2f} "
          f"{r['size']:>+7.2f} {r['psnr']:>7.2f} {r['ssim']:>8.5f} {r['ber']:>5.3f}")
print("\nproposed against each baseline, paired Wilcoxon on |delta p|")
for k, v in tests.items():
    meth, rate = k.split("@")
    print(f"  rate {float(rate):.2f} vs {meth:11s}: {v['prop']:.5f} vs {v['base']:.5f}  "
          f"reduction {v['reduction']:+.1f}%  p={v['p']:.2e}  Cliff={v['cliff']:+.3f}")
print(f"\nDC-only ceiling {dc_ceiling:.4f} bpnzAC-equivalent")
print("=" * 84)
