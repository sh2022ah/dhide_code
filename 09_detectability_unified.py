# 09_detectability_unified.py
# Tables 10 and 11 and the P(E) column of Table 8: one detectability run, identical payload bits.

# ============================================================
#  CELL B — Unified detectability: one protocol for Tables 6, 8 and 9
#
#  Every arm embeds IDENTICAL payload bits (rng restarted at each arm) into
#  the same 80 slices, and is scored by the same DCTR detector with the same
#  folds. P(E) and the classifier disturbance on those 80 slices come from
#  this single run, so the three tables can no longer disagree.
#
#  RUN FIRST: the RESTORE cell. Resumable per seed.
#  SEEDS = [42] takes about 2 h on a T4. Five seeds take about 10 h; add them
#  later if time allows, the cell skips seeds already on Drive.
# ============================================================
import os, sys, json, time, gc, base64, zlib, subprocess, warnings
import numpy as np, cv2, jpegio, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")

for _v in ("paths", "CACHE", "net", "to_tensor"):
    assert _v in dir(), f"{_v} missing: run the RESTORE cell first"

# J-UNIWARD module (identical to the one used for Table 6), written if absent
if not os.path.exists("/content/juniward.py"):
    _J = (
    "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"
    "dBYM7/zQZpLh/RuzCwk4"
    )
    open("/content/juniward.py", "wb").write(zlib.decompress(base64.b64decode("".join(_J))))
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "PyWavelets"], check=False)
sys.path.insert(0, "/content")
import embed_core as E, dctr, juniward as J

DRIVE = "/content/drive/MyDrive/dhide_ckpt"
OUT = f"{DRIVE}/pe_unified"; os.makedirs(OUT, exist_ok=True)
SEEDS = [42]                     # extend to [42, 123, 456, 789, 1011] if time allows
RATES = [0.05, 0.10]
sel = paths[:80]; ii = np.arange(len(sel))

print("J-UNIWARD maps for the 80 slices ...", flush=True)
for p in sel:
    if "JUNI" not in CACHE[p]:
        j = jpegio.read(p)
        C = np.array(j.coef_arrays[0], dtype=np.int64, copy=True)
        Q = j.quant_tables[0].astype(float); del j; gc.collect()
        CACHE[p]["JUNI"] = E.ranknorm(J.juniward_cost(C, Q, CACHE[p]["cover"]))

ARMS = {                                   # used in
    "Random":           None,               # Table 8
    "Q2 non-adaptive":  lambda c: c["D"],   # Table 8
    "UERD":             lambda c: c["UERD"],                      # 6, 8, 9
    "UERD+0.25G":       lambda c: 0.75 * c["UERD"] + 0.25 * c["G"],  # 6, 9
    "UERD+0.50G":       lambda c: 0.50 * c["UERD"] + 0.50 * c["G"],  # 6, 8, 9
    "UERD+0.75G":       lambda c: 0.25 * c["UERD"] + 0.75 * c["G"],  # 9
    "G only (w=1)":     lambda c: c["G"],                            # 9
    "J-UNIWARD":        lambda c: c["JUNI"],                         # 6, 8
    "JUNI+0.25G":       lambda c: 0.75 * c["JUNI"] + 0.25 * c["G"],  # 6
}

@torch.no_grad()
def probs(files):
    P = []
    for k in range(0, len(files), 32):
        xs = [to_tensor(cv2.imread(f, cv2.IMREAD_GRAYSCALE)) for f in files[k:k+32]]
        P.append(torch.softmax(net(torch.cat(xs)), 1).cpu().numpy())
    return np.vstack(P)

B0 = probs(sel); C0 = B0.argmax(1)
print("cover DCTR features ...", flush=True)
COVER = np.array([dctr.dctr(cv2.imread(p, cv2.IMREAD_GRAYSCALE)) for p in sel])

def detect(Xs):
    X = np.vstack([COVER, Xs])
    y = np.r_[np.zeros(len(COVER)), np.ones(len(Xs))]
    g = np.r_[np.arange(len(COVER)), np.arange(len(Xs))]   # cover and stego share a fold
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000))
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=42)
    acc = accuracy_score(y, cross_val_predict(clf, X, y, cv=cv, groups=g))
    return float(acc), float(1 - acc)

t0 = time.time()
for seed in SEEDS:
    fn = f"{OUT}/seed_{seed}.json"
    if os.path.exists(fn):
        print(f"seed {seed}: already on Drive, skipped"); continue
    rec = {}
    for rate in RATES:
        for arm, mk in ARMS.items():
            rng = np.random.default_rng(seed)           # identical bits in every arm
            outs, feats = [], []
            for i, p in enumerate(sel):
                c = CACHE[p]
                rho = (np.random.default_rng(seed + 10**6 + i).random(c["UERD"].shape)
                       if mk is None else mk(c))
                bits = rng.integers(0, 2, int(rate * c["n_ac"])).astype(np.uint8)
                o = f"/tmp/pe_{i}.jpg"
                E.embed_mode(p, o, bits, rho, mode="ac")
                outs.append(o); feats.append(dctr.dctr(cv2.imread(o, cv2.IMREAD_GRAYSCALE)))
            P1 = probs(outs)
            dp = np.abs(P1[ii, C0] - B0[ii, C0])
            acc, pe = detect(np.array(feats))
            rec[f"{arm}@{rate}"] = dict(acc=acc, pe=pe, dp=dp.tolist())
            for o in outs: os.remove(o)
            print(f"  seed {seed} rate {rate:.2f} {arm:16s} P_E {pe:.3f}  "
                  f"mean|dp| {dp.mean():.5f}  [{time.time()-t0:.0f}s]", flush=True)
    json.dump(dict(seed=seed, n=len(sel), rec=rec), open(fn, "w"))
    print(f"seed {seed} saved to Drive\n", flush=True)

# ---------------- summary ----------------
runs = [json.load(open(f"{OUT}/seed_{s}.json")) for s in SEEDS
        if os.path.exists(f"{OUT}/seed_{s}.json")]
print("=" * 86)
print(f"UNIFIED DETECTABILITY — 80 slices, {len(runs)} seed(s), identical bits across arms")
print("P(E) moves in steps of 1/160 = 0.00625 (160 classified files per arm)")
print(f"{'rate':>5} {'arm':>17} {'P_E':>7} {'P_E sd':>7} {'mean|dp|':>10} "
      f"{'dp gain':>9} {'P_E cost':>9} {'ratio':>7}")
summ = {}
for rate in RATES:
    base_dp = np.mean([np.mean(r["rec"][f"UERD@{rate}"]["dp"]) for r in runs])
    base_pe = np.mean([r["rec"][f"UERD@{rate}"]["pe"] for r in runs])
    for arm in ARMS:
        pes = [r["rec"][f"{arm}@{rate}"]["pe"] for r in runs]
        dps = [np.mean(r["rec"][f"{arm}@{rate}"]["dp"]) for r in runs]
        pe, dp = float(np.mean(pes)), float(np.mean(dps))
        gain = 100 * (1 - dp / base_dp); cost = 100 * (1 - pe / base_pe)
        ratio = gain / cost if arm.startswith("UERD+") or arm.startswith("G only") else float("nan")
        if cost <= 0: ratio = float("nan")
        summ[f"{arm}@{rate}"] = dict(pe=pe, pe_sd=float(np.std(pes)), dp=dp,
                                     gain=gain, cost=cost, ratio=ratio)
        print(f"{rate:>5.2f} {arm:>17} {pe:>7.3f} {np.std(pes):>7.3f} {dp:>10.5f} "
              f"{gain:>+8.1f}% {cost:>+8.1f}% {ratio:>7.2f}")
    print()
json.dump(summ, open(f"{OUT}/pe_unified_summary.json", "w"), indent=2)
print("summary saved to", f"{OUT}/pe_unified_summary.json")
print("=" * 86)
