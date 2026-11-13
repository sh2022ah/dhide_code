# 07_juniward.py
# Table 8 J-UNIWARD rows: re-implemented J-UNIWARD with sanity checks, five seeds.
# Extracted from the Colab notebook used for the paper; table numbers in the
# comments below are those of earlier drafts, see README for the mapping.

# ============================================================
#  J-UNIWARD baseline — the comparison the editor found missing
#
#  Re-implemented from Holub, Fridrich and Denemark (2014). Three sanity
#  checks run before any result is produced; if one fails the cell stops
#  rather than reporting a figure from a suspect implementation.
#
#  Produces, over 120 held-out slices and five payload seeds:
#    - diagnostic disturbance for J-UNIWARD and J-UNIWARD + 0.25 diagnostic
#    - the same for UERD, so all four sit in one table
#    - DCTR detection error for each
#  ~40 min on a T4.
# ============================================================
import base64, zlib, os, gc, json, time, warnings
_J = (
    "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"
    "/Tc/tNlkDP4NswsJOA=="
)
open("/content/juniward.py", "wb").write(zlib.decompress(base64.b64decode(_J)))
print("juniward.py written")

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "PyWavelets"], check=False)
sys.path.insert(0, "/content")

import numpy as np, cv2, jpegio, torch
from scipy.stats import wilcoxon
import juniward as J, embed_core as E, dctr
warnings.filterwarnings("ignore")

DRIVE = "/content/drive/MyDrive/dhide_ckpt"
SEEDS, RATES = [42, 123, 456, 789, 1011], [0.05, 0.10]
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("images:", len(paths), "| cache:", len(CACHE), flush=True)

# ---------------- sanity checks on one slice ----------------
p0 = paths[0]
j = jpegio.read(p0)
C0 = np.array(j.coef_arrays[0], dtype=np.int64, copy=True)
Q0 = j.quant_tables[0].astype(float); del j; gc.collect()
print("\nsanity checks on one slice:")
_, chk = J.sanity_checks(C0, Q0, cv2.imread(p0, cv2.IMREAD_GRAYSCALE),
                         uerd_map=E.uerd_cost(C0, Q0))
assert chk["finite"] and chk["positive"], "cost map is not finite and positive"
assert chk["rho_cost_texture"] < -0.5, \
    f"cost does not fall with texture (rho={chk['rho_cost_texture']:.3f})"
assert 0.25 < chk["rho_vs_uerd"] < 0.90, \
    f"agreement with UERD out of range (rho={chk['rho_vs_uerd']:.3f})"
print("  all three checks passed\n", flush=True)

# ---------------- add the J-UNIWARD map to the cache ----------------
print("computing J-UNIWARD cost maps ...", flush=True)
t0 = time.time()
for i, p in enumerate(paths):
    j = jpegio.read(p)
    C = np.array(j.coef_arrays[0], dtype=np.int64, copy=True)
    Q = j.quant_tables[0].astype(float); del j; gc.collect()
    CACHE[p]["JUNI"] = E.ranknorm(J.juniward_cost(C, Q, CACHE[p]["cover"]))
    if (i + 1) % 40 == 0:
        print(f"  {i+1}/{len(paths)}  [{time.time()-t0:.0f}s]", flush=True)

ARMS = {
    "UERD":        lambda c: c["UERD"],
    "UERD+0.25G":  lambda c: 0.75 * c["UERD"] + 0.25 * c["G"],
    "J-UNIWARD":   lambda c: c["JUNI"],
    "JUNI+0.25G":  lambda c: 0.75 * c["JUNI"] + 0.25 * c["G"],
}

@torch.no_grad()
def probs(files):
    P = []
    for k in range(0, len(files), 32):
        xs = [to_tensor(cv2.imread(f, cv2.IMREAD_GRAYSCALE)) for f in files[k:k+32]]
        P.append(torch.softmax(net(torch.cat(xs)), 1).cpu().numpy())
    return np.vstack(P)

BASE = probs(paths); CLS = BASE.argmax(1); ii = np.arange(len(paths))
lab = np.asarray(sub.label.values) - 1
print(f"clean cover accuracy {(CLS == lab).mean():.4f}")
assert (CLS == lab).mean() > 0.85, "inference path is wrong"

sel = paths[:80]
COVER = np.array([dctr.dctr(cv2.imread(p, cv2.IMREAD_GRAYSCALE)) for p in sel])
print(f"cover DCTR features {COVER.shape}\n", flush=True)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import accuracy_score

def detect_error(X1):
    X = np.vstack([COVER, X1])
    y = np.r_[np.zeros(len(COVER)), np.ones(len(X1))]
    g = np.r_[np.arange(len(COVER)), np.arange(len(X1))]
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000))
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=42)
    return float(1 - accuracy_score(y, cross_val_predict(clf, X, y, cv=cv, groups=g)))

raw = {a: {str(r): [] for r in RATES} for a in ARMS}
pe = {a: {} for a in ARMS}
t0 = time.time()

for rate in RATES:
    for arm, mk in ARMS.items():
        feats = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            outs = []
            for p in paths:
                c = CACHE[p]
                o = f"/tmp/jw_{os.path.basename(p)}"
                E.embed_mode(p, o, rng.integers(0, 2, int(rate * c["n_ac"])).astype(np.uint8),
                             mk(c), mode="ac")
                outs.append(o)
            P1 = probs(outs)
            raw[arm][str(rate)].append(np.abs(P1[ii, CLS] - BASE[ii, CLS]).tolist())
            if seed == SEEDS[0]:
                feats = [dctr.dctr(cv2.imread(o, cv2.IMREAD_GRAYSCALE)) for o in outs[:80]]
            for o in outs: os.remove(o)
        pe[arm][str(rate)] = detect_error(np.array(feats))
        m = np.array(raw[arm][str(rate)]).mean(axis=0)
        print(f"  rate {rate:.2f} {arm:12s} |dp| {m.mean():.5f}  P_E {pe[arm][str(rate)]:.3f} "
              f"[{time.time()-t0:.0f}s]", flush=True)
    print()

json.dump(dict(seeds=SEEDS, rates=RATES, arms=list(ARMS), raw=raw, pe=pe),
          open("/content/juniward_results.json", "w"))
subprocess.run(["cp", "/content/juniward_results.json", f"{DRIVE}/"], check=False)

def im(arm, rate): return np.array(raw[arm][str(rate)]).mean(axis=0)

print("=" * 92)
print("FOUR-WAY COMPARISON, image-level, n = %d" % len(paths))
print(f"{'rate':>5} {'criterion':>13} {'mean e-4':>9} {'median e-5':>11} {'P_E':>7} "
      f"{'vs base':>9} {'p':>10} {'Cliff':>7}")
out = {}
for rate in RATES:
    for base_arm, add_arm in (("UERD", "UERD+0.25G"), ("J-UNIWARD", "JUNI+0.25G")):
        b = im(base_arm, rate); a = im(add_arm, rate)
        s, pv = wilcoxon(a, b); d = np.sign(a - b)
        for nm, v, extra in ((base_arm, b, None), (add_arm, a, (pv, d))):
            red = "" if extra is None else f"{100*(1-a.mean()/b.mean()):+8.1f}%"
            pp = "" if extra is None else f"{pv:10.2e}"
            cf = "" if extra is None else f"{float((d<0).mean()-(d>0).mean()):+7.3f}"
            print(f"{rate:>5.2f} {nm:>13} {v.mean()*1e4:>9.3f} {np.median(v)*1e5:>11.3f} "
                  f"{pe[nm][str(rate)]:>7.3f} {red:>9} {pp:>10} {cf:>7}")
        out[f"{add_arm}@{rate}"] = dict(base=float(b.mean()), prop=float(a.mean()),
            red=float(100*(1-a.mean()/b.mean())), p=float(pv),
            cliff=float((d<0).mean()-(d>0).mean()), win=float((a<b).mean()))
print("\nJ-UNIWARD against UERD, no diagnostic term")
for rate in RATES:
    x = im("J-UNIWARD", rate); y = im("UERD", rate)
    s, pv = wilcoxon(x, y); d = np.sign(x - y)
    print(f"  rate {rate:.2f}: J-UNIWARD {x.mean()*1e4:.3f} vs UERD {y.mean()*1e4:.3f} e-4  "
          f"({100*(1-x.mean()/y.mean()):+.1f}%)  p={pv:.3f}  Cliff={float((d<0).mean()-(d>0).mean()):+.3f}")
print("=" * 92)
json.dump(dict(tests=out, pe=pe, checks=chk),
          open("/content/juniward_tables.json", "w"), indent=2)
subprocess.run(["cp", "/content/juniward_tables.json", f"{DRIVE}/"], check=False)
print("\nsaved to Drive.")
