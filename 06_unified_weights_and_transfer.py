# 06_unified_weights_and_transfer.py
# Tables 8 (UERD rows) and 9: five-seed, image-level analysis on three models.
# Extracted from the Colab notebook used for the paper; table numbers in the
# comments below are those of earlier drafts, see README for the mapping.

# ============================================================
#  UNIFIED RUN — Tables 7 and 8 from one execution
#
#  The two tables previously came from separate sessions on different
#  compute backends, and their surrogate rows disagreed by about ten
#  percent. This computes every cell from one pass over one cost cache,
#  so the two tables are slices of a single result matrix.
#
#  Arms   : UERD, UERD + 0.25 diagnostic, UERD + 0.50 diagnostic
#  Models : ResNet-50 (surrogate), EfficientNet-B0, DenseNet-121
#  Rates  : 0.05, 0.10        Seeds: 5
#  Unit of analysis: the image. Seed outcomes are averaged within an
#  image before testing, so n = 120 rather than 600.
#  ~35 min on a T4.
# ============================================================
import os, json, time, warnings
import numpy as np, torch, cv2, torch.nn as nn
from torchvision import models
from scipy.stats import wilcoxon
import embed_core as E
warnings.filterwarnings("ignore")

DRIVE = "/content/drive/MyDrive/dhide_ckpt"
SEEDS = [42, 123, 456, 789, 1011]
RATES = [0.05, 0.10]
ARMS = {"UERD": 0.0, "UERD+0.25G": 0.25, "UERD+0.50G": 0.50}

print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("images:", len(paths), "| cost maps cached:", len(CACHE), flush=True)


def build(arch):
    m = getattr(models, arch)(weights=None)
    if hasattr(m, "fc"):
        m.fc = nn.Linear(m.fc.in_features, 3)
    elif isinstance(m.classifier, nn.Sequential):
        last = [x for x in m.classifier if isinstance(x, nn.Linear)][-1]
        m.classifier[-1] = nn.Linear(last.in_features, 3)
    else:
        m.classifier = nn.Linear(m.classifier.in_features, 3)
    return m.to(DEV)


NETS = {"resnet50": net}
for arch in ["efficientnet_b0", "densenet121"]:
    ck = f"/content/{arch}.pt"
    if not os.path.exists(ck):
        import subprocess
        subprocess.run(["cp", f"{DRIVE}/{arch}.pt", ck], check=True)
    m = build(arch)
    m.load_state_dict(torch.load(ck, map_location=DEV))
    m.eval()
    for p_ in m.parameters():
        p_.requires_grad_(False)
    NETS[arch] = m
print("models:", list(NETS), flush=True)


@torch.no_grad()
def probs_of(model, files):
    P = []
    for k in range(0, len(files), 32):
        xs = [to_tensor(cv2.imread(f, cv2.IMREAD_GRAYSCALE)) for f in files[k:k + 32]]
        P.append(torch.softmax(model(torch.cat(xs)), 1).cpu().numpy())
    return np.vstack(P)


BASE = {nm: probs_of(m, paths) for nm, m in NETS.items()}
CLS = {nm: BASE[nm].argmax(1) for nm in NETS}
ii = np.arange(len(paths))
lab = np.asarray(sub.label.values) - 1
for nm in NETS:
    print(f"  {nm:16s} clean accuracy {(CLS[nm] == lab).mean():.4f}")

# raw[model][rate][arm] = 5 seed rows, each of length n_images
raw = {nm: {str(r): {a: [] for a in ARMS} for r in RATES} for nm in NETS}
t0 = time.time()

for rate in RATES:
    for arm, w in ARMS.items():
        for seed in SEEDS:
            rng = np.random.default_rng(seed)      # identical payloads across arms
            outs = []
            for p in paths:
                c = CACHE[p]
                rho = (1 - w) * c["UERD"] + w * c["G"]
                o = f"/tmp/u_{os.path.basename(p)}"
                E.embed_mode(p, o,
                             rng.integers(0, 2, int(rate * c["n_ac"])).astype(np.uint8),
                             rho, mode="ac")
                outs.append(o)
            for nm, m in NETS.items():
                P1 = probs_of(m, outs)
                dp = np.abs(P1[ii, CLS[nm]] - BASE[nm][ii, CLS[nm]])
                raw[nm][str(rate)][arm].append(dp.tolist())
            for o in outs:
                os.remove(o)
        print(f"  rate {rate:.2f} {arm:11s} done  [{time.time() - t0:.0f}s]", flush=True)
    print()

json.dump(dict(seeds=SEEDS, rates=RATES, arms=list(ARMS), n_images=len(paths),
               slice_ids=[os.path.basename(p)[:-4] for p in paths], raw=raw),
          open("/content/unified_raw.json", "w"))
import subprocess
subprocess.run(["cp", "/content/unified_raw.json", f"{DRIVE}/"], check=False)
print(f"unified_raw.json written, {os.path.getsize('/content/unified_raw.json')/1e6:.1f} MB\n")


def img_mean(nm, rate, arm):
    return np.array(raw[nm][str(rate)][arm]).mean(axis=0)


def compare(a, b):
    s, p = wilcoxon(a, b)
    d = np.sign(a - b)
    return dict(a=float(a.mean()), b=float(b.mean()),
                a_med=float(np.median(a)), b_med=float(np.median(b)),
                red=float(100 * (1 - a.mean() / b.mean())),
                red_med=float(100 * (1 - np.median(a) / np.median(b))),
                p=float(p), cliff=float((d < 0).mean() - (d > 0).mean()),
                win=float((a < b).mean()), n=int(len(a)))


# ---------------- Table 7: weights on the surrogate ----------------
t7 = {}
for rate in RATES:
    base = img_mean("resnet50", rate, "UERD")
    for arm in ("UERD+0.25G", "UERD+0.50G"):
        t7[f"{arm}@{rate}"] = compare(img_mean("resnet50", rate, arm), base)
o = sorted(t7.items(), key=lambda kv: kv[1]["p"])
for i, (k, v) in enumerate(o):
    v["holm"] = float(min(v["p"] * (len(o) - i), 1.0))

print("=" * 96)
print("TABLE 7  weights on the surrogate, image-level, n = %d" % len(paths))
print(f"{'rate':>5} {'arm':>12} {'mean e-4':>9} {'UERD e-4':>9} {'med e-5':>8} {'UERD e-5':>9} "
      f"{'red m/med':>13} {'Holm p':>9} {'Cliff':>7} {'lower on':>9}")
for rate in RATES:
    for arm in ("UERD+0.25G", "UERD+0.50G"):
        v = t7[f"{arm}@{rate}"]
        print(f"{rate:>5.2f} {arm:>12} {v['a']*1e4:>9.3f} {v['b']*1e4:>9.3f} "
              f"{v['a_med']*1e5:>8.3f} {v['b_med']*1e5:>9.3f} "
              f"{v['red']:>6.1f}%/{v['red_med']:>5.1f}% {v['holm']:>9.4f} "
              f"{v['cliff']:>+7.3f} {100*v['win']:>8.1f}%")

print("\nhead to head, 0.25 against 0.50")
for rate in RATES:
    v = compare(img_mean("resnet50", rate, "UERD+0.25G"),
                img_mean("resnet50", rate, "UERD+0.50G"))
    print(f"  rate {rate:.2f}: 0.25 lower by {v['red']:+.1f}%  p={v['p']:.3f}  "
          f"Cliff={v['cliff']:+.3f}  lower on {100*v['win']:.1f}%")

# ---------------- Table 8: transfer at the recommended weight ----------------
t8 = {}
for rate in RATES:
    for nm in NETS:
        t8[f"{nm}@{rate}"] = compare(img_mean(nm, rate, "UERD+0.25G"),
                                     img_mean(nm, rate, "UERD"))
o = sorted(t8.items(), key=lambda kv: kv[1]["p"])
for i, (k, v) in enumerate(o):
    v["holm"] = float(min(v["p"] * (len(o) - i), 1.0))

print("\n" + "=" * 96)
print("TABLE 8  transfer at the quarter weight, image-level, n = %d" % len(paths))
print(f"{'rate':>5} {'model':>17} {'prop':>9} {'UERD':>9} {'change':>8} "
      f"{'raw p':>10} {'Holm p':>9} {'Cliff':>7} {'lower on':>9}")
for rate in RATES:
    for nm in NETS:
        v = t8[f"{nm}@{rate}"]
        print(f"{rate:>5.2f} {nm:>17} {v['a']:>9.5f} {v['b']:>9.5f} {v['red']:>+7.1f}% "
              f"{v['p']:>10.2e} {v['holm']:>9.4f} {v['cliff']:>+7.3f} {100*v['win']:>8.1f}%")
print("=" * 96)

json.dump(dict(table7=t7, table8=t8), open("/content/unified_tables.json", "w"), indent=2)
subprocess.run(["cp", "/content/unified_tables.json", f"{DRIVE}/"], check=False)
print("\nunified_tables.json saved. The surrogate rows of Table 8 now come from the same")
print("matrix as Table 7, so the two can no longer disagree.")
