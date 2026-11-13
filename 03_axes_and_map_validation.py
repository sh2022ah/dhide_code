# 03_axes_and_map_validation.py
# Tables 5 and 6: cost-axis rank correlations (50 slices, 400 coefficients each) and map validation (40 probes per slice, Dice/IoU, Grad-CAM).
# Extracted from the Colab notebook used for the paper; table numbers in the
# comments below are those of earlier drafts, see README for the mapping.

# ============================================================
#  RERUN — Table 4 (gate) and Table 5 (map validation)
#  Both were measured with a model running at 512x512, outside the
#  resolution it was trained on. This repeats them under the corrected
#  pipeline. ~35 min.
# ============================================================
import os, gc, json, tempfile, warnings
import numpy as np, cv2, jpegio, torch
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")

ROOT, N_IMG, N_PROBE = "/content/data", 50, 40
rng = np.random.default_rng(42)
_t = df[df.split == "test"]
sub = _t.sample(min(N_IMG, len(_t)), random_state=42)

# sanity: the corrected pipeline must be in force
_p = f"{ROOT}/jpeg_qf75/{sub.slice_id.iloc[0]}.jpg"
assert tuple(to_tensor(cv2.imread(_p, cv2.IMREAD_GRAYSCALE)).shape) == (1, 3, 224, 224), \
    "to_tensor is not the corrected version"
_j = jpegio.read(_p); _Q = _j.quant_tables[0].astype(float); del _j
assert rho_G(net, cv2.imread(_p, cv2.IMREAD_GRAYSCALE), _Q).shape == (512, 512), \
    "rho_G is not the corrected version"
print("corrected pipeline confirmed\n")


# ---------- Table 4: the three cost axes ---------------------------------
print("TABLE 4 — cost-axis correlations", flush=True)
res4, pooled = {}, {"D": [], "R": [], "G": []}
for qf in (50, 75, 90):
    acc = {"D": [], "R": [], "G": []}
    for _, row in sub.iterrows():
        jpg = f"{ROOT}/jpeg_qf{qf}/{row.slice_id}.jpg"
        if not os.path.exists(jpg): continue
        j = jpegio.read(jpg)
        C = np.array(j.coef_arrays[0], dtype=np.int64, copy=True)
        Q = j.quant_tables[0].astype(float); del j; gc.collect()
        base = os.path.getsize(jpg)
        G = rho_G(net, cv2.imread(jpg, cv2.IMREAD_GRAYSCALE), Q)
        nz = np.argwhere(C != 0)
        if len(nz) < 50: continue
        sel = nz[rng.choice(len(nz), min(400, len(nz)), replace=False)]
        for r, c in sel:
            jj = jpegio.read(jpg); jj.coef_arrays[0][r, c] += 1
            t = tempfile.mktemp(suffix=".jpg"); jpegio.write(jj, t)
            acc["R"].append(os.path.getsize(t) - base); os.remove(t)
            acc["D"].append(float(Q[r % 8, c % 8]) ** 2)
            acc["G"].append(float(G[r, c]))
    for k in acc:
        acc[k] = np.asarray(acc[k]); pooled[k].append(acc[k])
    n = len(acc["D"]); res4[f"n_{qf}"] = n
    for a, b in (("D", "R"), ("D", "G"), ("R", "G")):
        res4[f"{a}{b}_{qf}"] = float(spearmanr(acc[a], acc[b])[0])
    R = acc["R"]
    res4[f"rate_{qf}"] = [float(R.min()), float(R.max()), float(R.mean()),
                          float((R == 0).mean())]
    print(f"  QF {qf}: n={n}  rho(D,R)={res4[f'DR_{qf}']:+.3f}  "
          f"rho(D,G)={res4[f'DG_{qf}']:+.3f}  rho(R,G)={res4[f'RG_{qf}']:+.3f}", flush=True)
P = {k: np.concatenate(v) for k, v in pooled.items()}
for a, b in (("D", "R"), ("D", "G"), ("R", "G")):
    res4[f"{a}{b}_all"] = float(spearmanr(P[a], P[b])[0])
res4["n_all"] = len(P["D"])
print(f"  pooled: n={res4['n_all']}  rho(D,G)={res4['DG_all']:+.3f}\n", flush=True)


# ---------- Table 5: validation of the map -------------------------------
print("TABLE 5 — sensitivity map validation", flush=True)
from scipy.fftpack import dct
import torch.nn.functional as F

_feat, _grad = {}, {}
_tg = net.layer4[-1]
_h1 = _tg.register_forward_hook(lambda m, i, o: _feat.__setitem__("v", o))
_h2 = _tg.register_full_backward_hook(lambda m, gi, go: _grad.__setitem__("v", go[0]))

def grad_cam(img8):
    x = _prep512(img8).requires_grad_(True)
    out = _forward(x); cls = int(out.argmax(1))
    net.zero_grad(); out[0, cls].backward()
    f, g = _feat["v"][0], _grad["v"][0]
    return torch.relu((g.mean(dim=(1, 2), keepdim=True) * f).sum(0)).detach().cpu().numpy()

def bpool(a, n):
    s = a.shape[0] // n
    return a.reshape(n, s, n, s).sum(axis=(1, 3))

def signed_map(img8, Q):
    x = _prep512(img8).requires_grad_(True)
    out = _forward(x); cls = int(out.argmax(1))
    net.zero_grad(); out[0, cls].backward()
    g = x.grad[0].sum(0).detach().cpu().numpy()
    nb = g.shape[0] // 8
    b = g.reshape(nb, 8, nb, 8).transpose(0, 2, 1, 3)
    S = dct(dct(b, axis=2, norm="ortho"), axis=3, norm="ortho") * Q
    return S.transpose(0, 2, 1, 3).reshape(g.shape), cls

sgn, absl, dice, iou, mass, blk, cam_r = [], [], [], [], [], [], []
for k, (_, row) in enumerate(sub.iterrows(), 1):
    jpg = f"{ROOT}/jpeg_qf75/{row.slice_id}.jpg"
    mp = f"{ROOT}/masks/{row.slice_id}.png"
    if not (os.path.exists(jpg) and os.path.exists(mp)): continue
    j = jpegio.read(jpg)
    C = np.array(j.coef_arrays[0], dtype=np.int64, copy=True)
    Q = j.quant_tables[0].astype(float); del j; gc.collect()
    img8 = cv2.imread(jpg, cv2.IMREAD_GRAYSCALE)
    S, cls = signed_map(img8, Q); G = np.abs(S)
    with torch.no_grad():
        l0 = float(net(to_tensor(img8))[0][cls])
    nz = np.argwhere(C != 0)
    sel = nz[rng.choice(len(nz), min(N_PROBE, len(nz)), replace=False)]
    d = []
    for r, c in sel:
        jj = jpegio.read(jpg); jj.coef_arrays[0][r, c] += 1
        t = tempfile.mktemp(suffix=".jpg"); jpegio.write(jj, t)
        with torch.no_grad():
            d.append(float(net(to_tensor(cv2.imread(t, cv2.IMREAD_GRAYSCALE)))[0][cls]) - l0)
        os.remove(t)
    d = np.asarray(d); pr = S[sel[:, 0], sel[:, 1]]
    if d.std() > 0 and pr.std() > 0:
        sgn.append(spearmanr(pr, d)[0]); absl.append(spearmanr(abs(pr), abs(d))[0])

    mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 127
    nb = 64
    tum = mask.reshape(nb, 8, nb, 8).transpose(0, 2, 1, 3).reshape(nb, nb, 64).any(2)
    gb = bpool(G, nb)
    if tum.sum() > 0:
        hot = gb >= np.percentile(gb, 100 * (1 - tum.mean()))
        inter = float((hot & tum).sum())
        dice.append(2 * inter / max(hot.sum() + tum.sum(), 1))
        iou.append(inter / max((hot | tum).sum(), 1))
        mass.append(gb[tum].sum() / max(gb.sum(), 1e-12)); blk.append(tum.mean())

    cam = grad_cam(img8); n = cam.shape[0]
    gs = bpool(G, n).ravel()
    if cam.std() > 0 and gs.std() > 0:
        cam_r.append(spearmanr(gs, cam.ravel())[0])
    if k % 10 == 0: print(f"  {k}/{len(sub)}", flush=True)

_h1.remove(); _h2.remove()
res5 = dict(signed=float(np.mean(sgn)), signed_sd=float(np.std(sgn)),
            absolute=float(np.mean(absl)), absolute_sd=float(np.std(absl)),
            dice=float(np.mean(dice)), dice_sd=float(np.std(dice)),
            iou=float(np.mean(iou)), mass=float(np.mean(mass)),
            blkfrac=float(np.mean(blk)), gradcam=float(np.mean(cam_r)),
            gradcam_sd=float(np.std(cam_r)), n=len(dice))
res5["concentration"] = res5["mass"] / max(res5["blkfrac"], 1e-9)

json.dump(dict(table4=res4, table5=res5), open("/content/tables45_fixed.json", "w"), indent=2)
print("\n" + "=" * 70)
print("TABLE 4")
print(f"{'QF':>5} {'n':>7} {'rho(D,R)':>10} {'rho(D,G)':>10} {'rho(R,G)':>10}")
for qf in (50, 75, 90):
    print(f"{qf:>5} {res4[f'n_{qf}']:>7} {res4[f'DR_{qf}']:>+10.3f} "
          f"{res4[f'DG_{qf}']:>+10.3f} {res4[f'RG_{qf}']:>+10.3f}")
print(f"{'all':>5} {res4['n_all']:>7} {res4['DR_all']:>+10.3f} "
      f"{res4['DG_all']:>+10.3f} {res4['RG_all']:>+10.3f}")
for qf in (50, 75, 90):
    lo, hi, mu, z = res4[f"rate_{qf}"]
    print(f"  rate QF {qf}: min {lo:+.0f}  max {hi:+.0f}  mean {mu:+.3f}  no effect {100*z:.1f}%")
print("\nTABLE 5")
print(f"  signed probe   : {res5['signed']:+.3f} +/- {res5['signed_sd']:.3f}  (threshold 0.70)")
print(f"  absolute probe : {res5['absolute']:+.3f} +/- {res5['absolute_sd']:.3f}")
print(f"  Dice           : {res5['dice']:.3f} +/- {res5['dice_sd']:.3f}")
print(f"  IoU            : {res5['iou']:.3f}")
print(f"  Grad-CAM       : {res5['gradcam']:+.3f} +/- {res5['gradcam_sd']:.3f}")
print(f"  mass in tumour : {100*res5['mass']:.1f}%   tumour blocks {100*res5['blkfrac']:.1f}%")
print(f"  concentration  : {res5['concentration']:.2f}x")
print("=" * 70)
print(open("/content/tables45_fixed.json").read())
