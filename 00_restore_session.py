# 00_restore_session.py
# Restore the full session from Google Drive (data, modules, model, unified inference pipeline, 120-slice evaluation set and cost-map cache). Run first in every new Colab session.
# Extracted from the Colab notebook used for the paper; table numbers in the
# comments below are those of earlier drafts, see README for the mapping.

# ============================================================
#  RESTORE — full session state from Drive. Do not interrupt during pip.
# ============================================================
import subprocess, sys, os, glob, json, time, gc
from google.colab import drive
drive.mount("/content/drive", force_remount=False)
DRIVE = "/content/drive/MyDrive/dhide_ckpt"
print("in Drive:", sorted(os.listdir(DRIVE)))

print("installing jpegio and PyWavelets, ~2 min, do not interrupt ...", flush=True)
subprocess.run([sys.executable, "-m", "pip", "-q", "install",
                "jpegio", "h5py", "PyWavelets"], check=False)

import numpy as np, torch, cv2, jpegio, h5py, torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from scipy.fftpack import dct
print("torch", torch.__version__, "| device:",
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

if not os.path.exists("/content/data/rows.json"):
    subprocess.run(["tar", "xzf", f"{DRIVE}/data.tar.gz", "-C", "/content"], check=True)
for f in ("gate_core.py", "embed_core.py", "dctr.py"):
    src = f"{DRIVE}/{f}"
    assert os.path.exists(src), f"{f} missing from Drive"
    subprocess.run(["cp", src, "/content/"], check=True)
for a in ("resnet50", "efficientnet_b0", "densenet121"):
    if not os.path.exists(f"/content/{a}.pt"):
        subprocess.run(["cp", f"{DRIVE}/{a}.pt", f"/content/{a}.pt"], check=True)
sys.path.insert(0, "/content")
assert "uerd_cost" in open("/content/embed_core.py").read(), "embed_core lacks the baselines"
print("images:", len(glob.glob("/content/data/jpeg_qf75/*.jpg")))


exec(open("/content/gate_core.py").read())
rows = [tuple(r) for r in json.load(open("/content/data/rows.json"))]
df, clean = split_by_patient(rows); assert clean
print(f"{len(df)} slices, {df.pid.nunique()} patients")

# ---- corrected inference path: build at 512, resize inside ----
def _prep512(img8):
    x = torch.from_numpy(img8.astype(np.float32) / 255.)[None, None].repeat(1, 3, 1, 1)
    return (x.to(DEV) - MEAN) / STD

def _forward(x512):
    return net(F.interpolate(x512, (224, 224), mode="bilinear",
                             align_corners=False, antialias=True))

def to_tensor(img8):
    return F.interpolate(_prep512(img8), (224, 224), mode="bilinear",
                         align_corners=False, antialias=True)

def rho_G(net_, img8, Q):
    x = _prep512(img8).requires_grad_(True)
    out = _forward(x); cls = int(out.argmax(1))
    net_.zero_grad(); out[0, cls].backward()
    g = x.grad[0].sum(0).detach().cpu().numpy()
    nb = g.shape[0] // 8
    b = g.reshape(nb, 8, nb, 8).transpose(0, 2, 1, 3)
    G = np.abs(dct(dct(b, axis=2, norm="ortho"), axis=3, norm="ortho") * Q)
    return G.transpose(0, 2, 1, 3).reshape(g.shape)

net = models.resnet50(weights=None)
net.fc = nn.Linear(net.fc.in_features, 3)
net.load_state_dict(torch.load("/content/resnet50.pt", map_location=DEV))

net = net.to(DEV).eval()
for p_ in net.parameters(): p_.requires_grad_(False)

# ---- evaluation set and cost cache, same seed as every other table ----
import embed_core as E
te = df[df.split == "test"]
sub = te.sample(120, random_state=42)
paths = [f"/content/data/jpeg_qf75/{s}.jpg" for s in sub.slice_id]

print("\nrate table and cost maps ...", flush=True)
t0 = time.time()
TAB = E.rate_table(paths[:20], per_img=150)
CACHE = {}
for i, p in enumerate(paths):
    j = jpegio.read(p)
    C = np.array(j.coef_arrays[0], dtype=np.int64, copy=True)
    Q = j.quant_tables[0].astype(float); del j; gc.collect()
    img8 = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    CACHE[p] = dict(D=E.ranknorm(E.rho_D(Q, C.shape)),
                    R=E.ranknorm(E.rho_R(TAB, C.shape)),
                    G=E.ranknorm(rho_G(net, img8, Q)),
                    UERD=E.ranknorm(E.uerd_cost(C, Q)), cover=img8,
                    n_ac=len(E.carrier(C)[0]), n_dc=len(E.carrier_dc(C)[0]))
    if (i + 1) % 40 == 0: print(f"  {i+1}/{len(paths)}  [{time.time()-t0:.0f}s]", flush=True)

from sklearn.metrics import roc_auc_score
P0 = np.vstack([torch.softmax(net(torch.cat(
        [to_tensor(cv2.imread(f, cv2.IMREAD_GRAYSCALE)) for f in paths[k:k+32]])), 1)
        .detach().cpu().numpy() for k in range(0, len(paths), 32)])
CLS0 = P0.argmax(1); lab = np.asarray(sub.label.values) - 1
acc = (CLS0 == lab).mean()
print(f"\nclean cover: accuracy {acc:.4f}, "

      f"AUC {roc_auc_score(lab, P0, multi_class='ovr', average='macro', labels=[0,1,2]):.4f}")
print("expected 0.9083 / 0.9858")
assert acc > 0.85, "inference path is wrong, stop here"
print("\nrestored. run the J-UNIWARD cell next.")
