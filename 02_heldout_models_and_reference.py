# 02_heldout_models_and_reference.py
# Trains EfficientNet-B0 and DenseNet-121 and computes the reference metrics of Table 4.
# Extracted from the Colab notebook used for the paper; table numbers in the
# comments below are those of earlier drafts, see README for the mapping.

# ============================================================
#  Tables 3 and 7 — held-out architectures and transfer
#
#  The optimiser sees only ResNet-50. If the 60 percent reduction is a
#  property of that network rather than of the criterion, it will not
#  survive on models the optimiser never touched. Embedding is performed
#  with the ResNet-50 sensitivity map and evaluated on all three.
#  ~60 min: two models to train, then the transfer sweep.
# ============================================================
import os, gc, json, time, warnings
import numpy as np, cv2, jpegio, torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score, f1_score
import embed_core as E
warnings.filterwarnings("ignore")

ROOT, QF = "/content/data", 75
DRIVE = "/content/drive/MyDrive/dhide_ckpt"
os.makedirs(DRIVE, exist_ok=True)
rng = np.random.default_rng(42)

def build(arch):
    m = getattr(models, arch)(weights="DEFAULT")
    if hasattr(m, "fc"):
        m.fc = nn.Linear(m.fc.in_features, 3)
    elif isinstance(m.classifier, nn.Sequential):
        last = [x for x in m.classifier if isinstance(x, nn.Linear)][-1]
        m.classifier[-1] = nn.Linear(last.in_features, 3)
    else:
        m.classifier = nn.Linear(m.classifier.in_features, 3)
    return m.to(DEV)

# ---------- train or restore the two held-out models ----------------------
NETS = {"resnet50": net}
for arch in ["efficientnet_b0", "densenet121"]:
    ck = f"{DRIVE}/{arch}.pt"
    m = build(arch)
    if os.path.exists(ck):
        m.load_state_dict(torch.load(ck, map_location=DEV))
        print(f"{arch}: restored from Drive", flush=True)
    else:
        print(f"training {arch} ...", flush=True)
        tr, va = df[df.split == "train"], df[df.split == "val"]
        dl_tr = DataLoader(DS(tr, ROOT, QF, True), batch_size=32, shuffle=True, num_workers=2)
        dl_va = DataLoader(DS(va, ROOT, QF, False), batch_size=32, num_workers=2)
        cnt = tr.label.value_counts().reindex([1, 2, 3]).fillna(0).values
        w = torch.tensor((cnt.sum() / np.maximum(cnt, 1)) / 3, dtype=torch.float32, device=DEV)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 15)
        lf, best = nn.CrossEntropyLoss(weight=w), -1
        for ep in range(1, 16):
            m.train()
            for x, y in dl_tr:
                x, y = x.to(DEV), y.to(DEV)
                opt.zero_grad(); lf(m(x), y).backward(); opt.step()
            sch.step()
            a, u = evaluate(m, dl_va)
            if u > best: best = u; torch.save(m.state_dict(), ck)
            print(f"  epoch {ep:02d}  val acc {a:.4f}  auc {u:.4f}", flush=True)
        m.load_state_dict(torch.load(ck, map_location=DEV))
    m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    NETS[arch] = m

# ---------- Table 3: reference metrics under the unified pipeline ---------
def infer(model, files):
    P = []
    with torch.no_grad():
        for k in range(0, len(files), 32):
            xs = [to_tensor(cv2.imread(f, cv2.IMREAD_GRAYSCALE)) for f in files[k:k+32]]
            P.append(torch.softmax(model(torch.cat(xs)), 1).cpu().numpy())
    return np.vstack(P)

te = df[df.split == "test"]
te_paths = [f"{ROOT}/jpeg_qf{QF}/{s}.jpg" for s in te.slice_id]
te_lab = np.asarray(te.label.values) - 1
table3 = {}
print("\nTable 3 — reference performance")
for name, m in NETS.items():
    P = infer(m, te_paths); pred = P.argmax(1)
    table3[name] = dict(
        acc=float((pred == te_lab).mean()),
        auc=float(roc_auc_score(te_lab, P, multi_class="ovr", average="macro", labels=[0,1,2])),
        f1=float(f1_score(te_lab, pred, average="weighted")),
        sens=[float((pred[te_lab == k] == k).mean()) for k in range(3)])
    v = table3[name]
    print(f"  {name:16s} acc {v['acc']:.4f} | AUC {v['auc']:.4f} | F1 {v['f1']:.4f} | "
          f"sens {v['sens'][0]:.3f}/{v['sens'][1]:.3f}/{v['sens'][2]:.3f}", flush=True)
