"""
Self-contained Phase-0 gate. Written to be embedded in a Colab notebook.
Tested locally on the sample .mat files before packaging.
"""
import os, glob, gc, warnings
import numpy as np
import h5py, cv2, jpegio, torch
import torch.nn as nn
from scipy.fftpack import dct
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

warnings.filterwarnings("ignore")
SEED = 42
P_LO, P_HI = 0.5, 99.5
LBL = {1: "meningioma", 2: "glioma", 3: "pituitary"}
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEV)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEV)


# ---------------------------------------------------------------- data
def read_mat(path):
    with h5py.File(path, "r") as f:
        d = f["cjdata"]
        return dict(pid="".join(chr(c) for c in d["PID"][:, 0]),
                    image=np.array(d["image"]).astype(np.float32),
                    mask=np.array(d["tumorMask"]).astype(np.uint8),
                    label=int(d["label"][0, 0]))


def convert(src, out, qfs=(50, 75, 90)):
    for qf in qfs:
        os.makedirs(f"{out}/jpeg_qf{qf}", exist_ok=True)
    os.makedirs(f"{out}/masks", exist_ok=True)
    rows = []
    files = sorted(glob.glob(f"{src}/**/*.mat", recursive=True),
                   key=lambda p: int("".join(c for c in os.path.basename(p)[:-4]
                                             if c.isdigit()) or 0))
    for p in files:
        sid = os.path.splitext(os.path.basename(p))[0]
        try:
            r = read_mat(p)
        except Exception:
            continue
        if r["image"].shape != (512, 512):
            continue
        lo, hi = np.percentile(r["image"], [P_LO, P_HI])
        if hi <= lo:
            continue
        img8 = np.clip((r["image"] - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(f"{out}/masks/{sid}.png", r["mask"] * 255)
        for qf in qfs:
            cv2.imwrite(f"{out}/jpeg_qf{qf}/{sid}.jpg", img8,
                        [int(cv2.IMWRITE_JPEG_QUALITY), qf])
        rows.append((sid, r["pid"], r["label"]))
    return rows


def split_by_patient(rows):
    import pandas as pd
    df = pd.DataFrame(rows, columns=["slice_id", "pid", "label"])
    df["label_name"] = df.label.map(LBL)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr, te = next(sgkf.split(df, df.label, groups=df.pid))
    df["split"] = "train"
    df.loc[df.index[te], "split"] = "test"
    sub = df[df.split == "train"]
    if sub.pid.nunique() >= 6:
        s2 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        _, va = next(s2.split(sub, sub.label, groups=sub.pid))
        df.loc[sub.index[va], "split"] = "val"
    sets = {s: set(g.pid) for s, g in df.groupby("split")}
    overlap = any(sets.get(a, set()) & sets.get(b, set())
                  for a, b in [("train", "val"), ("train", "test"), ("val", "test")])
    return df, (not overlap)


# ---------------------------------------------------------------- model
class DS(Dataset):
    def __init__(self, df, root, qf, train):
        self.df, self.root, self.qf = df.reset_index(drop=True), root, qf
        aug = [transforms.RandomHorizontalFlip(),
               transforms.RandomAffine(10, translate=(.05, .05), scale=(.95, 1.05))]
        self.tf = transforms.Compose([transforms.ToPILImage()] + (aug if train else []) +
                                     [transforms.Resize((224, 224)), transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        g = cv2.imread(f"{self.root}/jpeg_qf{self.qf}/{r.slice_id}.jpg",
                       cv2.IMREAD_GRAYSCALE)
        return self.tf(np.repeat(g[:, :, None], 3, 2)), int(r.label) - 1


@torch.no_grad()
def evaluate(net, dl):
    net.eval()
    P, Y = [], []
    for x, y in dl:
        P.append(torch.softmax(net(x.to(DEV)), 1).cpu().numpy())
        Y.append(y.numpy())
    P, Y = np.vstack(P), np.concatenate(Y)
    acc = (P.argmax(1) == Y).mean()
    try:
        auc = roc_auc_score(Y, P, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")
    return acc, auc


def train_cad(df, root, qf=75, epochs=15, bs=32, ckpt=None):
    ckpt = ckpt or os.path.join(root, "resnet50.pt")
    torch.manual_seed(SEED)
    tr, va = df[df.split == "train"], df[df.split == "val"]
    if len(va) == 0:
        va = df[df.split == "test"]
    net = models.resnet50(weights="DEFAULT")
    net.fc = nn.Linear(net.fc.in_features, 3)
    net = net.to(DEV)
    dl_tr = DataLoader(DS(tr, root, qf, True), batch_size=bs, shuffle=True, num_workers=2)
    dl_va = DataLoader(DS(va, root, qf, False), batch_size=bs, num_workers=2)
    cnt = tr.label.value_counts().reindex([1, 2, 3]).fillna(0).values
    w = torch.tensor((cnt.sum() / np.maximum(cnt, 1)) / 3, dtype=torch.float32, device=DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lossf = nn.CrossEntropyLoss(weight=w)
    best = -1
    for ep in range(1, epochs + 1):
        net.train()
        tot = 0
        for x, y in dl_tr:
            x, y = x.to(DEV), y.to(DEV)
            opt.zero_grad()
            l = lossf(net(x), y)
            l.backward()
            opt.step()
            tot += l.item() * len(y)
        sch.step()
        acc, auc = evaluate(net, dl_va)
        s = auc if not np.isnan(auc) else acc
        flag = ""
        if s > best:
            best, flag = s, "  <- saved"
            torch.save(net.state_dict(), ckpt)
        print(f"  epoch {ep:02d}  loss {tot/max(len(tr),1):.4f}  "
              f"val acc {acc:.4f}  auc {auc:.4f}{flag}", flush=True)
    net.load_state_dict(torch.load(ckpt, map_location=DEV))
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net, best


# ---------------------------------------------------------------- gate
def to_tensor(img8):
    x = torch.from_numpy(img8.astype(np.float32) / 255.)[None, None].repeat(1, 3, 1, 1)
    return (x.to(DEV) - MEAN) / STD


def rho_G(net, img8, Q):
    x = to_tensor(img8).requires_grad_(True)
    out = net(x)
    cls = int(out.argmax(1))
    out[0, cls].backward()
    g = x.grad[0].sum(0).detach().cpu().numpy()
    H, W = g.shape
    nb = H // 8
    blocks = g.reshape(nb, 8, W // 8, 8).transpose(0, 2, 1, 3)
    G = np.abs(dct(dct(blocks, axis=2, norm="ortho"), axis=3, norm="ortho") * Q)
    return G.transpose(0, 2, 1, 3).reshape(H, W)


def gate(net, df, root, qfs=(50, 75, 90), n_img=50, n_coef=400):
    import tempfile
    rng = np.random.default_rng(SEED)
    test = df[df.split == "test"]
    sub = test.sample(min(n_img, len(test)), random_state=SEED)
    res, pooled = {}, {"D": [], "R": [], "G": []}
    for qf in qfs:
        acc = {"D": [], "R": [], "G": []}
        for _, row in sub.iterrows():
            jpg = f"{root}/jpeg_qf{qf}/{row.slice_id}.jpg"
            if not os.path.exists(jpg):
                continue
            j = jpegio.read(jpg)
            C = np.array(j.coef_arrays[0], dtype=np.int64, copy=True)
            Q = j.quant_tables[0].astype(float)
            del j
            gc.collect()
            base = os.path.getsize(jpg)
            G = rho_G(net, cv2.imread(jpg, cv2.IMREAD_GRAYSCALE), Q)
            nz = np.argwhere(C != 0)
            if len(nz) < 50:
                continue
            sel = nz[rng.choice(len(nz), min(n_coef, len(nz)), replace=False)]
            for r, c in sel:
                jj = jpegio.read(jpg)
                jj.coef_arrays[0][r, c] += 1
                t = tempfile.mktemp(suffix=".jpg")
                jpegio.write(jj, t)
                acc["R"].append(os.path.getsize(t) - base)
                os.remove(t)
                acc["D"].append(float(Q[r % 8, c % 8]) ** 2)
                acc["G"].append(float(G[r, c]))
        for k in acc:
            acc[k] = np.asarray(acc[k])
            pooled[k].append(acc[k])
        n = len(acc["D"])
        res[f"n_{qf}"] = n
        for a, b in (("D", "R"), ("D", "G"), ("R", "G")):
            res[f"{a}{b}_{qf}"] = float(spearmanr(acc[a], acc[b])[0]) if n > 10 else float("nan")
        if n:
            res[f"rate_{qf}"] = (float(acc["R"].min()), float(acc["R"].max()),
                                 float(acc["R"].mean()), float((acc["R"] == 0).mean()))
        print(f"  QF {qf}: n={n}  rho(D,G)={res[f'DG_{qf}']:+.3f}", flush=True)
    P = {k: np.concatenate(v) for k, v in pooled.items()}
    for a, b in (("D", "R"), ("D", "G"), ("R", "G")):
        res[f"{a}{b}_all"] = float(spearmanr(P[a], P[b])[0])
    res["n_all"] = len(P["D"])
    return res


def verdict(rho):
    a = abs(rho)
    if a < 0.35:
        return "PROCEED — strong disagreement, a bolder title is justified"
    if a <= 0.80:
        return "PROCEED — gap confirmed as planned"
    return "STOP — distortion cost already captures diagnostic cost; no gap"
