# 01_data_and_surrogate.py
# Stage A (download, windowing, JPEG at Qf 50/75/90, patient-disjoint split) and Stage B (ResNet-50 surrogate).
# Extracted from the Colab notebook used for the paper; table numbers in the
# comments below are those of earlier drafts, see README for the mapping.

# ============================================================
#  STAGE A — data.  Resumable: a completed stage is restored from Drive
#  in about three minutes instead of being recomputed in twenty.
# ============================================================
import os, sys, glob, json, time, zipfile, urllib.request, subprocess, warnings
warnings.filterwarnings("ignore")
from google.colab import drive
drive.mount("/content/drive", force_remount=False)

DRIVE = "/content/drive/MyDrive/dhide_ckpt"
os.makedirs(DRIVE, exist_ok=True)
print("checkpoint folder:", DRIVE)
print("contents:", sorted(os.listdir(DRIVE)) or "empty")

subprocess.run([sys.executable, "-m", "pip", "-q", "install", "jpegio", "h5py"], check=False)
import numpy as np, torch, cv2, jpegio, h5py
print("torch", torch.__version__, "| GPU:",
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE  <-- set T4")

TAR = f"{DRIVE}/data.tar.gz"
if os.path.exists(TAR):
    print("\nrestoring converted data from Drive ...", flush=True)
    t0 = time.time()
    subprocess.run(["tar", "xzf", TAR, "-C", "/content"], check=True)
    print(f"  restored {len(glob.glob('/content/data/jpeg_qf75/*.jpg'))} images "
          f"in {time.time()-t0:.0f}s")
else:
    print("\nno cached data; downloading from figshare ...", flush=True)
    os.makedirs("/content/zips", exist_ok=True); os.makedirs("/content/mats", exist_ok=True)
    files = json.loads(urllib.request.urlopen(
        "https://api.figshare.com/v2/articles/1512427/files").read())
    for f in [x for x in files if x["name"].lower().endswith(".zip")]:
        dst = f"/content/zips/{f['name']}"
        if os.path.exists(dst) and os.path.getsize(dst) == f["size"]:
            print("  cached", f["name"]); continue
        t0 = time.time(); urllib.request.urlretrieve(f["download_url"], dst)
        print(f"  {f['name']} {os.path.getsize(dst)/1e6:.0f} MB in {time.time()-t0:.0f}s", flush=True)
    for z in sorted(glob.glob("/content/zips/*.zip")):
        with zipfile.ZipFile(z) as zf: zf.extractall("/content/mats")
    n = len(glob.glob("/content/mats/**/*.mat", recursive=True))
    print(f"  unpacked {n} .mat files"); assert n > 3000

    exec(open("/content/gate_core.py").read())
    print("\nconverting ...", flush=True)
    rows = convert("/content/mats", "/content/data")
    json.dump(rows, open("/content/data/rows.json", "w"))
    print(f"  {len(rows)} slices, {len({r[1] for r in rows})} patients")

    print("\narchiving to Drive (one file, not 9000) ...", flush=True)
    t0 = time.time()
    subprocess.run(["tar", "czf", TAR, "-C", "/content", "data"], check=True)
    print(f"  {os.path.getsize(TAR)/1e6:.0f} MB in {time.time()-t0:.0f}s")

exec(open("/content/gate_core.py").read())
rows = [tuple(r) for r in json.load(open("/content/data/rows.json"))]
df, clean = split_by_patient(rows)
assert clean, "patient overlap"
print("\nSTAGE A complete")
print(df.groupby(["split", "label_name"]).size().unstack(fill_value=0))

# ============================================================
#  STAGE B — the frozen surrogate.  Restored from Drive if present.
# ============================================================
import os, torch, torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader

CKPT_D = f"{DRIVE}/resnet50.pt"
if os.path.exists(CKPT_D):
    print("restoring the trained model from Drive ...")
    net = models.resnet50(weights=None); net.fc = nn.Linear(net.fc.in_features, 3)
    net.load_state_dict(torch.load(CKPT_D, map_location=DEV))
    net = net.to(DEV).eval()
    for p in net.parameters(): p.requires_grad_(False)
else:
    print("training ResNet-50, about forty minutes ...", flush=True)
    net, best = train_cad(df, "/content/data", qf=75, epochs=15, bs=32,
                          ckpt="/content/resnet50.pt")
    torch.save(net.state_dict(), CKPT_D)
    print(f"saved to Drive, best validation {best:.4f}")

acc, auc = evaluate(net, DataLoader(DS(df[df.split == "test"], "/content/data", 75, False),
                                    batch_size=32))
print(f"\nSTAGE B complete — test accuracy {acc:.4f}, AUC {auc:.4f}")
print("previous run gave 0.9375 / 0.9847; a match confirms the split was reproduced")
