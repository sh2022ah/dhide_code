"""embed_core: STC, DCT-domain embedding, and the three cost maps."""
import os, gc, tempfile
import numpy as np, cv2, jpegio
from scipy.fftpack import dct

INF = 1e18


# ------------------------------------------------------------ STC
def _submatrix(h, w, seed=42):
    rng = np.random.default_rng(seed)
    H = rng.integers(0, 2, size=(h, w), dtype=np.uint8)
    H[0, :] = 1; H[-1, :] = 1
    return H


def _cols(H):
    h, w = H.shape
    return np.array([int("".join(str(b) for b in H[::-1, j]), 2) for j in range(w)],
                    dtype=np.int64)


def stc_embed(x, rho, msg, h=7, seed=42):
    x = np.asarray(x, np.uint8); rho = np.asarray(rho, np.float64)
    msg = np.asarray(msg, np.uint8)
    n, m = len(x), len(msg)
    if n % m: raise ValueError("n must be divisible by m")
    w = n // m
    if w < 2: raise ValueError("payload too large")
    cols = _cols(_submatrix(h, w, seed)); ns = 1 << h
    wght = np.full(ns, INF); wght[0] = 0.0
    path = np.zeros((n, ns), np.uint8); idx = np.arange(ns)
    for i in range(m):
        for j in range(w):
            t = i * w + j; c = cols[j]
            c0 = wght + (x[t] == 1) * rho[t]
            c1 = wght[idx ^ c] + (x[t] == 0) * rho[t]
            take = c1 < c0; path[t] = take.astype(np.uint8)
            wght = np.where(take, c1, c0)
        keep = wght[msg[i]::2]
        wght = np.concatenate([keep, np.full(ns - len(keep), INF)])
    st = 0; y = np.zeros(n, np.uint8)
    for i in range(m - 1, -1, -1):
        st = 2 * st + int(msg[i])
        for j in range(w - 1, -1, -1):
            t = i * w + j; b = path[t, st]; y[t] = b
            if b: st ^= int(cols[j])
    return y


def stc_extract(y, m, h=7, seed=42):
    y = np.asarray(y, np.uint8); n = len(y)
    w = n // m; cols = _cols(_submatrix(h, w, seed))
    msg = np.zeros(m, np.uint8); st = 0
    for i in range(m):
        for j in range(w):
            if y[i * w + j]: st ^= int(cols[j])
        msg[i] = st & 1; st >>= 1
    return msg


# ------------------------------------------------------------ carrier
def carrier(C):
    C = np.array(C, dtype=np.int64, copy=True)
    r, c = np.nonzero(C)
    keep = ~((r % 8 == 0) & (c % 8 == 0))
    r, c = r[keep], c[keep]
    o = np.lexsort((c, r))
    return r[o], c[o]


def _dirn(v):
    if v == 1: return 1
    if v == -1: return -1
    return -1 if v > 0 else 1


def embed(jpg_in, jpg_out, bits, rho, h=7, seed=42):
    j = jpegio.read(jpg_in)
    C = j.coef_arrays[0]
    Cc = np.array(C, dtype=np.int64, copy=True)
    rows, cols = carrier(Cc)
    n, m = len(rows), len(bits)
    if m == 0 or n < 2 * m: raise ValueError("cover too small")
    nu = (n // m) * m
    rows, cols = rows[:nu], cols[:nu]
    x = (np.abs(Cc[rows, cols]) % 2).astype(np.uint8)
    cost = np.maximum(rho[rows, cols].astype(np.float64), 1e-9)
    y = stc_embed(x, cost, bits, h, seed)
    flip = np.nonzero(x != y)[0]
    for i in flip:
        r, c = int(rows[i]), int(cols[i])
        C[r, c] = int(Cc[r, c]) + _dirn(int(Cc[r, c]))
    jpegio.write(j, jpg_out)
    s0, s1 = os.path.getsize(jpg_in), os.path.getsize(jpg_out)
    return dict(n=nu, m=m, bpnzAC=m / nu, changes=len(flip),
                size_pct=100 * (s1 - s0) / s0)


def extract(jpg, m, h=7, seed=42):
    j = jpegio.read(jpg)
    C = np.array(j.coef_arrays[0], dtype=np.int64, copy=True)
    del j; gc.collect()
    rows, cols = carrier(C)
    nu = (len(rows) // m) * m
    y = (np.abs(C[rows[:nu], cols[:nu]]) % 2).astype(np.uint8)
    return stc_extract(y, m, h, seed)


# ------------------------------------------------------------ costs
def rho_D(Q, shape):
    return np.tile(Q.astype(np.float64) ** 2, (shape[0] // 8, shape[1] // 8))


def rate_table(jpgs, per_img=150, seed=42):
    """One rate table averaged over a few images. The cost is dominated by
    the frequency index, not by the block, so a global table is adequate and
    saves one re-encode per coefficient per image."""
    rng = np.random.default_rng(seed)
    tot = np.zeros(64); cnt = np.zeros(64)
    for p in jpgs:
        j = jpegio.read(p)
        C = np.array(j.coef_arrays[0], dtype=np.int64, copy=True)
        del j; gc.collect()
        base = os.path.getsize(p)
        nz = np.argwhere(C != 0)
        if len(nz) < 20: continue
        sel = nz[rng.choice(len(nz), min(per_img, len(nz)), replace=False)]
        for r, c in sel:
            jj = jpegio.read(p); jj.coef_arrays[0][r, c] += 1
            t = tempfile.mktemp(suffix=".jpg"); jpegio.write(jj, t)
            f = (r % 8) * 8 + (c % 8)
            tot[f] += os.path.getsize(t) - base; cnt[f] += 1
            os.remove(t)
    return (tot / np.maximum(cnt, 1)).reshape(8, 8)


def rho_R(table, shape):
    return np.tile(table, (shape[0] // 8, shape[1] // 8))


def ranknorm(a):
    f = a.ravel(); o = f.argsort()
    r = np.empty_like(o, dtype=np.float64); r[o] = np.arange(len(f))
    return (r / max(len(f) - 1, 1)).reshape(a.shape)
