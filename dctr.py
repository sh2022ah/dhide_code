"""
DCTR: low-complexity features for steganalysis of JPEG images.
Holub and Fridrich, IEEE TIFS 10(2), 2015.

The decompressed image is convolved with the 64 DCT basis patterns of an
8 by 8 block. Each residual is quantised, truncated, and histogrammed
separately for each of the 64 phases of the block grid. Symmetry merging
reduces the 64 phases to 25 classes, giving 64 x 25 x 5 = 8000 features.
"""
import numpy as np
from scipy.signal import fftconvolve


def dct_bases():
    """The 64 separable 8 by 8 DCT patterns, unit-norm."""
    k = np.arange(8)
    c = np.where(k == 0, np.sqrt(1 / 8), np.sqrt(2 / 8))
    B = np.array([[c[u] * np.cos((2 * x + 1) * u * np.pi / 16) for x in range(8)]
                  for u in range(8)])
    return np.array([np.outer(B[u], B[v]) for u in range(8) for v in range(8)])


BASES = dct_bases()
QUANT, TRUNC = 4.0, 4          # q and T from the paper


def _phase_hist(res, q=QUANT, T=TRUNC):
    """Quantise, truncate, and histogram each of the 64 grid phases.

    Phases are merged into 25 symmetry classes as in the original: the pair
    (a, b) and (b, a) are equivalent, as are a and 8 - a.
    """
    v = np.minimum(np.abs(np.round(res / q)).astype(np.int16), T)
    H, W = v.shape
    nb_r, nb_c = H // 8, W // 8
    v = v[:nb_r * 8, :nb_c * 8].reshape(nb_r, 8, nb_c, 8).transpose(1, 3, 0, 2)

    # phase -> symmetry class: fold each axis independently, a ~ 8 - a,
    # which maps 0..7 onto 0..4 and gives the 25 classes of the original
    cls, seen = np.zeros((8, 8), dtype=int), {}
    n = 0
    for a in range(8):
        for b in range(8):
            key = (min(a, 8 - a) if a else 0, min(b, 8 - b) if b else 0)
            if key not in seen:
                seen[key] = n; n += 1
            cls[a, b] = seen[key]
    n_cls = n

    out = np.zeros((n_cls, T + 1), dtype=np.float64)
    cnt = np.zeros(n_cls, dtype=np.float64)
    for a in range(8):
        for b in range(8):
            g = v[a, b].ravel()
            out[cls[a, b]] += np.bincount(g, minlength=T + 1)
            cnt[cls[a, b]] += g.size
    return (out / np.maximum(cnt, 1)[:, None]).ravel(), n_cls


def dctr(img8):
    """8000-dimensional DCTR feature vector for a decompressed grayscale image."""
    x = img8.astype(np.float64)
    feats = []
    for k in range(64):
        res = fftconvolve(x, BASES[k], mode="valid")
        h, _ = _phase_hist(res)
        feats.append(h)
    return np.concatenate(feats)


def dctr_dim():
    _, n_cls = _phase_hist(np.zeros((64, 64)))
    return 64 * n_cls * (TRUNC + 1)
