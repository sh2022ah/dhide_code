"""
J-UNIWARD: the JPEG-domain form of the universal wavelet relative distortion.

Holub, Fridrich and Denemark, EURASIP Journal on Information Security 2014.

Re-implemented from the published description. No reference implementation was
available for bit-exact comparison, so three sanity checks are provided at the
bottom of this file and their results are reported alongside any use of the
function.

Definition. Let W be the undecimated wavelet decomposition of the decompressed
cover with a first-level Daubechies-8 filter bank, giving three directional
residuals. Changing one quantised coefficient by one perturbs every wavelet
coefficient whose support overlaps the corresponding 8x8 block. The cost is the
sum over all three residuals and all positions of the relative change,

    rho = sum_k sum_u  |dW_k(u)| / (sigma + |W_k(u)|),

with the stabilising constant sigma = 2^-6 as in the original.

Computing this exactly for every coefficient would require one wavelet
decomposition per coefficient. The published algorithm exploits the fact that
changing coefficient (b, f) by one adds Q_f times the f-th DCT basis pattern to
block b in the pixel domain, so the wavelet perturbation is the fixed filter
response to that pattern, scaled by Q_f, and shifted to the position of block b.
The 64 responses are therefore precomputed once per quantisation table.
"""
import numpy as np
import pywt
from scipy.signal import fftconvolve
from scipy.fftpack import idct

SIGMA = 2.0 ** -6
WAVELET = "db8"


def dct_basis_patterns():
    """The 64 spatial patterns produced by a unit change in each DCT mode."""
    P = np.zeros((8, 8, 8, 8))
    for u in range(8):
        for v in range(8):
            C = np.zeros((8, 8))
            C[u, v] = 1.0
            P[u, v] = idct(idct(C, axis=0, norm="ortho"), axis=1, norm="ortho")
    return P.reshape(64, 8, 8)


BASIS = dct_basis_patterns()


def _filters():
    """First-level undecimated db8 filter bank: LH, HL and HH."""
    w = pywt.Wavelet(WAVELET)
    lo = np.array(w.dec_lo)[::-1]
    hi = np.array(w.dec_hi)[::-1]
    return [np.outer(lo, hi), np.outer(hi, lo), np.outer(hi, hi)]


FILTERS = _filters()


def _residuals(img):
    """The three directional wavelet residuals of a decompressed image."""
    return [fftconvolve(img.astype(np.float64), F, mode="same") for F in FILTERS]


def _perturbation_kernels(Q):
    """For each of the 64 modes, the wavelet response to a unit change.

    Returns a list of three arrays of shape (64, kh, kw): the response of each
    directional filter to Q_f times the f-th basis pattern.
    """
    out = []
    for F in FILTERS:
        stack = np.empty((64,) + tuple(np.array(F.shape) + 7))
        for f in range(64):
            q = Q.reshape(-1)[f]
            stack[f] = fftconvolve(BASIS[f] * q, F, mode="full")
        out.append(stack)
    return out


def juniward_cost(C, Q, img):
    """Cost map of shape C.shape, one value per quantised coefficient.

    C   quantised coefficients, as returned by jpegio
    Q   8x8 quantisation table
    img decompressed cover, uint8, same dimensions as C
    """
    C = np.asarray(C)
    H, W = C.shape
    nbr, nbc = H // 8, W // 8
    R = _residuals(img)
    KER = _perturbation_kernels(Q.astype(np.float64))

    # denominator sigma + |W| once per residual
    DEN = [SIGMA + np.abs(r) for r in R]

    rho = np.zeros((nbr, nbc, 64))
    for d in range(3):
        k = KER[d]
        kh, kw = k.shape[1], k.shape[2]
        pad_r = (kh - 8) // 2
        pad_c = (kw - 8) // 2
        inv = 1.0 / DEN[d]
        # a change in block (br, bc) touches rows br*8-pad_r .. br*8+8+pad_r
        Ipad = np.pad(inv, ((pad_r, pad_r + 8), (pad_c, pad_c + 8)), mode="edge")
        for br in range(nbr):
            r0 = br * 8
            for bc in range(nbc):
                c0 = bc * 8
                win = Ipad[r0:r0 + kh, c0:c0 + kw]
                rho[br, bc] += np.abs(k).reshape(64, -1) @ win.reshape(-1)

    out = rho.reshape(nbr, nbc, 8, 8).transpose(0, 2, 1, 3).reshape(H, W)
    return out


# ------------------------------------------------------------------ checks
def sanity_checks(C, Q, img, uerd_map=None, verbose=True):
    """Three properties any correct implementation must satisfy."""
    from scipy.stats import spearmanr
    rho = juniward_cost(C, Q, img)
    res = {}

    # 1. cost must be finite, positive, and vary across the image
    res["finite"] = bool(np.isfinite(rho).all())
    res["positive"] = bool((rho > 0).all())
    res["dynamic_range"] = float(np.log10(rho.max() / max(rho.min(), 1e-30)))

    # 2. smooth regions must cost more than textured ones
    lap = np.abs(fftconvolve(img.astype(np.float64),
                             np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
                             mode="same"))
    nb = img.shape[0] // 8
    tex = lap.reshape(nb, 8, nb, 8).mean(axis=(1, 3))
    cost = rho.reshape(nb, 8, nb, 8).mean(axis=(1, 3))
    res["rho_cost_texture"] = float(spearmanr(cost.ravel(), tex.ravel())[0])

    # 3. moderate agreement with UERD: related but not identical
    if uerd_map is not None:
        nz = np.argwhere(np.asarray(C) != 0)
        idx = nz[np.random.default_rng(0).choice(len(nz), min(4000, len(nz)),
                                                 replace=False)]
        res["rho_vs_uerd"] = float(spearmanr(rho[idx[:, 0], idx[:, 1]],
                                             uerd_map[idx[:, 0], idx[:, 1]])[0])
    if verbose:
        print("  finite:", res["finite"], "| all positive:", res["positive"])
        print(f"  dynamic range: {res['dynamic_range']:.1f} decades")
        print(f"  cost against local texture: rho = {res['rho_cost_texture']:+.3f}"
              "   (expected clearly negative)")
        if uerd_map is not None:
            print(f"  cost against UERD:          rho = {res['rho_vs_uerd']:+.3f}"
                  "   (expected 0.3 to 0.8)")
    return rho, res
