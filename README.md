# Diagnostic model-sensitivity-aware cost function for JPEG steganography in medical images

Code accompanying the paper *A Diagnostic Model-Sensitivity-Aware Cost Function for JPEG Steganography in Medical Images*.

## Contents
```
src/
  gate_core.py   data conversion (windowing, JPEG at Qf 50/75/90), patient-disjoint split, model training,
                 inference pipeline, axis-correlation gate
  embed_core.py  syndrome-trellis codes (h = 7), carrier (non-zero AC, parity of |C|), embedding/extraction,
                 rate table, UERD, DC-only carrier, rank normalization
  dctr.py        DCTR features (8000-D)
  juniward.py    J-UNIWARD cost (re-implemented) with three sanity checks
experiments/     one script per result; run 00 first in every new session
figures/         Figure 1 drawing script
```
The diagnostic cost of equation (7) is `rho_G` (defined in `00_restore_session.py` and `gate_core.py`):
`|Q ⊙ DCT8x8(∇x logit_y)|`, with the 512×512 input resized to 224×224 inside the graph so that the gradient reaches every pixel controlled by the coefficients. It accepts any differentiable classifier.

## Environment
Google Colab, T4 GPU. Data: the brain MRI collection of Cheng et al., Figshare 10.6084/m9.figshare.1512427 (downloaded by `01`). Intermediate files and checkpoints are cached in `/content/drive/MyDrive/dhide_ckpt`. Copy the four files of `src/` to `/content` (or to that Drive folder) before running the experiments.

## Results → scripts (table numbers of the revised manuscript)
| Result | Script |
|---|---|
| Table 2 (dataset), split | `01_data_and_surrogate.py` |
| Table 4 (baseline models) | `01`, `02_heldout_models_and_reference.py` |
| Table 5 (axis independence), Figure 3 | `03_axes_and_map_validation.py`, `12_figures.py` |
| Table 6 (map validation), Figure 4 | `03_axes_and_map_validation.py`, `04_radial_null.py`, `12_figures.py` |
| Table 7 (baselines), Figure 5a | `05_baselines.py` |
| Table 8 (two perceptual bases), Figure 5b | `06_unified_weights_and_transfer.py`, `07_juniward.py`; statistics `11_table8_statistics.py`; P(E) `09` |
| Table 9 (cross-model transfer), Figure 6 | `06_unified_weights_and_transfer.py` |
| Tables 10, 11 (detectability, weight sweep), Figure 7 | `09_detectability_unified.py` |
| Tables 12, 13 (ablation) | `08_ablation_five_seeds.py`, `10_ablation_factorial_tests.py` |
| Figure 2 | `13_figure2_final.py` |

Comments inside the extracted notebook cells use the table numbers of earlier drafts; the mapping above is authoritative.

## Seeds and protocol
Evaluation set: 120 slices sampled from the 608-slice test partition with `random_state=42`; the first 80 carry the detectability measurements. Payload seeds: 42, 123, 456, 789, 1011 (identical payload bits across compared arms). Payloads in the experiments are pseudo-random bit strings; in deployment the payload is AES-GCM output (nonce ‖ ciphertext ‖ tag). STC parity-check matrix seed and carrier order are fixed public constants in the experiments; in deployment both are derived from the key.
