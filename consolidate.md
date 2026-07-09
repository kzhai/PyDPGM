# DPGM consolidation note

This branch intentionally keeps the consolidation limited to **`dpgm.py`**.

## Scope kept
- `dpgm.py` is a standalone script that contains the DPGM API (`fit_dpgm`, `predict_dpgm`) and inlined model implementation required to run independently.

## Scope reverted
All other prior branch changes were reverted so that non-`dpgm.py` files remain aligned with the repository baseline.

## External runtime dependencies
- `numpy`
- `scipy`
