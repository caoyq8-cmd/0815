CBS adjoint physics correction on self-consistent sparse-64 test50.

Dataset:
- self_consistent_cbs/sparse_64_dobs_train300_test50
- test samples: 50
- measurement: sparse-64, 64 sources and 64 receivers
- frequency: 500 kHz
- CBS iterations: 80
- boundary: PML3, width=300, strength=225

Initial reconstruction:
- InversionNet direct inversion condition
- Coordinate rule: x_phys = x_img.T

Correction:
- num_iters = 8
- step_size_mps = 1.0
- prior_tether = 0.1
- smooth_kernel = 9
- backtracking enabled

Aggregate:
- image_mse_256: 176.9088 -> 125.4597
- image_mse_480: 165.6735 -> 116.9615
- dobs_abs_loss: 0.0096905 -> 0.0052362

Mean per-sample improvement:
- mse256 reduced by 32.36%
- mse480 reduced by 33.34%
- dobs_abs_loss reduced by 47.21%

Improved samples:
- mse256 improved: 50/50
- mse480 improved: 50/50
- dobs loss improved: 50/50

Conclusion:
CBS adjoint correction is stable and consistently improves both image-domain reconstruction and physical measurement consistency.
