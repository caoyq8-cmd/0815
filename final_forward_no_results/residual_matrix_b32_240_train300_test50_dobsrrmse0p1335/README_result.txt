Residual measurement neural operator.

Dataset:
- self_consistent_cbs/sparse_64_dobs_train300_test50
- train: 300 speed maps
- test: 50 speed maps
- input: speed map + coordinate maps
- output: residual dobs matrix relative to train mean dobs
- measurement shape: 64 x 64 complex
- image_size = 240
- residual_scale = 0.02

Model:
- DobsMatrixCNN
- base_ch = 32
- spatial_pool = 8
- head_dim = 1024

Evaluation on 50 test samples:
- mean baseline dobs_rrmse = 0.166824
- residual operator dobs_rrmse = 0.133519
- relative improvement over mean baseline = 19.96%
- dobs_mae = 0.01010
- residual_rrmse = 0.80235

Conclusion:
Residual measurement learning successfully improves over the mean observation baseline. This indicates that the model captures nontrivial speed-dependent measurement perturbations and can be used as a neural measurement surrogate for fast physics correction.