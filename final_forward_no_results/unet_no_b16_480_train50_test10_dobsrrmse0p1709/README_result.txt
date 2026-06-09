Lightweight Neural Operator forward surrogate.

Dataset:
- self_consistent_cbs/sparse_64_wavefield_train50_test10
- train: 50 speed maps × 64 sources = 3200 training pairs
- test: 10 speed maps × 64 sources = 640 testing pairs
- input: speed_480 + source_map
- output: complex wavefield real/imag
- wave_scale = 0.1

Model:
- Lightweight U-Net
- base_ch = 16
- image_size = 480
- parameters = 2.31M
- epochs = 50

Evaluation:
- wave_rrmse = 0.6946
- wave_mae = 0.02227
- dobs_rrmse = 0.1709
- dobs_mae = 0.01351

Conclusion:
The lightweight U-Net surrogate captures receiver measurements reasonably well, although full wavefield prediction remains inaccurate. It is suitable as a first neural forward surrogate, but further receiver-focused training is needed before replacing CBS adjoint correction.