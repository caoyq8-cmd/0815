Unconditional DDPM speed-map prior on OpenBreastUS processed oldstyle dataset.

Dataset:
- train: 897 samples
- test: 100 samples
- training target: target_256
- image size: 256 x 256
- normalization range: [1400, 1605] mapped to [-1, 1]

Model:
- Simple DDPM U-Net
- base_ch = 48
- time_dim = 192
- timesteps = 1000
- epochs = 200
- batch_size = 8
- optimizer = AdamW
- EMA decay = 0.999
- DDIM sampling steps = 100 for final sampling

Final training:
- epoch 200 train_loss = 0.007367
- epoch 200 val_loss = 0.008353

Best checkpoint sampling:
- sample speed min = 1400.0
- sample speed max = 1605.0
- sample speed mean = 1498.8511
- sample speed std = 35.2270

Observation:
- The model has learned breast ROI boundaries and heterogeneous internal structures.
- Some samples remain noisy or over-saturated.
- The prior is usable for the next conditional refinement stage, but still weaker than a full EDM/CM prior.