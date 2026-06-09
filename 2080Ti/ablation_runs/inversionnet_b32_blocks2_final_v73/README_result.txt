Direct Inversion / InversionNet baseline on OpenBreastUS processed oldstyle dataset.

Dataset:
- train: 897 samples
- test: 100 samples
- input: input_2ch, shape = 2 x 256 x 256
- target: target_256, shape = 256 x 256
- input_2ch[0] = real(dobs_complex)
- input_2ch[1] = imag(dobs_complex)

Model:
- InversionNet
- base_ch = 32
- bottleneck_blocks = 2
- loss = L1 + 0.2 MSE + 0.1 gradient loss
- optimizer = AdamW
- early stopping = on

Best checkpoint:
- epoch = 67

Evaluation on full test set:
- num_test = 100
- MSE = 170.7008 ± 111.4457
- MAE = 3.9173 ± 1.7152
- RMSE = 12.3909 ± 4.1433
- PSNR = 24.6656 ± 3.0231 dB
- SSIM = 0.8664 ± 0.0388

Best case:
- sample_index = 72
- PSNR = 31.7270
- SSIM = 0.9253

Worst case:
- sample_index = 36
- PSNR = 18.3555
- SSIM = 0.7453