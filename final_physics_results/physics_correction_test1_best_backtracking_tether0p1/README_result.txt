Best CBS physics correction result on self-consistent sparse-64 test_1.

Initial InversionNet condition:
- image_mse_256 = 53.7091
- image_mse_480 = 48.0212
- dobs_rel_l1 = 0.1522
- dobs_rel_l2 = 0.0944
- dobs_abs_loss = 0.0068737

After 8 backtracking CBS adjoint correction steps with prior_tether=0.1:
- image_mse_256 = 35.2701
- image_mse_480 = 30.5475
- dobs_rel_l1 = 0.0664
- dobs_rel_l2 = 0.0376
- dobs_abs_loss = 0.0029984

Relative improvement:
- image_mse_256 reduced by about 34.3%
- image_mse_480 reduced by about 36.4%
- dobs_abs_loss reduced by about 56.4%

Conclusion:
Self-consistent CBS adjoint correction improves both measurement consistency and reconstruction accuracy. This supports the Diff-ANO physical correction route and motivates replacing slow CBS adjoint computation with a neural operator surrogate.