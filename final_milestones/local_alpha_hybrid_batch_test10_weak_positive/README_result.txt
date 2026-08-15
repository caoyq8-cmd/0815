Local-alpha residual neural operator + CBS validation hybrid correction.

Setting:
- local-alpha residual measurement operator
- train base samples: 100
- test base samples: 20
- alphas: 0.0, 0.25, 0.5, 0.75, 1.0
- hybrid test samples: 10
- step_size_mps = 0.5
- prior_tether = 0.0
- step_factors = 1.0 0.5 0.25 0.1 0.05 0.02
- acceptance criterion = true CBS forward measurement loss

Aggregate:
- image_mse_240: 104.2069 -> 104.1326
- image_mse_480: 106.2498 -> 106.1813
- true_cbs_abs_loss: 0.00841688 -> 0.00840472

Mean improvement:
- mse240 reduced by 0.0947%
- mse480 reduced by 0.0835%
- dobs loss reduced by 0.1393%

Improved samples:
- mse240 improved: 9/10
- mse480 improved: 9/10
- dobs loss improved: 10/10

Conclusion:
Local-alpha training makes hybrid correction more image-stable, but the improvement is still very small compared with full CBS adjoint correction.
