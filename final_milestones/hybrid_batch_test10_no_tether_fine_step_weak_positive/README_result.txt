Hybrid neural proposal + CBS validation on test10.

Setting:
- residual measurement neural operator
- step_size_mps = 0.5
- prior_tether = 0.0
- step_factors = 1.0 0.5 0.25 0.1 0.05 0.02
- num_iters = 6
- acceptance criterion = true CBS forward measurement loss

Aggregate:
- image_mse_240: 104.2069 -> 104.1681
- image_mse_480: 106.2498 -> 106.2321
- true_cbs_abs_loss: 0.00841688 -> 0.00836594

Mean improvement:
- mse240 reduced by 0.0669%
- mse480 reduced by 0.0313%
- dobs loss reduced by 0.6257%

Improved samples:
- mse240 improved: 7/10
- mse480 improved: 7/10
- dobs loss improved: 9/10

Conclusion:
Hybrid neural proposal + CBS validation is weakly positive but much weaker than full CBS adjoint correction.
The current residual neural operator provides weak proposal directions, but its gradient is not strong enough to replace the true adjoint gradient.
