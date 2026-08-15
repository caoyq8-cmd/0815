Hybrid neural proposal + CBS validation on test_1.

Setting:
- residual measurement neural operator
- step_size_mps = 0.5
- prior_tether = 0.0
- step_factors = 1.0 0.5 0.25 0.1 0.05 0.02
- acceptance criterion = true CBS forward measurement loss

Initial:
- true_cbs_abs_loss = 0.00684651
- image_mse_480_up = 47.24794

Final:
- true_cbs_abs_loss = 0.00681087
- image_mse_480_up = 47.17947

Observation:
- Neural loss increases, so neural loss is not a reliable acceptance criterion.
- CBS validation prevents degradation.
- Neural proposal provides a weak but valid improvement direction.
- Improvement is much smaller than full CBS adjoint correction.
EOFcat > ./final_milestones/hybrid_neural_proposal_cbs_validated_test1_no_tether_best/README_result.txt << 'EOF'
Hybrid neural proposal + CBS validation on test_1.

Setting:
- residual measurement neural operator
- step_size_mps = 0.5
- prior_tether = 0.0
- step_factors = 1.0 0.5 0.25 0.1 0.05 0.02
- acceptance criterion = true CBS forward measurement loss

Initial:
- true_cbs_abs_loss = 0.00684651
- image_mse_480_up = 47.24794

Final:
- true_cbs_abs_loss = 0.00681087
- image_mse_480_up = 47.17947

Observation:
- Neural loss increases, so neural loss is not a reliable acceptance criterion.
- CBS validation prevents degradation.
- Neural proposal provides a weak but valid improvement direction.
- Improvement is much smaller than full CBS adjoint correction.
