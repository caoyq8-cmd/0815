Hybrid neural proposal + CBS validation on test_1.

Initial:
- true_cbs_abs_loss = 0.00684651
- image_mse_480_up = 47.24794

Final:
- true_cbs_abs_loss = 0.00681520
- image_mse_480_up = 47.18361

Observation:
- Neural proposal can produce physically valid candidates.
- CBS validation prevents degradation.
- Improvement is small and saturates quickly; iter=5 chooses zero.
- Neural loss increases, confirming that neural loss is not a reliable acceptance criterion.
