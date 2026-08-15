Local-alpha residual measurement neural operator.

Dataset:
- self_consistent_cbs/sparse_64_local_alpha_train100_test20
- train base samples: 100
- test base samples: 20
- alphas: 0.0, 0.25, 0.5, 0.75, 1.0
- train files: 500
- test files: 100

Mean baseline:
- dobs_rrmse = 0.1671006
- dobs_mae = 0.0125502

Residual operator:
- dobs_rrmse = 0.1387741
- dobs_mae = 0.0104050
- residual_rrmse = 0.83051
- relative improvement over mean = 16.95%

Conclusion:
The local-alpha residual operator is slightly worse in forward RRMSE than the previous GT-only residual operator, but it covers the correction path from InversionNet condition to GT. The next step is to test whether its gradient gives better neural correction or hybrid proposal.
