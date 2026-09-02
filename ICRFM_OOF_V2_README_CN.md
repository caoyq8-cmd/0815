# IC-RFM G1-v2：5-fold OOF条件实验

## 为什么必须重建训练condition

现有epoch-67 InversionNet在897个训练样本上的condition MSE为98.03，而在GEN-VAL20与DEV30上分别为149.92和194.90。用训练内预测训练第二阶段Flow，会导致明显的stacking leakage与残差分布偏移。

G1-v2采用5-fold cross-fitting：每个fold模型只使用另外80%的样本，固定训练67 epoch，并只预测从未见过的holdout fold。合并五折后，每个训练condition均为样本外预测。

## 固定设置

- folds: 5，seed=20260902
- OOF InversionNet: base_ch=32, bottleneck_blocks=2
- fixed epochs per fold: 67
- loss: L1 + 0.2 MSE + 0.1 gradient
- target normalization for InversionNet: 1400–1600 m/s
- Flow/cache normalization: 1400–1605 m/s
- Flow start: exactly the condition, no start noise
- Flow endpoint loss: 1.0 L1 + 0.20 gradient
- primary inference: deterministic 4-step Heun

## 执行顺序

```bash
bash run_oof5_icrfm_v2.sh fold 0
bash run_oof5_icrfm_v2.sh fold 1
bash run_oof5_icrfm_v2.sh fold 2
bash run_oof5_icrfm_v2.sh fold 3
bash run_oof5_icrfm_v2.sh fold 4
bash run_oof5_icrfm_v2.sh finalize
```

也可顺序执行五折：

```bash
bash run_oof5_icrfm_v2.sh folds_all
```

fold意外中断时：

```bash
bash run_oof5_icrfm_v2.sh resume_fold 2
```

`finalize`必须显示PASS、897个OOF train conditions和100个test conditions，才允许继续：

```bash
bash run_oof5_icrfm_v2.sh flow_smoke
bash run_oof5_icrfm_v2.sh flow_train
```

只有VAL20通过MSE、MAE、SSIM门槛后，才运行DEV30和NFE消融。
