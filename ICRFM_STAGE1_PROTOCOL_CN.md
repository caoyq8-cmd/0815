# 生成式阶段 G1：IC-RFM 冻结实验协议

## 1. 方法定位

IC-RFM（Inversion-Conditioned Residual Flow Matching）在 256×256 干净声速图空间中学习条件速度场

\[
v_\theta(x_t,c,t),
\]

其中 \(c\) 是固定 epoch-67 InversionNet 重建。训练起点为

\[
z=c+\sigma\epsilon,
\]

路径与监督速度分别为

\[
x_t=(1-t)z+t x^*,\qquad v^*=x^*-z.
\]

推理的论文主配置为从 \(z=c\) 出发的确定性 4-step Heun ODE。随机起点只用于不确定性消融，不用于主表。

这是一种“条件中心、干净空间、少步”的生成式细化器，不宣称复现 Diff-ANO 的 EDM→Consistency Model→ControlNet 全部训练过程。

## 2. 数据纪律

| 子集 | 用途 | 是否选择模型 |
|---|---|---|
| 897 train condition cache | IC-RFM训练 | 否 |
| old test1–20 | checkpoint与early stopping | 是 |
| old test21–50 | 冻结后的DEV30报告 | 否 |
| old test51–100 | 暂不作为独立最终留出集 | 否 |

模型训练和图像评价只读取 `condition_norm/target_norm/condition_speed/target_speed`。G1不读取任何观测数据，因此不会混用旧 `dobs` 与当前 self-consistent CBS forward。

## 3. 冻结主配置

- speed range: 1400–1605 m/s
- base channels: 32
- continuous time embedding: 128
- optimizer: AdamW, lr=2e-4, weight decay=1e-4
- epochs: at most 100; early stopping patience=20
- EMA: 0.999
- condition-centered start noise: sigma_max=0.15 (normalized)
- exact zero-noise start probability: 0.25
- loss: velocity MSE + 0.10 endpoint L1 + 0.05 endpoint gradient L1
- primary inference: deterministic 4-step Heun
- checkpoint guard: validation SSIM may not fall more than 0.001 below InversionNet

## 4. 通过门槛

G1进入物理耦合阶段至少需要在DEV30上满足：

1. mean image MSE lower than InversionNet;
2. mean MAE no worse than InversionNet;
3. mean SSIM degradation not larger than 0.001;
4. paired improvements must report bootstrap 95% CI and Wilcoxon tests;
5. NFE=1/2/4/8 ablation shows the result is not an accidental single-step setting。

若未通过，不在DEV30上反复搜索参数；只允许回到 test1–20 检查训练失败、欠拟合或数值问题。

## 5. 运行顺序

将两个脚本放在当前 `USCT_download` 根目录：

```bash
chmod +x run_icrfm_stage1.sh
bash run_icrfm_stage1.sh smoke
bash run_icrfm_stage1.sh train
bash run_icrfm_stage1.sh dev30
bash run_icrfm_stage1.sh nfe
```

如果训练中断：

```bash
bash run_icrfm_stage1.sh resume
```

先检查 `smoke` 的模型、缓存shape和显存接口；其checkpoint不得进入论文结果。

## 6. 后续 G2（暂不提前调参）

若G1通过，将4-step Flow轨迹的 \(t=0,0.25,0.5,0.75,1\) 状态插值到480×480，并使用与Local-AA完全一致的500 kHz、sparse64、CBS-80 true-forward loss选出物理最优候选。G2保持CBS adjoint=0，并与Original、Local-AA、E3.5、Full-CBS比较图像质量、物理误差、forward调用数和运行时间。
