import ml_collections
import torch


def get_default_configs():
  config = ml_collections.ConfigDict()
  # training parameters
  config.training = training = ml_collections.ConfigDict() # 创建 train 为 ConfigDict(）来记录
  config.training.batch_size = 64 # 训练 batch 数目

  training.epochs = 3000 # 总训练 epoch 轮数
  training.snapshot_freq = 50 # 每 snapshot_freq 个 epoch 进行一次 unconditional sampling 并且 保存一下 checkpoints;
  training.snapshot_freq_for_preemption = 10 # 每 snapshot_freq_for_preemption 个 epoch 进行一次 实时checkpoint 存储, 以便之后可以恢复继续训练
  training.log_freq = 1 # 每 log_freq 个 epoch 记录一次 training state
  training.eval_freq = 5 # 每 eval_freq 个 epoch 记录一次 evaluation state
  
  ## produce samples at each snapshot.
  training.snapshot_sampling = True # 每 snapshot_freq 个 epoch 进行一次 unconditional sampling 并生成图片
  training.likelihood_weighting = False # 使用 SDE 来训练时，对 training loss 对每个样本都有 g2-weight （SMLD和DDPM没有）
  training.continuous = True # training 的 score 模型采用对 [0,1] 上连续的 SDE-time 进行数值采样, 还是 [0,1,2,...,999] 上离散 Markov-chain 采样 #这两种训练好的模型可以互相转化
  training.reduce_mean = False # training loss 进行平均还是仅仅求和
  
  # sampling
  config.sampling = sampling = ml_collections.ConfigDict() # 创建 sampling 为 ConfigDict(）来记录
  sampling.n_steps_each = 1 # Sampling 每次向前采样步数
  sampling.noise_removal = True # 在 Sampling 的最后一步多采样一步来去噪
  sampling.probability_flow = False #每一步是否采用 Reverse-ODE 的形式，但是可以 PC 或 ODE_Flow 的 Sampling Methods！ 
  sampling.snr = 0.16 # LangevinCorrector 中修正分布使用的 stepsize, 对应于 SDE_paper 中 Algorithm 5 的 r；

  # evaluation 
  # bpd 是指 "bits per dimension"，是一种衡量数据压缩效率的度量单位。在这里，bpd 用于评估模型生成的样本的似然性。
  config.eval = evaluate = ml_collections.ConfigDict() 
  evaluate.begin_ckpt = 9 # 从9号模型开始，
  evaluate.end_ckpt = 26 # 到26号模型结束，依次评估各自模型的表现；
  evaluate.batch_size = 16 # Evalution 函数下的batchsize
  evaluate.enable_sampling = False # 创建采样函数；
  evaluate.num_samples = 400 # ?
  evaluate.enable_loss = True # Create the one-step evaluation function 计算loss 的公式 when loss computation is enabled；
  evaluate.enable_bpd = False # 创建计算样本似然函数的公式，SDE_paper 中的 likelihood_computation (39) 式；
  evaluate.bpd_dataset = 'test' # 不是重新使用另一个 “test” 数据集，而是在 "test" 的过程中使用前面的 eval 数据集来计算 likehood; 

  # data
  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'AI4Scup2_simulated' # 数据集名称，在 datasets.py 文件中需要自定义 训练数据和验证数据 的路径
  data.image_size = 256 # 数据图像大小
  data.random_flip = False # 增广数据集是否使用 随机翻转
  data.centered = False # 决定数据是否从 [0,1] 变成 [-1, 1]
  data.uniform_dequantization = False # 增广数据集是否使用 均匀量化: 在 255 数值图片上加 [0,1] 的随机噪声再变回 [0,1] 数值图片
  data.num_channels = 1 # 数据图像通道数

  # model
  config.model = model = ml_collections.ConfigDict()
  model.sigma_min = 0.01
  model.sigma_max = 50
  model.num_scales = 1000
  model.beta_min = 0.1
  model.beta_max = 20.
  model.dropout = 0.1
  model.embedding_type = 'fourier'

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.

  config.seed = 23456
  config.device_ids = [2,3,4,5]
  config.device = torch.device('cuda:' + str(config.device_ids[0])) if torch.cuda.is_available() else torch.device('cpu')
  
  return config