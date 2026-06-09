import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import glob
import sys 
import numpy as np
from PIL import Image
from skimage.transform import resize

def get_data_scaler(config):
  """Data normalizer. Assume data are always in [0, 1]."""
  if config.data.centered:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x


def get_data_inverse_scaler(config):
  """Inverse data normalizer."""
  if config.data.centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x


def get_dataset_pytorch(config):
  """Create data loaders for training and evaluation. But using the torch.Dataset
  
  """
  # Compute batch size for this worker.
  if config.data.dataset == 'Marmousi':
    data_dir = "/home/caoxiang/Desktop/diffusion_model/score_sde_pytorch/datasets/Marmousi/samples_fig"
    train_split_name, eval_split_name = 'train', 'eval'

    class Create_Custom_Dataset(data.Dataset):
      def __init__(self, data_dir, split):
        super(Create_Custom_Dataset, self).__init__()
        self.image_files = glob.glob(os.path.join(data_dir, split, '*.png'))
        self.transform = transforms.Compose([
        transforms.Resize((128, 128)),  # 调整图片大小
        transforms.ToTensor(),  # 将图片转换为PyTorch张量
      ])
            
      def __len__(self):
        return len(self.image_files)

      def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert('L')  # 'L' mode means single-channel (grayscale)
        image = self.transform(image)
        return image

  elif config.data.dataset == 'AI4Scup2_full':
    data_dir = "/home/caoxiang/Desktop/Datasets/AI4Scup2_simulated_CBS/speed"
    train_split_name, eval_split_name= 'train', 'eval'

    class Create_Custom_Dataset(data.Dataset):
      def __init__(self, data_dir, split):
        super(Create_Custom_Dataset, self).__init__()
        self.image_files = glob.glob(os.path.join(data_dir, split, '*.npz'))
        
        self.max_output = 1595.1279
        self.min_output = 1408.692
            
      def __len__(self):
        return len(self.image_files)

      def __getitem__(self, idx):
        output_full = np.load(self.image_files[idx])["data"]
        output = resize(output_full[90:390,90:390], (256,256), mode='reflect', anti_aliasing=True)
        output = 2 * (output - self.min_output) / (self.max_output - self.min_output) - 1.
        output = torch.tensor(output, dtype=torch.float32).unsqueeze(0)

        return output

  else:
    raise NotImplementedError(
      f'Dataset {config.data.dataset} not yet supported.')

  
  train_dataset = Create_Custom_Dataset(data_dir, train_split_name)
  train_loader = data.DataLoader(train_dataset, shuffle = True, batch_size = config.training.batch_size, num_workers = 8)
  eval_dataset = Create_Custom_Dataset(data_dir, eval_split_name)
  eval_loader = data.DataLoader(eval_dataset, shuffle = False, batch_size = config.eval.batch_size, num_workers = 2)

  return train_loader, eval_loader, data_dir   


