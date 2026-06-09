from torch.utils.data import Dataset
import numpy as np
import random, os
from skimage.transform import resize
import torch
import torch.nn.functional as F

class USCT_Dataset(Dataset):
    def __init__(self, data_dict):
        self.input_path = data_dict["base_dir_dobs_" + data_dict["frequency"] + "_" + data_dict["stage"]]
        self.speed_path = data_dict["base_dir_speed_" + data_dict["stage"]]
        self.length = len(os.listdir(self.input_path))
         
        self.max_output = data_dict["max_output"]
        self.min_output = data_dict["min_output"]
        self.data_dict = data_dict

    def __len__(self):
        return self.length
    
    def add_awgn(self, signal, snr_dB):
        signal_power = np.mean(signal**2)

        snr_linear = 10**(snr_dB / 10.0)
        noise_power = signal_power / snr_linear

        noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)
        noisy_signal = signal + noise
        return noisy_signal
    
    def add_random_noise(self, input_freq):
        random_number = random.random()
        if random_number < 0.33:
            input_freq_noised = self.add_awgn(input_freq,10)
        elif random_number < 0.66:
            input_freq_noised = self.add_awgn(input_freq,5)
        else:
            input_freq_noised = input_freq
        return input_freq_noised
    
    def __getitem__(self, index):
        if self.data_dict["stage"] == "train":
            file_name = "train_" + str(index+1) + ".npy"
        elif self.data_dict["stage"] == "eval":
            file_name = "train_" + str(index+6601) + ".npy"
        
        # 先加噪
        input_freq = np.load(os.path.join(self.input_path, file_name))
        input_freq = np.stack((input_freq.real, input_freq.imag))  # 2,256,256
        input_freq_noised = self.add_random_noise(input_freq)
        
        # 再 Mask
        input_freq_noised_masked = input_freq_noised * self.data_dict["mask"]
        
        # 最后 取不同测量模式   
        if self.data_dict["measurement_mode"] == "sparse":
            sparse_ratio = 4
            invalid_rows = [i for i in range(input_freq_noised_masked.shape[1]) if i % sparse_ratio != 0] 
            invalid_cols = [j for j in range(input_freq_noised_masked.shape[2]) if j % sparse_ratio != 0]
            input_freq_noised_masked[:,  invalid_rows, :] = 0  
            input_freq_noised_masked[:, :,  invalid_cols] = 0  

            # 创建新矩阵并将有效值移动到左上角
            input_freq_noised_masked_rearranged = np.zeros_like(input_freq_noised_masked)  
            valid_block = input_freq_noised_masked[:, ::sparse_ratio, ::sparse_ratio]  
            input_freq_noised_masked_rearranged[:, :valid_block.shape[1], :valid_block.shape[2]] = valid_block  

            input_freq_noised_masked = np.stack([input_freq_noised_masked_rearranged], axis= 0)
        
        elif self.data_dict["measurement_mode"] == "sparse":
            sparse_ratio = 8
            invalid_rows = [i for i in range(input_freq_noised_masked.shape[1]) if i % sparse_ratio != 0] 
            invalid_cols = [j for j in range(input_freq_noised_masked.shape[2]) if j % sparse_ratio != 0]
            input_freq_noised_masked[:, invalid_rows, :] = 0  
            input_freq_noised_masked[:, :, invalid_cols] = 0  

            # 创建新矩阵并将有效值移动到左上角
            input_freq_noised_masked_rearranged = np.zeros_like(input_freq_noised_masked)  
            valid_block = input_freq_noised_masked[:, ::sparse_ratio, ::sparse_ratio]  
            input_freq_noised_masked_rearranged[:, :valid_block.shape[1], :valid_block.shape[2]] = valid_block  

            input_freq_noised_masked = np.stack([input_freq_noised_masked_rearranged], axis= 0)


        elif self.data_dict["measurement_mode"] == "partial":
            invalid_rows = [i for i in range(64,256)] 
            invalid_cols = [j for j in range(0,128)] + [j for j in range(128+64,256)]
            input_freq_noised_masked[:, invalid_rows, :] = 0
            input_freq_noised_masked[:, :, invalid_cols] = 0 
        
        elif self.data_dict["measurement_mode"] == "partial_2":
            invalid_rows = [i for i in range(32,256)] 
            invalid_cols = [j for j in range(0,128)] + [j for j in range(128+32,256)]
            input_freq_noised_masked[:, invalid_rows, :] = 0
            input_freq_noised_masked[:, :, invalid_cols] = 0 
            
        input_freq_noised_masked_tensor = torch.tensor(input_freq_noised_masked, dtype=torch.float32).view((2, 256, 256))

        # 加载对应的 真实图像
        output_full = np.load(os.path.join(self.speed_path, file_name))
        
        output = resize(output_full[90:390,90:390], self.data_dict["resize_size"], mode='reflect', anti_aliasing=True)
        output = 2 * (output - self.min_output) / (self.max_output - self.min_output) - 1.
        output = torch.tensor(output, dtype=torch.float32).unsqueeze(0)
        
        return input_freq_noised_masked_tensor, output

class USCT_Dataset_Valid(Dataset):
    def __init__(self, data_dict):
        self.input_path = data_dict["base_dir_dobs_" + data_dict["frequency"] + "_" + data_dict["stage"]]
        self.speed_path = data_dict["base_dir_speed_" + data_dict["stage"]]
        self.length = len(os.listdir(self.input_path))
         
        self.max_output = data_dict["max_output"]
        self.min_output = data_dict["min_output"]
        self.data_dict = data_dict

    def __len__(self):
        return self.length
    
    def add_awgn(self, signal, snr_dB):
        signal_power = np.mean(signal**2)

        snr_linear = 10**(snr_dB / 10.0)
        noise_power = signal_power / snr_linear

        noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)
        noisy_signal = signal + noise
        return noisy_signal

    
    def __getitem__(self, index):
        if self.data_dict["stage"] == "train":
            file_name = "train_" + str(index+1) + ".npy"
        elif self.data_dict["stage"] == "eval":
            file_name = "train_" + str(index+6601) + ".npy"
        
        # 先加噪
        input_freq = np.load(os.path.join(self.input_path, file_name))
        input_freq = np.stack((input_freq.real, input_freq.imag))  # 2,256,256

        inputs_freq_noised_free = input_freq
        inputs_freq_noised_10db = self.add_awgn(input_freq,10)
        inputs_freq_noised_5db = self.add_awgn(input_freq,5)
        input_freq_noised = np.stack([inputs_freq_noised_free,inputs_freq_noised_10db,inputs_freq_noised_5db], axis= 0)
        input_freq_noised_masked = input_freq_noised * self.data_dict["mask"] # 3, 2, 256, 256
        
        # 最后 取不同测量模式   
        if self.data_dict["measurement_mode"] == "sparse":
            sparse_ratio = 4
            invalid_rows = [i for i in range(input_freq_noised_masked.shape[2]) if i % sparse_ratio != 0] 
            invalid_cols = [j for j in range(input_freq_noised_masked.shape[3]) if j % sparse_ratio != 0]
            input_freq_noised_masked[:, :, invalid_rows, :] = 0  # 置为 0 的行
            input_freq_noised_masked[:, :, :, invalid_cols] = 0  # 置为 0 的列

            # 创建新矩阵并将有效值移动到左上角
            input_freq_noised_masked_rearranged = np.zeros_like(input_freq_noised_masked)  # 假设使用 PyTorch
            valid_block = input_freq_noised_masked[:, :, ::sparse_ratio, ::sparse_ratio]  # 提取有效块
            input_freq_noised_masked_rearranged[:, :, :valid_block.shape[2], :valid_block.shape[3]] = valid_block  # 填充左上角
            
            input_freq_noised_masked = np.stack([input_freq_noised_masked_rearranged], axis= 0)

        elif self.data_dict["measurement_mode"] == "sparse_2":
            sparse_ratio = 8
            invalid_rows = [i for i in range(input_freq_noised_masked.shape[2]) if i % sparse_ratio != 0] 
            invalid_cols = [j for j in range(input_freq_noised_masked.shape[3]) if j % sparse_ratio != 0]
            input_freq_noised_masked[:, :, invalid_rows, :] = 0  # 置为 0 的行
            input_freq_noised_masked[:, :, :, invalid_cols] = 0  # 置为 0 的列

            # 创建新矩阵并将有效值移动到左上角
            input_freq_noised_masked_rearranged = np.zeros_like(input_freq_noised_masked)  # 假设使用 PyTorch
            valid_block = input_freq_noised_masked[:, :, ::sparse_ratio, ::sparse_ratio]  # 提取有效块
            input_freq_noised_masked_rearranged[:, :, :valid_block.shape[2], :valid_block.shape[3]] = valid_block  # 填充左上角
            
            input_freq_noised_masked = np.stack([input_freq_noised_masked_rearranged], axis= 0)

        elif self.data_dict["measurement_mode"] == "partial":
            invalid_rows = [i for i in range(64,256)] 
            invalid_cols = [j for j in range(0,128)] + [j for j in range(128+64,256)]
            input_freq_noised_masked[:, :, invalid_rows, :] = 0
            input_freq_noised_masked[:, :, :, invalid_cols] = 0

            input_freq_noised_masked = np.stack([input_freq_noised_masked], axis= 0)

        elif self.data_dict["measurement_mode"] == "partial_2":
            invalid_rows = [i for i in range(32,256)] 
            invalid_cols = [j for j in range(0,128)] + [j for j in range(128+32,256)]
            input_freq_noised_masked[:, :, invalid_rows, :] = 0
            input_freq_noised_masked[:, :, :, invalid_cols] = 0  

            input_freq_noised_masked = np.stack([input_freq_noised_masked], axis= 0)
            
        input_freq_noised_masked_tensor = torch.tensor(input_freq_noised_masked, dtype=torch.float32).view((3, 2, 256, 256))
        
        # 加载对应的 真实图像
        output_full = np.load(os.path.join(self.speed_path, file_name))
        
        output = resize(output_full[90:390,90:390], self.data_dict["resize_size"], mode='reflect', anti_aliasing=True)
        output = 2 * (output - self.min_output) / (self.max_output - self.min_output) - 1.
        output = torch.tensor(output, dtype=torch.float32).unsqueeze(0)
        
        return input_freq_noised_masked_tensor, output


class USCT_Dataset_CBS(Dataset):
    # 这里的 Base Size 就分别是 Partial/Sparse 对应的 [64,64], 然后 Partial_2/Sparse_2 在此基础上继续变成[32,32];
    def __init__(self, data_dict):
        self.input_path = data_dict["base_dir_dobs_" + data_dict["frequency"] + "_" + data_dict["stage"]]
        self.speed_path = data_dict["base_dir_speed_" + data_dict["stage"]]
        self.length = len(os.listdir(self.input_path))
         
        self.max_output = data_dict["max_output"]
        self.min_output = data_dict["min_output"]
        self.data_dict = data_dict

    def __len__(self):
        return self.length
    
    def add_awgn(self, signal, snr_dB):
        signal_power = np.mean(signal**2)

        snr_linear = 10**(snr_dB / 10.0)
        noise_power = signal_power / snr_linear

        noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)
        noisy_signal = signal + noise
        return noisy_signal
    
    def add_random_noise(self, input_freq):
        random_number = random.random()
        if random_number < 0.33:
            input_freq_noised = self.add_awgn(input_freq,10)
        elif random_number < 0.66:
            input_freq_noised = self.add_awgn(input_freq,5)
        else:
            input_freq_noised = input_freq
        return input_freq_noised

    def add_fixed_noise(self, input_freq, snr_dB_list):
        input_freq_noised_list = []
        for snr_dB in snr_dB_list:
            input_freq_noised = self.add_awgn(input_freq, snr_dB)
            input_freq_noised_list.append(input_freq_noised)
        return  np.stack(input_freq_noised_list, axis=0)


    def __getitem__(self, index):
        if self.data_dict["stage"] == "train":
            file_name = "train_" + str(index+1) + ".npy"
        elif self.data_dict["stage"] == "eval":
            file_name = "train_" + str(index+6601) + ".npy"
        
        input_freq = np.load(os.path.join(self.input_path, file_name))
        input_freq = np.stack((input_freq.real, input_freq.imag))

        # 预处理: Mask + 加噪 + Mask + Tensor
        input_freq = input_freq * self.data_dict["mask"]
        # input_freq_noised = self.add_random_noise(input_freq)
        input_freq_noised = self.add_fixed_noise(input_freq, [100,10,5])
        input_freq_noised_masked = input_freq_noised * self.data_dict["mask"]
        input_freq_noised_masked = torch.tensor(input_freq_noised_masked, dtype=torch.float32)
        
        # 然后取不同测量模式 Partial: 
        if self.data_dict["measurement_mode"] == "sparse":
            input_freq_noised_masked_padded = input_freq_noised_masked
        
        elif self.data_dict["measurement_mode"] == "sparse_2":
            input_freq_noised_masked = input_freq_noised_masked[..., ::2, ::2]  # 置为 0 的行
            input_freq_noised_masked_padded = F.pad(input_freq_noised_masked,  (0, 32, 0, 32), mode='constant', value=0)
            
        elif self.data_dict["measurement_mode"] == "partial":
            input_freq_noised_masked_padded = input_freq_noised_masked

        elif self.data_dict["measurement_mode"] == "partial_2":
            valid_rows, valid_cols = [i for i in range(32)], [j for j in range(32)]
            input_freq_noised_masked = input_freq_noised_masked[..., valid_rows, :][..., :, valid_cols]
            input_freq_noised_masked_padded = F.pad(input_freq_noised_masked,  (0, 32, 0, 32), mode='constant', value=0)
            
        elif self.data_dict["measurement_mode"] == "full":
            input_freq_noised_masked_padded = input_freq_noised_masked
            
        # 加载对应的 真实图像
        output_full = np.load(os.path.join(self.speed_path, file_name))
        
        output = resize(output_full[90:390,90:390], self.data_dict["resize_size"], mode='reflect', anti_aliasing=True)
        output = 2 * (output - self.min_output) / (self.max_output - self.min_output) - 1.
        output = torch.tensor(output, dtype=torch.float32).unsqueeze(0)
        
        return input_freq_noised_masked_padded, output
