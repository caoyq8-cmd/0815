import numpy as np
import torch
import scipy.io as scio

def usct_file_defaults_modified(measure_mode_name):
    # Setting Measurement_Mode Parameters
    file = {
        "max_output" : 1595.1279, 
        "min_output" : 1408.692, 
        "resize_size": (256,256), 
        "measurement_mode": measure_mode_name, # "sparse", "partial"， "sparse_2", "partial_2"
        "stage": "train", # "train", "eval" 
        "frequency": "500k", # "500k"
        "base_mask": np.load("auxiliary_data/mask.npy"), # 256 * 256 测量域相邻 Detectors 的屏蔽 
        "x_pos": scio.loadmat('auxiliary_data/x_pos.mat')['x_pos256'],
        "y_pos": scio.loadmat('auxiliary_data/y_pos.mat')['y_pos256'],
        }
    if file["measurement_mode"] == "sparse" or file["measurement_mode"] == "sparse_2":
        file["base_dir_dobs_500k_train"] = "/home/caoxiang/Desktop/Datasets/AI4Scup2_simulated_CBS_sparse/dobs_500k/train"
        file["base_dir_dobs_500k_eval"] = "/home/caoxiang/Desktop/Datasets/AI4Scup2_simulated_CBS_sparse/dobs_500k/eval"
        file["base_dir_speed_train"] = "/home/caoxiang/Desktop/Datasets/AI4Scup2_simulated_CBS_sparse/speed/train"
        file["base_dir_speed_eval"] = "/home/caoxiang/Desktop/Datasets/AI4Scup2_simulated_CBS_sparse/speed/eval"
        
        file["mask"] = file["base_mask"][0::4,0::4]
        if  file["measurement_mode"] == "sparse":
            file["receiver_indices"] = torch.cat((torch.tensor(file["x_pos"][0::4].astype(np.int64)), torch.tensor(file["y_pos"][0::4].astype(np.int64))), dim = 1)
            file["transmitter_indices"] = torch.cat((torch.tensor(file["x_pos"][0::4].astype(np.int64)), torch.tensor(file["y_pos"][0::4].astype(np.int64))), dim = 1)
        
        elif file["measurement_mode"] == "sparse_2":
            file["receiver_indices"] = torch.cat((torch.tensor(file["x_pos"][0::8].astype(np.int64)), torch.tensor(file["y_pos"][0::8].astype(np.int64))), dim = 1)
            file["transmitter_indices"] = torch.cat((torch.tensor(file["x_pos"][0::8].astype(np.int64)), torch.tensor(file["y_pos"][0::8].astype(np.int64))), dim = 1)
        
    elif file["measurement_mode"] == "partial" or file["measurement_mode"] == "partial_2":
        file["base_dir_dobs_500k_train"] = "/home/caoxiang/Desktop/Datasets/AI4Scup2_simulated_CBS_partial/dobs_500k/train"
        file["base_dir_dobs_500k_eval"] = "/home/caoxiang/Desktop/Datasets/AI4Scup2_simulated_CBS_partial/dobs_500k/eval"
        file["base_dir_speed_train"] = "/home/caoxiang/Desktop/Datasets/AI4Scup2_simulated_CBS_partial/speed/train"
        file["base_dir_speed_eval"] = "/home/caoxiang/Desktop/Datasets/AI4Scup2_simulated_CBS_partial/speed/eval"
        
        base_valid_rows, base_valid_cols = [i for i in range(64)], [j for j in range(128,128+64)] 
        file["mask"] = file["base_mask"][base_valid_rows,:][:,base_valid_cols]
        if  file["measurement_mode"] == "partial":
            valid_rows, valid_cols = [i for i in range(64)], [j for j in range(128,128+64)] 
            file["receiver_indices"] = torch.cat((torch.tensor(file["x_pos"][valid_cols].astype(np.int64)), torch.tensor(file["y_pos"][valid_cols].astype(np.int64))), dim = 1)
            file["transmitter_indices"] = torch.cat((torch.tensor(file["x_pos"][valid_rows].astype(np.int64)), torch.tensor(file["y_pos"][valid_rows].astype(np.int64))), dim = 1)
        
        elif file["measurement_mode"] == "partial_2":
            valid_rows, valid_cols = [i for i in range(32)], [j for j in range(128,128+32)] 
            file["receiver_indices"] = torch.cat((torch.tensor(file["x_pos"][valid_cols].astype(np.int64)), torch.tensor(file["y_pos"][valid_cols].astype(np.int64))), dim = 1)
            file["transmitter_indices"] = torch.cat((torch.tensor(file["x_pos"][valid_rows].astype(np.int64)), torch.tensor(file["y_pos"][valid_rows].astype(np.int64))), dim = 1)
    
    return file



def usct_file_defaults():
    file = {
    "max_output" : 1595.1279, 
    "min_output" : 1408.692,
    "resize_size": (256,256), 
    "mask": np.load("/home/caoxiang/Desktop/score_sde_FWI_posterior_sampling/auxiliary_data/mask.npy"), # 256 * 256 测量域相邻 Detectors 的屏蔽 
    "measurement_mode": "partial", # "sparse", "paritial"
    "stage": "train", # "train", "eval"
    "frequency": "300k", # "300k", "400k", "500k"
    }

    file["base_dir_dobs_300k_train"] = "/home/caoxiang/Desktop/Datasets/AI4Scup2_simulated/dobs_300k/train"
    file["base_dir_dobs_400k_train"] = "/home/caoxiang/Desktop/Datasets/AI4Scup2_simulated/dobs_400k/train"
    file["base_dir_dobs_500k_train"] = "/home/caoxiang/Desktop/Datasets/AI4Scup2_simulated/dobs_500k/train"
    file["base_dir_dobs_300k_eval"] = "/home/caoxiang/Desktop/Datasets/AI4Scup2_simulated/dobs_300k/eval"
    file["base_dir_dobs_400k_eval"] = "/home/caoxiang/Desktop/Datasets/AI4Scup2_simulated/dobs_400k/eval"
    file["base_dir_dobs_500k_eval"] = "/home/caoxiang/Desktop/Datasets/AI4Scup2_simulated/dobs_500k/eval"

    file["base_dir_speed_train"] = "/home/caoxiang/Desktop/Datasets/AI4Scup2_simulated/speed/train"
    file["base_dir_speed_eval"] = "/home/caoxiang/Desktop/Datasets/AI4Scup2_simulated/speed/eval"
    

    return file

def train_file_defaults():
    train_file = {
    "gamma": 1,
    "epochs": 2000,
    "train_batch_size": 128, #这个选 128 以内的吧！
    "eval_batch_size": 16, #这个选 128 以内的吧！
    "learning_rate": 0.001, #这个不能太大采用 OneCycle 时要取 0.00005, 循环过程中的最大学习率！
    "weight_decay": 1e-06,
    }

    return train_file