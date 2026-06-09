import os
import sys
import math
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.io import loadmat


def add_repo_to_path(repo_root: str):
    repo_root = os.path.abspath(repo_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def resolve_device(device_arg: str):
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_file_dict(repo_root: str, measurement_mode: str, stage: str):
    repo_root = Path(repo_root)
    aux = repo_root / "auxiliary_data"
    base_mask = np.load(aux / "mask.npy")
    x_pos = loadmat(aux / "x_pos.mat")["x_pos256"]
    y_pos = loadmat(aux / "y_pos.mat")["y_pos256"]

    file = {
        "max_output": 1595.1279,
        "min_output": 1408.692,
        "resize_size": (256, 256),
        "measurement_mode": measurement_mode,
        "stage": stage,
        "frequency": "500k",
        "base_mask": base_mask,
        "x_pos": x_pos,
        "y_pos": y_pos,
    }

    if measurement_mode in ["sparse", "sparse_2"]:
        dataset_root = repo_root / "Datasets_test" / "AI4Scup2_simulated_CBS_sparse"
        file["base_dir_dobs_500k_train"] = str(dataset_root / "dobs_500k" / "train")
        file["base_dir_dobs_500k_eval"] = str(dataset_root / "dobs_500k" / "eval")
        file["base_dir_speed_train"] = str(dataset_root / "speed" / "train")
        file["base_dir_speed_eval"] = str(dataset_root / "speed" / "eval")
        file["mask"] = base_mask[0::4, 0::4]
        if measurement_mode == "sparse":
            file["downsampled_input_mask"] = file["mask"]
            file["receiver_indices"] = torch.cat(
                (torch.tensor(x_pos[0::4].astype(np.int64)), torch.tensor(y_pos[0::4].astype(np.int64))), dim=1
            )
            file["transmitter_indices"] = torch.cat(
                (torch.tensor(x_pos[0::4].astype(np.int64)), torch.tensor(y_pos[0::4].astype(np.int64))), dim=1
            )
        else:
            file["downsampled_input_mask"] = file["mask"][::2, ::2]
            file["receiver_indices"] = torch.cat(
                (torch.tensor(x_pos[0::8].astype(np.int64)), torch.tensor(y_pos[0::8].astype(np.int64))), dim=1
            )
            file["transmitter_indices"] = torch.cat(
                (torch.tensor(x_pos[0::8].astype(np.int64)), torch.tensor(y_pos[0::8].astype(np.int64))), dim=1
            )
    elif measurement_mode in ["partial", "partial_2"]:
        dataset_root = repo_root / "Datasets_test" / "AI4Scup2_simulated_CBS_partial"
        file["base_dir_dobs_500k_train"] = str(dataset_root / "dobs_500k" / "train")
        file["base_dir_dobs_500k_eval"] = str(dataset_root / "dobs_500k" / "eval")
        file["base_dir_speed_train"] = str(dataset_root / "speed" / "train")
        file["base_dir_speed_eval"] = str(dataset_root / "speed" / "eval")
        base_valid_rows, base_valid_cols = [i for i in range(64)], [j for j in range(128, 128 + 64)]
        file["mask"] = base_mask[base_valid_rows, :][:, base_valid_cols]
        if measurement_mode == "partial":
            file["downsampled_input_mask"] = file["mask"]
            valid_rows, valid_cols = [i for i in range(64)], [j for j in range(128, 128 + 64)]
            file["receiver_indices"] = torch.cat(
                (torch.tensor(x_pos[valid_rows].astype(np.int64)), torch.tensor(y_pos[valid_cols].astype(np.int64))), dim=1
            )
            file["transmitter_indices"] = file["receiver_indices"].clone()
        else:
            file["downsampled_input_mask"] = file["mask"][::2, ::2]
            valid_rows, valid_cols = [i for i in range(0, 64, 2)], [j for j in range(128, 128 + 64, 2)]
            file["receiver_indices"] = torch.cat(
                (torch.tensor(x_pos[valid_rows].astype(np.int64)), torch.tensor(y_pos[valid_cols].astype(np.int64))), dim=1
            )
            file["transmitter_indices"] = file["receiver_indices"].clone()
    else:
        raise ValueError(f"Unsupported measurement_mode: {measurement_mode}")

    return file


def safe_load_state_dict(model, checkpoint_path, device, wrap_dataparallel=True):
    ckpt = torch.load(checkpoint_path, map_location=device)
    raw_state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    model_state = model.state_dict()

    def add_module_prefix(sd):
        return {(("module." + k) if not k.startswith("module.") else k): v for k, v in sd.items()}

    def strip_module_prefix(sd):
        return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}

    candidates = [raw_state, add_module_prefix(raw_state), strip_module_prefix(raw_state)]
    best = None
    best_count = -1
    for cand in candidates:
        count = 0
        filtered = {}
        for k, v in cand.items():
            if k in model_state and getattr(v, "shape", None) == model_state[k].shape:
                filtered[k] = v
                count += 1
        if count > best_count:
            best_count = count
            best = filtered

    missing = [k for k in model_state.keys() if k not in best]
    unexpected = [k for k in raw_state.keys() if k not in model_state]
    model.load_state_dict(best, strict=False)
    return ckpt, best_count, len(model_state), len(missing), len(unexpected)


def save_image(arr2d, path, title=None, cmap="inferno", vmin=None, vmax=None):
    plt.figure(figsize=(4, 4))
    plt.imshow(arr2d, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=str, default=".", help="默认当前 USCT_download 仓库根目录")
    parser.add_argument("--checkpoint", type=str, default="./mini_score_ckpt_sparse_fixednorm_e100/checkpoint_best.pth", help="默认使用当前会话训练得到的 EMA 权重")
    parser.add_argument("--measurement_mode", type=str, default="sparse", choices=["sparse", "sparse_2", "partial", "partial_2"], help="默认与你当前训练/测试最一致的 sparse 模式")
    parser.add_argument("--stage", type=str, default="train", choices=["train", "eval"], help="默认先用 train split 做 smoke test")
    parser.add_argument("--sample_index", type=int, default=0, help="默认取第 0 个样本")
    parser.add_argument("--noise_index", type=int, default=1, help="默认沿用 notebook 常用的 [1:2]，即 noise_index=1")
    parser.add_argument("--num_steps", type=int, default=20, help="默认只跑 3 个时间步做最小验证")
    parser.add_argument("--eta", type=float, default=0.0, help="默认更稳的 eta=0")
    parser.add_argument("--grad_step", type=float, default=0.1, help="默认的 CBS 梯度步长")
    parser.add_argument("--grad_epochs", type=int, default=1, help="默认每个时间步只做 1 次梯度修正")
    parser.add_argument("--cbs_iters", type=int, default=150, help="默认比 notebook 更轻的 60 次 CBS 迭代")
    parser.add_argument("--device", type=str, default="cuda:0", help="默认直接用 cuda:0；没有 GPU 时可手动改成 cpu")
    parser.add_argument("--use_dataparallel", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./dps_smoke_outputs", help="默认输出目录")
    args = parser.parse_args()

    print("=" * 80)
    print("[默认运行说明]")
    print("=" * 80)
    print("你可以直接运行: python run_dps_smoke_test.py")
    print("当前默认参数已经写入代码：")
    print("  repo_root        =", args.repo_root)
    print("  checkpoint       =", args.checkpoint)
    print("  measurement_mode =", args.measurement_mode)
    print("  stage            =", args.stage)
    print("  sample_index     =", args.sample_index)
    print("  noise_index      =", args.noise_index)
    print("  num_steps        =", args.num_steps)
    print("  cbs_iters        =", args.cbs_iters)
    print("  output_dir       =", args.output_dir)

    add_repo_to_path(args.repo_root)

    from utils_cbs import normalize, denormalize
    from image_datasets import USCT_Dataset_CBS
    from cbs_model import ConvergentBornSeries_Batch, ConvergentBornSeries_Batch_Adjoint
    import sde_lib
    from models import ddpm as ddpm_model
    from models import utils as mutils
    from configs.vp import AI4Scup2_ddpm_continuous as configs

    device = resolve_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("[0] 基本信息")
    print("=" * 80)
    print("repo_root         =", args.repo_root)
    print("checkpoint        =", args.checkpoint)
    print("measurement_mode  =", args.measurement_mode)
    print("stage             =", args.stage)
    print("sample_index      =", args.sample_index)
    print("noise_index       =", args.noise_index)
    print("num_steps         =", args.num_steps)
    print("eta               =", args.eta)
    print("cbs_iters         =", args.cbs_iters)
    print("resolved device   =", device)

    config = configs.get_config()
    config.device_ids = [0]
    config.device = device

    file = build_file_dict(args.repo_root, args.measurement_mode, args.stage)
    dataset = USCT_Dataset_CBS(file)
    input_batch, output_batch = dataset[args.sample_index]

    # input_batch: [3, 2, 64,64] or [3,2,32,32] padded; output_batch: [1,256,256]
    print("=" * 80)
    print("[1] 样本信息")
    print("=" * 80)
    print("input_batch shape =", tuple(input_batch.shape))
    print("output_batch shape=", tuple(output_batch.shape))

    output_batch = output_batch.unsqueeze(0).to(device)  # [1,1,256,256]
    output_cbs_batch = denormalize(output_batch).to(device)  # [1,1,480,480]
    cbs_batch_one = 1500 * torch.ones_like(output_cbs_batch, device=device)

    image_mask = (output_cbs_batch > 1500.1) | (output_cbs_batch < 1499.9)
    kernel = torch.ones(1, 1, 3, 3, device=device)
    dilated_mask = (F.conv2d(image_mask.float(), kernel, padding=1) > 0)

    # 选择一个噪声层：notebook 用 [1:2]，即 noise_index=1
    input_batch = input_batch.to(device)
    dobs_complex_all = torch.complex(input_batch[:, 0], input_batch[:, 1])
    if args.measurement_mode in ["sparse_2", "partial_2"]:
        dobs_complex_all = dobs_complex_all[..., :32, :32]
    downsampled_dobs_500k_batch = dobs_complex_all[args.noise_index:args.noise_index + 1]

    print("downsampled_dobs shape =", tuple(downsampled_dobs_500k_batch.shape))
    print("dilated_mask shape     =", tuple(dilated_mask.shape))

    print("=" * 80)
    print("[2] 加载 score model")
    print("=" * 80)
    score_model = ddpm_model.DDPM(config)
    if args.use_dataparallel and torch.cuda.is_available():
        score_model = torch.nn.DataParallel(score_model, device_ids=[0])
    score_model = score_model.to(device)
    _, matched, total, missing, unexpected = safe_load_state_dict(score_model, args.checkpoint, device)
    score_model.eval()
    print(f"checkpoint matched keys = {matched}/{total}")
    print(f"missing keys            = {missing}")
    print(f"unexpected keys         = {unexpected}")

    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    score_fn = mutils.get_score_fn(sde, score_model, train=False, continuous=config.training.continuous)

    def neural_cbs_gradient_descent_dps(pred_batch, image_mask, downsampled_input_batch, downsampled_input_mask,
                                        ds=0.1, cbs_num_epoch=1):
        pred_cbs_batch = denormalize(pred_batch)
        last_loss = None
        for epoch in range(cbs_num_epoch):
            model_pred = ConvergentBornSeries_Batch(
                f=500e3,
                sos=pred_cbs_batch,
                boundary_width=[300, 300],
                boundary_strength=225,
                boundary_type='PML3',
                src_loc_set=file["transmitter_indices"],
                device=device,
            )
            model_pred_grad = ConvergentBornSeries_Batch_Adjoint(
                batch_model=model_pred,
                rec_loc=file["receiver_indices"],
                dobs_500k_batch=downsampled_input_batch,
                dobs_500k_mask=file["downsampled_input_mask"],
            )
            u_pred = model_pred(max_iters=args.cbs_iters)
            pred_cbs_batch_grad, loss_value = model_pred_grad(u_pred, max_iters=args.cbs_iters)
            step_scale = ds / torch.sqrt(loss_value + 1e-12)
            pred_cbs_batch = pred_cbs_batch.clone()
            pred_cbs_batch[image_mask] -= step_scale * pred_cbs_batch_grad[image_mask]
            last_loss = float(loss_value.detach().cpu())
            print(f"  [CBS grad] epoch={epoch} loss={last_loss:.6f} step_scale={float(step_scale.detach().cpu()):.6f}")
        x_norm = normalize(pred_cbs_batch)
        x_norm = torch.clamp(x_norm, -1.0, 1.0)
        return x_norm, last_loss

    def ancestral_sampling(pred_batch, dilated_mask, downsampled_input_batch, downsampled_input_mask,
                           ts_value, eta=0.0, mrsde_sampling=False):
        history = []
        t_start = ts_value[0] * torch.ones(pred_batch.shape[0], device=pred_batch.device)
        t_start_i = (torch.floor(t_start * sde.N) + 1) / sde.N

        x_0_hat_after_cbs, loss_value = neural_cbs_gradient_descent_dps(
            pred_batch, dilated_mask, downsampled_input_batch, downsampled_input_mask,
            ds=args.grad_step, cbs_num_epoch=args.grad_epochs,
        )
        history.append({"stage": "init_cbs", "loss": loss_value})

        with torch.no_grad():
            z = torch.randn_like(x_0_hat_after_cbs)
            mean, std, coef = sde.marginal_prob(x_0_hat_after_cbs, t_start_i)
            perturbed_data = mean + std[:, None, None, None] * z
            perturbed_score = score_fn(perturbed_data, t_start_i)
            x_0_hat_tweedie = (perturbed_data + perturbed_score * std[:, None, None, None] ** 2) / coef[:, None, None, None]
            x_0_hat = coef[:, None, None, None] * x_0_hat_tweedie + (1 - coef)[:, None, None, None] * pred_batch if mrsde_sampling else x_0_hat_tweedie
            x_0_hat = torch.clamp(x_0_hat, -1.0, 1.0)
            history.append({
                "stage": "init_tweedie",
                "mean": float(x_0_hat.mean().detach().cpu()),
                "std": float(x_0_hat.std().detach().cpu()),
            })

        for step_id, t_value in enumerate(ts_value[1:], start=1):
            print(f"[sampling] step {step_id}/{len(ts_value)-1}, t={t_value:.4f}")
            t = t_value * torch.ones(x_0_hat.shape[0], device=x_0_hat.device)
            t_i = (torch.floor(t * sde.N) + 1) / sde.N
            x_0_hat_after_cbs, loss_value = neural_cbs_gradient_descent_dps(
                x_0_hat, dilated_mask, downsampled_input_batch, downsampled_input_mask,
                ds=args.grad_step, cbs_num_epoch=args.grad_epochs,
            )
            with torch.no_grad():
                z = torch.randn_like(x_0_hat_after_cbs)
                std_before, coef_before = std.clone(), coef.clone()
                mean, std, coef = sde.marginal_prob(x_0_hat_after_cbs, t_i)
                beta_tilde_t = (std / std_before) * torch.sqrt(torch.clamp(1 - (coef_before / coef) ** 2, min=1e-12))
                deterministic_noise_weight = torch.sqrt(torch.clamp(std ** 2 - (eta * beta_tilde_t) ** 2, min=1e-12))
                perturbed_noise_before = -perturbed_score.detach() * std_before[:, None, None, None]
                perturbed_data = (
                    mean
                    + deterministic_noise_weight[:, None, None, None] * perturbed_noise_before
                    + (eta * beta_tilde_t)[:, None, None, None] * z
                )
                perturbed_score = score_fn(perturbed_data, t_i)
                x_0_hat_tweedie = (perturbed_data + perturbed_score * std[:, None, None, None] ** 2) / coef[:, None, None, None]
                x_0_hat = coef[:, None, None, None] * x_0_hat_tweedie + (1 - coef)[:, None, None, None] * x_0_hat if mrsde_sampling else x_0_hat_tweedie
                x_0_hat = torch.clamp(x_0_hat, -1.0, 1.0)
                history.append({
                    "stage": f"step_{step_id}",
                    "t": float(t_value),
                    "loss": loss_value,
                    "mean": float(x_0_hat.mean().detach().cpu()),
                    "std": float(x_0_hat.std().detach().cpu()),
                })

        with torch.no_grad():
            t_end = ts_value[-1] * torch.ones(x_0_hat.shape[0], device=x_0_hat.device)
            t_end_i = (torch.floor(t_end * sde.N) + 1) / sde.N
            z = torch.randn_like(x_0_hat)
            mean, std, coef = sde.marginal_prob(x_0_hat, t_end_i)
            perturbed_data = mean + std[:, None, None, None] * z
            perturbed_score = score_fn(perturbed_data, t_end_i)
            x_0_hat_tweedie = (perturbed_data + perturbed_score * std[:, None, None, None] ** 2) / coef[:, None, None, None]
            x_0_hat = coef[:, None, None, None] * x_0_hat_tweedie + (1 - coef)[:, None, None, None] * x_0_hat if mrsde_sampling else x_0_hat_tweedie
            history.append({
                "stage": "final_denoise",
                "mean": float(x_0_hat.mean().detach().cpu()),
                "std": float(x_0_hat.std().detach().cpu()),
            })
        return x_0_hat, history

    ts_values = np.linspace(0.99, 0.01, args.num_steps).tolist()
    print("=" * 80)
    print("[3] 开始 DPS smoke test")
    print("=" * 80)
    print("ts_values =", [round(v, 4) for v in ts_values])

    pred_init = torch.zeros_like(output_batch, device=device)
    result, history = ancestral_sampling(
        pred_init,
        dilated_mask,
        downsampled_dobs_500k_batch,
        file["downsampled_input_mask"],
        ts_value=ts_values,
        eta=args.eta,
        mrsde_sampling=False,
    )

    result_denorm = denormalize(result).detach().cpu().numpy()
    gt_denorm = output_cbs_batch.detach().cpu().numpy()
    mask_np = dilated_mask.detach().cpu().numpy()

    center_pred = result_denorm[0, 0, 90:390, 90:390]
    center_gt = gt_denorm[0, 0, 90:390, 90:390]
    center_mask = mask_np[0, 0, 90:390, 90:390]

    np.save(os.path.join(args.output_dir, "result_norm.npy"), result.detach().cpu().numpy())
    np.save(os.path.join(args.output_dir, "result_denorm.npy"), result_denorm)
    np.save(os.path.join(args.output_dir, "gt_denorm.npy"), gt_denorm)
    np.save(os.path.join(args.output_dir, "dilated_mask.npy"), mask_np)

    save_image(center_pred, os.path.join(args.output_dir, "result_center.png"), title="DPS smoke result",
               cmap="inferno", vmin=file["min_output"], vmax=file["max_output"])
    save_image(center_gt, os.path.join(args.output_dir, "gt_center.png"), title="GT center",
               cmap="inferno", vmin=file["min_output"], vmax=file["max_output"])
    save_image(center_mask.astype(np.float32), os.path.join(args.output_dir, "mask_center.png"), title="Dilated mask",
               cmap="gray")

    print("=" * 80)
    print("[4] 结果统计")
    print("=" * 80)
    print("result norm shape   =", tuple(result.shape))
    print("result denorm shape =", tuple(result_denorm.shape))
    print("result mean/std     =", float(result.mean().detach().cpu()), float(result.std().detach().cpu()))
    print("denorm min/max      =", float(result_denorm.min()), float(result_denorm.max()))
    print("history entries     =", len(history))
    for item in history:
        print(item)

    print("=" * 80)
    print("[完成]")
    print("=" * 80)
    print("已保存输出到:", os.path.abspath(args.output_dir))
    print("说明: 这是 notebook-faithful 的 smoke test，其中 dilated_mask 来自真值图像，仅用于先验证采样链路是否能跑通。")


if __name__ == "__main__":
    main()
