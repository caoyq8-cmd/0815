import os
import sys
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


def safe_load_state_dict(model, checkpoint_path, device):
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


def save_curve(values, path, title, ylabel):
    plt.figure(figsize=(6, 4))
    plt.plot(range(len(values)), values, marker="o")
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def crop_center_300_from_480(x480):
    return x480[..., 90:390, 90:390]


def save_denorm_center_image(x_norm, path, title, denormalize_fn, file_dict):
    x_denorm = denormalize_fn(x_norm).detach().cpu().numpy()
    center = x_denorm[0, 0, 90:390, 90:390]
    save_image(
        center,
        path,
        title=title,
        cmap="inferno",
        vmin=file_dict["min_output"],
        vmax=file_dict["max_output"],
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--repo_root", type=str, default=".", help="USCT_download 仓库根目录")
    parser.add_argument("--checkpoint", type=str, default="./mini_score_ckpt_sparse_fixednorm_e100/checkpoint_best.pth")
    parser.add_argument("--measurement_mode", type=str, default="sparse",
                        choices=["sparse", "sparse_2", "partial", "partial_2"])
    parser.add_argument("--stage", type=str, default="train", choices=["train", "eval"])
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--noise_index", type=int, default=1)

    parser.add_argument("--num_steps", type=int, default=10, help="外层 diffusion 采样步数")
    parser.add_argument("--eta", type=float, default=0.0, help="保留 DDIM 风格随机项控制，默认 0")
    parser.add_argument("--grad_step", type=float, default=0.02, help="DDS 子问题内循环步长")
    parser.add_argument("--grad_epochs", type=int, default=2, help="DDS 子问题内循环次数")
    parser.add_argument("--lam", type=float, default=3.0, help="DDS 中 prior tether 权重 lambda")
    parser.add_argument("--cbs_iters", type=int, default=60, help="每次 CBS/adjoint 的内部迭代数")

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use_dataparallel", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./dds_smoke_outputs")
    args = parser.parse_args()

    print("=" * 80)
    print("[DDS smoke test with Tweedie/after-DDS saving]")
    print("=" * 80)
    print("repo_root        =", args.repo_root)
    print("checkpoint       =", args.checkpoint)
    print("measurement_mode =", args.measurement_mode)
    print("stage            =", args.stage)
    print("sample_index     =", args.sample_index)
    print("noise_index      =", args.noise_index)
    print("num_steps        =", args.num_steps)
    print("grad_step        =", args.grad_step)
    print("grad_epochs      =", args.grad_epochs)
    print("lam              =", args.lam)
    print("cbs_iters        =", args.cbs_iters)
    print("output_dir       =", args.output_dir)

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
    print("resolved device =", device)

    config = configs.get_config()
    config.device_ids = [0]
    config.device = device

    file = build_file_dict(args.repo_root, args.measurement_mode, args.stage)
    dataset = USCT_Dataset_CBS(file)
    input_batch, output_batch = dataset[args.sample_index]

    print("=" * 80)
    print("[1] 样本信息")
    print("=" * 80)
    print("input_batch shape =", tuple(input_batch.shape))
    print("output_batch shape=", tuple(output_batch.shape))

    output_batch = output_batch.unsqueeze(0).to(device)   # [1,1,256,256]
    output_cbs_batch = denormalize(output_batch).to(device)  # [1,1,480,480]

    image_mask = (output_cbs_batch > 1500.1) | (output_cbs_batch < 1499.9)
    kernel = torch.ones(1, 1, 3, 3, device=device)
    dilated_mask = (F.conv2d(image_mask.float(), kernel, padding=1) > 0)

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

    sde = sde_lib.VPSDE(
        beta_min=config.model.beta_min,
        beta_max=config.model.beta_max,
        N=config.model.num_scales
    )
    score_fn = mutils.get_score_fn(sde, score_model, train=False, continuous=config.training.continuous)

    def neural_cbs_dds_update(
        pred_batch,
        image_mask,
        downsampled_input_batch,
        downsampled_input_mask,
        lam=3.0,
        ds=0.02,
        cbs_num_epoch=2,
        smooth_kernel=9,
    ):
        """
        DDS-style regularized data-consistency subproblem:

            min_x  L_meas(x) + 0.5 * lam * ||x - x_prior||^2

        这里:
        - x_prior = 当前 diffusion / Tweedie 给出的干净图估计
        - L_meas  = CBS 测量一致性误差
        - 内循环用 regularized gradient descent 近似求解
        """
        x_prior_cbs = denormalize(pred_batch).detach()
        x_cbs = x_prior_cbs.clone()

        last_loss = None
        last_prior_mse = None

        for epoch in range(cbs_num_epoch):
            model_pred = ConvergentBornSeries_Batch(
                f=500e3,
                sos=x_cbs,
                boundary_width=[300, 300],
                boundary_strength=225,
                boundary_type="PML3",
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
            meas_grad, loss_value = model_pred_grad(u_pred, max_iters=args.cbs_iters)

            meas_grad = F.avg_pool2d(
                meas_grad,
                kernel_size=smooth_kernel,
                stride=1,
                padding=smooth_kernel // 2,
            )

            prior_grad = lam * (x_cbs - x_prior_cbs)
            total_grad = meas_grad + prior_grad

            step_scale = ds / torch.sqrt(loss_value + 1e-12)

            x_cbs = x_cbs.clone()
            x_cbs[image_mask] -= step_scale * total_grad[image_mask]
            x_cbs = torch.clamp(x_cbs, 1400.0, 1605.0)

            last_loss = float(loss_value.detach().cpu())
            last_prior_mse = float(torch.mean((x_cbs - x_prior_cbs) ** 2).detach().cpu())

            print(
                f"  [DDS DC] epoch={epoch} "
                f"meas_loss={last_loss:.6f} "
                f"prior_mse={last_prior_mse:.6f} "
                f"step_scale={float(step_scale.detach().cpu()):.6f}"
            )

            del u_pred, model_pred, model_pred_grad, meas_grad, prior_grad, total_grad
            torch.cuda.empty_cache()

        x_norm = normalize(x_cbs)
        x_norm = torch.clamp(x_norm, -1.0, 1.0)
        return x_norm, last_loss, last_prior_mse

    def dds_sampling(
        pred_batch,
        dilated_mask,
        downsampled_input_batch,
        downsampled_input_mask,
        ts_value,
        eta=0.0,
        mrsde_sampling=False,
    ):
        history = []

        # 初始时刻
        t_start = ts_value[0] * torch.ones(pred_batch.shape[0], device=pred_batch.device)
        t_start_i = (torch.floor(t_start * sde.N) + 1) / sde.N

        # 先做一次 Tweedie
        with torch.no_grad():
            z = torch.randn_like(pred_batch)
            mean, std, coef = sde.marginal_prob(pred_batch, t_start_i)
            perturbed_data = mean + std[:, None, None, None] * z
            perturbed_score = score_fn(perturbed_data, t_start_i)
            x_0_hat_tweedie = (
                perturbed_data + perturbed_score * std[:, None, None, None] ** 2
            ) / coef[:, None, None, None]
            x_0_hat = (
                coef[:, None, None, None] * x_0_hat_tweedie
                + (1 - coef)[:, None, None, None] * pred_batch
                if mrsde_sampling else x_0_hat_tweedie
            )
            x_0_hat = torch.clamp(x_0_hat, -1.0, 1.0)

        # 保存初始 Tweedie
        save_denorm_center_image(
            x_0_hat,
            os.path.join(args.output_dir, "init_tweedie.png"),
            title="Init Tweedie",
            denormalize_fn=denormalize,
            file_dict=file,
        )

        # DDS 子问题
        current, meas_loss, prior_mse = neural_cbs_dds_update(
            x_0_hat,
            dilated_mask,
            downsampled_input_batch,
            downsampled_input_mask,
            lam=args.lam,
            ds=args.grad_step,
            cbs_num_epoch=args.grad_epochs,
        )

        # 保存初始 after DDS
        save_denorm_center_image(
            current,
            os.path.join(args.output_dir, "init_after_dds.png"),
            title="Init After DDS",
            denormalize_fn=denormalize,
            file_dict=file,
        )

        history.append({
            "stage": "init_dds",
            "meas_loss": meas_loss,
            "prior_mse": prior_mse,
            "mean": float(current.mean().detach().cpu()),
            "std": float(current.std().detach().cpu()),
        })

        for step_id, t_value in enumerate(ts_value[1:], start=1):
            print(f"[DDS sampling] step {step_id}/{len(ts_value)-1}, t={t_value:.4f}")

            t = t_value * torch.ones(current.shape[0], device=current.device)
            t_i = (torch.floor(t * sde.N) + 1) / sde.N

            with torch.no_grad():
                z = torch.randn_like(current)
                mean, std, coef = sde.marginal_prob(current, t_i)
                perturbed_data = mean + std[:, None, None, None] * z
                perturbed_score = score_fn(perturbed_data, t_i)
                x_0_hat_tweedie = (
                    perturbed_data + perturbed_score * std[:, None, None, None] ** 2
                ) / coef[:, None, None, None]
                x_0_hat = (
                    coef[:, None, None, None] * x_0_hat_tweedie
                    + (1 - coef)[:, None, None, None] * current
                    if mrsde_sampling else x_0_hat_tweedie
                )
                x_0_hat = torch.clamp(x_0_hat, -1.0, 1.0)

            # 保存 Tweedie
            save_denorm_center_image(
                x_0_hat,
                os.path.join(args.output_dir, f"step_{step_id:02d}_tweedie.png"),
                title=f"Tweedie step {step_id}",
                denormalize_fn=denormalize,
                file_dict=file,
            )

            # DDS 子问题
            current, meas_loss, prior_mse = neural_cbs_dds_update(
                x_0_hat,
                dilated_mask,
                downsampled_input_batch,
                downsampled_input_mask,
                lam=args.lam,
                ds=args.grad_step,
                cbs_num_epoch=args.grad_epochs,
            )

            # 保存 after DDS
            save_denorm_center_image(
                current,
                os.path.join(args.output_dir, f"step_{step_id:02d}_after_dds.png"),
                title=f"After DDS step {step_id}",
                denormalize_fn=denormalize,
                file_dict=file,
            )

            history.append({
                "stage": f"step_{step_id}",
                "t": float(t_value),
                "meas_loss": meas_loss,
                "prior_mse": prior_mse,
                "mean": float(current.mean().detach().cpu()),
                "std": float(current.std().detach().cpu()),
            })

        # 最后再做一次轻微去噪
        with torch.no_grad():
            t_end = ts_value[-1] * torch.ones(current.shape[0], device=current.device)
            t_end_i = (torch.floor(t_end * sde.N) + 1) / sde.N
            z = torch.randn_like(current)
            mean, std, coef = sde.marginal_prob(current, t_end_i)
            perturbed_data = mean + std[:, None, None, None] * z
            perturbed_score = score_fn(perturbed_data, t_end_i)
            x_0_hat_tweedie = (
                perturbed_data + perturbed_score * std[:, None, None, None] ** 2
            ) / coef[:, None, None, None]
            current = (
                coef[:, None, None, None] * x_0_hat_tweedie
                + (1 - coef)[:, None, None, None] * current
                if mrsde_sampling else x_0_hat_tweedie
            )
            current = torch.clamp(current, -1.0, 1.0)

        save_denorm_center_image(
            current,
            os.path.join(args.output_dir, "final_denoise.png"),
            title="Final denoise",
            denormalize_fn=denormalize,
            file_dict=file,
        )

        history.append({
            "stage": "final_denoise",
            "mean": float(current.mean().detach().cpu()),
            "std": float(current.std().detach().cpu()),
        })

        return current, history

    ts_values = np.linspace(0.99, 0.01, args.num_steps).tolist()

    print("=" * 80)
    print("[3] 开始 DDS smoke test")
    print("=" * 80)
    print("ts_values =", [round(v, 4) for v in ts_values])

    # 保留和你现在 DPS / DDS smoke test 一样的零初始化，方便对比
    pred_init = torch.zeros_like(output_batch, device=device)

    result, history = dds_sampling(
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

    save_image(
        center_pred,
        os.path.join(args.output_dir, "result_center.png"),
        title="DDS smoke result",
        cmap="inferno",
        vmin=file["min_output"],
        vmax=file["max_output"],
    )
    save_image(
        center_gt,
        os.path.join(args.output_dir, "gt_center.png"),
        title="GT center",
        cmap="inferno",
        vmin=file["min_output"],
        vmax=file["max_output"],
    )
    save_image(
        center_mask.astype(np.float32),
        os.path.join(args.output_dir, "mask_center.png"),
        title="Dilated mask",
        cmap="gray",
    )

    meas_loss_list = [item["meas_loss"] for item in history if "meas_loss" in item]
    prior_mse_list = [item["prior_mse"] for item in history if "prior_mse" in item]

    if len(meas_loss_list) > 0:
        save_curve(
            meas_loss_list,
            os.path.join(args.output_dir, "curve_meas_loss.png"),
            title="DDS measurement loss",
            ylabel="meas loss",
        )
    if len(prior_mse_list) > 0:
        save_curve(
            prior_mse_list,
            os.path.join(args.output_dir, "curve_prior_mse.png"),
            title="DDS prior tether MSE",
            ylabel="prior mse",
        )

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

    with open(os.path.join(args.output_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"measurement_mode={args.measurement_mode}\n")
        f.write(f"stage={args.stage}\n")
        f.write(f"sample_index={args.sample_index}\n")
        f.write(f"noise_index={args.noise_index}\n")
        f.write(f"num_steps={args.num_steps}\n")
        f.write(f"grad_step={args.grad_step}\n")
        f.write(f"grad_epochs={args.grad_epochs}\n")
        f.write(f"lam={args.lam}\n")
        f.write(f"cbs_iters={args.cbs_iters}\n")
        if len(meas_loss_list) > 0:
            f.write(f"last_meas_loss={meas_loss_list[-1]:.8f}\n")
        if len(prior_mse_list) > 0:
            f.write(f"last_prior_mse={prior_mse_list[-1]:.8f}\n")

    print("=" * 80)
    print("[完成]")
    print("=" * 80)
    print("已保存输出到:", os.path.abspath(args.output_dir))
    print("本次额外保存了 Tweedie / after DDS 的逐步图像。")


if __name__ == "__main__":
    main()