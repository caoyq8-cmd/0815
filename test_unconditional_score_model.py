import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt


def add_repo_to_path(repo_root: str):
    repo_root = os.path.abspath(repo_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def resolve_device(device_arg: str):
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


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


def make_grid(images, nrow=4):
    """
    images: [B, H, W]
    """
    B, H, W = images.shape
    ncol = nrow
    nrows = int(np.ceil(B / ncol))
    canvas = np.zeros((nrows * H, ncol * W), dtype=images.dtype)
    for i in range(B):
        r = i // ncol
        c = i % ncol
        canvas[r * H:(r + 1) * H, c * W:(c + 1) * W] = images[i]
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=str, default=".")
    parser.add_argument("--checkpoint", type=str, default="./mini_score_ckpt_sparse_fixednorm_e100/checkpoint_best.pth")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output_dir", type=str, default="./uncond_score_test_outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    add_repo_to_path(args.repo_root)

    from utils_cbs import denormalize
    import sde_lib
    from models import ddpm as ddpm_model
    from models import utils as mutils
    from configs.vp import AI4Scup2_ddpm_continuous as configs

    device = resolve_device(args.device)

    print("=" * 80)
    print("[Unconditional Score Model Test]")
    print("=" * 80)
    print("repo_root   =", args.repo_root)
    print("checkpoint  =", args.checkpoint)
    print("device      =", device)
    print("num_samples =", args.num_samples)
    print("num_steps   =", args.num_steps)
    print("output_dir  =", args.output_dir)

    config = configs.get_config()
    config.device_ids = [0]
    config.device = device

    print("=" * 80)
    print("[1] 加载模型")
    print("=" * 80)
    score_model = ddpm_model.DDPM(config).to(device)
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
    score_fn = mutils.get_score_fn(
        sde,
        score_model,
        train=False,
        continuous=config.training.continuous
    )

    # 初始噪声
    x = torch.randn(args.num_samples, 1, 256, 256, device=device)

    ts_values = np.linspace(0.99, 0.01, args.num_steps).tolist()

    print("=" * 80)
    print("[2] 开始 unconditional sampling")
    print("=" * 80)

    step_stats = []

    for step_id, t_value in enumerate(ts_values, start=1):
        print(f"[sampling] step {step_id}/{len(ts_values)}, t={t_value:.4f}")

        t = t_value * torch.ones(x.shape[0], device=device)
        t_i = (torch.floor(t * sde.N) + 1) / sde.N

        with torch.no_grad():
            z = torch.randn_like(x)
            mean, std, coef = sde.marginal_prob(x, t_i)
            perturbed_data = mean + std[:, None, None, None] * z
            perturbed_score = score_fn(perturbed_data, t_i)
            x0_hat_tweedie = (
                perturbed_data + perturbed_score * std[:, None, None, None] ** 2
            ) / coef[:, None, None, None]
            x = torch.clamp(x0_hat_tweedie, -1.0, 1.0)

        step_mean = float(x.mean().detach().cpu())
        step_std = float(x.std().detach().cpu())
        step_stats.append((step_id, t_value, step_mean, step_std))
        print(f"  x mean/std = {step_mean:.6f} / {step_std:.6f}")

        # 保存若干关键步
        if step_id in [1, 2, 5, 10, len(ts_values)]:
            x_denorm = denormalize(x).detach().cpu().numpy()[:, 0, 90:390, 90:390]
            grid = make_grid(x_denorm, nrow=min(args.num_samples, 4))
            save_image(
                grid,
                os.path.join(args.output_dir, f"samples_step_{step_id:02d}.png"),
                title=f"Uncond samples step {step_id}",
                cmap="inferno",
                vmin=1408.692,
                vmax=1595.1279,
            )

    print("=" * 80)
    print("[3] 保存最终结果")
    print("=" * 80)

    x_denorm = denormalize(x).detach().cpu().numpy()
    x_center = x_denorm[:, 0, 90:390, 90:390]
    grid = make_grid(x_center, nrow=min(args.num_samples, 4))

    save_image(
        grid,
        os.path.join(args.output_dir, "uncond_samples_final_grid.png"),
        title="Unconditional samples final",
        cmap="inferno",
        vmin=1408.692,
        vmax=1595.1279,
    )

    np.save(os.path.join(args.output_dir, "uncond_samples_final_norm.npy"), x.detach().cpu().numpy())
    np.save(os.path.join(args.output_dir, "uncond_samples_final_denorm.npy"), x_denorm)

    with open(os.path.join(args.output_dir, "stats.txt"), "w", encoding="utf-8") as f:
        for item in step_stats:
            f.write(f"step={item[0]}, t={item[1]:.6f}, mean={item[2]:.8f}, std={item[3]:.8f}\n")

    print("done.")
    print("saved to:", os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()