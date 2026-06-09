import argparse
import json
import math
import os
from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from diffano_neural_adjoint import NeuralAdjointOperator, neural_adjoint_gradient
from utils_cbs import normalize as cbs_normalize
from utils_cbs import score_denormalize


@dataclass
class SmokeConfig:
    ckpt_path: str = "diffano_runs/neural_adjoint_sparse/best.pth"
    dobs_path: str = "Datasets_test/AI4Scup2_simulated_CBS_sparse/dobs_500k/eval/train_6601.npy"
    speed_path: str = "Datasets_test/AI4Scup2_simulated_CBS_sparse/speed/eval/train_6601.npy"
    output_dir: str = "diffano_runs/neural_adjoint_smoke"
    num_iters: int = 300
    lr: float = 0.03
    tv_weight: float = 1e-4
    clamp_min: float = -1.0
    clamp_max: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def image_tv(x: torch.Tensor) -> torch.Tensor:
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return dx.abs().mean() + dy.abs().mean()


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 2.0) -> float:
    mse = F.mse_loss(pred, target).item()
    if mse <= 1e-12:
        return 99.0
    return 20.0 * math.log10(data_range / math.sqrt(mse))


def save_map(x_norm: torch.Tensor, path: str, title: str) -> None:
    x_speed = score_denormalize(x_norm.detach().cpu())
    img = x_speed[0, 0].numpy()
    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap="inferno", vmin=1408.692, vmax=1595.1279)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def parse_args() -> SmokeConfig:
    cfg = SmokeConfig()
    parser = argparse.ArgumentParser(description="Run DIFF-ANO neural adjoint reconstruction smoke test.")
    for name, value in asdict(cfg).items():
        parser.add_argument(f"--{name}", type=type(value), default=value)
    return SmokeConfig(**vars(parser.parse_args()))


def main() -> None:
    cfg = parse_args()
    ensure_dir(cfg.output_dir)

    ckpt = torch.load(cfg.ckpt_path, map_location=cfg.device, weights_only=False)
    model_cfg = ckpt["config"]
    model = NeuralAdjointOperator(
        measurement_shape=tuple(ckpt["measurement_shape"]),
        width=model_cfg["width"],
        modes=model_cfg["modes"],
        depth=model_cfg["depth"],
        dropout=model_cfg["dropout"],
    ).to(cfg.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dobs = np.load(cfg.dobs_path)
    target = np.stack([dobs.real, dobs.imag], axis=0).astype(np.float32)
    target_t = torch.tensor(target, dtype=torch.float32, device=cfg.device).unsqueeze(0)
    target_t = target_t / float(ckpt["dobs_scale"])

    speed_gt = np.load(cfg.speed_path)
    gt_t = torch.tensor(speed_gt, dtype=torch.float32).view(1, 1, *speed_gt.shape)
    gt_norm = cbs_normalize(gt_t).to(cfg.device)

    x = torch.zeros_like(gt_norm, device=cfg.device, requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=cfg.lr)

    history = []
    for step in range(1, cfg.num_iters + 1):
        pred = model(x)
        meas_loss = F.mse_loss(pred, target_t)
        tv_loss = image_tv(x)
        loss = meas_loss + cfg.tv_weight * tv_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            x.clamp_(cfg.clamp_min, cfg.clamp_max)

        if step % 10 == 0 or step == 1:
            recon_psnr = psnr(x.detach(), gt_norm)
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "meas_loss": float(meas_loss.detach().cpu()),
                "tv_loss": float(tv_loss.detach().cpu()),
                "psnr_norm": recon_psnr,
            }
            history.append(row)
            print(
                f"[{step:04d}/{cfg.num_iters}] "
                f"loss={row['loss']:.6e} meas={row['meas_loss']:.6e} "
                f"psnr_norm={recon_psnr:.2f}"
            )

    grad, final_loss = neural_adjoint_gradient(model, x.detach(), target_t)
    torch.save(
        {
            "recon_norm": x.detach().cpu(),
            "gt_norm": gt_norm.detach().cpu(),
            "adjoint_grad": grad.detach().cpu(),
            "final_loss": float(final_loss.cpu()),
            "history": history,
        },
        os.path.join(cfg.output_dir, "smoke_result.pt"),
    )

    save_map(gt_norm, os.path.join(cfg.output_dir, "gt_speed.png"), "Ground truth")
    save_map(x.detach(), os.path.join(cfg.output_dir, "recon_speed.png"), "Neural adjoint recon")

    plt.figure(figsize=(6, 4))
    plt.plot([r["step"] for r in history], [r["meas_loss"] for r in history])
    plt.xlabel("step")
    plt.ylabel("measurement MSE")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, "curve_measurement_loss.png"), dpi=180)
    plt.close()

    with open(os.path.join(cfg.output_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
