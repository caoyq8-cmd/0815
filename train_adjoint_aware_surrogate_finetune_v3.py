#!/usr/bin/env python3
"""
Adjoint-aware fine-tuning for the existing residual measurement surrogate.

Starting point:
    pairwise-delta pretrained DobsMatrixCNN.

Fine-tuning objective:
    L_total = L_forward + lambda_grad * L_gradient_direction

where L_forward preserves the original pointwise measurement prediction and
L_gradient_direction matches the neural inverse-loss input gradient to a cached
CBS adjoint teacher gradient.

Important:
    Computing grad(neural inverse loss, speed, create_graph=True) and then
    differentiating the gradient-alignment loss w.r.t. network parameters requires
    SECOND-ORDER autodiff. Use batch_size=1 first and no AMP.

The gradient teacher is scale-normalized per sample, so legacy CBS source/adjoint
scaling constants do not affect the supervision.
"""

import os
import csv
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from train_dobs_matrix_no import DobsMatrixCNN


def complex_rrmse_np(pred, target):
    num = np.sqrt(np.mean(np.abs(pred - target) ** 2))
    den = np.sqrt(np.mean(np.abs(target) ** 2)) + 1e-12
    return float(num / den)


def make_coord_maps(size, device, batch_size=1):
    yy = torch.linspace(-1.0, 1.0, size, device=device).view(1, 1, size, 1)
    xx = torch.linspace(-1.0, 1.0, size, device=device).view(1, 1, 1, size)
    y = yy.repeat(batch_size, 1, 1, size)
    x = xx.repeat(batch_size, 1, size, 1)
    return y, x


class AdjointTeacherDataset(Dataset):
    def __init__(
        self,
        cache_root,
        mean_dobs,
        teacher_mode="raw",
        residual_scale=0.02,
        max_files=-1,
    ):
        self.root = Path(cache_root)
        self.files = sorted(self.root.glob("*_adjoint_teacher.npz"))
        if max_files > 0:
            self.files = self.files[:max_files]
        if not self.files:
            raise RuntimeError(f"No teacher cache files found in {cache_root}")

        self.mean_dobs = mean_dobs.astype(np.complex64)
        self.teacher_key = (
            "teacher_grad_raw_240"
            if teacher_mode == "raw"
            else "teacher_grad_smooth_240"
        )
        self.residual_scale = float(residual_scale)

        print(
            f"[AdjointTeacherDataset] root={cache_root}, files={len(self.files)}, "
            f"teacher={self.teacher_key}"
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        z = np.load(p, allow_pickle=True)

        speed = z["speed_240"].astype(np.float32)
        point_dobs = z["point_dobs_original"].astype(np.complex64)
        target_gt = z["target_gt_dobs"].astype(np.complex64)
        teacher = z[self.teacher_key].astype(np.float32)

        point_residual = point_dobs - self.mean_dobs
        point_target_scaled = np.stack(
            [
                point_residual.real / self.residual_scale,
                point_residual.imag / self.residual_scale,
            ],
            axis=0,
        ).astype(np.float32)

        return {
            "speed": torch.from_numpy(speed[None]),
            "point_target_scaled": torch.from_numpy(point_target_scaled),
            "point_dobs_real": torch.from_numpy(point_dobs.real.astype(np.float32)),
            "point_dobs_imag": torch.from_numpy(point_dobs.imag.astype(np.float32)),
            "target_gt_real": torch.from_numpy(target_gt.real.astype(np.float32)),
            "target_gt_imag": torch.from_numpy(target_gt.imag.astype(np.float32)),
            "teacher_grad": torch.from_numpy(teacher[None]),
            "alpha": float(z["alpha"]),
            "base_sample": str(z["base_sample"]),
            "file": str(p),
        }


def mixed_complex_loss(pred_scaled, target_scaled, lambda_l1=1.0, lambda_mse=1.0):
    dr = pred_scaled[:, 0] - target_scaled[:, 0]
    di = pred_scaled[:, 1] - target_scaled[:, 1]
    abs_err = torch.sqrt(dr ** 2 + di ** 2 + 1e-12)
    loss_l1 = abs_err.mean()
    loss_mse = (dr ** 2 + di ** 2).mean()
    return lambda_l1 * loss_l1 + lambda_mse * loss_mse


def normalized_gradient_cosine(g_pred, g_teacher, eps=1e-12):
    # Per-sample cosine over HxW.
    gp = g_pred.flatten(1)
    gt = g_teacher.flatten(1)
    gp = gp / (torch.linalg.vector_norm(gp, dim=1, keepdim=True) + eps)
    gt = gt / (torch.linalg.vector_norm(gt, dim=1, keepdim=True) + eps)
    return torch.sum(gp * gt, dim=1)


def construct_prediction(
    model,
    speed,
    mean_real,
    mean_imag,
    residual_scale,
    speed_center,
    speed_scale,
    y_map,
    x_map,
):
    speed_norm = (speed - speed_center) / speed_scale
    inp = torch.cat([speed_norm, y_map, x_map], dim=1)
    pred_scaled = model(inp)
    pred_real = mean_real + pred_scaled[:, 0] * residual_scale
    pred_imag = mean_imag + pred_scaled[:, 1] * residual_scale
    return pred_scaled, pred_real, pred_imag


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    mean_real_scalar,
    mean_imag_scalar,
    residual_scale,
    speed_center,
    speed_scale,
):
    model.eval()
    rrmses = []
    grad_cosines = []

    for batch in loader:
        # Forward metric first under no_grad.
        speed0 = batch["speed"].to(device)
        bsz = speed0.shape[0]
        y_map, x_map = make_coord_maps(speed0.shape[-1], device, bsz)

        mean_real = mean_real_scalar.expand(bsz, -1, -1)
        mean_imag = mean_imag_scalar.expand(bsz, -1, -1)

        pred_scaled, pred_real, pred_imag = construct_prediction(
            model, speed0, mean_real, mean_imag, residual_scale,
            speed_center, speed_scale, y_map, x_map
        )

        pred_np = (
            pred_real.cpu().numpy() + 1j * pred_imag.cpu().numpy()
        ).astype(np.complex64)
        target_np = (
            batch["point_dobs_real"].numpy()
            + 1j * batch["point_dobs_imag"].numpy()
        ).astype(np.complex64)
        for i in range(bsz):
            rrmses.append(complex_rrmse_np(pred_np[i], target_np[i]))

    # Need gradients, so a second pass with grad enabled.
    model.eval()
    with torch.enable_grad():
        for batch in loader:
            speed = batch["speed"].to(device).clone().detach().requires_grad_(True)
            teacher = batch["teacher_grad"].to(device)
            gt_real = batch["target_gt_real"].to(device)
            gt_imag = batch["target_gt_imag"].to(device)

            bsz = speed.shape[0]
            y_map, x_map = make_coord_maps(speed.shape[-1], device, bsz)
            mean_real = mean_real_scalar.expand(bsz, -1, -1)
            mean_imag = mean_imag_scalar.expand(bsz, -1, -1)

            _, pred_real, pred_imag = construct_prediction(
                model, speed, mean_real, mean_imag, residual_scale,
                speed_center, speed_scale, y_map, x_map
            )

            dr = pred_real - gt_real
            di = pred_imag - gt_imag
            inv_loss = (dr ** 2 + di ** 2).mean()

            g = torch.autograd.grad(
                inv_loss, speed, create_graph=False, retain_graph=False
            )[0]
            cos = normalized_gradient_cosine(g, teacher)
            grad_cosines.extend(cos.detach().cpu().numpy().tolist())

    return {
        "forward_rrmse_mean": float(np.mean(rrmses)),
        "forward_rrmse_std": float(np.std(rrmses)),
        "grad_cos_mean": float(np.mean(grad_cosines)),
        "grad_cos_std": float(np.std(grad_cosines)),
        "num_eval": len(rrmses),
    }


def save_checkpoint(path, model, epoch, init_args, ft_args, metrics):
    args_for_loader = dict(init_args)
    args_for_loader.update({
        "adjoint_aware_finetune": True,
        "teacher_mode": ft_args.teacher_mode,
        "lambda_grad": ft_args.lambda_grad,
        "finetune_lr": ft_args.lr,
        "finetune_epoch": epoch,
    })
    ckpt = {
        "model": model.state_dict(),
        "epoch": epoch,
        "args": args_for_loader,
        "finetune_args": vars(ft_args),
        "metrics": metrics,
    }
    torch.save(ckpt, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init_ckpt", type=str, required=True)
    ap.add_argument("--train_cache_root", type=str, required=True)
    ap.add_argument("--val_cache_root", type=str, required=True)
    ap.add_argument(
        "--val_local_cache_root",
        type=str,
        default="",
        help="Optional local-alpha validation cache for domain-specific reporting.",
    )
    ap.add_argument(
        "--val_offpath_cache_root",
        type=str,
        default="",
        help="Optional off-path validation cache for domain-specific reporting.",
    )
    ap.add_argument("--mean_dobs_path", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)

    ap.add_argument("--teacher_mode", choices=["raw", "smooth"], default="raw")
    ap.add_argument("--lambda_grad", type=float, default=0.1)
    ap.add_argument("--lambda_forward", type=float, default=1.0)
    ap.add_argument("--lambda_l1", type=float, default=1.0)
    ap.add_argument("--lambda_mse", type=float, default=1.0)

    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--eval_batch_size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--max_train_files", type=int, default=-1)
    ap.add_argument("--max_val_files", type=int, default=-1)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    out = Path(args.output_dir)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    init_ckpt = torch.load(args.init_ckpt, map_location=device)
    init_args = init_ckpt["args"]

    model = DobsMatrixCNN(
        in_ch=3,
        base_ch=init_args.get("base_ch", 32),
        num_sources=64,
        num_receivers=64,
        spatial_pool=init_args.get("spatial_pool", 8),
        head_dim=init_args.get("head_dim", 1024),
    ).to(device)
    model.load_state_dict(init_ckpt["model"], strict=True)

    image_size = int(init_args.get("image_size", 240))
    residual_scale = float(init_args.get("residual_scale", 0.02))
    speed_center = float(init_args.get("speed_center", 1500.0))
    speed_scale = float(init_args.get("speed_scale", 100.0))

    mean_dobs = np.load(args.mean_dobs_path)["mean_dobs"].astype(np.complex64)
    mean_real_scalar = torch.from_numpy(mean_dobs.real.astype(np.float32)).to(device)
    mean_imag_scalar = torch.from_numpy(mean_dobs.imag.astype(np.float32)).to(device)

    train_ds = AdjointTeacherDataset(
        args.train_cache_root,
        mean_dobs,
        teacher_mode=args.teacher_mode,
        residual_scale=residual_scale,
        max_files=args.max_train_files,
    )
    val_ds = AdjointTeacherDataset(
        args.val_cache_root,
        mean_dobs,
        teacher_mode=args.teacher_mode,
        residual_scale=residual_scale,
        max_files=args.max_val_files,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    val_local_loader = None
    if args.val_local_cache_root:
        val_local_ds = AdjointTeacherDataset(
            args.val_local_cache_root,
            mean_dobs,
            teacher_mode=args.teacher_mode,
            residual_scale=residual_scale,
            max_files=-1,
        )
        val_local_loader = DataLoader(
            val_local_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    val_offpath_loader = None
    if args.val_offpath_cache_root:
        val_offpath_ds = AdjointTeacherDataset(
            args.val_offpath_cache_root,
            mean_dobs,
            teacher_mode=args.teacher_mode,
            residual_scale=residual_scale,
            max_files=-1,
        )
        val_offpath_loader = DataLoader(
            val_offpath_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    baseline = evaluate(
        model, val_loader, device,
        mean_real_scalar, mean_imag_scalar,
        residual_scale, speed_center, speed_scale,
    )
    print("=" * 100)
    print("[Before fine-tune | combined validation]")
    print(json.dumps(baseline, indent=2))

    baseline_local = None
    if val_local_loader is not None:
        baseline_local = evaluate(
            model, val_local_loader, device,
            mean_real_scalar, mean_imag_scalar,
            residual_scale, speed_center, speed_scale,
        )
        print("[Before fine-tune | local-alpha validation]")
        print(json.dumps(baseline_local, indent=2))

    baseline_offpath = None
    if val_offpath_loader is not None:
        baseline_offpath = evaluate(
            model, val_offpath_loader, device,
            mean_real_scalar, mean_imag_scalar,
            residual_scale, speed_center, speed_scale,
        )
        print("[Before fine-tune | off-path validation]")
        print(json.dumps(baseline_offpath, indent=2))

    print("=" * 100)

    history = []
    best_grad_cos = baseline["grad_cos_mean"]

    save_checkpoint(
        out / "checkpoints" / "initial.pth",
        model, 0, init_args, args, baseline
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        point_losses = []
        grad_losses = []
        cosines = []

        for step, batch in enumerate(train_loader, start=1):
            speed = batch["speed"].to(device).clone().detach().requires_grad_(True)
            point_target_scaled = batch["point_target_scaled"].to(device)
            gt_real = batch["target_gt_real"].to(device)
            gt_imag = batch["target_gt_imag"].to(device)
            teacher = batch["teacher_grad"].to(device)

            bsz = speed.shape[0]
            y_map, x_map = make_coord_maps(image_size, device, bsz)
            mean_real = mean_real_scalar.expand(bsz, -1, -1)
            mean_imag = mean_imag_scalar.expand(bsz, -1, -1)

            optimizer.zero_grad(set_to_none=True)

            pred_scaled, pred_real, pred_imag = construct_prediction(
                model, speed, mean_real, mean_imag,
                residual_scale, speed_center, speed_scale,
                y_map, x_map
            )

            point_loss = mixed_complex_loss(
                pred_scaled,
                point_target_scaled,
                lambda_l1=args.lambda_l1,
                lambda_mse=args.lambda_mse,
            )

            dr = pred_real - gt_real
            di = pred_imag - gt_imag
            inv_loss = (dr ** 2 + di ** 2).mean()

            # This create_graph=True is the key adjoint-aware second-order step.
            g_nn = torch.autograd.grad(
                inv_loss,
                speed,
                create_graph=True,
                retain_graph=True,
            )[0]

            cos = normalized_gradient_cosine(g_nn, teacher)
            grad_loss = (1.0 - cos).mean()

            loss = (
                args.lambda_forward * point_loss
                + args.lambda_grad * grad_loss
            )

            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            losses.append(float(loss.detach().cpu()))
            point_losses.append(float(point_loss.detach().cpu()))
            grad_losses.append(float(grad_loss.detach().cpu()))
            cosines.extend(cos.detach().cpu().numpy().tolist())

            if step % args.log_every == 0 or step == 1:
                print(
                    f"Epoch {epoch:03d} | step {step:04d}/{len(train_loader):04d} | "
                    f"loss={np.mean(losses):.6f} | "
                    f"point={np.mean(point_losses):.6f} | "
                    f"grad={np.mean(grad_losses):.6f} | "
                    f"cos={np.mean(cosines):.4f}"
                )

        metrics = evaluate(
            model, val_loader, device,
            mean_real_scalar, mean_imag_scalar,
            residual_scale, speed_center, speed_scale,
        )
        local_metrics = None
        if val_local_loader is not None:
            local_metrics = evaluate(
                model, val_local_loader, device,
                mean_real_scalar, mean_imag_scalar,
                residual_scale, speed_center, speed_scale,
            )

        offpath_metrics = None
        if val_offpath_loader is not None:
            offpath_metrics = evaluate(
                model, val_offpath_loader, device,
                mean_real_scalar, mean_imag_scalar,
                residual_scale, speed_center, speed_scale,
            )

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_point_loss": float(np.mean(point_losses)),
            "train_grad_loss": float(np.mean(grad_losses)),
            "train_grad_cos": float(np.mean(cosines)),
        }
        row.update(metrics)

        if local_metrics is not None:
            for k, v in local_metrics.items():
                row[f"local_{k}"] = v

        if offpath_metrics is not None:
            for k, v in offpath_metrics.items():
                row[f"offpath_{k}"] = v

        history.append(row)

        print("=" * 100)
        msg = (
            f"Epoch {epoch:03d} | "
            f"train_cos={row['train_grad_cos']:.4f} | "
            f"combined_cos={metrics['grad_cos_mean']:.4f} | "
            f"combined_fwd_rrmse={metrics['forward_rrmse_mean']:.5f}"
        )
        print(msg)
        if local_metrics is not None:
            print(
                f"          local   | cos={local_metrics['grad_cos_mean']:.4f} | "
                f"fwd_rrmse={local_metrics['forward_rrmse_mean']:.5f}"
            )
        if offpath_metrics is not None:
            print(
                f"          offpath | cos={offpath_metrics['grad_cos_mean']:.4f} | "
                f"fwd_rrmse={offpath_metrics['forward_rrmse_mean']:.5f}"
            )
        print("=" * 100)

        with open(out / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        # Save every epoch so Pareto model selection can be done on the validation set
        # without losing intermediate forward/gradient trade-off checkpoints.
        save_checkpoint(
            out / "checkpoints" / f"epoch_{epoch:03d}.pth",
            model, epoch, init_args, args, metrics
        )

        save_checkpoint(
            out / "checkpoints" / "latest.pth",
            model, epoch, init_args, args, metrics
        )

        if metrics["grad_cos_mean"] > best_grad_cos:
            best_grad_cos = metrics["grad_cos_mean"]
            save_checkpoint(
                out / "checkpoints" / "best_gradcos.pth",
                model, epoch, init_args, args, metrics
            )
            with open(out / "best_metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            print(
                f"[Saved best_gradcos] epoch={epoch}, "
                f"val_cos={best_grad_cos:.6f}, "
                f"val_rrmse={metrics['forward_rrmse_mean']:.6f}"
            )

    print("[DONE]")
    print("best val grad cosine =", best_grad_cos)
    print("output =", out.resolve())


if __name__ == "__main__":
    main()
