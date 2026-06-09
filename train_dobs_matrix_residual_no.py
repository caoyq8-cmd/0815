import os
import re
import json
import argparse
from pathlib import Path

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from train_dobs_matrix_no import (
    DobsMatrixCNN,
    compute_loss,
    numeric_key,
    complex_rrmse,
    complex_mae,
)


class ResidualDobsMatrixDataset(Dataset):
    def __init__(
        self,
        data_root,
        split="train",
        image_size=240,
        speed_center=1500.0,
        speed_scale=100.0,
        residual_scale=0.02,
        mean_dobs_path="",
        max_files=-1,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.speed_center = speed_center
        self.speed_scale = speed_scale
        self.residual_scale = residual_scale

        self.files = sorted(
            list((self.data_root / split).glob(f"{split}_*.npz")),
            key=numeric_key,
        )

        if max_files > 0:
            self.files = self.files[:max_files]

        if len(self.files) == 0:
            raise RuntimeError(f"No npz files found in {self.data_root / split}")

        if mean_dobs_path == "":
            raise ValueError("Residual learning 需要指定 --mean_dobs_path")

        m = np.load(mean_dobs_path)
        self.mean_dobs = m["mean_dobs"].astype(np.complex64)

        first = np.load(self.files[0])
        self.num_sources, self.num_receivers = first["dobs_complex"].shape

        assert self.mean_dobs.shape == (self.num_sources, self.num_receivers), (
            self.mean_dobs.shape,
            (self.num_sources, self.num_receivers),
        )

        h = image_size
        w = image_size

        yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
        xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]

        self.y_map = np.repeat(yy, w, axis=1)[None]
        self.x_map = np.repeat(xx, h, axis=0)[None]

        print(
            f"[ResidualDataset] split={split}, files={len(self.files)}, "
            f"image_size={image_size}, dobs_shape=({self.num_sources},{self.num_receivers}), "
            f"residual_scale={residual_scale}"
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx])

        speed = d["target_480"].astype(np.float32)
        dobs = d["dobs_complex"].astype(np.complex64)

        residual = dobs - self.mean_dobs

        speed_t = torch.from_numpy(speed[None])
        speed_t = (speed_t - self.speed_center) / self.speed_scale

        if self.image_size != 480:
            speed_t = F.interpolate(
                speed_t[None],
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )[0]

        inp = np.concatenate(
            [
                speed_t.numpy().astype(np.float32),
                self.y_map.astype(np.float32),
                self.x_map.astype(np.float32),
            ],
            axis=0,
        )

        target_residual = np.stack(
            [
                residual.real.astype(np.float32) / self.residual_scale,
                residual.imag.astype(np.float32) / self.residual_scale,
            ],
            axis=0,
        )

        return {
            "input": torch.from_numpy(inp.astype(np.float32)),
            "target": torch.from_numpy(target_residual.astype(np.float32)),
        }


@torch.no_grad()
def evaluate(model, loader, args, device):
    model.eval()

    losses = []
    dobs_rrmse_list = []
    dobs_mae_list = []
    residual_rrmse_list = []
    residual_mae_list = []
    mean_rrmse_list = []
    mean_mae_list = []

    mean_dobs = loader.dataset.mean_dobs.astype(np.complex64)

    for batch in loader:
        x = batch["input"].to(device)
        y_res = batch["target"].to(device)

        pred_res = model(x)

        loss, loss_items = compute_loss(
            pred_res,
            y_res,
            lambda_l1=args.lambda_l1,
            lambda_mse=args.lambda_mse,
        )
        losses.append(loss_items["loss"])

        pred_np = pred_res.detach().cpu().numpy()
        y_np = y_res.detach().cpu().numpy()

        for i in range(pred_np.shape[0]):
            pred_res_c = (
                pred_np[i, 0] + 1j * pred_np[i, 1]
            ) * args.residual_scale

            true_res_c = (
                y_np[i, 0] + 1j * y_np[i, 1]
            ) * args.residual_scale

            pred_dobs = mean_dobs + pred_res_c
            true_dobs = mean_dobs + true_res_c

            # 原始 dobs 指标：这是最重要的
            dobs_rrmse_list.append(complex_rrmse(pred_dobs, true_dobs))
            dobs_mae_list.append(complex_mae(pred_dobs, true_dobs))

            # residual 指标：判断 residual 是否真正学到了
            residual_rrmse_list.append(complex_rrmse(pred_res_c, true_res_c))
            residual_mae_list.append(complex_mae(pred_res_c, true_res_c))

            # mean baseline：pred_residual = 0
            mean_rrmse_list.append(complex_rrmse(mean_dobs, true_dobs))
            mean_mae_list.append(complex_mae(mean_dobs, true_dobs))

    model.train()

    return {
        "loss": float(np.mean(losses)),

        "dobs_rrmse": float(np.mean(dobs_rrmse_list)),
        "dobs_rrmse_std": float(np.std(dobs_rrmse_list)),
        "dobs_mae": float(np.mean(dobs_mae_list)),

        "residual_rrmse": float(np.mean(residual_rrmse_list)),
        "residual_rrmse_std": float(np.std(residual_rrmse_list)),
        "residual_mae": float(np.mean(residual_mae_list)),

        "mean_baseline_dobs_rrmse": float(np.mean(mean_rrmse_list)),
        "mean_baseline_dobs_rrmse_std": float(np.std(mean_rrmse_list)),
        "mean_baseline_dobs_mae": float(np.mean(mean_mae_list)),

        "improve_over_mean_rrmse": float(
            np.mean(mean_rrmse_list) - np.mean(dobs_rrmse_list)
        ),
        "relative_improve_over_mean": float(
            (np.mean(mean_rrmse_list) - np.mean(dobs_rrmse_list))
            / (np.mean(mean_rrmse_list) + 1e-12)
        ),
    }


def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("device =", device)

    train_set = ResidualDobsMatrixDataset(
        data_root=args.data_root,
        split="train",
        image_size=args.image_size,
        speed_center=args.speed_center,
        speed_scale=args.speed_scale,
        residual_scale=args.residual_scale,
        mean_dobs_path=args.mean_dobs_path,
        max_files=args.max_train_files,
    )

    val_set = ResidualDobsMatrixDataset(
        data_root=args.data_root,
        split="test",
        image_size=args.image_size,
        speed_center=args.speed_center,
        speed_scale=args.speed_scale,
        residual_scale=args.residual_scale,
        mean_dobs_path=args.mean_dobs_path,
        max_files=args.max_test_files,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    model = DobsMatrixCNN(
        in_ch=3,
        base_ch=args.base_ch,
        num_sources=train_set.num_sources,
        num_receivers=train_set.num_receivers,
        spatial_pool=args.spatial_pool,
        head_dim=args.head_dim,
    ).to(device)

    print(f"model parameters = {sum(p.numel() for p in model.parameters()) / 1e6:.3f} M")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    best_metric = float("inf")
    best_epoch = -1

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_dobs_rrmse": [],
        "val_residual_rrmse": [],
        "val_mean_baseline_dobs_rrmse": [],
        "relative_improve_over_mean": [],
    }

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for step, batch in enumerate(train_loader, start=1):
            x = batch["input"].to(device, non_blocking=True)
            y = batch["target"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.use_amp):
                pred = model(x)
                loss, loss_items = compute_loss(
                    pred,
                    y,
                    lambda_l1=args.lambda_l1,
                    lambda_mse=args.lambda_mse,
                )

            scaler.scale(loss).backward()

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            train_losses.append(float(loss.detach().cpu()))

            if step % args.log_every == 0:
                print(
                    f"Epoch [{epoch:03d}/{args.epochs:03d}] "
                    f"Step [{step:04d}/{len(train_loader):04d}] "
                    f"loss={np.mean(train_losses[-args.log_every:]):.6f}"
                )

        scheduler.step()

        train_loss = float(np.mean(train_losses))
        val = evaluate(model, val_loader, args, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val["loss"])
        history["val_dobs_rrmse"].append(val["dobs_rrmse"])
        history["val_residual_rrmse"].append(val["residual_rrmse"])
        history["val_mean_baseline_dobs_rrmse"].append(val["mean_baseline_dobs_rrmse"])
        history["relative_improve_over_mean"].append(val["relative_improve_over_mean"])

        print("=" * 80)
        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val['loss']:.6f} "
            f"dobs_rrmse={val['dobs_rrmse']:.6f} "
            f"mean_base={val['mean_baseline_dobs_rrmse']:.6f} "
            f"rel_improve={val['relative_improve_over_mean']:.4%} "
            f"residual_rrmse={val['residual_rrmse']:.6f}"
        )
        print("=" * 80)

        with open(os.path.join(args.output_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        ckpt = {
            "model": model.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "val": val,
            "best_metric": best_metric,
        }

        torch.save(ckpt, os.path.join(args.output_dir, "checkpoints", "latest.pth"))

        if val["dobs_rrmse"] < best_metric:
            best_metric = val["dobs_rrmse"]
            best_epoch = epoch

            torch.save(ckpt, os.path.join(args.output_dir, "checkpoints", "best.pth"))

            with open(os.path.join(args.output_dir, "best_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "best_epoch": best_epoch,
                        "best_dobs_rrmse": best_metric,
                        "val": val,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            print(f"Saved best checkpoint at epoch {epoch}, dobs_rrmse={best_metric:.6f}")

    print("[Done]")
    print("best_epoch =", best_epoch)
    print("best_dobs_rrmse =", best_metric)


@torch.no_grad()
def eval_only(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("device =", device)

    ckpt = torch.load(args.ckpt_path, map_location=device)
    ckpt_args = ckpt["args"]

    args.residual_scale = ckpt_args.get("residual_scale", args.residual_scale)
    args.lambda_l1 = ckpt_args.get("lambda_l1", args.lambda_l1)
    args.lambda_mse = ckpt_args.get("lambda_mse", args.lambda_mse)
    args.mean_dobs_path = ckpt_args.get("mean_dobs_path", args.mean_dobs_path)

    test_set = ResidualDobsMatrixDataset(
        data_root=args.data_root,
        split=args.eval_split,
        image_size=ckpt_args.get("image_size", args.image_size),
        speed_center=ckpt_args.get("speed_center", args.speed_center),
        speed_scale=ckpt_args.get("speed_scale", args.speed_scale),
        residual_scale=ckpt_args.get("residual_scale", args.residual_scale),
        mean_dobs_path=ckpt_args.get("mean_dobs_path", args.mean_dobs_path),
        max_files=args.max_test_files,
    )

    loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    model = DobsMatrixCNN(
        in_ch=3,
        base_ch=ckpt_args.get("base_ch", args.base_ch),
        num_sources=test_set.num_sources,
        num_receivers=test_set.num_receivers,
        spatial_pool=ckpt_args.get("spatial_pool", args.spatial_pool),
        head_dim=ckpt_args.get("head_dim", args.head_dim),
    ).to(device)

    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    val = evaluate(model, loader, args, device)

    with open(os.path.join(args.output_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(val, f, indent=2, ensure_ascii=False)

    print(json.dumps(val, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, default="")
    parser.add_argument("--eval_split", type=str, default="test")

    parser.add_argument("--mean_dobs_path", type=str, required=True)
    parser.add_argument("--residual_scale", type=float, default=0.02)

    parser.add_argument("--image_size", type=int, default=240)
    parser.add_argument("--speed_center", type=float, default=1500.0)
    parser.add_argument("--speed_scale", type=float, default=100.0)

    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--spatial_pool", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=1024)

    parser.add_argument("--max_train_files", type=int, default=-1)
    parser.add_argument("--max_test_files", type=int, default=-1)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--lambda_l1", type=float, default=1.0)
    parser.add_argument("--lambda_mse", type=float, default=1.0)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--log_every", type=int, default=10)

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    else:
        if args.ckpt_path == "":
            raise ValueError("--mode eval 需要指定 --ckpt_path")
        eval_only(args)


if __name__ == "__main__":
    main()