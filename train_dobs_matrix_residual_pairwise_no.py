import os
import re
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from train_dobs_matrix_no import DobsMatrixCNN


def numeric_key(path):
    nums = re.findall(r"\d+", str(path))
    return int(nums[-1]) if nums else -1


def resize_np(x, size):
    x_t = torch.from_numpy(x.astype(np.float32))[None, None]
    y_t = F.interpolate(
        x_t,
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )
    return y_t[0, 0].cpu().numpy().astype(np.float32)


def complex_rrmse(pred, target):
    pred = pred.astype(np.complex64)
    target = target.astype(np.complex64)
    num = np.sqrt(np.mean(np.abs(pred - target) ** 2))
    den = np.sqrt(np.mean(np.abs(target) ** 2)) + 1e-12
    return float(num / den)


def complex_mae(pred, target):
    pred = pred.astype(np.complex64)
    target = target.astype(np.complex64)
    return float(np.mean(np.abs(pred - target)))


def make_coord_maps_np(size):
    yy = np.linspace(-1.0, 1.0, size, dtype=np.float32).reshape(1, size, 1)
    xx = np.linspace(-1.0, 1.0, size, dtype=np.float32).reshape(1, 1, size)
    y_map = np.repeat(yy, size, axis=2)
    x_map = np.repeat(xx, size, axis=1)
    return y_map, x_map


def make_input(speed_480, image_size, speed_center, speed_scale, y_map, x_map):
    speed_240 = resize_np(speed_480, image_size)
    speed_norm = (speed_240 - speed_center) / speed_scale
    inp = np.concatenate(
        [
            speed_norm[None].astype(np.float32),
            y_map.astype(np.float32),
            x_map.astype(np.float32),
        ],
        axis=0,
    )
    return inp.astype(np.float32)


def make_scaled_residual_target(dobs, mean_dobs, residual_scale):
    residual = dobs.astype(np.complex64) - mean_dobs.astype(np.complex64)
    target = np.stack(
        [
            residual.real / residual_scale,
            residual.imag / residual_scale,
        ],
        axis=0,
    )
    return target.astype(np.float32)


def mixed_complex_loss(pred, target, lambda_l1=1.0, lambda_mse=1.0):
    """
    pred, target: [B, 2, 64, 64], scaled residual channels.
    """
    dr = pred[:, 0] - target[:, 0]
    di = pred[:, 1] - target[:, 1]
    abs_err = torch.sqrt(dr ** 2 + di ** 2 + 1e-12)

    loss_l1 = abs_err.mean()
    loss_mse = (dr ** 2 + di ** 2).mean()

    return lambda_l1 * loss_l1 + lambda_mse * loss_mse


class LocalAlphaPointDataset(Dataset):
    """
    用于 eval：逐点评估 c_alpha -> dobs(c_alpha)-mean。
    """
    def __init__(
        self,
        data_root,
        split,
        mean_dobs,
        image_size=240,
        residual_scale=0.02,
        speed_center=1500.0,
        speed_scale=100.0,
        max_files=-1,
    ):
        super().__init__()

        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.mean_dobs = mean_dobs.astype(np.complex64)
        self.residual_scale = residual_scale
        self.speed_center = speed_center
        self.speed_scale = speed_scale

        self.files = sorted(
            list((self.data_root / split).glob(f"{split}_*.npz")),
            key=numeric_key,
        )

        if max_files > 0:
            self.files = self.files[:max_files]

        self.y_map, self.x_map = make_coord_maps_np(image_size)

        first = np.load(self.files[0])
        self.dobs_shape = first["dobs_complex"].shape

        print(
            f"[PointDataset] split={split}, files={len(self.files)}, "
            f"image_size={image_size}, dobs_shape={self.dobs_shape}, "
            f"residual_scale={residual_scale}"
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        data = np.load(path)

        speed_480 = data["target_480"].astype(np.float32)
        dobs = data["dobs_complex"].astype(np.complex64)

        inp = make_input(
            speed_480,
            self.image_size,
            self.speed_center,
            self.speed_scale,
            self.y_map,
            self.x_map,
        )

        target = make_scaled_residual_target(
            dobs,
            self.mean_dobs,
            self.residual_scale,
        )

        return {
            "input": torch.from_numpy(inp),
            "target": torch.from_numpy(target),
            "dobs_real": torch.from_numpy(dobs.real.astype(np.float32)),
            "dobs_imag": torch.from_numpy(dobs.imag.astype(np.float32)),
            "file": str(path),
            "alpha": float(data["alpha"]),
            "base_sample": str(data["base_sample"]),
        }


class LocalAlphaPairDataset(Dataset):
    """
    用于训练：同一个 base sample 的相邻 alpha 组成 pair。
    例如 alpha: 0.0 -> 0.25, 0.25 -> 0.5, ...
    """
    def __init__(
        self,
        data_root,
        split,
        mean_dobs,
        image_size=240,
        residual_scale=0.02,
        speed_center=1500.0,
        speed_scale=100.0,
        max_pairs=-1,
    ):
        super().__init__()

        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.mean_dobs = mean_dobs.astype(np.complex64)
        self.residual_scale = residual_scale
        self.speed_center = speed_center
        self.speed_scale = speed_scale

        self.files = sorted(
            list((self.data_root / split).glob(f"{split}_*.npz")),
            key=numeric_key,
        )

        self.y_map, self.x_map = make_coord_maps_np(image_size)

        groups = {}

        for path in self.files:
            data = np.load(path, allow_pickle=True)
            base_sample = str(data["base_sample"])
            alpha = float(data["alpha"])

            if base_sample not in groups:
                groups[base_sample] = []
            groups[base_sample].append((alpha, path))

        pairs = []
        for base_sample, items in groups.items():
            items = sorted(items, key=lambda x: x[0])
            for i in range(len(items) - 1):
                a0, p0 = items[i]
                a1, p1 = items[i + 1]

                if a1 <= a0:
                    continue

                pairs.append(
                    {
                        "base_sample": base_sample,
                        "alpha0": a0,
                        "alpha1": a1,
                        "path0": p0,
                        "path1": p1,
                    }
                )

        pairs = sorted(
            pairs,
            key=lambda r: (numeric_key(r["path0"]), r["alpha0"], r["alpha1"]),
        )

        if max_pairs > 0:
            pairs = pairs[:max_pairs]

        self.pairs = pairs

        first = np.load(self.pairs[0]["path0"])
        self.dobs_shape = first["dobs_complex"].shape

        print(
            f"[PairDataset] split={split}, pairs={len(self.pairs)}, "
            f"image_size={image_size}, dobs_shape={self.dobs_shape}, "
            f"residual_scale={residual_scale}"
        )

    def __len__(self):
        return len(self.pairs)

    def _load_one(self, path):
        data = np.load(path)

        speed_480 = data["target_480"].astype(np.float32)
        dobs = data["dobs_complex"].astype(np.complex64)

        inp = make_input(
            speed_480,
            self.image_size,
            self.speed_center,
            self.speed_scale,
            self.y_map,
            self.x_map,
        )

        target = make_scaled_residual_target(
            dobs,
            self.mean_dobs,
            self.residual_scale,
        )

        return inp, target, dobs

    def __getitem__(self, idx):
        item = self.pairs[idx]

        inp0, target0, dobs0 = self._load_one(item["path0"])
        inp1, target1, dobs1 = self._load_one(item["path1"])

        return {
            "input0": torch.from_numpy(inp0),
            "target0": torch.from_numpy(target0),
            "input1": torch.from_numpy(inp1),
            "target1": torch.from_numpy(target1),
            "dobs0_real": torch.from_numpy(dobs0.real.astype(np.float32)),
            "dobs0_imag": torch.from_numpy(dobs0.imag.astype(np.float32)),
            "dobs1_real": torch.from_numpy(dobs1.real.astype(np.float32)),
            "dobs1_imag": torch.from_numpy(dobs1.imag.astype(np.float32)),
            "alpha0": item["alpha0"],
            "alpha1": item["alpha1"],
            "base_sample": item["base_sample"],
        }


@torch.no_grad()
def evaluate_model(model, loader, mean_dobs, residual_scale, device, args):
    model.eval()

    losses = []
    dobs_rrmses = []
    dobs_maes = []
    residual_rrmses = []
    residual_maes = []

    mean_baseline_rrmses = []
    mean_baseline_maes = []

    rows = []

    mean_dobs = mean_dobs.astype(np.complex64)

    for batch in loader:
        inp = batch["input"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        pred = model(inp)

        loss = mixed_complex_loss(
            pred,
            target,
            lambda_l1=args.lambda_l1,
            lambda_mse=args.lambda_mse,
        )

        losses.append(float(loss.detach().cpu()))

        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        dobs_real = batch["dobs_real"].numpy()
        dobs_imag = batch["dobs_imag"].numpy()
        true_dobs_batch = dobs_real + 1j * dobs_imag

        bsz = pred_np.shape[0]

        for i in range(bsz):
            pred_res = (
                pred_np[i, 0] * residual_scale
                + 1j * pred_np[i, 1] * residual_scale
            ).astype(np.complex64)

            true_res = (
                target_np[i, 0] * residual_scale
                + 1j * target_np[i, 1] * residual_scale
            ).astype(np.complex64)

            pred_dobs = mean_dobs + pred_res
            true_dobs = true_dobs_batch[i].astype(np.complex64)

            dobs_rrmses.append(complex_rrmse(pred_dobs, true_dobs))
            dobs_maes.append(complex_mae(pred_dobs, true_dobs))

            residual_rrmses.append(complex_rrmse(pred_res, true_res))
            residual_maes.append(complex_mae(pred_res, true_res))

            mean_baseline_rrmses.append(complex_rrmse(mean_dobs, true_dobs))
            mean_baseline_maes.append(complex_mae(mean_dobs, true_dobs))

            file_name = batch["file"][i] if isinstance(batch["file"], list) else str(batch["file"])
            alpha_val = float(batch["alpha"][i]) if torch.is_tensor(batch["alpha"]) else float(batch["alpha"])

            rows.append(
                {
                    "file": file_name,
                    "alpha": alpha_val,
                    "dobs_rrmse": dobs_rrmses[-1],
                    "dobs_mae": dobs_maes[-1],
                    "mean_baseline_rrmse": mean_baseline_rrmses[-1],
                }
            )

    out = {
        "loss": float(np.mean(losses)),

        "dobs_rrmse": float(np.mean(dobs_rrmses)),
        "dobs_rrmse_std": float(np.std(dobs_rrmses)),
        "dobs_mae": float(np.mean(dobs_maes)),

        "residual_rrmse": float(np.mean(residual_rrmses)),
        "residual_rrmse_std": float(np.std(residual_rrmses)),
        "residual_mae": float(np.mean(residual_maes)),

        "mean_baseline_dobs_rrmse": float(np.mean(mean_baseline_rrmses)),
        "mean_baseline_dobs_rrmse_std": float(np.std(mean_baseline_rrmses)),
        "mean_baseline_dobs_mae": float(np.mean(mean_baseline_maes)),

        "improve_over_mean_rrmse": float(
            np.mean(mean_baseline_rrmses) - np.mean(dobs_rrmses)
        ),
        "relative_improve_over_mean": float(
            (np.mean(mean_baseline_rrmses) - np.mean(dobs_rrmses))
            / (np.mean(mean_baseline_rrmses) + 1e-12)
        ),

        "rows": rows,
    }

    return out


def save_checkpoint(path, model, optimizer, epoch, args, metrics):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "args": vars(args),
        "metrics": metrics,
        "best_dobs_rrmse": metrics.get("dobs_rrmse", None),
    }
    torch.save(ckpt, path)


def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("device =", device)

    mean_npz = np.load(args.mean_dobs_path)
    mean_dobs = mean_npz["mean_dobs"].astype(np.complex64)

    train_ds = LocalAlphaPairDataset(
        data_root=args.data_root,
        split="train",
        mean_dobs=mean_dobs,
        image_size=args.image_size,
        residual_scale=args.residual_scale,
        speed_center=args.speed_center,
        speed_scale=args.speed_scale,
        max_pairs=args.max_train_pairs,
    )

    val_ds = LocalAlphaPointDataset(
        data_root=args.data_root,
        split="test",
        mean_dobs=mean_dobs,
        image_size=args.image_size,
        residual_scale=args.residual_scale,
        speed_center=args.speed_center,
        speed_scale=args.speed_scale,
        max_files=args.max_test_files,
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

    model = DobsMatrixCNN(
        in_ch=3,
        base_ch=args.base_ch,
        num_sources=64,
        num_receivers=64,
        spatial_pool=args.spatial_pool,
        head_dim=args.head_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    best_rrmse = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()

        epoch_losses = []
        epoch_point_losses = []
        epoch_delta_losses = []

        for step, batch in enumerate(train_loader, start=1):
            inp0 = batch["input0"].to(device, non_blocking=True)
            tgt0 = batch["target0"].to(device, non_blocking=True)
            inp1 = batch["input1"].to(device, non_blocking=True)
            tgt1 = batch["target1"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.use_amp):
                pred0 = model(inp0)
                pred1 = model(inp1)

                point_loss0 = mixed_complex_loss(
                    pred0,
                    tgt0,
                    lambda_l1=args.lambda_l1,
                    lambda_mse=args.lambda_mse,
                )

                point_loss1 = mixed_complex_loss(
                    pred1,
                    tgt1,
                    lambda_l1=args.lambda_l1,
                    lambda_mse=args.lambda_mse,
                )

                point_loss = 0.5 * (point_loss0 + point_loss1)

                pred_delta = pred1 - pred0
                true_delta = tgt1 - tgt0

                delta_loss = mixed_complex_loss(
                    pred_delta,
                    true_delta,
                    lambda_l1=args.lambda_l1,
                    lambda_mse=args.lambda_mse,
                )

                loss = point_loss + args.lambda_delta * delta_loss

            scaler.scale(loss).backward()

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(float(loss.detach().cpu()))
            epoch_point_losses.append(float(point_loss.detach().cpu()))
            epoch_delta_losses.append(float(delta_loss.detach().cpu()))

            if step % args.log_every == 0:
                print(
                    f"Epoch {epoch:03d} | Step {step:04d}/{len(train_loader):04d} | "
                    f"loss={np.mean(epoch_losses):.6f} | "
                    f"point={np.mean(epoch_point_losses):.6f} | "
                    f"delta={np.mean(epoch_delta_losses):.6f}"
                )

        if epoch % args.eval_every == 0 or epoch == 1 or epoch == args.epochs:
            metrics = evaluate_model(
                model=model,
                loader=val_loader,
                mean_dobs=mean_dobs,
                residual_scale=args.residual_scale,
                device=device,
                args=args,
            )

            row = {
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_losses)),
                "train_point_loss": float(np.mean(epoch_point_losses)),
                "train_delta_loss": float(np.mean(epoch_delta_losses)),
            }
            row.update(metrics)
            history.append(row)

            print("=" * 100)
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={row['train_loss']:.6f} | "
                f"point={row['train_point_loss']:.6f} | "
                f"delta={row['train_delta_loss']:.6f} | "
                f"val_dobs_rrmse={metrics['dobs_rrmse']:.6f} | "
                f"mean_base={metrics['mean_baseline_dobs_rrmse']:.6f} | "
                f"rel_improve={100*metrics['relative_improve_over_mean']:.4f}%"
            )
            print("=" * 100)

            with open(os.path.join(args.output_dir, "history.json"), "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2, ensure_ascii=False)

            save_checkpoint(
                os.path.join(args.output_dir, "checkpoints", "latest.pth"),
                model,
                optimizer,
                epoch,
                args,
                metrics,
            )

            if metrics["dobs_rrmse"] < best_rrmse:
                best_rrmse = metrics["dobs_rrmse"]

                save_checkpoint(
                    os.path.join(args.output_dir, "checkpoints", "best.pth"),
                    model,
                    optimizer,
                    epoch,
                    args,
                    metrics,
                )

                with open(os.path.join(args.output_dir, "best_metrics.json"), "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2, ensure_ascii=False)

                print(f"[Saved best] epoch={epoch}, dobs_rrmse={best_rrmse:.6f}")

    print("[Done]")
    print("best_dobs_rrmse =", best_rrmse)


def eval_only(args):
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("device =", device)

    mean_npz = np.load(args.mean_dobs_path)
    mean_dobs = mean_npz["mean_dobs"].astype(np.complex64)

    ckpt = torch.load(args.ckpt_path, map_location=device)
    ckpt_args = ckpt["args"]

    model = DobsMatrixCNN(
        in_ch=3,
        base_ch=ckpt_args.get("base_ch", args.base_ch),
        num_sources=64,
        num_receivers=64,
        spatial_pool=ckpt_args.get("spatial_pool", args.spatial_pool),
        head_dim=ckpt_args.get("head_dim", args.head_dim),
    ).to(device)

    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    eval_ds = LocalAlphaPointDataset(
        data_root=args.data_root,
        split=args.eval_split,
        mean_dobs=mean_dobs,
        image_size=ckpt_args.get("image_size", args.image_size),
        residual_scale=ckpt_args.get("residual_scale", args.residual_scale),
        speed_center=ckpt_args.get("speed_center", args.speed_center),
        speed_scale=ckpt_args.get("speed_scale", args.speed_scale),
        max_files=args.max_test_files,
    )

    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # 保证 evaluate_model 使用 ckpt 内的 loss 参数
    args.lambda_l1 = ckpt_args.get("lambda_l1", args.lambda_l1)
    args.lambda_mse = ckpt_args.get("lambda_mse", args.lambda_mse)
    args.residual_scale = ckpt_args.get("residual_scale", args.residual_scale)

    metrics = evaluate_model(
        model=model,
        loader=eval_loader,
        mean_dobs=mean_dobs,
        residual_scale=args.residual_scale,
        device=device,
        args=args,
    )

    with open(os.path.join(args.output_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--mean_dobs_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, default="")

    parser.add_argument("--eval_split", type=str, default="test")

    parser.add_argument("--image_size", type=int, default=240)
    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--spatial_pool", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=1024)

    parser.add_argument("--speed_center", type=float, default=1500.0)
    parser.add_argument("--speed_scale", type=float, default=100.0)
    parser.add_argument("--residual_scale", type=float, default=0.02)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--lambda_l1", type=float, default=1.0)
    parser.add_argument("--lambda_mse", type=float, default=1.0)
    parser.add_argument("--lambda_delta", type=float, default=2.0)

    parser.add_argument("--max_train_pairs", type=int, default=-1)
    parser.add_argument("--max_test_files", type=int, default=-1)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=5)

    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    else:
        if not args.ckpt_path:
            raise ValueError("--ckpt_path is required for eval mode")
        eval_only(args)


if __name__ == "__main__":
    main()