import os
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_openbreastus_oldstyle import OpenBreastUSOldStyleDataset
from train_inversionnet_baseline import InversionNetBaseline


def normalize_speed(x, vmin=1400.0, vmax=1605.0):
    mid = 0.5 * (vmin + vmax)
    half = 0.5 * (vmax - vmin)
    return (x - mid) / half


def denormalize_target(y_norm, vmin=1400.0, vmax=1600.0):
    y = (y_norm + 1.0) * 0.5
    return y * (vmax - vmin) + vmin


def load_inversionnet(ckpt_path, device, base_ch=None, bottleneck_blocks=None, dropout=None):
    ckpt = torch.load(ckpt_path, map_location=device)

    cfg = ckpt.get("config", {})
    if base_ch is None:
        base_ch = cfg.get("base_ch", 32)
    if bottleneck_blocks is None:
        bottleneck_blocks = cfg.get("bottleneck_blocks", 2)
    if dropout is None:
        dropout = cfg.get("dropout", 0.0)

    target_min = cfg.get("target_min", 1400.0)
    target_max = cfg.get("target_max", 1600.0)

    model = InversionNetBaseline(
        in_channels=2,
        out_channels=1,
        base_ch=base_ch,
        bottleneck_blocks=bottleneck_blocks,
        dropout=dropout,
    ).to(device)

    state = ckpt.get("model_state", ckpt.get("model", ckpt))
    model.load_state_dict(state, strict=True)
    model.eval()

    print("[Model]")
    print("ckpt_path         =", ckpt_path)
    print("base_ch           =", base_ch)
    print("bottleneck_blocks =", bottleneck_blocks)
    print("dropout           =", dropout)
    print("target_min/max    =", target_min, target_max)

    return model, target_min, target_max


@torch.no_grad()
def precompute_split(args, split, model, inv_target_min, inv_target_max, device):
    dataset = OpenBreastUSOldStyleDataset(
        root_dir=args.data_root,
        split=split,
        normalize_input=True,
        normalize_target=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    out_dir = Path(args.output_root) / split
    out_dir.mkdir(parents=True, exist_ok=True)

    global_index = 0

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred_norm_inv = model(x)
        pred_speed = denormalize_target(pred_norm_inv, inv_target_min, inv_target_max)
        pred_speed = torch.clamp(pred_speed, args.speed_min, args.speed_max)

        cond_norm = normalize_speed(pred_speed, args.speed_min, args.speed_max)
        target_norm = normalize_speed(y, args.speed_min, args.speed_max)

        pred_speed_np = pred_speed.detach().cpu().numpy().astype(np.float32)
        cond_norm_np = cond_norm.detach().cpu().numpy().astype(np.float32)
        target_speed_np = y.detach().cpu().numpy().astype(np.float32)
        target_norm_np = target_norm.detach().cpu().numpy().astype(np.float32)

        b = x.shape[0]
        for i in range(b):
            sample_idx = global_index + 1
            save_path = out_dir / f"{split}_{sample_idx}.npz"

            np.savez_compressed(
                save_path,
                condition_speed=pred_speed_np[i],
                condition_norm=cond_norm_np[i],
                target_speed=target_speed_np[i],
                target_norm=target_norm_np[i],
                sample_index=np.array([sample_idx], dtype=np.int32),
            )

            global_index += 1

        if (batch_idx + 1) % args.log_every == 0:
            print(f"[{split}] batch {batch_idx + 1}/{len(loader)}, saved {global_index} samples")

    print(f"[{split}] done. saved {global_index} samples to {out_dir}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--base_ch", type=int, default=None)
    parser.add_argument("--bottleneck_blocks", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)

    parser.add_argument("--speed_min", type=float, default=1400.0)
    parser.add_argument("--speed_max", type=float, default=1605.0)

    parser.add_argument("--log_every", type=int, default=10)

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("device =", device)

    os.makedirs(args.output_root, exist_ok=True)

    model, inv_target_min, inv_target_max = load_inversionnet(
        args.ckpt_path,
        device,
        base_ch=args.base_ch,
        bottleneck_blocks=args.bottleneck_blocks,
        dropout=args.dropout,
    )

    config = vars(args).copy()
    config["inversion_target_min"] = inv_target_min
    config["inversion_target_max"] = inv_target_max

    with open(os.path.join(args.output_root, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    precompute_split(args, "train", model, inv_target_min, inv_target_max, device)
    precompute_split(args, "test", model, inv_target_min, inv_target_max, device)

    print("[Done] condition cache generated.")


if __name__ == "__main__":
    main()