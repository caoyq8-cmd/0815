#!/usr/bin/env python3
"""Build leakage-free out-of-fold InversionNet conditions for IC-RFM.

Each fold model is trained from scratch for a fixed number of epochs on the
other K-1 folds, then predicts only its unseen holdout fold.  Combining all
folds yields one out-of-fold condition for every training sample.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from dataset_openbreastus_oldstyle import OpenBreastUSOldStyleDataset
from train_inversionnet_baseline import (
    CompositeLoss,
    InversionNetBaseline,
    denormalize_target,
    normalize_target,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def sample_id(path: str | Path) -> int:
    nums = re.findall(r"\d+", Path(path).stem)
    return int(nums[-1]) if nums else -1


def dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)


def normalize_speed_np(x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    return ((x - 0.5 * (vmin + vmax)) / (0.5 * (vmax - vmin))).astype(np.float32)


def make_folds(n: int, num_folds: int, seed: int) -> List[List[int]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    return [sorted(map(int, x)) for x in np.array_split(order, num_folds)]


class IndexedSubset(Dataset):
    def __init__(self, dataset: Dataset, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        x, y = self.dataset[idx]
        return x, y, torch.tensor(idx, dtype=torch.long)


def amp_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(enabled: bool):
    try:
        return torch.amp.autocast(device_type="cuda", enabled=enabled)
    except AttributeError:
        return torch.cuda.amp.autocast(enabled=enabled)


def train_fold(args) -> None:
    if not (0 <= args.fold < args.num_folds):
        raise ValueError(f"fold must be in [0,{args.num_folds-1}]")
    seed_everything(args.seed + args.fold)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp = args.use_amp and device.type == "cuda"

    full = OpenBreastUSOldStyleDataset(
        root_dir=args.data_root,
        split="train",
        normalize_input=True,
        normalize_target=False,
    )
    folds = make_folds(len(full), args.num_folds, args.seed)
    holdout = folds[args.fold]
    holdout_set = set(holdout)
    fitting = [i for i in range(len(full)) if i not in holdout_set]
    if holdout_set & set(fitting):
        raise RuntimeError("Fold leakage detected")

    root = Path(args.output_root)
    fold_dir = root / f"fold_{args.fold}"
    ckpt_dir = fold_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    oof_dir = root / "train"
    oof_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "fold": args.fold,
        "num_folds": args.num_folds,
        "seed": args.seed,
        "fit_indices_zero_based": fitting,
        "fit_sample_ids": [sample_id(full.files[i]) for i in fitting],
        "holdout_indices_zero_based": holdout,
        "holdout_sample_ids": [sample_id(full.files[i]) for i in holdout],
    }
    dump_json(fold_dir / "manifest.json", manifest)
    dump_json(fold_dir / "config.json", vars(args))

    train_loader = DataLoader(
        Subset(full, fitting),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    model = InversionNetBaseline(
        in_channels=2,
        out_channels=1,
        base_ch=args.base_ch,
        bottleneck_blocks=args.bottleneck_blocks,
        dropout=args.dropout,
    ).to(device)
    criterion = CompositeLoss(args.lambda_l1, args.lambda_mse, args.lambda_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = amp_scaler(amp)

    latest = ckpt_dir / "latest.pth"
    history: List[Dict] = []
    start_epoch = 1
    if args.resume:
        if not latest.exists():
            raise RuntimeError(f"Missing resume checkpoint: {latest}")
        ckpt = torch.load(latest, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        history_path = fold_dir / "history.json"
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
    elif latest.exists():
        raise RuntimeError(
            f"Fold output already exists: {fold_dir}. Use --resume or a new output_root."
        )

    print("=" * 100)
    print("OOF InversionNet fold", args.fold)
    print("device / AMP      =", device, amp)
    print("fit / holdout     =", len(fitting), len(holdout))
    print("epochs            =", args.epochs)
    print("base / blocks     =", args.base_ch, args.bottleneck_blocks)
    print("=" * 100)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total = 0.0
        total_l1 = 0.0
        total_grad = 0.0
        count = 0
        t0 = time.time()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yn = normalize_target(y, args.target_min, args.target_max)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(amp):
                pred = model(x)
                losses = criterion(pred, yn)
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            b = x.shape[0]
            total += float(losses["total"].detach().cpu()) * b
            total_l1 += float(losses["l1"].cpu()) * b
            total_grad += float(losses["grad"].cpu()) * b
            count += b
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": total / count,
            "train_l1": total_l1 / count,
            "train_grad": total_grad / count,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "seconds": float(time.time() - t0),
        }
        history.append(row)
        dump_json(fold_dir / "history.json", history)
        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "config": vars(args),
            "manifest": manifest,
        }
        torch.save(state, latest)
        if epoch == args.epochs:
            torch.save(state, ckpt_dir / "final.pth")
        print(
            f"[fold {args.fold} epoch {epoch:03d}/{args.epochs:03d}] "
            f"loss={row['train_loss']:.6f} l1={row['train_l1']:.6f} "
            f"grad={row['train_grad']:.6f} time={row['seconds']:.1f}s"
        )

    # Predict only the unseen fold with the fixed final-epoch model.
    final_ckpt = torch.load(ckpt_dir / "final.pth", map_location=device)
    model.load_state_dict(final_ckpt["model_state"])
    model.eval()
    holdout_loader = DataLoader(
        IndexedSubset(full, holdout),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    rows: List[Dict] = []
    with torch.no_grad():
        for x, y, dataset_indices in holdout_loader:
            x = x.to(device, non_blocking=True)
            pred_norm = model(x)
            pred_speed = denormalize_target(pred_norm, args.target_min, args.target_max)
            pred_speed = pred_speed.clamp(args.speed_min, args.speed_max).cpu().numpy().astype(np.float32)
            target_speed = y.numpy().astype(np.float32)
            for j, dataset_idx in enumerate(dataset_indices.tolist()):
                sid = sample_id(full.files[dataset_idx])
                condition_speed = pred_speed[j]
                target = target_speed[j]
                condition_norm = normalize_speed_np(condition_speed, args.speed_min, args.speed_max)
                target_norm = normalize_speed_np(target, args.speed_min, args.speed_max)
                save_path = oof_dir / f"train_{sid}.npz"
                np.savez_compressed(
                    save_path,
                    condition_speed=condition_speed,
                    condition_norm=condition_norm,
                    target_speed=target,
                    target_norm=target_norm,
                    sample_index=np.asarray([sid], dtype=np.int32),
                    oof_fold=np.asarray([args.fold], dtype=np.int32),
                )
                err = target.astype(np.float64) - condition_speed.astype(np.float64)
                rows.append({
                    "sample_id": sid,
                    "fold": args.fold,
                    "mse": float(np.mean(err * err)),
                    "mae": float(np.mean(np.abs(err))),
                })
    with (fold_dir / "holdout_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    dump_json(fold_dir / "holdout_summary.json", {
        "num_samples": len(rows),
        "mse_mean": float(np.mean([r["mse"] for r in rows])),
        "mse_std": float(np.std([r["mse"] for r in rows], ddof=1)),
        "mae_mean": float(np.mean([r["mae"] for r in rows])),
        "mae_std": float(np.std([r["mae"] for r in rows], ddof=1)),
    })
    print(f"[DONE fold {args.fold}] wrote {len(rows)} unseen predictions to {oof_dir}")


def finalize(args) -> None:
    root = Path(args.output_root)
    train_dir = root / "train"
    files = sorted(train_dir.glob("train_*.npz"), key=sample_id)
    expected = OpenBreastUSOldStyleDataset(
        root_dir=args.data_root, split="train", normalize_input=True, normalize_target=False
    )
    expected_ids = [sample_id(p) for p in expected.files]
    actual_ids = [sample_id(p) for p in files]
    missing = sorted(set(expected_ids) - set(actual_ids))
    extra = sorted(set(actual_ids) - set(expected_ids))
    if missing or extra or len(files) != len(expected):
        raise RuntimeError(
            f"OOF cache incomplete: expected={len(expected)} actual={len(files)} "
            f"missing={missing[:20]} extra={extra[:20]}"
        )

    folds_seen = []
    mse, mae = [], []
    for p in files:
        with np.load(p) as d:
            fold = int(d["oof_fold"].reshape(-1)[0])
            c = d["condition_speed"].astype(np.float64)
            y = d["target_speed"].astype(np.float64)
        folds_seen.append(fold)
        err = y - c
        mse.append(float(np.mean(err * err)))
        mae.append(float(np.mean(np.abs(err))))
    if set(folds_seen) != set(range(args.num_folds)):
        raise RuntimeError(f"Unexpected folds in cache: {sorted(set(folds_seen))}")

    source_test = Path(args.original_condition_root) / "test"
    target_test = root / "test"
    target_test.mkdir(parents=True, exist_ok=True)
    test_files = sorted(source_test.glob("test_*.npz"), key=sample_id)
    if not test_files:
        raise RuntimeError(f"No original test cache files: {source_test}")
    for p in test_files:
        shutil.copy2(p, target_test / p.name)

    summary = {
        "status": "PASS",
        "num_train_oof": len(files),
        "num_test_copied": len(test_files),
        "fold_counts": {str(k): int(folds_seen.count(k)) for k in range(args.num_folds)},
        "oof_mse_mean": float(np.mean(mse)),
        "oof_mse_std": float(np.std(mse, ddof=1)),
        "oof_mae_mean": float(np.mean(mae)),
        "oof_mae_std": float(np.std(mae, ddof=1)),
        "note": "train conditions are OOF; test conditions retain the frozen epoch-67 InversionNet",
    }
    dump_json(root / "oof_audit.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("[FINALIZED]", root.resolve())


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=("fold", "finalize"), required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--original_condition_root", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--num_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260902)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=67)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--base_ch", type=int, default=32)
    p.add_argument("--bottleneck_blocks", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lambda_l1", type=float, default=1.0)
    p.add_argument("--lambda_mse", type=float, default=0.2)
    p.add_argument("--lambda_grad", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--target_min", type=float, default=1400.0)
    p.add_argument("--target_max", type=float, default=1600.0)
    p.add_argument("--speed_min", type=float, default=1400.0)
    p.add_argument("--speed_max", type=float, default=1605.0)
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p


def main() -> None:
    args = parser().parse_args()
    if args.mode == "fold":
        train_fold(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
