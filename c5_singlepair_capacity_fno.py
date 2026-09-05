import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from c5_train_fno_bg_wavefield import (
    WavefieldBGDataset,
    BackgroundFNO,
    estimate_wave_scale,
    numeric_key,
    sample_receivers,
)


def metrics(pred, target, dobs, rec):

    full = torch.sqrt(
        torch.sum((pred-target)**2)
        /
        torch.clamp(
            torch.sum(target**2),
            min=1e-12,
        )
    )

    pred_rec = sample_receivers(
        pred,
        rec,
    )

    recv = torch.sqrt(
        torch.sum((pred_rec-dobs)**2)
        /
        torch.clamp(
            torch.sum(dobs**2),
            min=1e-12,
        )
    )

    return (
        float(full.detach().cpu()),
        float(recv.detach().cpu()),
    )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--data_root", required=True)
    ap.add_argument("--background_path", required=True)
    ap.add_argument("--output_dir", required=True)

    ap.add_argument("--source_idx", type=int, default=0)

    ap.add_argument("--modes", type=int, default=25)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)

    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)

    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--device", default="cuda:0")

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    root = Path(args.data_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_files = sorted(
        (root/"train").glob("*.npz"),
        key=numeric_key,
    )[:1]

    wave_scale = estimate_wave_scale(
        train_files
    )

    ds = WavefieldBGDataset(
        root=root,
        split="train",
        background_path=args.background_path,
        wave_scale=wave_scale,
        max_images=1,
    )

    if args.source_idx < 0 or args.source_idx >= 8:
        raise ValueError("source_idx must be 0..7")

    batch = ds[args.source_idx]

    device = torch.device(
        args.device
        if torch.cuda.is_available()
        else "cpu"
    )

    x = batch["input"].unsqueeze(0).to(device)
    target = batch["target"].unsqueeze(0).to(device)
    bg = batch["background"].unsqueeze(0).to(device)
    dobs = batch["dobs"].unsqueeze(0).to(device)
    rec = batch["rec"].unsqueeze(0).to(device)

    model = BackgroundFNO(
        modes=args.modes,
        width=args.width,
        depth=args.depth,
    ).to(device)

    n_params = sum(
        p.numel()
        for p in model.parameters()
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.0,
    )

    # Capacity audit, not formal paper training:
    # use step-based cosine decay rather than short epoch schedule.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.steps,
        eta_min=args.lr * 0.01,
    )

    print("="*100)
    print("C5.1g SINGLE-PAIR FNO-BG CAPACITY AUDIT")
    print("="*100)

    print("source_idx  =", args.source_idx)
    print("modes       =", args.modes)
    print("width       =", args.width)
    print("depth       =", args.depth)
    print("parameters  =", f"{n_params:,}")
    print("wave_scale  =", f"{wave_scale:.8e}")
    print("steps       =", args.steps)
    print("lr          =", args.lr)

    model.eval()

    with torch.no_grad():

        pred0 = model(x, bg)

        init_full, init_rec = metrics(
            pred0,
            target,
            dobs,
            rec,
        )

    print()
    print(
        f"INITIAL full={init_full:.6f} "
        f"receiver={init_rec:.6f}"
    )

    history = []

    best_full = float("inf")
    best_row = None

    model.train()

    for step in range(1, args.steps+1):

        pred = model(x, bg)

        # Full-wavefield MSE only.
        loss = F.mse_loss(
            pred,
            target,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            5.0,
        )

        optimizer.step()
        scheduler.step()

        if (
            step == 1
            or step % 25 == 0
            or step == args.steps
        ):

            model.eval()

            with torch.no_grad():

                pred_eval = model(
                    x,
                    bg,
                )

                full_rr, rec_rr = metrics(
                    pred_eval,
                    target,
                    dobs,
                    rec,
                )

            row = {
                "step": step,
                "loss": float(
                    loss.detach().cpu()
                ),
                "full_rrmse": full_rr,
                "receiver_rrmse": rec_rr,
                "lr": float(
                    scheduler.get_last_lr()[0]
                ),
            }

            history.append(row)

            if full_rr < best_full:

                best_full = full_rr
                best_row = row.copy()

                torch.save(
                    {
                        "model_state":
                            model.state_dict(),
                        "args":
                            vars(args),
                        "wave_scale":
                            wave_scale,
                        "best":
                            best_row,
                    },
                    out/"best.pth",
                )

            print(
                f"[{step:04d}/{args.steps}] "
                f"loss={row['loss']:.6e} "
                f"lr={row['lr']:.3e} | "
                f"full={full_rr:.6f} "
                f"rec={rec_rr:.6f}"
            )

            model.train()

    with open(
        out/"history.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            history,
            f,
            indent=2,
        )

    print()
    print("="*100)
    print("C5.1g SINGLE-PAIR CAPACITY SUMMARY")
    print("="*100)

    print(
        "initial full =",
        f"{init_full:.6f}",
    )

    print(
        "best full    =",
        f"{best_row['full_rrmse']:.6f}",
        "@ step",
        best_row["step"],
    )

    print(
        "best rec     =",
        f"{best_row['receiver_rrmse']:.6f}",
    )

    if best_row["full_rrmse"] < 0.05:

        print(
            "[STRONG PASS] single-wavefield "
            "capacity confirmed."
        )

    elif best_row["full_rrmse"] < 0.10:

        print(
            "[PASS] single-wavefield "
            "capacity confirmed."
        )

    elif best_row["full_rrmse"] < 0.20:

        print(
            "[PARTIAL] learnable but "
            "representation remains difficult."
        )

    else:

        print(
            "[FAIL] even one wavefield "
            "cannot be fitted adequately."
        )


if __name__ == "__main__":
    main()
