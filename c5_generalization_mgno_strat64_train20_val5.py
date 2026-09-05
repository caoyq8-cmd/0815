import csv
import json
import math
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from c5_mgno_paper import (
    MgNOBackgroundWavefield,
    count_parameters,
)

from c5_shared8_capacity_fno import (
    set_seed,
    compute_wave_scale,
)

from c5_generalization_fno_rot64_utils import (
    find_split_files,
    WavefieldSplit,
    make_micro_batch,
    evaluate_split,
    metric_summary,
    print_metric_summary,
    save_pair_csv,
    save_source_csv,
    save_image_csv,
)


def audit_model(
    label,
    model,
    ckpt_path,
    train_set,
    val_set,
    device,
    micro_batch_size,
    output_dir,
    save_predictions=False,
):
    ckpt = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        ckpt["model_state"]
    )

    print()
    print("=" * 100)
    print(
        f"{label} CHECKPOINT AUDIT"
    )
    print("=" * 100)

    print(
        "epoch  =",
        ckpt["epoch"]
    )

    print(
        "update =",
        ckpt["update"]
    )

    train_metrics = evaluate_split(
        model,
        train_set,
        device,
        micro_batch_size=micro_batch_size,
        save_predictions=False,
    )

    val_metrics = evaluate_split(
        model,
        val_set,
        device,
        micro_batch_size=micro_batch_size,
        save_predictions=save_predictions,
    )

    print()
    print("TRAIN")

    print_metric_summary(
        "train full",
        train_metrics["full"],
    )

    print_metric_summary(
        "train receiver",
        train_metrics["rec"],
    )

    print()
    print("VALIDATION")

    print_metric_summary(
        "val full",
        val_metrics["full"],
    )

    print_metric_summary(
        "val receiver",
        val_metrics["rec"],
    )

    result = {
        "epoch":
            int(ckpt["epoch"]),

        "update":
            int(ckpt["update"]),

        "train_full":
            metric_summary(
                train_metrics["full"]
            ),

        "train_rec":
            metric_summary(
                train_metrics["rec"]
            ),

        "val_full":
            metric_summary(
                val_metrics["full"]
            ),

        "val_rec":
            metric_summary(
                val_metrics["rec"]
            ),
    }

    prefix = (
        "best"
        if label == "BEST"
        else "latest"
    )

    save_pair_csv(
        output_dir
        / f"{prefix}_train_per_pair.csv",
        train_set,
        train_metrics["full"],
        train_metrics["rec"],
    )

    save_pair_csv(
        output_dir
        / f"{prefix}_val_per_pair.csv",
        val_set,
        val_metrics["full"],
        val_metrics["rec"],
    )

    save_source_csv(
        output_dir
        / f"{prefix}_val_per_source.csv",
        val_metrics["full"],
        val_metrics["rec"],
    )

    save_image_csv(
        output_dir
        / f"{prefix}_val_per_image.csv",
        val_set,
        val_metrics["full"],
        val_metrics["rec"],
    )

    if save_predictions:
        np.savez_compressed(
            output_dir
            / f"{prefix}_val_predictions.npz",

            pred_wavefields=
                val_metrics[
                    "predictions"
                ],

            target_wavefields=
                val_set.wave_raw,

            background_wavefields=
                val_set.background_raw,

            src_indices=
                val_set.src_indices,

            rec_indices=
                val_set.rec_indices,
        )

    return result


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--data_root",
        required=True,
    )

    parser.add_argument(
        "--background_path",
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        required=True,
    )

    parser.add_argument(
        "--channels",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--recurrent_iters",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--accum_pairs",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--sources_per_image",
        type=int,
        default=8,
        help="Random sources sampled per image per epoch from the 64-source pool.",
    )

    parser.add_argument(
        "--measurement_source_ids",
        default="0,8,16,24,32,40,48,56",
        help="Comma-separated source IDs used by measurement TX geometry.",
    )

    parser.add_argument(
        "--measurement_sources_per_image",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--eval_every",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--speed_mean",
        type=float,
        default=1488.39,
    )

    parser.add_argument(
        "--speed_std",
        type=float,
        default=27.53,
    )

    parser.add_argument(
        "--wave_scale",
        type=float,
        default=3.72290883e-02,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260904,
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--disable_checkpoint",
        action="store_true",
    )

    args = parser.parse_args()

    out = Path(
        args.output_dir
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    set_seed(
        args.seed
    )

    device = torch.device(
        args.device
    )

    train_files = find_split_files(
        args.data_root,
        "train",
    )

    val_files = find_split_files(
        args.data_root,
        "val",
    )

    print("=" * 100)
    print(
        "C5.2a MgNO-I TRUE-BG "
        "OPERATOR GENERALIZATION"
    )
    print("=" * 100)

    print(
        "train images =",
        len(train_files)
    )

    print(
        "val images   =",
        len(val_files)
    )

    estimated_scale = compute_wave_scale(
        train_files
    )

    print(
        "estimated wave_scale =",
        f"{estimated_scale:.8e}"
    )

    print(
        "locked wave_scale    =",
        f"{args.wave_scale:.8e}"
    )

    train_set = WavefieldSplit(
        files=train_files,
        background_path=args.background_path,
        wave_scale=args.wave_scale,
        speed_mean=args.speed_mean,
        speed_std=args.speed_std,
    )

    val_set = WavefieldSplit(
        files=val_files,
        background_path=args.background_path,
        wave_scale=args.wave_scale,
        speed_mean=args.speed_mean,
        speed_std=args.speed_std,
        reference_src_indices=
            train_set.src_indices,
        reference_rec_indices=
            train_set.rec_indices,
    )

    model = MgNOBackgroundWavefield(
        channels=args.channels,
        recurrent_iters=
            args.recurrent_iters,
        use_checkpoint=
            not args.disable_checkpoint,
    ).to(
        device
    )

    nparams = count_parameters(
        model
    )

    print()
    print(
        "channels        =",
        args.channels
    )

    print(
        "recurrent_iters =",
        args.recurrent_iters
    )

    print(
        "parameters      =",
        f"{nparams:,}"
    )

    print(
        "epochs          =",
        args.epochs
    )

    print(
        "accum_pairs     =",
        args.accum_pairs
    )

    measurement_source_ids = np.asarray(
        [
            int(x)
            for x in args.measurement_source_ids.split(",")
            if x.strip()
        ],
        dtype=np.int64,
    )

    if len(np.unique(measurement_source_ids)) != len(
        measurement_source_ids
    ):
        raise ValueError(
            "measurement_source_ids contains duplicates"
        )

    if np.any(measurement_source_ids < 0) or np.any(
        measurement_source_ids >= train_set.num_sources
    ):
        raise ValueError(
            "measurement_source_ids out of range"
        )

    all_source_ids = np.arange(
        train_set.num_sources,
        dtype=np.int64,
    )

    other_source_ids = np.setdiff1d(
        all_source_ids,
        measurement_source_ids,
    )

    if args.measurement_sources_per_image < 0:
        raise ValueError(
            "measurement_sources_per_image must be >= 0"
        )

    other_sources_per_image = (
        args.sources_per_image
        - args.measurement_sources_per_image
    )

    if (
        args.measurement_sources_per_image
        > len(measurement_source_ids)
    ):
        raise ValueError(
            "too many measurement sources requested"
        )

    if other_sources_per_image > len(other_source_ids):
        raise ValueError(
            "too many other sources requested"
        )

    if args.sources_per_image <= 0:
        raise ValueError("sources_per_image must be positive")

    if args.sources_per_image > train_set.num_sources:
        raise ValueError(
            f"sources_per_image={args.sources_per_image} "
            f"> num_sources={train_set.num_sources}"
        )

    train_pairs_per_epoch = (
        train_set.num_images
        * args.sources_per_image
    )

    updates_per_epoch = math.ceil(
        train_pairs_per_epoch
        / args.accum_pairs
    )

    total_updates = (
        updates_per_epoch
        * args.epochs
    )

    print(
        "train pairs     =",
        train_set.num_pairs
    )

    print(
        "val pairs       =",
        val_set.num_pairs
    )

    print(
        "source pool     =",
        train_set.num_sources
    )

    print(
        "sources/image   =",
        args.sources_per_image
    )

    print(
        "sampled train pairs/epoch =",
        train_pairs_per_epoch
    )

    print(
        "measurement pool =",
        measurement_source_ids.tolist()
    )

    print(
        "measurement/image=",
        args.measurement_sources_per_image
    )

    print(
        "other/image      =",
        other_sources_per_image
    )

    print(
        "updates/epoch   =",
        updates_per_epoch
    )

    print(
        "total updates   =",
        total_updates
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=total_updates,
            eta_min=args.min_lr,
        )
    )

    initial_val = evaluate_split(
        model,
        val_set,
        device,
        micro_batch_size=
            args.micro_batch_size,
        save_predictions=False,
    )

    print()
    print("INITIAL VALIDATION")

    print_metric_summary(
        "val full",
        initial_val["full"],
    )

    print_metric_summary(
        "val receiver",
        initial_val["rec"],
    )

    rng = np.random.default_rng(
        args.seed
    )

    history = []

    global_update = 0

    best_val = float("inf")

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        model.train()

        # --------------------------------------------------------------
        # C5.5a rotating-source sampling:
        # each image contributes exactly sources_per_image randomly
        # selected sources from the full source pool.
        # --------------------------------------------------------------
        epoch_pair_ids = []

        for image_idx in range(
            train_set.num_images
        ):
            selected_measurement = rng.choice(
                measurement_source_ids,
                size=args.measurement_sources_per_image,
                replace=False,
            )

            selected_other = rng.choice(
                other_source_ids,
                size=other_sources_per_image,
                replace=False,
            )

            selected_sources = np.concatenate(
                [
                    selected_measurement,
                    selected_other,
                ]
            )

            rng.shuffle(
                selected_sources
            )

            pair_ids = (
                image_idx
                * train_set.num_sources
                + selected_sources
            )

            epoch_pair_ids.extend(
                pair_ids.tolist()
            )

        permutation = np.asarray(
            epoch_pair_ids,
            dtype=np.int64,
        )

        # Randomize across images/sources while preserving
        # exactly Nimage * sources_per_image pairs per epoch.
        rng.shuffle(permutation)

        epoch_loss_sum = 0.0
        epoch_count = 0

        for group_start in range(
            0,
            len(permutation),
            args.accum_pairs,
        ):
            group_ids = permutation[
                group_start:
                group_start
                + args.accum_pairs
            ]

            optimizer.zero_grad(
                set_to_none=True
            )

            group_loss_value = 0.0

            for micro_start in range(
                0,
                len(group_ids),
                args.micro_batch_size,
            ):
                micro_ids = group_ids[
                    micro_start:
                    micro_start
                    + args.micro_batch_size
                ]

                xb, yb = make_micro_batch(
                    train_set,
                    micro_ids,
                )

                xb = xb.to(
                    device
                )

                yb = yb.to(
                    device
                )

                pred = model(
                    xb
                )

                micro_loss = F.mse_loss(
                    pred,
                    yb,
                    reduction="mean",
                )

                weight = (
                    len(micro_ids)
                    / float(
                        len(group_ids)
                    )
                )

                loss = (
                    micro_loss
                    * weight
                )

                loss.backward()

                group_loss_value += (
                    float(
                        micro_loss
                        .detach()
                        .cpu()
                    )
                    * weight
                )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()
            scheduler.step()

            global_update += 1

            epoch_loss_sum += (
                group_loss_value
                * len(group_ids)
            )

            epoch_count += (
                len(group_ids)
            )

        epoch_loss = (
            epoch_loss_sum
            / epoch_count
        )

        # Always preserve latest state.
        torch.save(
            {
                "model_state":
                    model.state_dict(),

                "epoch":
                    epoch,

                "update":
                    global_update,

                "train_loss":
                    epoch_loss,

                "args":
                    vars(args),
            },
            out / "latest.pt",
        )

        do_eval = (
            epoch == 1
            or epoch % args.eval_every == 0
            or epoch == args.epochs
        )

        if not do_eval:
            continue

        val_metrics = evaluate_split(
            model,
            val_set,
            device,
            micro_batch_size=
                args.micro_batch_size,
            save_predictions=False,
        )

        vf = float(
            val_metrics[
                "full"
            ].mean()
        )

        vr = float(
            val_metrics[
                "rec"
            ].mean()
        )

        vmax = float(
            val_metrics[
                "full"
            ].max()
        )

        lr_now = (
            scheduler.get_last_lr()[0]
        )

        row = {
            "epoch":
                epoch,

            "update":
                global_update,

            "train_loss":
                epoch_loss,

            "lr":
                lr_now,

            "val_full_mean":
                vf,

            "val_full_max":
                vmax,

            "val_rec_mean":
                vr,
        }

        history.append(
            row
        )

        print(
            f"[epoch {epoch:03d}/{args.epochs}] "
            f"update={global_update:04d}/{total_updates} "
            f"loss={epoch_loss:.6e} "
            f"lr={lr_now:.3e} | "
            f"val_full={vf:.6f} "
            f"(max={vmax:.6f}) | "
            f"val_rec={vr:.6f}"
        )

        if vf < best_val:
            best_val = vf

            torch.save(
                {
                    "model_state":
                        model.state_dict(),

                    "epoch":
                        epoch,

                    "update":
                        global_update,

                    "train_loss":
                        epoch_loss,

                    "val_full_mean":
                        vf,

                    "val_rec_mean":
                        vr,

                    "args":
                        vars(args),
                },
                out / "best.pt",
            )

    with (
        out / "history.csv"
    ).open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                history[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            history
        )

    # ------------------------------------------------------------
    # Best and latest audits
    # ------------------------------------------------------------

    best = audit_model(
        "BEST",
        model,
        out / "best.pt",
        train_set,
        val_set,
        device,
        args.micro_batch_size,
        out,
        save_predictions=True,
    )

    latest = audit_model(
        "LATEST",
        model,
        out / "latest.pt",
        train_set,
        val_set,
        device,
        args.micro_batch_size,
        out,
        save_predictions=False,
    )

    summary = {
        "architecture":
            "MgNO-I",

        "channels":
            args.channels,

        "recurrent_iters":
            args.recurrent_iters,

        "parameters":
            nparams,

        "best":
            best,

        "latest":
            latest,
    }

    with (
        out / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("=" * 100)
    print(
        "C5.2a MgNO-I FINAL SUMMARY"
    )
    print("=" * 100)

    print(
        f"BEST epoch       = "
        f"{best['epoch']}"
    )

    print(
        f"BEST train full  = "
        f"{best['train_full']['mean']:.6f}"
    )

    print(
        f"BEST val full    = "
        f"{best['val_full']['mean']:.6f}"
    )

    print(
        f"BEST val rec     = "
        f"{best['val_rec']['mean']:.6f}"
    )

    print()

    print(
        f"LATEST epoch     = "
        f"{latest['epoch']}"
    )

    print(
        f"LATEST train full= "
        f"{latest['train_full']['mean']:.6f}"
    )

    print(
        f"LATEST val full  = "
        f"{latest['val_full']['mean']:.6f}"
    )

    print(
        f"LATEST val rec   = "
        f"{latest['val_rec']['mean']:.6f}"
    )

    print()

    print(
        "FNO reference:"
    )

    print(
        "  best val full = 0.551782"
    )

    print(
        "  best val rec  = 0.192650"
    )

    gain = (
        0.551782
        - best["val_full"]["mean"]
    ) / 0.551782

    print()
    print(
        "MgNO-I relative val-full gain "
        f"vs FNO = {100*gain:.2f}%"
    )

    if (
        best["val_full"]["mean"]
        < 0.551782
    ):
        print(
            "[PASS] MgNO-I improves over "
            "the frozen FNO smoke baseline."
        )
    else:
        print(
            "[NO GAIN] MgNO-I does not yet "
            "beat the FNO smoke baseline."
        )


if __name__ == "__main__":
    main()
