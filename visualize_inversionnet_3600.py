import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from pathlib import Path

from dataset_openbreastus_oldstyle import OpenBreastUSOldStyleDataset
from inversionnet import InversionNet


def save_img(x, path, title, vmin=None, vmax=None):
    plt.figure(figsize=(4,4))
    plt.imshow(
        x,
        cmap="inferno",
        vmin=vmin,
        vmax=vmax
    )
    plt.colorbar()
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_root",
        required=True
    )

    parser.add_argument(
        "--ckpt_path",
        required=True
    )

    parser.add_argument(
        "--output_dir",
        default="./vis_results"
    )

    parser.add_argument(
        "--indices",
        nargs="+",
        type=int,
        default=[155,278]
    )

    args = parser.parse_args()


    device=torch.device(
        "cuda:0" if torch.cuda.is_available()
        else "cpu"
    )

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )


    dataset=OpenBreastUSOldStyleDataset(
        root=args.data_root,
        split="test"
    )


    print("test size =",len(dataset))


    model=InversionNet(
        base_ch=32,
        bottleneck_blocks=2,
        dropout=0
    )


    ckpt=torch.load(
        args.ckpt_path,
        map_location=device
    )

    state=ckpt.get(
        "model",
        ckpt
    )

    model.load_state_dict(
        state,
        strict=False
    )

    model.to(device)
    model.eval()


    for idx in args.indices:

        x,y=dataset[idx]

        x=x.unsqueeze(0).to(device)
        y=y.unsqueeze(0).to(device)


        with torch.no_grad():

            pred=model(x)


        pred=pred.cpu().numpy()[0,0]
        gt=y.cpu().numpy()[0,0]


        err=np.abs(pred-gt)


        out=Path(args.output_dir)/f"sample_{idx:04d}"

        out.mkdir(
            exist_ok=True
        )


        save_img(
            gt,
            out/"gt.png",
            "Ground Truth",
            1400,
            1600
        )


        save_img(
            pred,
            out/"prediction.png",
            "InversionNet",
            1400,
            1600
        )


        save_img(
            err,
            out/"error.png",
            "Absolute Error"
        )


        real=x.cpu().numpy()[0,0]
        imag=x.cpu().numpy()[0,1]


        save_img(
            real,
            out/"input_real.png",
            "Input Real"
        )

        save_img(
            imag,
            out/"input_imag.png",
            "Input Imag"
        )


        mse=np.mean(
            (pred-gt)**2
        )

        mae=np.mean(
            np.abs(pred-gt)
        )


        print(
            f"sample {idx}:",
            "MSE=",mse,
            "MAE=",mae
        )


if __name__=="__main__":
    main()