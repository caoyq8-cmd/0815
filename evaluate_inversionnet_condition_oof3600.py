import os
import glob
import json

import numpy as np
import torch
import torch.nn.functional as F

from skimage.metrics import structural_similarity as ssim



def calc_metrics(pred, gt):

    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)

    mse = np.mean((pred - gt) ** 2)

    mae = np.mean(
        np.abs(pred - gt)
    )

    rmse = np.sqrt(mse)


    data_range = gt.max() - gt.min()

    psnr = (
        20 *
        np.log10(
            data_range / (rmse + 1e-12)
        )
    )


    ssim_value = ssim(
        gt,
        pred,
        data_range=data_range
    )


    return {
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse),
        "psnr": float(psnr),
        "ssim": float(ssim_value)
    }



def main():

    root = (
        "./condition_cache/"
        "inversionnet_oof5_3600/train"
    )


    files = sorted(
        glob.glob(
            os.path.join(
                root,
                "*.npz"
            )
        )
    )


    print(
        "num files =",
        len(files)
    )


    all_metrics=[]


    folds=[]


    for i,f in enumerate(files):

        data=np.load(f)


        pred=data[
            "condition_speed"
        ][0]


        gt=data[
            "target_speed"
        ][0]


        m=calc_metrics(
            pred,
            gt
        )


        m["sample_index"]=int(
            data["sample_index"][0]
        )

        m["fold"]=int(
            data["oof_fold"][0]
        )


        all_metrics.append(m)

        folds.append(
            m["fold"]
        )


        if (i+1)%500==0:
            print(
                f"{i+1}/{len(files)}"
            )



    keys=[
        "mse",
        "mae",
        "rmse",
        "psnr",
        "ssim"
    ]


    result={}


    for k in keys:

        values=[
            x[k]
            for x in all_metrics
        ]

        result[k+"_mean"]=float(
            np.mean(values)
        )

        result[k+"_std"]=float(
            np.std(values)
        )


    result["num_samples"]=len(files)


    fold_count={}

    for f in folds:
        fold_count[str(f)] = (
            fold_count.get(str(f),0)+1
        )


    result["fold_count"]=fold_count


    print("="*80)

    print(json.dumps(
        result,
        indent=2,
        ensure_ascii=False
    ))


    out="./condition_cache/inversionnet_oof5_3600_condition_metrics.json"

    with open(out,"w") as fp:

        json.dump(
            result,
            fp,
            indent=2
        )


    print(
        "saved:",
        out
    )



if __name__=="__main__":
    main()