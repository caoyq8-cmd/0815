import os
import math
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from dataset_openbreastus_oldstyle import OpenBreastUSOldStyleDataset

from InversionNet_modules.Baselines_openbreastus_256 import InversionNet



# ======================
# utils
# ======================

def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def denormalize(
    x,
    vmin=1400,
    vmax=1600
):

    x=(x+1)/2

    return x*(vmax-vmin)+vmin



# ======================
# loss
# ======================


def image_gradient(x):

    dx=x[:,:,:,1:]-x[:,:,:,:-1]

    dy=x[:,:,1:,:]-x[:,:,:-1,:]

    return dx,dy



class GradientLoss(nn.Module):

    def forward(self,pred,target):

        pdx,pdy=image_gradient(pred)

        tdx,tdy=image_gradient(target)


        return (
            F.l1_loss(pdx,tdx)
            +
            F.l1_loss(pdy,tdy)
        )



class CompositeLoss(nn.Module):

    def __init__(
        self,
        l1=1.0,
        mse=0.2,
        grad=0.05
    ):

        super().__init__()

        self.l1=l1
        self.mse=mse
        self.grad=grad

        self.grad_loss=GradientLoss()



    def forward(self,pred,target):

        loss = (
            self.l1*F.l1_loss(pred,target)
            +
            self.mse*F.mse_loss(pred,target)
            +
            self.grad*self.grad_loss(pred,target)
        )

        return loss



# ======================
# metrics
# ======================


def calc_psnr(pred,target):

    mse=torch.mean(
        (pred-target)**2
    ).item()

    if mse<1e-12:
        return 99

    return (
        20*math.log10(
            2.0/mse**0.5
        )
    )



# ======================
# train
# ======================


def main():

    parser=argparse.ArgumentParser()


    parser.add_argument(
        "--data_root",
        required=True
    )

    parser.add_argument(
        "--output_dir",
        default="./experiments_inversionnet_official"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4
    )


    args=parser.parse_args()



    os.makedirs(
        args.output_dir,
        exist_ok=True
    )


    seed_everything(42)


    device="cuda" if torch.cuda.is_available() else "cpu"


    print("device =",device)


    train_set=OpenBreastUSOldStyleDataset(
        args.data_root,
        split="train"
    )

    test_set=OpenBreastUSOldStyleDataset(
        args.data_root,
        split="test"
    )


    train_loader=DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )


    test_loader=DataLoader(
        test_set,
        batch_size=1,
        shuffle=False
    )


    print(
        "train size=",
        len(train_set)
    )

    print(
        "test size=",
        len(test_set)
    )


    model=InversionNet().to(device)


    print(
        "params=",
        sum(
            p.numel()
            for p in model.parameters()
        )
    )


    optimizer=torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )


    criterion=CompositeLoss()


    scaler=torch.cuda.amp.GradScaler()



    best=1e9



    for epoch in range(1,args.epochs+1):


        model.train()

        total=0


        for x,y in train_loader:


            x=x.to(device)

            y=y.to(device)


            optimizer.zero_grad()



            with torch.cuda.amp.autocast():

                pred=model(x)

                loss=criterion(
                    pred,
                    y
                )


            scaler.scale(loss).backward()

            scaler.step(
                optimizer
            )

            scaler.update()


            total+=loss.item()



        avg=total/len(train_loader)



        # validation

        model.eval()

        mse_all=[]
        psnr_all=[]


        with torch.no_grad():

            for x,y in test_loader:


                x=x.to(device)
                y=y.to(device)


                pred=model(x)


                pred=denormalize(pred)
                gt=denormalize(y)


                mse=torch.mean(
                    (pred-gt)**2
                ).item()


                mse_all.append(mse)

                psnr_all.append(
                    20*math.log10(
                        200/(mse**0.5)
                    )
                )



        val_mse=np.mean(mse_all)

        val_psnr=np.mean(psnr_all)



        print(
            f"Epoch {epoch:03d} "
            f"train={avg:.6f} "
            f"MSE={val_mse:.4f} "
            f"PSNR={val_psnr:.4f}"
        )



        if val_mse<best:


            best=val_mse


            torch.save(
                {
                    "epoch":epoch,
                    "model":model.state_dict(),
                    "val_mse":best
                },
                os.path.join(
                    args.output_dir,
                    "best.pth"
                )
            )

            print("saved best")



if __name__=="__main__":

    main()