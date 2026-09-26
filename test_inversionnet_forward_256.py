import torch

from InversionNet_modules.Baselines_modified import InversionNet


device="cuda"

model=InversionNet().to(device)

x=torch.randn(
    1,
    2,
    256,
    256
).to(device)


with torch.no_grad():

    y=model(x)


print("input:",x.shape)
print("output:",y.shape)