from dataset_openbreastus_oldstyle import OpenBreastUSOldStyleDataset

root = r"/home/featurize/datasets/90024ebe-ceca-4e0d-aab7-55f496b4b5f5"

train_set = OpenBreastUSOldStyleDataset(root_dir=root, split="train")
test_set = OpenBreastUSOldStyleDataset(root_dir=root, split="test")

print("train size =", len(train_set))
print("test size  =", len(test_set))

x, y = train_set[0]
print("x shape =", x.shape)
print("y shape =", y.shape)
print("x dtype =", x.dtype)
print("y dtype =", y.dtype)
print("x min/max =", x.min().item(), x.max().item())
print("y min/max =", y.min().item(), y.max().item())