from utils.dataset import ChangeDetectionDataset

ds = ChangeDetectionDataset("data/train")

x, y = ds[0]

print(x.shape)
print(y.shape)