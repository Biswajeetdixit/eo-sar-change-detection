import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.unet_siamese import UNet
from utils.dataset import ChangeDetectionDataset

# -----------------------------------
# Device
# -----------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using Device:", device)

# -----------------------------------
# Dataset
# -----------------------------------
train_ds = ChangeDetectionDataset(
    "data/train",
    image_size=256
)

val_ds = ChangeDetectionDataset(
    "data/val",
    image_size=256
)

print("Train Samples:", len(train_ds))
print("Val Samples:", len(val_ds))

# -----------------------------------
# DataLoader
# -----------------------------------
train_loader = DataLoader(
    train_ds,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_ds,
    batch_size=4,
    shuffle=False,
    num_workers=0
)

# -----------------------------------
# Model
# -----------------------------------
model = UNet(in_channels=4).to(device)

# -----------------------------------
# Optimizer
# -----------------------------------
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)

# -----------------------------------
# Loss Functions
# -----------------------------------
bce = torch.nn.BCELoss()


def dice_loss(pred, target):

    smooth = 1.0

    intersection = (pred * target).sum()

    dice = (
        (2.0 * intersection + smooth)
        /
        (pred.sum() + target.sum() + smooth)
    )

    return 1 - dice


# -----------------------------------
# Training Settings
# -----------------------------------
epochs = 10

best_loss = float("inf")

# -----------------------------------
# Training Loop
# -----------------------------------
for epoch in range(epochs):

    model.train()

    total_train_loss = 0

    train_bar = tqdm(
        train_loader,
        desc=f"Training Epoch {epoch+1}/{epochs}"
    )

    for x, y in train_bar:

        x = x.to(device)
        y = y.to(device)

        # Forward
        pred = model(x)

        # Loss
        loss = (
            0.5 * bce(pred, y)
            +
            0.5 * dice_loss(pred, y)
        )

        # Backprop
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_train_loss += loss.item()

        train_bar.set_postfix({
            "loss": f"{loss.item():.4f}"
        })

    avg_train_loss = total_train_loss / len(train_loader)

    # -----------------------------------
    # Validation
    # -----------------------------------
    model.eval()

    total_val_loss = 0

    with torch.no_grad():

        val_bar = tqdm(
            val_loader,
            desc=f"Validation Epoch {epoch+1}/{epochs}"
        )

        for x, y in val_bar:

            x = x.to(device)
            y = y.to(device)

            pred = model(x)

            val_loss = (
                0.5 * bce(pred, y)
                +
                0.5 * dice_loss(pred, y)
            )

            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(val_loader)

    print(f"\nEpoch {epoch+1}/{epochs}")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Val Loss  : {avg_val_loss:.4f}\n")

    # -----------------------------------
    # Save Best Model
    # -----------------------------------
    if avg_val_loss < best_loss:

        best_loss = avg_val_loss

        torch.save(
            model.state_dict(),
            "best_model.pth"
        )

        print("Best model saved!\n")

print("Training Finished!")