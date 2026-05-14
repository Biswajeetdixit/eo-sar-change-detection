import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.unet_siamese import UNet
from utils.dataset import ChangeDetectionDataset
from utils.metrics import compute_metrics

# -----------------------------------
# Device
# -----------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using Device:", device)

# -----------------------------------
# Dataset
# -----------------------------------
test_ds = ChangeDetectionDataset(
    "data/test",
    image_size=128
)

test_loader = DataLoader(
    test_ds,
    batch_size=1,
    shuffle=False
)

print("Test Samples:", len(test_ds))

# -----------------------------------
# Model
# -----------------------------------
model = UNet(in_channels=4).to(device)

# IMPORTANT:
# Load BEST trained model
model.load_state_dict(
    torch.load(
        "best_model.pth",
        map_location=device
    )
)

model.eval()

# -----------------------------------
# Metrics Storage
# -----------------------------------
precision_scores = []
recall_scores = []
iou_scores = []
f1_scores = []

# -----------------------------------
# Evaluation
# -----------------------------------
with torch.no_grad():

    progress_bar = tqdm(
        test_loader,
        desc="Evaluating"
    )

    for x, y in progress_bar:

        x = x.to(device)
        y = y.to(device)

        # Forward
        pred = model(x)

        # Convert probabilities to binary mask
        pred = (pred > 0.5).float()

        # Metrics
        precision, recall, iou, f1 = compute_metrics(
            pred,
            y
        )

        precision_scores.append(precision)
        recall_scores.append(recall)
        iou_scores.append(iou)
        f1_scores.append(f1)

# -----------------------------------
# Average Metrics
# -----------------------------------
avg_precision = (
    sum(precision_scores)
    /
    len(precision_scores)
)

avg_recall = (
    sum(recall_scores)
    /
    len(recall_scores)
)

avg_iou = (
    sum(iou_scores)
    /
    len(iou_scores)
)

avg_f1 = (
    sum(f1_scores)
    /
    len(f1_scores)
)

# -----------------------------------
# Final Results
# -----------------------------------
print("\n===== FINAL TEST RESULTS =====")

print(f"Precision : {avg_precision:.4f}")
print(f"Recall    : {avg_recall:.4f}")
print(f"IoU       : {avg_iou:.4f}")
print(f"F1 Score  : {avg_f1:.4f}")