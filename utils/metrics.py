import torch


def compute_metrics(pred, target, threshold=0.5):

    # -----------------------------------
    # Convert probabilities to binary
    # -----------------------------------
    pred = (pred > threshold).float()

    target = target.float()

    # -----------------------------------
    # True Positive
    # -----------------------------------
    tp = (pred * target).sum().item()

    # -----------------------------------
    # False Positive
    # -----------------------------------
    fp = ((pred == 1) & (target == 0)).sum().item()

    # -----------------------------------
    # False Negative
    # -----------------------------------
    fn = ((pred == 0) & (target == 1)).sum().item()

    # -----------------------------------
    # Metrics
    # -----------------------------------
    precision = tp / (tp + fp + 1e-8)

    recall = tp / (tp + fn + 1e-8)

    iou = tp / (tp + fp + fn + 1e-8)

    f1 = (
        2 * precision * recall
        /
        (precision + recall + 1e-8)
    )

    return precision, recall, iou, f1