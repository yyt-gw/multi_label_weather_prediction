import torch


def criterion(loss_func, preds, gts, device):
    losses = 0
    for i, key in enumerate(preds):
        losses += loss_func(
            preds[key],
            torch.unsqueeze(gts[key], 1).float().to(device),
        )
    return losses
