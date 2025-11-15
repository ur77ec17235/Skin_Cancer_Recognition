import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5, reduction: str = "mean"):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, predictions, targets):
        # predictions: [B, C, H, W], targets: [B, H, W]
        predictions = F.softmax(predictions, dim=1)
        
        # One-hot encode targets
        num_classes = predictions.shape[1]
        targets_onehot = F.one_hot(targets, num_classes=num_classes)
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()
        
        # Flatten
        predictions_flat = predictions.view(predictions.shape[0], num_classes, -1)
        targets_flat = targets_onehot.view(targets_onehot.shape[0], num_classes, -1)
        
        # Dice coefficient
        intersection = (predictions_flat * targets_flat).sum(dim=2)
        union = predictions_flat.sum(dim=2) + targets_flat.sum(dim=2)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - dice
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, predictions, targets):
        # predictions: [B, C], targets: [B]
        ce_loss = F.cross_entropy(predictions, targets, reduction="none")
        p_t = torch.exp(-ce_loss)
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = self.alpha * focal_weight * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss

class CombinedLoss(nn.Module):
    def __init__(self, loss_type: str = "focal", **kwargs):
        super().__init__()
        if loss_type.lower() == "dice":
            self.loss = DiceLoss(**kwargs)
        elif loss_type.lower() == "focal":
            self.loss = FocalLoss(**kwargs)
        else:
            self.loss = nn.CrossEntropyLoss()
    
    def forward(self, predictions, targets):
        return self.loss(predictions, targets)
