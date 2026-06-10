import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

def create_mirror_unet() -> nn.Module:
    return smp.MAnet("resnet34", encoder_weights="imagenet", in_channels=5, classes=1, activation=None)

def _forward_pass(model: nn.Module, rgb: torch.Tensor, y_apple: torch.Tensor, mask_sloppy: torch.Tensor):
    net_input = torch.cat([rgb, y_apple, mask_sloppy], dim=1)

    R = model(net_input)

    y_pred = y_apple + R

    return R, y_pred

class ResidualLoss(nn.Module):
    def __init__(self, w_sup=10.0, w_reg=30.0, w_smooth=1.0):
        super().__init__()
        self.w_sup = w_sup      # Peso della correzione dello specchio
        self.w_reg = w_reg      # Peso della penalizzazione fuori dallo specchio
        self.w_smooth = w_smooth

    def forward(self, y_pred, target_perfect, mask_perfect, R):
        # SUPERVISION LOSS (Dentro lo specchio reale)
        # Forza la rete a calcolare il residuo corretto per pareggiare il RANSAC perfetto
        loss_sup = torch.sum(mask_perfect * torch.abs(y_pred - target_perfect)) / (torch.sum(mask_perfect) + 1e-8)

        # REGULARIZATION LOSS (Fuori dallo specchio reale)
        # Punisce SEVERAMENTE la rete se il residuo R è diverso da zero
        # nelle zone che sappiamo essere muro sano (dove mask_perfect è 0)
        outside_perfect = 1.0 - mask_perfect
        loss_reg = torch.mean(outside_perfect * torch.abs(R))

        # SMOOTHNESS LOSS SU Y_PRED (Solo dentro lo specchio)
        dy_dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])
        dy_dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        mask_x = mask_perfect[:, :, :, 1:] * mask_perfect[:, :, :, :-1]
        mask_y = mask_perfect[:, :, 1:, :] * mask_perfect[:, :, :-1, :]
        smooth_x = torch.sum(dy_dx * mask_x) / (torch.sum(mask_x) + 1e-8)
        smooth_y = torch.sum(dy_dy * mask_y) / (torch.sum(mask_y) + 1e-8)
        loss_smooth = smooth_x + smooth_y

        total = (self.w_sup * loss_sup) + (self.w_reg * loss_reg) + (self.w_smooth * loss_smooth)
        return total, {"superv": loss_sup.item(), "regulariz": loss_reg.item(), "smooth": loss_smooth.item()}