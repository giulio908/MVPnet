import numpy as np
import matplotlib.pyplot as plt
import albumentations as A

def get_inverse_depth_viz(depth_matrix, min_inv=None, max_inv=None):
    inv_depth = 1 / depth_matrix

    if max_inv is None:
        max_inv = min(inv_depth.max(), 1 / 0.1)
    if min_inv is None:
        min_inv = max(1 / 250, inv_depth.min())

    if max_inv == min_inv:
        return np.zeros_like(inv_depth), min_inv, max_inv

    norm_depth = (inv_depth - min_inv) / (max_inv - min_inv)
    return norm_depth, min_inv, max_inv

def plot_learning_curves(train_losses, val_losses):
    """Disegna il grafico di Train vs Validation per individuare l'overfitting."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='orange', linewidth=2)
    
    plt.xlabel('Epoche')
    plt.ylabel('Loss Totale')
    plt.title('Curva di Apprendimento')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    
def _transforms():
    train_transform = A.Compose([
        A.Resize(height=512, width=512),
        
        A.HorizontalFlip(p=0.5),
        
        A.OneOf([
            A.Affine(translate_percent=(-0.05, 0.05), scale=(0.95, 1.05), rotate=(-10, 10), mode=0),
            A.Perspective(scale=(0.02, 0.05)),
        ], p=0.4),

        A.OneOf([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10),
        ], p=0.6),
        
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MedianBlur(blur_limit=3),
        ], p=0.3),

    ], additional_targets={
        'mask_sloppy': 'mask',
        'depth': 'mask',
        'r_sloppy': 'mask',
        't_perfect': 'mask'
    })

    val_test_transform = A.Compose([
        A.Resize(height=512, width=512),
    ], additional_targets={
        'mask_sloppy': 'mask',
        'depth': 'mask',
        'r_sloppy': 'mask',
        't_perfect': 'mask'
    })
    
    return train_transform, val_test_transform