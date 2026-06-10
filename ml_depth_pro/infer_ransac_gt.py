import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
import argparse 
import depth_pro

# ==========================================
# 1. Setup & Arguments
# ==========================================
parser = argparse.ArgumentParser(description="Run Depth Pro with RANSAC correction.")
parser.add_argument("--img", type=str, required=True, help="Image suffix (e.g., _test7, 001)")
args = parser.parse_args()

IMG = args.img
IMG_PATH = f"./data/image/img{IMG}.jpg"
MASK_SLOPPY = f"./data/mask_sloppy/img{IMG}.png"
MASK = f"./data/mask/img{IMG}.png"
DEPTH = f"./data/depth/img{IMG}.npy"
DEPTH_CORR = f"./data/depth_corr/img{IMG}.npy"
DEPTH_GT = f"./data/depth_gt/img{IMG}.npy"

# ==========================================
# 2. Helper Functions
# ==========================================
def get_torch_device() -> torch.device:
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    return device

def get_inverse_depth_viz(depth_matrix):
    inv_depth = 1 / depth_matrix
    max_inv = min(inv_depth.max(), 1 / 0.1)
    min_inv = max(1 / 250, inv_depth.min())
    return (inv_depth - min_inv) / (max_inv - min_inv)

def apply_ransac_correction(depth_map, mask_path):
    print(f"  -> Loading sloppy mask: {mask_path}")
    mirror_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, mirror_mask = cv2.threshold(mirror_mask, 127, 255, cv2.THRESH_BINARY)

    if mirror_mask.shape != depth_map.shape:
        mirror_mask = cv2.resize(mirror_mask, (depth_map.shape[1], depth_map.shape[0]), interpolation=cv2.INTER_NEAREST)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask_dilated = cv2.dilate(mirror_mask, kernel, iterations=2)
    cornice_mask = cv2.bitwise_xor(mask_dilated, mirror_mask)

    if np.sum(mirror_mask == 255) > 0 and np.sum(cornice_mask == 255) > 0:
        print("  -> Applying RANSAC geometric correction...")
        y_cornice, x_cornice = np.where(cornice_mask == 255)
        z_cornice = depth_map[y_cornice, x_cornice]

        X_inputs = np.column_stack((x_cornice, y_cornice))
        
        ransac = RANSACRegressor(residual_threshold=0.05)
        ransac.fit(X_inputs, z_cornice)
        
        y_specchio, x_specchio = np.where(mirror_mask == 255)
        X_da_predire = np.column_stack((x_specchio, y_specchio))
        
        z_corretti = ransac.predict(X_da_predire)
        
        corrected_depth = depth_map.copy()
        corrected_depth[y_specchio, x_specchio] = z_corretti
    else:
        print("  -> No valid mirror/frame found. Skipping RANSAC.")
        corrected_depth = depth_map.copy()
        
    return corrected_depth

# ==========================================
# 3. Main Execution
# ==========================================
device = get_torch_device()

print("Loading Apple Depth Pro...")
model, transform = depth_pro.create_model_and_transforms(device=device)
model.eval()

print(f"Loading image: {IMG_PATH}")
image, _, f_px = depth_pro.load_rgb(IMG_PATH)
image_tensor = transform(image)

print("Running Apple Depth Pro inference...")
prediction = model.infer(image_tensor, f_px=f_px)
depth = prediction["depth"].detach().cpu().numpy().squeeze()

print("Processing Predicted Depth with perfect Mask")
depth_gt = apply_ransac_correction(depth, MASK)

print("\nProcessing Predicted Depth with SLOPPY Mask...")
depth_corr = apply_ransac_correction(depth, MASK_SLOPPY)

# ==========================================
# 4. Visualization & Saving
# ==========================================
print("\nPreparing final plots...")

inv_pred = get_inverse_depth_viz(depth)
inv_pred_corr = get_inverse_depth_viz(depth_corr)
inv_gt = get_inverse_depth_viz(depth_gt)

fig, axs = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Depth Correction Comparison (Normalized Inverse)", fontsize=16)

axs[0].imshow(inv_pred, cmap="turbo")
axs[0].set_title("1. Original Prediction")
axs[0].axis("off")

axs[1].imshow(inv_pred_corr, cmap="turbo")
axs[1].set_title("2. Corrected (Sloppy Mask + RANSAC)")
axs[1].axis("off")

axs[2].imshow(inv_gt, cmap="turbo")
axs[2].set_title("3. Ground Truth (Clean Mask + RANSAC)")
axs[2].axis("off")

plt.tight_layout()

print("\nSaving predicted depth maps...")
np.save(DEPTH, depth)
np.save(DEPTH_CORR, depth_corr)
np.save(DEPTH_GT, depth_gt)

plt.show()