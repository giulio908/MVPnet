import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor

import depth_pro

IMG="_test7"
IMG_PATH = f"./data/image/img{IMG}.jpg"
MASK = f"./data/mask_sloppy/img{IMG}.png"
DEPTH =f"./data/depth/img{IMG}.npy"
DEPTH_CORR =f"./data/depth_corr/img{IMG}.npy"

def get_torch_device() -> torch.device:
    """Get the Torch device."""
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

device = get_torch_device()

print("Loading Apple Depth Pro...")
model,transform=depth_pro.create_model_and_transforms(device=device)

model.eval()

print(f"Loading image: {IMG_PATH}")
image, _, f_px = depth_pro.load_rgb(IMG_PATH)
image_tensor = transform(image)

print(f"Loading custom mask: {MASK}")
mirror_mask = cv2.imread(MASK, cv2.IMREAD_GRAYSCALE)

_, mirror_mask = cv2.threshold(mirror_mask, 127, 255, cv2.THRESH_BINARY)

print("Running Apple Depth Pro inference...")
prediction = model.infer(image_tensor, f_px=f_px)
depth = prediction["depth"].detach().cpu().numpy().squeeze()

if mirror_mask.shape != depth.shape:
    print("Resizing mask to match depth map resolution...")
    mirror_mask = cv2.resize(mirror_mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)

print("Extracting frame around the mirror...")
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
mask_dilated = cv2.dilate(mirror_mask, kernel, iterations=2)
cornice_mask = cv2.bitwise_xor(mask_dilated, mirror_mask)

if np.sum(mirror_mask == 255) > 0 and np.sum(cornice_mask == 255) > 0:
    print("Applying RANSAC geometric correction...")
    
    y_cornice, x_cornice = np.where(cornice_mask == 255)
    z_cornice = depth[y_cornice, x_cornice]

    X_inputs = np.column_stack((x_cornice, y_cornice))
    
    ransac = RANSACRegressor(residual_threshold=0.05)
    ransac.fit(X_inputs, z_cornice)
    
    y_specchio, x_specchio = np.where(mirror_mask == 255)
    X_da_predire = np.column_stack((x_specchio, y_specchio))
    
    z_corretti = ransac.predict(X_da_predire)
    
    corrected_depth = depth.copy()
    corrected_depth[y_specchio, x_specchio] = z_corretti
else:
    corrected_depth = depth.copy()

print("Preparing final plots...")

inv_depth_original = get_inverse_depth_viz(depth)
inv_depth_corrected = get_inverse_depth_viz(corrected_depth)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(inv_depth_original, cmap="turbo")
plt.title("Original Apple Depth Pro")
plt.colorbar()
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(inv_depth_corrected, cmap="turbo")
plt.title("Corrected Depth (Custom Mask + RANSAC)")
plt.colorbar()
plt.axis("off")
plt.tight_layout()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(depth, cmap="turbo")
plt.title("Original Apple Depth Pro No Norm")
plt.colorbar()
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(corrected_depth, cmap="turbo")
plt.title("Corrected Depth (Custom Mask + RANSAC) No Norm")
plt.colorbar()
plt.axis("off")
plt.tight_layout()

#Saving results
np.save(DEPTH, depth)
np.save(DEPTH_CORR, corrected_depth)

plt.show()