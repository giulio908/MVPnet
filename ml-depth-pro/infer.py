from PIL import Image
import depth_pro
import cv2
import numpy as np

# Load model and preprocessing transform
model, transform = depth_pro.create_model_and_transforms()
model.eval()

# Load and preprocess an image.
image, _, f_px = depth_pro.load_rgb("data/img001.jpg")
image = transform(image)

# Run inference.
prediction = model.infer(image, f_px=f_px)

depth = prediction["depth"].detach().cpu().numpy().squeeze()
print(depth.shape)
np.save('img001.npy', depth)

inverse_depth = 1 / depth
# Visualize inverse depth instead of depth, clipped to [0.1m;250m] range for better visualization.
max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
min_invdepth_vizu = max(1 / 250, inverse_depth.min())
inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (
    max_invdepth_vizu - min_invdepth_vizu
)
        
import matplotlib.pyplot as plt
import numpy as np

# Visualization
plt.figure()
plt.imshow(inverse_depth, cmap="turbo")
plt.title("Original Depth Pro")
plt.axis("off")

plt.show()