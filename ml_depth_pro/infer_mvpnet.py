import depth_pro
import matplotlib.pyplot as plt

MONODEPTH_CONFIG_DICT = depth_pro.depth_pro.DepthProConfig(
    patch_encoder_preset="dinov2l16_384",
    image_encoder_preset="dinov2l16_384",
    checkpoint_uri="./ml_depth_pro/checkpoints/depth_pro.pt",
    decoder_features=256,
    use_fov_head=True,
    fov_encoder_preset="dinov2l16_384",
)

def infer_depth_pro(img_path):
    # Load model and preprocessing transform
    model, transform = depth_pro.create_model_and_transforms(config=MONODEPTH_CONFIG_DICT)
    model.eval()

    # Load and preprocess an image.
    image, _, f_px = depth_pro.load_rgb(img_path)
    image = transform(image)

    # Run inference.
    prediction = model.infer(image, f_px=f_px)

    depth = prediction["depth"].detach().cpu().numpy().squeeze()
    return depth

if __name__=='__main__':
    depth = infer_depth_pro("path_to_image")
    
    inverse_depth = 1 / depth
    # Visualize inverse depth instead of depth, clipped to [0.1m;250m] range for better visualization.
    max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
    min_invdepth_vizu = max(1 / 250, inverse_depth.min())
    inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (
        max_invdepth_vizu - min_invdepth_vizu
    )
    
    # Visualization
    plt.figure()
    plt.imshow(inverse_depth, cmap="turbo")
    plt.title("Original Depth Pro")
    plt.axis("off")

    plt.show()