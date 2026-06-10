import os
import torch
import cv2
from config import mirrornet_weights
import numpy as np
import matplotlib.pyplot as plt
from MVPNet.utils import get_inverse_depth_viz
from MVPNet.model import _forward_pass, create_mirror_unet
from ml_depth_pro.infer_mvpnet import infer_depth_pro
from ICCV2019_MirrorNet.infer_mvpnet import infer_mirrornet
from PIL import Image

IMG_PATH = "./data"
DEPTH_PATH = "./data"
MASK_PATH = "./data"
OUT_DIR = "./data/res/"
CKPT = "./checkpoints/mvpnet_5_10_02.pth"

def test_external_image(model, img_path, depth_pro_path, mask_sloppy_path, device, out_dir, target_size=(512, 512)):
    """Carica un'immagine esterna con la depth e la maschera, valutandola in modalità End-to-End."""

    if not os.path.exists(img_path): raise FileNotFoundError(f"Immagine non trovata: {img_path}")
    if not os.path.exists(depth_pro_path): raise FileNotFoundError(f"Depth originale non trovata: {depth_pro_path}")
    if not os.path.exists(mask_sloppy_path): raise FileNotFoundError(f"Mask Sloppy non trovata: {mask_sloppy_path}")

    model.eval()
    print(f"\nElaborazione di: {os.path.basename(img_path)}")

    rgb = cv2.imread(img_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, target_size).astype(np.float32) / 255.0
    rgb = np.transpose(rgb, (2, 0, 1))

    depth_raw = np.load(depth_pro_path).astype(np.float32)
    depth_raw_resized = cv2.resize(depth_raw, target_size, interpolation=cv2.INTER_LINEAR)

    depth_norm, min_inv, max_inv = get_inverse_depth_viz(depth_raw)
    depth_norm = cv2.resize(depth_norm, target_size)

    mask_sloppy = cv2.imread(mask_sloppy_path, cv2.IMREAD_GRAYSCALE)
    mask_sloppy = cv2.resize(mask_sloppy, target_size, interpolation=cv2.INTER_NEAREST)
    mask_sloppy = (mask_sloppy > 127).astype(np.float32)

    t_rgb    = torch.from_numpy(rgb).unsqueeze(0).to(device)
    t_depth  = torch.from_numpy(depth_norm).unsqueeze(0).unsqueeze(0).to(device)
    t_mask   = torch.from_numpy(mask_sloppy).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        R_norm, y_pred_norm = _forward_pass(model, t_rgb, t_depth, t_mask)

    rgb_vis = t_rgb.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    pred_norm_np = y_pred_norm.squeeze().cpu().numpy()
    r_vis = R_norm.squeeze().cpu().numpy()

    inv_depth_corretta = (pred_norm_np * (max_inv - min_inv)) + min_inv

    metric_depth_pred = 1.0 / (inv_depth_corretta + 1e-8)

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"Test Immagine Esterna (Scale in Metri Reali): {os.path.basename(img_path)}", fontsize=14, fontweight='bold')

    axs[0].imshow(rgb_vis)
    axs[0].set_title("Input RGB")
    axs[0].axis("off")

    im1 = axs[1].imshow(depth_raw_resized, cmap='turbo')
    axs[1].set_title("Depth Pro (Originale)")
    axs[1].axis("off")
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04, label="Metri")

    vmax = max(abs(r_vis.min()), abs(r_vis.max()), 0.01)
    im2 = axs[2].imshow(r_vis, cmap='coolwarm', vmin=-vmax, vmax=vmax)
    axs[2].set_title("Residuo Predetto")
    axs[2].axis("off")
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    im3 = axs[3].imshow(metric_depth_pred, cmap='turbo')
    axs[3].set_title("MVPNet")
    axs[3].axis("off")
    plt.colorbar(im3, ax=axs[3], fraction=0.046, pad=0.04, label="Metri")

    base_name = os.path.splitext(os.path.basename(img_path))[0]
    os.makedirs(out_dir, exist_ok=True)

    rgb_bgr = cv2.cvtColor((rgb_vis * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{out_dir}{base_name}_rgb.png", rgb_bgr)

    depth_uint8 = (cv2.resize(depth_norm, target_size) * 255.0).astype(np.uint8)
    cv2.imwrite(f"{out_dir}{base_name}_depth_orig.png", cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO))

    pred_uint8 = (pred_norm_np * 255.0).astype(np.uint8)
    cv2.imwrite(f"{out_dir}{base_name}_pred.png", cv2.applyColorMap(pred_uint8, cv2.COLORMAP_TURBO))

    np.save(f"{out_dir}{base_name}_pred_metric.npy", metric_depth_pred)

    plt.tight_layout()
    plt.show()

# Utils function to clear memory    
def clear_memory():
    import gc
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nEsecuzione su device: {device}")

    print("\n[1/3] Generazione Depth...")

    depth_pro_res = infer_depth_pro(f"{IMG_PATH}/img_test.jpg")

    np.save(f"{DEPTH_PATH}/depth.npy", depth_pro_res)

    clear_memory()

    print("\n[2/3] Generazione Mask...")

    mask = infer_mirrornet(
        f"{IMG_PATH}/img_test.jpg",
        mirrornet_weights
    )

    Image.fromarray(mask).save(
        os.path.join(MASK_PATH, "mask.png")
    )

    clear_memory()

    print("\n[3/3] MVPNet inference...")

    model = create_mirror_unet().to(device)

    model.load_state_dict(
        torch.load(CKPT, map_location=device)
    )

    model.eval()

    test_external_image(
        model=model,
        img_path=f"{IMG_PATH}/img_test.jpg",
        depth_pro_path=f"{DEPTH_PATH}/depth.npy",
        mask_sloppy_path=f"{MASK_PATH}/mask.png",
        device=device,
        out_dir=OUT_DIR,
        target_size=(512, 512)
    )

    del model

    clear_memory()
    
if __name__=='__main__':    
    main()