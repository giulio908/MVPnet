import os
import glob
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset import MirrorDataset
from model import _forward_pass, ResidualLoss, create_mirror_unet
from MVPNet.utils import _transforms


def get_test_dataloader(data_dir, transform, batch_size=20, num_workers=2):
    image_paths = glob.glob(os.path.join(data_dir, "image", "*.jpg"))
    all_base_names = sorted([os.path.splitext(os.path.basename(p))[0] for p in image_paths])

    valid_base_names = [
        name for name in all_base_names
        if os.path.exists(os.path.join(data_dir, "image", f"{name}.jpg")) and
           os.path.exists(os.path.join(data_dir, "depth_corr", f"{name}.npy"))
    ]
        
    train_val_names, test_names = train_test_split(valid_base_names, test_size=20, random_state=42)
    
    test_dataset = MirrorDataset(data_dir, test_names, transform)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return test_loader

def compute_metrics(pred, gt, mask=None, eps=1e-8, gt_threshold=1e-3):
    valid_gt_mask = (gt > gt_threshold)
    total_mask = mask.bool() & valid_gt_mask if mask is not None else valid_gt_mask

    if torch.sum(total_mask) == 0:
        return {"mae": 0.0, "rmse": 0.0, "abs_rel": 0.0}

    pred_valid = pred[total_mask].clamp(min=eps)
    gt_valid = gt[total_mask].clamp(min=eps)

    diff = pred_valid - gt_valid
    mae = torch.mean(torch.abs(diff))
    rmse = torch.sqrt(torch.mean(diff ** 2))
    abs_rel = torch.mean(torch.abs(diff) / gt_valid)

    return {"mae": mae.item(), "rmse": rmse.item(), "abs_rel": abs_rel.item()}

def evaluate_test_set(model, test_dataloader, criterion, device):
    model.eval()
    test_loss = 0.0
    
    totals = {k: 0.0 for k in [
        "superv", "regulariz", "smooth", "mae", "rmse", "abs_rel",
        "mirror_mae", "mirror_rmse", "mirror_abs_rel",
        "base_mae", "base_rmse", "base_abs_rel",
        "base_mirror_mae", "base_mirror_rmse", "base_mirror_abs_rel"
    ]}

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Test"):
            rgb = batch["rgb"].to(device)
            y_apple = batch["depth"].to(device)
            mask_sloppy = batch["mask_sloppy"].to(device)
            target_perfect = batch["target_perfect"].to(device)
            mask_perfect = batch["mask_perfect"].to(device)

            R, y_pred = _forward_pass(model, rgb, y_apple, mask_sloppy)
            loss, breakdown = criterion(y_pred, target_perfect, mask_perfect, R)
            test_loss += loss.item()
            
            for k, v in breakdown.items():
                totals[k] += v

            mirror_mask = mask_perfect.bool()

            gm = compute_metrics(y_pred, target_perfect)
            bgm = compute_metrics(y_apple, target_perfect)
            mm = compute_metrics(y_pred, target_perfect, mirror_mask)
            bmm = compute_metrics(y_apple, target_perfect, mirror_mask)

            for key in ["mae", "rmse", "abs_rel"]:
                totals[key] += gm[key]
                totals[f"base_{key}"] += bgm[key]
                totals[f"mirror_{key}"] += mm[key]
                totals[f"base_mirror_{key}"] += bmm[key]

    n = len(test_dataloader)
    
    print(f"\n{'='*65}\nRISULTATI TEST SET ({len(test_dataloader.dataset)} immagini)\n{'='*65}")
    print(f"\n[LOSS]\nLoss Totale:       {test_loss / n:.5f}")
    print(f"Supervisione:      {totals['superv'] / n:.5f}")
    print(f"Regolarizzazione:  {totals['regulariz'] / n:.5f}")
    print(f"Smoothness:        {totals['smooth'] / n:.5f}")

    print("\n[GLOBAL METRICS]\n  Depth Pro Baseline:")
    print(f"    MAE:      {totals['base_mae'] / n:.5f}\n    RMSE:     {totals['base_rmse'] / n:.5f}\n    AbsRel:   {totals['base_abs_rel'] / n:.5f}")
    print("  Ours (Modello Corretto):")
    print(f"    MAE:      {totals['mae'] / n:.5f}\n    RMSE:     {totals['rmse'] / n:.5f}\n    AbsRel:   {totals['abs_rel'] / n:.5f}")

    print("\n[MIRROR-ONLY METRICS]\n  Depth Pro Baseline:")
    print(f"    MAE:      {totals['base_mirror_mae'] / n:.5f}\n    RMSE:     {totals['base_mirror_rmse'] / n:.5f}\n    AbsRel:   {totals['base_mirror_abs_rel'] / n:.5f}")
    print("  Ours (Modello Corretto):")
    print(f"    MAE:      {totals['mirror_mae'] / n:.5f}\n    RMSE:     {totals['mirror_rmse'] / n:.5f}\n    AbsRel:   {totals['mirror_abs_rel'] / n:.5f}")

    res = {k: v / n for k, v in totals.items()}
    res["loss"] = test_loss / n
    return res

def main():
    DATA_DIR = "path_to_dataset"
    WEIGHTS_PATH = "path_to_weights.pth"
    BATCH_SIZE = 20
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizzo device: {device}")
    
    print("Inizializzazione Dataloader del Test Set...")
    
    _, val_test_transform = _transforms()

    test_loader = get_test_dataloader(
        data_dir=DATA_DIR, 
        transform= val_test_transform,
        batch_size=BATCH_SIZE
    )
    
    print("Caricamento MVPNet e Pesi...")
    model = create_mirror_unet().to(device)
    
    if os.path.exists(WEIGHTS_PATH):
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
        print(f"Pesi caricati correttamente da: {WEIGHTS_PATH}")
    else:
        print(f"ATTENZIONE: File dei pesi non trovato in {WEIGHTS_PATH}. Il modello userà pesi casuali.")

    criterion = ResidualLoss(w_sup=5.0, w_reg=10.0, w_smooth=0.2)
    
    evaluate_test_set(model, test_loader, criterion, device)

if __name__ == "__main__":
    main()