import torch
import os
import glob
from tqdm import tqdm
from torch.utils.data import DataLoader
from MVPNet.model import _forward_pass, ResidualLoss, create_mirror_unet
from MVPNet.dataset import MirrorDataset
from MVPNet.utils import plot_learning_curves, _transforms
from sklearn.model_selection import train_test_split

def train_epoch(model, dataloader, optimizer, criterion, device, epoch, num_epochs):
    """Esegue un'epoca di addestramento e aggiorna i pesi."""
    model.train()
    running_loss = 0.0
    totals = {"superv": 0.0, "regulariz": 0.0, "smooth": 0.0}

    for batch in tqdm(dataloader, desc=f"Train Epoca {epoch + 1}/{num_epochs}"):
        rgb            = batch["rgb"].to(device)
        y_apple        = batch["depth"].to(device)
        target_perfect = batch["target_perfect"].to(device)
        mask_sloppy    = batch["mask_sloppy"].to(device)    
        mask_perfect   = batch["mask_perfect"].to(device) 

        R, y_pred = _forward_pass(model, rgb, y_apple, mask_sloppy)
        loss, breakdown = criterion(y_pred, target_perfect, mask_perfect, R)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        for k, v in breakdown.items():
            totals[k] += v

    n = len(dataloader)
    return running_loss / n, totals["superv"] / n, totals["regulariz"] / n, totals["smooth"] / n

def validate_epoch(model, dataloader, criterion, device, epoch, num_epochs):
    """Esegue un'epoca di validazione sui dati unseen (senza aggiornare i pesi)."""
    model.eval()
    running_loss = 0.0
    totals = {"superv": 0.0, "regulariz": 0.0, "smooth": 0.0}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Val Epoca {epoch + 1}/{num_epochs}", leave=False):
            rgb            = batch["rgb"].to(device)
            y_apple        = batch["depth"].to(device)
            target_perfect = batch["target_perfect"].to(device)
            mask_perfect   = batch["mask_perfect"].to(device)   
            mask_sloppy    = batch["mask_sloppy"].to(device)    

            R, y_pred = _forward_pass(model, rgb, y_apple, mask_sloppy)
            loss, breakdown = criterion(y_pred, target_perfect, mask_perfect, R)

            running_loss += loss.item()
            for k, v in breakdown.items():
                totals[k] += v

    n = len(dataloader)
    return running_loss / n, totals["superv"] / n, totals["regulariz"] / n, totals["smooth"] / n
    
def train_model(model, train_dataloader, val_dataloader, optimizer, criterion, device, num_epochs=30):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    history_train_loss = []
    history_val_loss = []

    for epoch in range(num_epochs):
        # Fase di Addestramento
        t_loss, t_sup, t_reg, t_smooth = train_epoch(
            model, train_dataloader, optimizer, criterion, device, epoch, num_epochs
        )

        # Fase di Validazione
        v_loss, v_sup, v_reg, v_smooth = validate_epoch(
            model, val_dataloader, criterion, device, epoch, num_epochs
        )

        scheduler.step()

        history_train_loss.append(t_loss)
        history_val_loss.append(v_loss)

        print(f"Epoca {epoch+1:02d} | "
              f"Train Loss: {t_loss:.4f} (Sup: {t_sup:.4f}, Reg: {t_reg:.4f}, Sm: {t_smooth:.4f}) | "
              f"Val Loss: {v_loss:.4f} (Sup: {v_sup:.4f}, Reg: {v_reg:.4f}, Sm: {v_smooth:.4f})")

    print("\nAddestramento concluso. Generazione del grafico delle losses...")
    plot_learning_curves(history_train_loss, history_val_loss)

    return model

DEPTHPRO_DIR = "path_to_dataset"
image_paths = glob.glob(os.path.join(DEPTHPRO_DIR, "image", "*.jpg"))

all_base_names = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]
all_base_names.sort()

print(f"Trovate {len(all_base_names)} istanze totali.")

valid_base_names = []
for name in all_base_names:
    if (os.path.exists(os.path.join(DEPTHPRO_DIR, "image", f"{name}.jpg")) and
        os.path.exists(os.path.join(DEPTHPRO_DIR, "depth_corr", f"{name}.npy"))):
        valid_base_names.append(name)

print(f"Istanze valide e complete: {len(valid_base_names)}\n")

train_val_names, test_names = train_test_split(
    valid_base_names, 
    test_size=20, 
    random_state=42
)

train_names, val_names = train_test_split(
    train_val_names, 
    test_size=0.15,  
    random_state=42
)

print("--- RIEPILOGO PARTIZIONAMENTO ---")
print(f"Train:      {len(train_names)} immagini")
print(f"Validation: {len(val_names)} immagini")
print(f"Test:       {len(test_names)} immagini\n")

train_transform, val_test_transform = _transforms()

train_dataset = MirrorDataset(base_dir=DEPTHPRO_DIR,base_names_list=train_names,transform=train_transform)
val_dataset   = MirrorDataset(base_dir=DEPTHPRO_DIR,base_names_list=val_names,transform=val_test_transform)
test_dataset  = MirrorDataset(base_dir=DEPTHPRO_DIR,base_names_list=test_names,transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=2)
test_loader  = DataLoader(test_dataset, batch_size=20, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nEsecuzione su device: {device}")

model = create_mirror_unet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = ResidualLoss(w_sup=5.0, w_reg=10.0, w_smooth=0.2)

print("Inizio addestramento della rete")
trained_model = train_model(
   model=model,
   train_dataloader=train_loader,
   val_dataloader=val_loader,
   optimizer=optimizer,
   criterion=criterion,
   device=device,
   num_epochs=75
)
print("Addestramento completato e modello salvato!")
torch.save(trained_model.state_dict(), "mirror_unet_freeres_5_10_02.pth")