import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import logging
from datetime import datetime
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler


from model import GeoLG3DFaultNet
from dataset import SeismicDataset3D


from utils import dice_score, iou_score, precision_recall, accuracy_score, f1_score
from utils import GeoLGFaultLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

BATCH_SIZE = 1
LR = 1e-4
NUM_EPOCHS = 300
DATA_SHAPE = (128, 128, 128)
CROP_SIZE = (64, 64, 64)
GRAD_ACCUM_STEPS = 4


base_dir = './data'  
train_seis_dir = os.path.join(base_dir, 'train/seis')
train_fault_dir = os.path.join(base_dir, 'train/fault')
val_seis_dir = os.path.join(base_dir, 'validation/seis')
val_fault_dir = os.path.join(base_dir, 'validation/fault')

checkpoint_dir = './checkpoints'
best_dir = os.path.join(checkpoint_dir, 'best')
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(best_dir, exist_ok=True)


logging.basicConfig(level=logging.INFO, format='%(message)s')


def validate(model, loader):
    model.eval()
    val_dice_list = []
    
    with torch.no_grad():
        for seis, fault in loader:
            seis, fault = seis.to(device), fault.to(device)
   
            preds = model(seis)

            d_score = dice_score(preds, fault).item()
            val_dice_list.append(d_score)
            
    return np.mean(val_dice_list)


if __name__ == "__main__":
    

    print("⏳ Loading Dataset...")
    train_dataset = SeismicDataset3D(train_seis_dir, train_fault_dir, shape=DATA_SHAPE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    val_dataset = SeismicDataset3D(val_seis_dir, val_fault_dir, shape=DATA_SHAPE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print("✅ Dataset Loaded successfully!")


    model = GeoLG3DFaultNet(in_channels=1, num_classes=2).to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler()


    criterion = GeoLGFaultLoss(weight_bce=1.0, weight_dice=1.0, weight_cldice=0.5, weight_edge=0.5)


    best_dice = 0.0
    print("🎯 Starting Training Process...")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        start_time = time.time()
        
        for i, (seis, fault) in enumerate(train_loader):
            seis, fault = seis.to(device), fault.to(device)
            

            with autocast():
                preds = model(seis)

                loss = criterion(preds, fault) 
                loss = loss / GRAD_ACCUM_STEPS


            scaler.scale(loss).backward()
            

            if (i + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * GRAD_ACCUM_STEPS
            

        epoch_time = time.time() - start_time
        avg_train_loss = epoch_loss / len(train_loader)
        logging.info(f"Epoch [{epoch+1:03d}/{NUM_EPOCHS}] | Time: {epoch_time:.1f}s | Train Loss (Total): {avg_train_loss:.4f}")
        

        current_val_dice = validate(model, val_loader)


        if current_val_dice > best_dice:
            diff = current_val_dice - best_dice
            best_dice = current_val_dice

            state_dict = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_dice': best_dice
            }

            save_path = os.path.join(best_dir, 'best_model.pth')
            torch.save(state_dict, save_path)

            logging.info(f"  🔥 [NEW RECORD] Val Dice improved by {diff:.4f}! Current Best: {best_dice:.4f} (Saved to best_model.pth)")
        else:
            logging.info(f"  ... Val Dice: {current_val_dice:.4f} (No improvement)")


        if (epoch + 1) % 20 == 0:
            ck = os.path.join(checkpoint_dir, f'geolg_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch + 1, 
                'model_state_dict': model.state_dict()
            }, ck)
            logging.info(f"  💾 Saved periodic checkpoint to {ck}")