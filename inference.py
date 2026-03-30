import numpy as np
import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_objects

from model import GeoLG3DFaultNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("🚀 Initializing GeoLG-3DFaultNet Architecture...")
model = GeoLG3DFaultNet(in_channels=1, num_classes=2).to(device)


checkpoint_path = "./best_model.pth"

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    

    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('lgfe.', 'lg.').replace('fcom.', 'aco.')
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict, strict=False)
    print(f"✅ Successfully loaded and mapped pre-trained weights from {checkpoint_path}")
    model.eval()
else:
    print(f"❌ Error: Weights not found at {checkpoint_path}")
    exit()


def predict_with_tta(model, x):
    pred1 = model(x)
    pred1 = F.softmax(pred1, dim=1)[:, 1:2]  

    x_flip_w = torch.flip(x, dims=[-1])
    pred2 = model(x_flip_w)
    pred2 = F.softmax(pred2, dim=1)[:, 1:2]
    pred2 = torch.flip(pred2, dims=[-1]) 

    x_flip_h = torch.flip(x, dims=[-2])
    pred3 = model(x_flip_h)
    pred3 = F.softmax(pred3, dim=1)[:, 1:2]
    pred3 = torch.flip(pred3, dims=[-2])  

    x_flip_d = torch.flip(x, dims=[-3])
    pred4 = model(x_flip_d)
    pred4 = F.softmax(pred4, dim=1)[:, 1:2]
    pred4 = torch.flip(pred4, dims=[-3])  

    avg_pred = (pred1 + pred2 + pred3 + pred4) / 4.0
    return avg_pred

def infer(volume, shape=(128, 128, 128), chunk_size=96):
    mean = volume.mean()
    std = volume.std() + 1e-6
    volume_norm = (volume - mean) / std

    pred_label = np.zeros(shape, dtype=np.float32)
    count_map = np.zeros(shape, dtype=np.float32)

    with torch.no_grad():
        stride = chunk_size // 2
        for z in range(0, shape[0], stride):
            for y in range(0, shape[1], stride):
                for x in range(0, shape[2], stride):
                    z_start = min(z, shape[0] - chunk_size)
                    y_start = min(y, shape[1] - chunk_size)
                    x_start = min(x, shape[2] - chunk_size)

                    z_end = z_start + chunk_size
                    y_end = y_start + chunk_size
                    x_end = x_start + chunk_size

                    chunk = volume_norm[z_start:z_end, y_start:y_end, x_start:x_end]
                    chunk_tensor = torch.tensor(chunk).unsqueeze(0).unsqueeze(0).float().to(device)

                    pred_prob = predict_with_tta(model, chunk_tensor).squeeze().cpu().numpy()

                    pred_label[z_start:z_end, y_start:y_end, x_start:x_end] += pred_prob
                    count_map[z_start:z_end, y_start:y_end, x_start:x_end] += 1.0

    pred_label = pred_label / count_map
    pred_binary = (pred_label > 0.5).astype(np.float32)
    return pred_binary


def post_process(pred_mask, min_size=50):

    mask_bool = pred_mask > 0.5
    clean_mask = remove_small_objects(mask_bool, min_size=min_size)
    return clean_mask.astype(np.float32)

def save_visualization(seis_data, fault_data, pred_data, save_path, shape=(128, 128, 128)):

    slice_indices = [shape[0] // 2, shape[1] // 2, shape[2] // 2]
    

    items = [
        ("Seismic Data", seis_data, "gray"),
        ("Ground Truth", fault_data, "gray"),
        ("GeoLG-3DFaultNet (Ours)", pred_data, "hot") 
    ]

    fig, axes = plt.subplots(len(items), 3, figsize=(15, 5 * len(items)))

    for row, (title, data, cmap) in enumerate(items):

        axes[row, 0].imshow(data[slice_indices[0], :, :], cmap=cmap, aspect="auto")
        axes[row, 0].set_title(f"{title} - X slice {slice_indices[0]}")

        axes[row, 1].imshow(data[:, slice_indices[1], :], cmap=cmap, aspect="auto")
        axes[row, 1].set_title(f"{title} - Y slice {slice_indices[1]}")

        axes[row, 2].imshow(data[:, :, slice_indices[2]], cmap=cmap, aspect="auto")
        axes[row, 2].set_title(f"{title} - Z slice {slice_indices[2]}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":

    val_seis_dir = "./data/seis"
    val_fault_dir = "./data/fault"
    save_img_dir = "./predict_images"

    os.makedirs(save_img_dir, exist_ok=True)


    files = sorted([f for f in os.listdir(val_seis_dir) if f.endswith(".dat")])
    print(f"🔍 Found {len(files)} samples for Quick Test.")

    for i, f in enumerate(files):
        print(f"\n[{i + 1}/{len(files)}] Processing {f} ...")
        
        seis_path = os.path.join(val_seis_dir, f)
        fault_path = os.path.join(val_fault_dir, f)
        save_img_path = os.path.join(save_img_dir, f.replace(".dat", "_result.png"))


        try:
            seis_vol = np.fromfile(seis_path, dtype=np.float32).reshape((128, 128, 128))
            fault_vol = np.fromfile(fault_path, dtype=np.float32).reshape((128, 128, 128))
        except Exception as e:
            print(f"⚠️ Error loading data for {f}: {e}")
            continue


        print("  -> Running TTA Inference...")
        pred_vol = infer(seis_vol, chunk_size=96)


        print("  -> Applying Morphology Post-Processing...")
        pred_vol = post_process(pred_vol)


        print(f"  -> Saving visualization to {save_img_path}...")
        save_visualization(seis_vol, fault_vol, pred_vol, save_img_path)

    print("\n🎉 All tests completed successfully!")