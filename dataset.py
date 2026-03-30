import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class SeismicDataset3D(Dataset):
    def __init__(self, seis_dir, fault_dir, shape=(128, 128, 128), seis_dtype=np.float32, fault_dtype=np.float32):
        self.seis_dir = seis_dir
        self.fault_dir = fault_dir
        self.shape = shape
        self.seis_dtype = seis_dtype
        self.fault_dtype = fault_dtype


        self.file_list = sorted([f for f in os.listdir(seis_dir) if f.endswith('.dat')])


        self.data_cache = []
        print(f"🚀 Pre-loading {len(self.file_list)} original samples into RAM...")

        for filename in tqdm(self.file_list):
            seis_path = os.path.join(self.seis_dir, filename)
            fault_path = os.path.join(self.fault_dir, filename)

            try:

                s_np = np.fromfile(seis_path, dtype=self.seis_dtype).reshape(self.shape)

                f_np = np.fromfile(fault_path, dtype=self.fault_dtype).reshape(self.shape)

                mean = s_np.mean()
                std = s_np.std() + 1e-6
                s_np = (s_np - mean) / std



                s_tensor = torch.tensor(s_np).unsqueeze(0).float()
                f_tensor = torch.tensor(f_np).unsqueeze(0).float()

                self.data_cache.append((s_tensor, f_tensor))
            except Exception as e:
                print(f"Skipping {filename}: {e}")

        print(f"✅ Loaded {len(self.data_cache)} samples into RAM.")

    def __len__(self):
        return len(self.data_cache)

    def __getitem__(self, idx):
        return self.data_cache[idx]