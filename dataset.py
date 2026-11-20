from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


SUPPORTED_SUFFIXES = {".dat", ".npy"}


def read_volume(path: Path, shape: Optional[Sequence[int]] = (128, 128, 128), dtype=np.float32):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        volume = np.load(path).astype(dtype, copy=False)
    elif path.suffix.lower() == ".dat":
        if shape is None:
            raise ValueError(f"A shape is required to read raw .dat file: {path}")
        raw = np.fromfile(path, dtype=dtype)
        expected = int(np.prod(tuple(shape)))
        if raw.size != expected:
            raise ValueError(f"{path}: found {raw.size} values, expected {expected} for {tuple(shape)}")
        volume = raw.reshape(tuple(shape))
    else:
        raise ValueError(f"Unsupported volume extension: {path.suffix}")
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got {volume.shape} from {path}")
    if shape is not None and tuple(volume.shape) != tuple(shape):
        raise ValueError(f"{path}: shape {volume.shape} does not match expected {tuple(shape)}")
    return np.asarray(volume, dtype=np.float32)


def zscore_normalize(volume: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = float(np.mean(volume))
    std = float(np.std(volume))
    return ((volume - mean) / max(std, eps)).astype(np.float32, copy=False)


class SeismicDataset3D(Dataset):
    def __init__(
        self,
        seis_dir: str,
        fault_dir: str,
        shape: Optional[Sequence[int]] = (128, 128, 128),
        crop_size: Optional[Sequence[int]] = None,
        random_crop: bool = False,
        seis_dtype=np.float32,
        fault_dtype=np.float32,
    ):
        self.seis_dir = Path(seis_dir)
        self.fault_dir = Path(fault_dir)
        self.shape = tuple(shape) if shape is not None else None
        self.crop_size = tuple(crop_size) if crop_size is not None else None
        self.random_crop = bool(random_crop)
        self.seis_dtype = seis_dtype
        self.fault_dtype = fault_dtype

        if not self.seis_dir.is_dir():
            raise FileNotFoundError(f"Seismic directory not found: {self.seis_dir}")
        if not self.fault_dir.is_dir():
            raise FileNotFoundError(f"Fault directory not found: {self.fault_dir}")
        self.file_list = sorted(
            p.name for p in self.seis_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
        )
        if not self.file_list:
            raise RuntimeError(f"No .dat/.npy files found in {self.seis_dir}")
        missing = [name for name in self.file_list if not (self.fault_dir / name).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} matching labels; first missing file: {missing[0]}"
            )

    def __len__(self):
        return len(self.file_list)

    def _pad_to_crop(self, seismic: np.ndarray, fault: np.ndarray):
        if self.crop_size is None:
            return seismic, fault
        pads = [(0, max(int(crop) - int(current), 0))
                for current, crop in zip(seismic.shape, self.crop_size)]
        if any(right > 0 for _, right in pads):
            seismic = np.pad(seismic, pads, mode="reflect")
            fault = np.pad(fault, pads, mode="constant", constant_values=0)
        return seismic, fault

    def _crop(self, seismic: np.ndarray, fault: np.ndarray):
        if self.crop_size is None:
            return seismic, fault
        starts = []
        for current, crop in zip(seismic.shape, self.crop_size):
            max_start = max(int(current) - int(crop), 0)
            if self.random_crop:
                start = np.random.randint(0, max_start + 1) if max_start else 0
            else:
                start = max_start // 2
            starts.append(start)
        z, y, x = starts
        cd, ch, cw = self.crop_size
        slices = (slice(z, z + cd), slice(y, y + ch), slice(x, x + cw))
        return seismic[slices], fault[slices]

    def __getitem__(self, index):
        filename = self.file_list[index]
        seismic = read_volume(self.seis_dir / filename, self.shape, self.seis_dtype)
        fault = read_volume(self.fault_dir / filename, self.shape, self.fault_dtype)
        if seismic.shape != fault.shape:
            raise ValueError(f"Seismic/label shape mismatch for {filename}: {seismic.shape} vs {fault.shape}")

        seismic = zscore_normalize(seismic)
        fault = (fault > 0.5).astype(np.float32)
        seismic, fault = self._pad_to_crop(seismic, fault)
        seismic, fault = self._crop(seismic, fault)

        seismic_tensor = torch.from_numpy(np.ascontiguousarray(seismic)).unsqueeze(0).float()
        fault_tensor = torch.from_numpy(np.ascontiguousarray(fault)).unsqueeze(0).float()
        return seismic_tensor, fault_tensor
