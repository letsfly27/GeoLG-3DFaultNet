import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import SUPPORTED_SUFFIXES, read_volume, zscore_normalize
from model import GeoLG3DFaultNet
from utils import sliding_window_predict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seis_dir", default="./data/test/seis")
    parser.add_argument("--fault_dir", default=None, help="Optional labels for side-by-side visualization.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", choices=("geolg", "unet"), default="geolg")
    parser.add_argument("--output_dir", default="./predictions")
    parser.add_argument("--shape", type=int, nargs=3, default=(128, 128, 128),
                        help="Raw .dat volume shape; .npy files are read using their stored shape.")
    parser.add_argument("--patch_size", type=int, nargs=3, default=(64, 64, 64))
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--heads", type=int, default=4)
    return parser.parse_args()


def save_preview(seismic, probability, binary, output_path, ground_truth=None):
    d, h, w = seismic.shape
    z, y, x = d // 2, h // 2, w // 2
    views = [
        ("Inline", seismic[z, :, :], probability[z, :, :], binary[z, :, :],
         None if ground_truth is None else ground_truth[z, :, :]),
        ("Crossline", seismic[:, y, :], probability[:, y, :], binary[:, y, :],
         None if ground_truth is None else ground_truth[:, y, :]),
        ("Timeline", seismic[:, :, x], probability[:, :, x], binary[:, :, x],
         None if ground_truth is None else ground_truth[:, :, x]),
    ]
    ncols = 4 if ground_truth is not None else 3
    fig, axes = plt.subplots(3, ncols, figsize=(4 * ncols, 10))
    for row, (name, seis_slice, prob_slice, pred_slice, gt_slice) in enumerate(views):
        axes[row, 0].imshow(seis_slice, cmap="gray", aspect="auto")
        axes[row, 0].set_title(f"{name} - Seismic")
        if ground_truth is not None:
            axes[row, 1].imshow(gt_slice, cmap="gray", vmin=0, vmax=1, aspect="auto")
            axes[row, 1].set_title(f"{name} - Ground truth")
            prob_col, pred_col = 2, 3
        else:
            prob_col, pred_col = 1, 2
        axes[row, prob_col].imshow(prob_slice, cmap="hot", vmin=0, vmax=1, aspect="auto")
        axes[row, prob_col].set_title(f"{name} - Fault probability")
        axes[row, pred_col].imshow(pred_slice, cmap="gray", vmin=0, vmax=1, aspect="auto")
        axes[row, pred_col].set_title(f"{name} - Binary prediction")
        for axis in axes[row]:
            axis.axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        args.model, in_channels=1, num_classes=2, base_channels=32,
        window_size=args.window_size, heads=args.heads,
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=True)
    model.eval()

    seis_dir = Path(args.seis_dir)
    fault_dir = Path(args.fault_dir) if args.fault_dir else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in seis_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)
    if not files:
        raise RuntimeError(f"No supported seismic volumes found in {seis_dir}")

    for index, seismic_path in enumerate(files, start=1):
        volume_shape = None if seismic_path.suffix.lower() == ".npy" else args.shape
        seismic = read_volume(seismic_path, volume_shape)
        normalized = zscore_normalize(seismic)
        tensor = torch.from_numpy(np.ascontiguousarray(normalized)).float()[None, None]
        probability = sliding_window_predict(
            model, tensor, patch_size=args.patch_size, overlap=args.overlap,
            device=device, blend_mode="gaussian",
        )[0, 0].cpu().numpy()
        binary = (probability >= args.threshold).astype(np.uint8)

        np.save(output_dir / f"{seismic_path.stem}_probability.npy", probability.astype(np.float32))
        np.save(output_dir / f"{seismic_path.stem}_binary.npy", binary)

        ground_truth = None
        if fault_dir is not None:
            label_path = fault_dir / seismic_path.name
            if label_path.exists():
                label_shape = None if label_path.suffix.lower() == ".npy" else args.shape
                ground_truth = (read_volume(label_path, label_shape) > 0.5).astype(np.uint8)
        save_preview(
            seismic, probability, binary,
            output_dir / f"{seismic_path.stem}_preview.png", ground_truth,
        )
        print(f"[{index}/{len(files)}] Saved predictions for {seismic_path.name}")


if __name__ == "__main__":
    main()
