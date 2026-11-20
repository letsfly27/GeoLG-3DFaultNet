import argparse
import csv
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import SeismicDataset3D
from model import GeoLG3DFaultNet
from utils import GeoLGFaultLoss, dice_score, sliding_window_predict


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@torch.no_grad()
def validate(model, loader, device, patch_size, overlap):
    model.eval()
    volume_dice = []
    for seismic, fault in loader:
        seismic = seismic.to(device, non_blocking=True)
        fault = fault.to(device, non_blocking=True)
        probability = sliding_window_predict(
            model, seismic, patch_size=patch_size, overlap=overlap,
            device=device, blend_mode="gaussian",
        )
        volume_dice.append(float(dice_score(probability, fault).item()))
    return float(np.mean(volume_dice)) if volume_dice else 0.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--model", choices=("geolg", "unet"), default="geolg")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--patch_size", type=int, nargs=3, default=(64, 64, 64))
    parser.add_argument("--volume_shape", type=int, nargs=3, default=(128, 128, 128))
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--validation_overlap", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="./runs")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root)
    train_dataset = SeismicDataset3D(
        str(data_root / "train" / "seis"), str(data_root / "train" / "fault"),
        shape=args.volume_shape, crop_size=args.patch_size, random_crop=True,
    )
    val_dataset = SeismicDataset3D(
        str(data_root / "validation" / "seis"), str(data_root / "validation" / "fault"),
        shape=args.volume_shape, crop_size=None, random_crop=False,
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker, generator=generator, drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(
        args.model, in_channels=1, num_classes=2, base_channels=32,
        window_size=args.window_size, heads=args.heads,
    ).to(device)
    criterion = GeoLGFaultLoss(
        weight_bce=1.0, weight_dice=1.0,
        lambda_edge=0.5, lambda_line=0.5,
        w_positive=1.0, w_negative=1.0,
        edge_alpha=1.0, edge_beta=0.1,
        template_boundary_gain=0.5,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    run_dir = Path(args.output_dir) / args.model
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "training_history.csv"
    with log_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "epoch", "train_total", "train_bce", "train_dice", "train_edge",
            "train_geometry", "validation_dice", "seconds",
        ])

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    best_validation_dice = -1.0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals = {"total": 0.0, "bce": 0.0, "dice": 0.0, "edge": 0.0, "geometric": 0.0}
        start_time = time.time()

        for step, (seismic, fault) in enumerate(train_loader):
            seismic = seismic.to(device, non_blocking=True)
            fault = fault.to(device, non_blocking=True)
            group_start = (step // args.grad_accum_steps) * args.grad_accum_steps
            group_size = min(args.grad_accum_steps, len(train_loader) - group_start)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(seismic)
                loss, components = criterion(logits, fault, return_components=True)
                scaled_loss = loss / group_size

            scaler.scale(scaled_loss).backward()
            totals["total"] += float(loss.detach().item())
            for key in ("bce", "dice", "edge", "geometric"):
                totals[key] += float(components[key].item())

            accumulation_step = (step + 1) % args.grad_accum_steps == 0
            last_batch = (step + 1) == len(train_loader)
            if accumulation_step or last_batch:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        count = max(len(train_loader), 1)
        for key in totals:
            totals[key] /= count
        val_dice = validate(
            model, val_loader, device, tuple(args.patch_size), args.validation_overlap
        )
        elapsed = time.time() - start_time
        logging.info(
            f"Epoch [{epoch + 1:03d}/{args.epochs}] "
            f"Loss={totals['total']:.6f} (BCE={totals['bce']:.5f}, "
            f"Dice={totals['dice']:.5f}, Edge={totals['edge']:.5f}, "
            f"Geometry={totals['geometric']:.5f}) | "
            f"Val Dice={val_dice:.6f} | {elapsed:.1f}s"
        )
        with log_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch + 1, totals["total"], totals["bce"], totals["dice"],
                totals["edge"], totals["geometric"], val_dice, elapsed,
            ])

        if val_dice > best_validation_dice:
            best_validation_dice = val_dice
            torch.save({
                "epoch": epoch + 1,
                "model_name": args.model,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_validation_dice": best_validation_dice,
                "config": vars(args),
                "loss_weights": {
                    "bce": 1.0, "dice": 1.0, "lambda_edge": 0.5,
                    "lambda_line": 0.5, "alpha": 1.0, "beta": 0.1,
                    "template_boundary_gain": 0.5,
                },
            }, checkpoint_dir / "best_model.pth")

        if (epoch + 1) % 20 == 0:
            torch.save({
                "epoch": epoch + 1,
                "model_name": args.model,
                "model_state_dict": model.state_dict(),
            }, checkpoint_dir / f"epoch_{epoch + 1:03d}.pth")

    print(f"Training loop completed. Best validation Dice: {best_validation_dice:.6f}")


if __name__ == "__main__":
    main()
