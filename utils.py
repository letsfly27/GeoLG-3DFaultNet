from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage


def foreground_probability(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 5:
        raise ValueError(f"Expected BxCxDxHxW logits, got {tuple(logits.shape)}")
    if logits.shape[1] == 2:
        return torch.softmax(logits, dim=1)[:, 1:2]
    if logits.shape[1] == 1:
        return torch.sigmoid(logits)
    raise ValueError("Expected one- or two-channel segmentation output.")


def _as_5d_target(target: torch.Tensor) -> torch.Tensor:
    if target.ndim == 4:
        target = target.unsqueeze(1)
    if target.ndim != 5 or target.shape[1] != 1:
        raise ValueError(f"Expected Bx1xDxHxW or BxDxHxW target, got {tuple(target.shape)}")
    return target.float()


def _sobel_3d_kernels(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    smooth = torch.tensor([1.0, 2.0, 1.0], device=device, dtype=dtype)
    deriv = torch.tensor([-1.0, 0.0, 1.0], device=device, dtype=dtype)
    kz = torch.einsum("i,j,k->ijk", deriv, smooth, smooth)
    ky = torch.einsum("i,j,k->ijk", smooth, deriv, smooth)
    kx = torch.einsum("i,j,k->ijk", smooth, smooth, deriv)
    return torch.stack((kz, ky, kx), dim=0).unsqueeze(1) / 32.0


def _laplacian_3d_kernel(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    kernel = torch.zeros((3, 3, 3), device=device, dtype=dtype)
    kernel[1, 1, 1] = -6.0
    kernel[0, 1, 1] = kernel[2, 1, 1] = 1.0
    kernel[1, 0, 1] = kernel[1, 2, 1] = 1.0
    kernel[1, 1, 0] = kernel[1, 1, 2] = 1.0
    return kernel.view(1, 1, 3, 3, 3)


def edge_smoothing_loss(
    pred_prob: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.1,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target = _as_5d_target(target).to(device=pred_prob.device, dtype=pred_prob.dtype)
    sobel = _sobel_3d_kernels(pred_prob.device, pred_prob.dtype)
    grad_pred = F.conv3d(pred_prob, sobel, padding=1)
    grad_true = F.conv3d(target, sobel, padding=1)
    mag_pred = torch.sqrt(grad_pred.square().sum(dim=1, keepdim=True) + eps)
    mag_true = torch.sqrt(grad_true.square().sum(dim=1, keepdim=True) + eps)
    edge_consistency = F.l1_loss(mag_pred, mag_true)

    laplacian = F.conv3d(pred_prob, _laplacian_3d_kernel(pred_prob.device, pred_prob.dtype), padding=1)
    local_smoothness = laplacian.square().mean()
    total = float(alpha) * edge_consistency + float(beta) * local_smoothness
    return total, edge_consistency, local_smoothness


def _pca_oriented_extents(coords: np.ndarray) -> np.ndarray:
    if coords.shape[0] <= 1:
        return np.ones(3, dtype=np.float64)
    centered = coords.astype(np.float64) - coords.mean(axis=0, keepdims=True)
    if coords.shape[0] >= 3:
        covariance = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        axes = eigenvectors[:, np.argsort(eigenvalues)[::-1]]
        projected = centered @ axes
    else:
        projected = centered
    extents = np.ptp(projected, axis=0) + 1.0
    return np.maximum(extents, 1.0)


def build_geometric_template(
    target: torch.Tensor,
    connectivity: int = 26,
    boundary_gain: float = 0.5,
) -> torch.Tensor:
    target_5d = _as_5d_target(target)
    masks = target_5d.detach().cpu().numpy()[:, 0] > 0.5
    templates = np.zeros(masks.shape, dtype=np.float32)

    if connectivity == 6:
        structure = ndimage.generate_binary_structure(3, 1)
    elif connectivity == 18:
        structure = ndimage.generate_binary_structure(3, 2)
    elif connectivity == 26:
        structure = np.ones((3, 3, 3), dtype=bool)
    else:
        raise ValueError("connectivity must be 6, 18, or 26")

    for batch_index, mask in enumerate(masks):
        labels, component_count = ndimage.label(mask, structure=structure)
        if component_count == 0:
            continue

        components = []
        for component_id in range(1, component_count + 1):
            coords = np.column_stack(np.where(labels == component_id))
            voxel_count = int(coords.shape[0])
            if voxel_count == 0:
                continue
            extents = _pca_oriented_extents(coords)
            aspect_ratio = float(extents.max() / max(extents.min(), 1.0))
            components.append((component_id, voxel_count, aspect_ratio))

        if not components:
            continue

        max_log_area = max(float(np.log1p(item[1])) for item in components)
        max_log_ratio = max(float(np.log(max(item[2], 1.0))) for item in components)
        geometry_scores: Dict[int, float] = {}
        for component_id, voxel_count, aspect_ratio in components:
            area_score = float(np.log1p(voxel_count) / max_log_area) if max_log_area > 0 else 1.0
            log_ratio = float(np.log(max(aspect_ratio, 1.0)))
            ratio_score = log_ratio / max_log_ratio if max_log_ratio > 1e-8 else 1.0
            geometry_scores[component_id] = 0.5 * area_score + 0.5 * ratio_score

        local_occupancy = ndimage.uniform_filter(
            mask.astype(np.float32), size=3, mode="constant", cval=0.0
        )
        local_boundary_response = mask.astype(np.float32) * (1.0 - local_occupancy)

        raw = np.zeros(mask.shape, dtype=np.float32)
        for component_id, score in geometry_scores.items():
            region = labels == component_id
            raw[region] = score * (
                1.0 + float(boundary_gain) * local_boundary_response[region]
            )

        maximum = float(raw.max())
        if maximum > 1e-8:
            minimum = 0.0 if np.any(~mask) else float(raw.min())
            value_range = maximum - minimum
            if value_range > 1e-8:
                templates[batch_index] = (raw - minimum) / value_range
            else:
                templates[batch_index] = mask.astype(np.float32)

    return torch.from_numpy(templates[:, None]).to(
        device=target_5d.device, dtype=target_5d.dtype
    ).detach()


class GeoLGFaultLoss(nn.Module):
    def __init__(
        self,
        weight_bce: float = 1.0,
        weight_dice: float = 1.0,
        lambda_edge: float = 0.5,
        lambda_line: float = 0.5,
        w_positive: float = 1.0,
        w_negative: float = 1.0,
        edge_alpha: float = 1.0,
        edge_beta: float = 0.1,
        template_boundary_gain: float = 0.5,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.weight_bce = float(weight_bce)
        self.weight_dice = float(weight_dice)
        self.lambda_edge = float(lambda_edge)
        self.lambda_line = float(lambda_line)
        self.w_positive = float(w_positive)
        self.w_negative = float(w_negative)
        self.edge_alpha = float(edge_alpha)
        self.edge_beta = float(edge_beta)
        self.template_boundary_gain = float(template_boundary_gain)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, target: torch.Tensor,
                return_components: bool = False):
        target = _as_5d_target(target).to(device=logits.device, dtype=logits.dtype)
        probability = foreground_probability(logits)

        if logits.shape[1] == 2:
            fault_logit = logits[:, 1:2] - logits[:, 0:1]
        else:
            fault_logit = logits
        bce_map = F.binary_cross_entropy_with_logits(fault_logit, target, reduction="none")
        class_weight = self.w_positive * target + self.w_negative * (1.0 - target)
        loss_bce = (bce_map * class_weight).mean()

        reduce_dims = tuple(range(1, probability.ndim))
        intersection = (probability * target).sum(dim=reduce_dims)
        denominator = probability.sum(dim=reduce_dims) + target.sum(dim=reduce_dims)
        loss_dice = (1.0 - (2.0 * intersection + self.eps) /
                     (denominator + self.eps)).mean()

        loss_edge, edge_consistency, local_smoothness = edge_smoothing_loss(
            probability, target, self.edge_alpha, self.edge_beta, self.eps
        )
        template = build_geometric_template(
            target, connectivity=26, boundary_gain=self.template_boundary_gain
        )
        loss_line = ((probability * template - template).square()).mean()

        total = (
            self.weight_bce * loss_bce
            + self.weight_dice * loss_dice
            + self.lambda_edge * loss_edge
            + self.lambda_line * loss_line
        )
        if return_components:
            return total, {
                "bce": loss_bce.detach(),
                "dice": loss_dice.detach(),
                "edge": loss_edge.detach(),
                "edge_consistency": edge_consistency.detach(),
                "laplacian_smoothness": local_smoothness.detach(),
                "geometric": loss_line.detach(),
            }
        return total


def _binary_prediction(prediction: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    if prediction.ndim == 5 and prediction.shape[1] == 2:
        return foreground_probability(prediction) >= threshold
    if prediction.ndim == 5 and prediction.shape[1] == 1:
        return prediction >= threshold
    if prediction.ndim == 4:
        return prediction.unsqueeze(1) >= threshold
    raise ValueError(f"Unsupported prediction shape: {tuple(prediction.shape)}")


def segmentation_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    target_bin = _as_5d_target(target).to(prediction.device) >= 0.5
    pred_bin = _binary_prediction(prediction, threshold).bool()
    dims = tuple(range(1, pred_bin.ndim))
    tp = (pred_bin & target_bin).sum(dim=dims).float()
    fp = (pred_bin & ~target_bin).sum(dim=dims).float()
    fn = (~pred_bin & target_bin).sum(dim=dims).float()
    tn = (~pred_bin & ~target_bin).sum(dim=dims).float()

    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    return {
        "dice": dice.mean(),
        "iou": iou.mean(),
        "precision": precision.mean(),
        "recall": recall.mean(),
        "f1": dice.mean(),
        "accuracy": accuracy.mean(),
    }


def dice_score(prediction: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6):
    return segmentation_metrics(prediction, target, eps=smooth)["dice"]


def iou_score(prediction: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6):
    return segmentation_metrics(prediction, target, eps=smooth)["iou"]


def precision_recall(prediction: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6):
    metrics = segmentation_metrics(prediction, target, eps=smooth)
    return metrics["precision"], metrics["recall"]


def f1_score(prediction: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6):
    return segmentation_metrics(prediction, target, eps=smooth)["f1"]


def _window_starts(length: int, patch: int, stride: int):
    if length <= patch:
        return [0]
    starts = list(range(0, length - patch + 1, stride))
    final_start = length - patch
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def gaussian_importance_map(
    patch_size: Sequence[int],
    device: torch.device,
    dtype: torch.dtype,
    sigma_scale: float = 0.25,
) -> torch.Tensor:
    axes = []
    for length in patch_size:
        coord = torch.linspace(-1.0, 1.0, int(length), device=device, dtype=dtype)
        axes.append(torch.exp(-0.5 * (coord / max(float(sigma_scale), 1e-3)).square()))
    weights = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    weights = weights / weights.max().clamp_min(1e-8)
    return weights.clamp_min(1e-3)[None, None]


@torch.no_grad()
def sliding_window_predict(
    model: nn.Module,
    volume: torch.Tensor,
    patch_size: Sequence[int] = (64, 64, 64),
    overlap: float = 0.5,
    device: Optional[torch.device] = None,
    blend_mode: str = "gaussian",
) -> torch.Tensor:
    if volume.ndim == 4:
        volume = volume.unsqueeze(0)
    if volume.ndim != 5 or volume.shape[0] != 1 or volume.shape[1] != 1:
        raise ValueError("Expected one B=1,C=1 3D volume (B,C,D,H,W).")
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = volume.device
    patch_size = tuple(int(v) for v in patch_size)
    if len(patch_size) != 3 or any(v <= 0 for v in patch_size):
        raise ValueError("patch_size must contain three positive integers.")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0,1).")
    if blend_mode not in {"gaussian", "constant"}:
        raise ValueError("blend_mode must be 'gaussian' or 'constant'.")

    volume = volume.to(device=device, dtype=torch.float32)
    original_shape = tuple(volume.shape[2:])
    pad_d = max(patch_size[0] - original_shape[0], 0)
    pad_h = max(patch_size[1] - original_shape[1], 0)
    pad_w = max(patch_size[2] - original_shape[2], 0)
    if pad_d or pad_h or pad_w:
        volume = F.pad(volume, (0, pad_w, 0, pad_h, 0, pad_d), mode="replicate")

    d, h, w = volume.shape[2:]
    strides = tuple(max(int(round(p * (1.0 - overlap))), 1) for p in patch_size)
    starts_d = _window_starts(d, patch_size[0], strides[0])
    starts_h = _window_starts(h, patch_size[1], strides[1])
    starts_w = _window_starts(w, patch_size[2], strides[2])

    probability_sum = torch.zeros((1, 1, d, h, w), device=device, dtype=torch.float32)
    weight_sum = torch.zeros_like(probability_sum)
    if blend_mode == "gaussian":
        importance = gaussian_importance_map(patch_size, device, torch.float32)
    else:
        importance = torch.ones((1, 1, *patch_size), device=device, dtype=torch.float32)

    was_training = model.training
    model.eval()
    try:
        for z in starts_d:
            for y in starts_h:
                for x in starts_w:
                    patch = volume[:, :, z:z + patch_size[0],
                                   y:y + patch_size[1], x:x + patch_size[2]]
                    logits = model(patch)
                    probability = foreground_probability(logits).float()
                    probability_sum[:, :, z:z + patch_size[0],
                                    y:y + patch_size[1], x:x + patch_size[2]] += probability * importance
                    weight_sum[:, :, z:z + patch_size[0],
                               y:y + patch_size[1], x:x + patch_size[2]] += importance
    finally:
        model.train(was_training)

    probability = probability_sum / weight_sum.clamp_min(1e-8)
    return probability[:, :, :original_shape[0], :original_shape[1], :original_shape[2]]
