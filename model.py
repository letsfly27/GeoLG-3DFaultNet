from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2) if downsample else nn.Identity()
        self.conv = ConvBlock3D(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class MCBridge(nn.Module):
    def __init__(self, channels: int, dilations=(6, 12, 18)):
        super().__init__()
        branch_channels = max(channels // 5, 1)
        self.pointwise = nn.Sequential(
            nn.Conv3d(channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(branch_channels),
            nn.ReLU(inplace=True),
        )
        self.dilated = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(
                    channels, branch_channels, kernel_size=3,
                    padding=int(rate), dilation=int(rate), bias=False,
                ),
                nn.BatchNorm3d(branch_channels),
                nn.ReLU(inplace=True),
            )
            for rate in dilations
        ])
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, branch_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv3d(branch_channels * (len(dilations) + 2), channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_size = x.shape[2:]
        branches = [self.pointwise(x)]
        branches.extend(branch(x) for branch in self.dilated)
        pooled = self.image_pool(x)
        pooled = F.interpolate(pooled, size=spatial_size, mode="trilinear", align_corners=False)
        branches.append(pooled)
        return self.project(torch.cat(branches, dim=1))


class LGModule(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 1)

        self.local_conv3 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.local_conv1 = nn.Conv3d(channels, channels, kernel_size=1, bias=False)

        self.channel_reduce = nn.Conv3d(channels, hidden, kernel_size=1, bias=True)
        self.channel_expand = nn.Conv3d(hidden, channels, kernel_size=1, bias=True)

        self.spatial_reduce = nn.Conv3d(channels, hidden, kernel_size=1, bias=True)
        self.spatial_expand = nn.Conv3d(hidden, 1, kernel_size=1, bias=True)

        self.global_conv3 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.global_conv1 = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_loc = x + self.local_conv1(self.local_conv3(x))

        channel_descriptor = F.adaptive_avg_pool3d(x, output_size=1)
        m_c = torch.sigmoid(
            self.channel_expand(self.relu(self.channel_reduce(channel_descriptor)))
        )
        m_s = torch.sigmoid(
            self.spatial_expand(self.relu(self.spatial_reduce(x)))
        )

        f_channel = m_c * x
        f_spatial = m_s * x
        f_global = self.global_conv1(self.relu(self.global_conv3(f_channel + f_spatial)))
        return f_loc + f_global


def _window_partition(x: torch.Tensor, window_size: int):
    b, c, d, h, w = x.shape
    ws = int(window_size)
    pd = (ws - d % ws) % ws
    ph = (ws - h % ws) % ws
    pw = (ws - w % ws) % ws
    if pd or ph or pw:
        x = F.pad(x, (0, pw, 0, ph, 0, pd))

    valid = torch.ones((b, 1, d, h, w), device=x.device, dtype=x.dtype)
    if pd or ph or pw:
        valid = F.pad(valid, (0, pw, 0, ph, 0, pd), value=0.0)

    dp, hp, wp = x.shape[2:]
    windows = x.view(b, c, dp // ws, ws, hp // ws, ws, wp // ws, ws)
    windows = windows.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
    windows = windows.view(-1, ws ** 3, c)

    valid_windows = valid.view(b, 1, dp // ws, ws, hp // ws, ws, wp // ws, ws)
    valid_windows = valid_windows.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
    valid_windows = valid_windows.view(-1, ws ** 3).bool()
    meta = (b, c, d, h, w, dp, hp, wp)
    return windows, valid_windows, meta


def _window_reverse(windows: torch.Tensor, meta, window_size: int) -> torch.Tensor:
    b, c, d, h, w, dp, hp, wp = meta
    ws = int(window_size)
    x = windows.view(b, dp // ws, hp // ws, wp // ws, ws, ws, ws, c)
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
    x = x.view(b, c, dp, hp, wp)
    return x[:, :, :d, :h, :w]


class WindowAttention3D(nn.Module):
    def __init__(self, dim: int, window_size: int = 8, heads: int = 4,
                 window_chunk_size: int = 4):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads}).")
        self.dim = int(dim)
        self.window_size = int(window_size)
        self.heads = int(heads)
        self.head_dim = self.dim // self.heads
        self.scale = self.head_dim ** -0.5
        self.window_chunk_size = max(int(window_chunk_size), 1)

        self.norm = nn.LayerNorm(self.dim)
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)

        ws = self.window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(((2 * ws - 1) ** 3), self.heads)
        )
        coords = torch.stack(torch.meshgrid(
            torch.arange(ws), torch.arange(ws), torch.arange(ws), indexing="ij"
        ))
        coords_flat = coords.flatten(1)
        relative = coords_flat[:, :, None] - coords_flat[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        relative[:, :, 0] += ws - 1
        relative[:, :, 1] += ws - 1
        relative[:, :, 2] += ws - 1
        relative[:, :, 0] *= (2 * ws - 1) ** 2
        relative[:, :, 1] *= (2 * ws - 1)
        relative_index = relative.sum(-1)
        self.register_buffer("relative_position_index", relative_index, persistent=False)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _, _ = x.shape
        windows, valid_windows, meta = _window_partition(x, self.window_size)
        n_windows, n_tokens, _ = windows.shape
        windows = self.norm(windows)

        qkv = self.qkv(windows).reshape(
            n_windows, n_tokens, 3, self.heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ].view(n_tokens, n_tokens, self.heads).permute(2, 0, 1).contiguous()

        outputs = []
        for start in range(0, n_windows, self.window_chunk_size):
            end = min(start + self.window_chunk_size, n_windows)
            q_part, k_part, v_part = q[start:end], k[start:end], v[start:end]
            attn = torch.matmul(q_part, k_part.transpose(-2, -1)) * self.scale
            attn = attn + bias.unsqueeze(0)
            valid_keys = valid_windows[start:end][:, None, None, :]
            attn = attn.masked_fill(~valid_keys, torch.finfo(attn.dtype).min)
            attn = torch.softmax(attn, dim=-1)
            outputs.append(torch.matmul(attn, v_part))

        out = torch.cat(outputs, dim=0)
        out = out.transpose(1, 2).reshape(n_windows, n_tokens, c)
        out = self.proj(out)
        return _window_reverse(out, meta, self.window_size)


class FCOUnit(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, f_lg: torch.Tensor, f_wa: torch.Tensor) -> torch.Tensor:
        if f_lg.shape != f_wa.shape:
            raise ValueError(
                "FCOUnit requires identical LG/WA tensor dimensions; "
                f"got {tuple(f_lg.shape)} and {tuple(f_wa.shape)}."
            )
        fused = f_lg + f_wa
        return fused + self.conv2(self.relu(self.conv1(fused)))


class ACOModule(nn.Module):
    def __init__(self, channels: int, window_size: int = 8, heads: int = 4):
        super().__init__()
        self.window_attention = WindowAttention3D(channels, window_size, heads)
        self.fco = FCOUnit(channels)

    def forward(self, f_lg: torch.Tensor, f_fuse: torch.Tensor) -> torch.Tensor:
        f_wa = self.window_attention(f_fuse)
        return self.fco(f_lg, f_wa)


class DecoderStage(nn.Module):
    def __init__(self, deep_channels: int, skip_channels: int, out_channels: int,
                 upsample: bool, window_size: int, heads: int):
        super().__init__()
        self.upsample = bool(upsample)
        self.channel_projection = nn.Sequential(
            nn.Conv3d(deep_channels + skip_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.lg = LGModule(out_channels)
        self.aco = ACOModule(out_channels, window_size=window_size, heads=heads)

    def forward(self, deep: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if self.upsample:
            deep = F.interpolate(deep, size=skip.shape[2:], mode="trilinear", align_corners=False)
        elif deep.shape[2:] != skip.shape[2:]:
            raise ValueError("D4 expects the MC-Bridge and E4 feature sizes to match.")
        f_fuse = self.channel_projection(torch.cat((deep, skip), dim=1))
        f_lg = self.lg(f_fuse)
        return self.aco(f_lg, f_fuse)


class GeoLG3DFaultNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 2,
                 base_channels: int = 32, window_size: int = 8, heads: int = 4):
        super().__init__()
        if num_classes != 2:
            raise ValueError("The manuscript defines binary segmentation with two output channels.")
        if base_channels % heads != 0:
            raise ValueError("base_channels must be divisible by the attention head count.")

        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.enc1 = EncoderStage(in_channels, c1, downsample=False)
        self.enc2 = EncoderStage(c1, c2, downsample=True)
        self.enc3 = EncoderStage(c2, c3, downsample=True)
        self.enc4 = EncoderStage(c3, c4, downsample=True)

        self.mc_bridge = MCBridge(c4, dilations=(6, 12, 18))

        self.dec4 = DecoderStage(c4, c4, c4, False, window_size, heads)
        self.dec3 = DecoderStage(c4, c3, c3, True, window_size, heads)
        self.dec2 = DecoderStage(c3, c2, c2, True, window_size, heads)
        self.dec1 = DecoderStage(c2, c1, c1, True, window_size, heads)
        self.out_conv = nn.Conv3d(c1, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        deep = self.mc_bridge(e4)
        d4 = self.dec4(deep, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        return self.out_conv(d1)


class UNetDecoderStage(nn.Module):
    def __init__(self, deep_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock3D(deep_channels + skip_channels, out_channels)

    def forward(self, deep: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        deep = F.interpolate(deep, size=skip.shape[2:], mode="trilinear", align_corners=False)
        return self.conv(torch.cat((deep, skip), dim=1))


class UNet3DBaseline(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 2, base_channels: int = 32):
        super().__init__()
        if num_classes != 2:
            raise ValueError("Binary segmentation baseline expects two output channels.")
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.enc1 = EncoderStage(in_channels, c1, False)
        self.enc2 = EncoderStage(c1, c2, True)
        self.enc3 = EncoderStage(c2, c3, True)
        self.enc4 = EncoderStage(c3, c4, True)
        self.dec3 = UNetDecoderStage(c4, c3, c3)
        self.dec2 = UNetDecoderStage(c3, c2, c2)
        self.dec1 = UNetDecoderStage(c2, c1, c1)
        self.out_conv = nn.Conv3d(c1, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        d3 = self.dec3(e4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        return self.out_conv(d1)


def build_model(name: str = "geolg", **kwargs) -> nn.Module:
    name = name.lower()
    if name in {"geolg", "geolg-3dfaultnet", "proposed"}:
        return GeoLG3DFaultNet(**kwargs)
    if name in {"unet", "3d-unet", "baseline"}:
        kwargs.pop("window_size", None)
        kwargs.pop("heads", None)
        return UNet3DBaseline(**kwargs)
    raise ValueError(f"Unknown model name: {name}")


if __name__ == "__main__":
    net = GeoLG3DFaultNet(base_channels=8, window_size=4, heads=4)
    sample = torch.randn(1, 1, 32, 32, 32)
    with torch.no_grad():
        result = net(sample)
    print("Input:", tuple(sample.shape), "Output logits:", tuple(result.shape))
