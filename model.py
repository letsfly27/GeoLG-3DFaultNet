import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffZ = x2.size(2) - x1.size(2)
        diffY = x2.size(3) - x1.size(3)
        diffX = x2.size(4) - x1.size(4)

        if diffZ > 0 or diffY > 0 or diffX > 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2,
                            diffZ // 2, diffZ - diffZ // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)



class LGFEModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        channels = in_channels
        self.lfe = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True)
        )
        self.gfe = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=2, dilation=2),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True)
        )
        self.fusion = nn.ReLU(inplace=True)

    def forward(self, x):
        local_feat = self.lfe(x)
        global_feat = self.gfe(x)
        return self.fusion(local_feat + global_feat)


class WindowAttention3D(nn.Module):
    def __init__(self, dim=32, window_size=4, heads=4):
        super().__init__()
        channels = dim
        self.window_size = window_size if isinstance(window_size, int) else window_size[0]
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        B, C, D, H, W = x.shape
        ws = self.window_size

        pad_d = (ws - D % ws) % ws
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
        _, _, Dp, Hp, Wp = x.shape

        x_windows = x.view(B, C, Dp // ws, ws, Hp // ws, ws, Wp // ws, ws)
        x_windows = x_windows.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
        x_flat = x_windows.view(-1, ws * ws * ws, C)

        x_flat = self.norm(x_flat)
        chunk_size = 512
        out_list = []
        total_windows = x_flat.shape[0]

        for i in range(0, total_windows, chunk_size):
            end = min(i + chunk_size, total_windows)
            batch_chunk = x_flat[i:end]
            attn_chunk, _ = self.attn(batch_chunk, batch_chunk, batch_chunk)
            out_list.append(attn_chunk)

        attn_out = torch.cat(out_list, dim=0)
        attn_out = attn_out.view(B, Dp // ws, Hp // ws, Wp // ws, ws, ws, ws, C)
        attn_out = attn_out.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        out = attn_out.view(B, C, Dp, Hp, Wp)

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            out = out[:, :, :D, :H, :W]

        return out



class FCOM(nn.Module):
    def __init__(self, in_channels=64, out_channels=32):
        super().__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.continuity_conv = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x_f = self.fusion_conv(x)

        out = x_f + self.continuity_conv(x_f)
        return out

class GeoLG3DFaultNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        

        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)


        self.up1 = Up(384, 128)
        self.up2 = Up(192, 64)
        self.up3 = Up(96, 32)


        self.lg = LGFEModule(in_channels=32)
        self.wa = WindowAttention3D(dim=32, window_size=8, heads=4)
        

        self.aco = FCOM(in_channels=64, out_channels=32)


        self.outc = nn.Conv3d(32, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x_up = self.up1(x4, x3)
        x_up = self.up2(x_up, x2)
        x_up = self.up3(x_up, x1)

        feat_lg = self.lg(x_up)
        feat_wa = self.wa(x_up)
        

        feat_concat = torch.cat([feat_lg, feat_wa], dim=1)
        feat_aco = self.aco(feat_concat)

        out = self.outc(feat_aco)
        return out