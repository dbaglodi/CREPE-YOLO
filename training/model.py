import torch
import torch.nn as nn


class VisualBranch(nn.Module):
    """
    Branch A: 2D CNN processing the visual pitch posteriorgram + CoordiConv.
    Downsamples spatial and temporal dimensions by a factor of 32.
    """

    def __init__(self, in_channels=2, out_channels=512):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.SiLU(),
                nn.MaxPool2d(2),
            )

        self.net = nn.Sequential(
            conv_block(in_channels, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SemanticBranch(nn.Module):
    """
    Branch B: CRNN processing the 1D semantic state.
    Downsamples Time by 32, applies BiLSTM, and forces a 64-channel bottleneck.
    """

    def __init__(self, in_dim=2050, hidden_dim=128, out_channels=64):
        super().__init__()

        def conv1d_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(out_c),
                nn.SiLU(),
                nn.MaxPool1d(2),
            )

        self.cnn = nn.Sequential(
            conv1d_block(in_dim, 512),
            conv1d_block(512, 256),
            conv1d_block(256, 128),
            conv1d_block(128, 128),
            conv1d_block(128, hidden_dim),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.bottleneck = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x.permute(0, 2, 1)
        x = self.bottleneck(x)
        return x.unsqueeze(-2)


class DualStreamBackbone(nn.Module):
    """Shared feature extractor used by the anchor-free YOLOX head."""

    def __init__(self):
        super().__init__()
        self.visual_branch = VisualBranch()
        self.semantic_branch = SemanticBranch()

    def forward(self, posteriorgram, embedding, confidence, gradient):
        bsz, _, height, time_steps = posteriorgram.shape
        device = posteriorgram.device

        y_coords = torch.linspace(0, 1, steps=height, device=device).view(1, 1, height, 1)
        y_channel = y_coords.expand(bsz, 1, height, time_steps)
        visual_input = torch.cat([posteriorgram, y_channel], dim=1)
        feat_a = self.visual_branch(visual_input)

        semantic_input = torch.cat([embedding, confidence, gradient], dim=1)
        feat_b = self.semantic_branch(semantic_input)
        feat_b_tiled = feat_b.repeat(1, 1, feat_a.shape[2], 1)

        return torch.cat([feat_a, feat_b_tiled], dim=1)


class MusicYOLOXHead(nn.Module):
    """Anchor-free decoupled head that predicts one box and objectness score per cell."""

    def __init__(self, in_channels=576, hidden_channels=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(),
        )
        self.reg_branch = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(),
        )
        self.obj_branch = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(),
        )
        self.reg_pred = nn.Conv2d(hidden_channels, 4, kernel_size=1)
        self.obj_pred = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        reg_features = self.reg_branch(x)
        obj_features = self.obj_branch(x)
        return torch.cat([self.reg_pred(reg_features), self.obj_pred(obj_features)], dim=1)


class DualStreamMusicYOLOX(nn.Module):
    """End-to-end network with the anchor-free YOLOX head."""

    def __init__(self):
        super().__init__()
        self.backbone = DualStreamBackbone()
        self.head = MusicYOLOXHead(in_channels=576)

    def forward(self, posteriorgram, embedding, confidence, gradient):
        fused_features = self.backbone(
            posteriorgram,
            embedding,
            confidence,
            gradient,
        )
        return self.head(fused_features)
