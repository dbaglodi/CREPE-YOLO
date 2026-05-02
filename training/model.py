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

    def __init__(self, use_raw_shape=True):
        super().__init__()
        self.use_raw_shape = use_raw_shape
        self.visual_branch = VisualBranch()
        self.semantic_branch = SemanticBranch(in_dim=2050)

    def forward(self, posteriorgram, embedding, confidence, gradient, raw_shape=None):
        bsz, _, height, time_steps = posteriorgram.shape
        device = posteriorgram.device

        y_coords = torch.linspace(0, 1, steps=height, device=device).view(1, 1, height, 1)
        y_channel = y_coords.expand(bsz, 1, height, time_steps)
        visual_input = torch.cat([posteriorgram, y_channel], dim=1)
        feat_a = self.visual_branch(visual_input)

        semantic_input = torch.cat([embedding, confidence, gradient], dim=1)
        feat_b = self.semantic_branch(semantic_input)
        feat_b_tiled = feat_b.repeat(1, 1, feat_a.shape[2], 1)

        if self.use_raw_shape and raw_shape is not None:
            target_t = feat_a.shape[-1]
            raw_shape_downsampled = torch.nn.functional.adaptive_avg_pool1d(raw_shape, output_size=target_t)
            raw_shape_tiled = raw_shape_downsampled.unsqueeze(2).repeat(1, 1, feat_a.shape[2], 1)
            return torch.cat([feat_a, feat_b_tiled, raw_shape_tiled], dim=1) # 577 channels
        else:
            return torch.cat([feat_a, feat_b_tiled], dim=1) # 576 channels

class TemporalContextModule(nn.Module):
    """
    Dilated Temporal Convolution Network (TCN) Block.
    Expands the receptive field along the time axis without destroying pitch resolution.
    """
    def __init__(self, channels):
        super().__init__()
        # Dilation 1 (looks 1 step back/forward)
        self.d1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )
        # Dilation 2 (looks 2 steps back/forward)
        self.d2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 2), dilation=(1, 2), bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )
        # Dilation 4 (looks 4 steps back/forward)
        self.d4 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 4), dilation=(1, 4), bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )

    def forward(self, x):
        identity = x
        x = self.d1(x)
        x = self.d2(x)
        x = self.d4(x)
        return x + identity # Residual connection ensures stable gradient flow

class MusicYOLOXHead(nn.Module):
    """Anchor-free decoupled head that predicts one box and objectness score per cell."""

    def __init__(self, in_channels=576, hidden_channels=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(),
        )
        self.temporal_context = TemporalContextModule(hidden_channels)
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
        x = self.temporal_context(x)
        reg_features = self.reg_branch(x)
        obj_features = self.obj_branch(x)
        return torch.cat([self.reg_pred(reg_features), self.obj_pred(obj_features)], dim=1)


class MusicYOLOHead(nn.Module):
    """Anchor-based YOLO head that predicts one box/objectness tuple per anchor."""

    def __init__(self, in_channels=577, num_anchors=3):
        super().__init__()
        self.num_anchors = num_anchors
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.Conv2d(256, num_anchors * 5, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DualStreamMusicYOLO(nn.Module):
    """End-to-end network with the anchor-based YOLO head."""

    def __init__(self, num_anchors=3, use_raw_shape=True):
        super().__init__()
        self.backbone = DualStreamBackbone(use_raw_shape=use_raw_shape)
        head_in_channels = 577 if use_raw_shape else 576
        self.head = MusicYOLOHead(in_channels=head_in_channels, num_anchors=num_anchors)

    def forward(self, posteriorgram, embedding, confidence, gradient, raw_shape=None):
        fused_features = self.backbone(
            posteriorgram,
            embedding,
            confidence,
            gradient,
            raw_shape
        )
        return self.head(fused_features)


class DualStreamMusicYOLOX(nn.Module):
    """End-to-end network with the anchor-free YOLOX head."""

    def __init__(self, use_raw_shape=True):
        super().__init__()
        self.backbone = DualStreamBackbone(use_raw_shape=use_raw_shape)
        head_in_channels = 577 if use_raw_shape else 576
        self.head = MusicYOLOXHead(in_channels=head_in_channels)

    def forward(self, posteriorgram, embedding, confidence, gradient, raw_shape=None):
        fused_features = self.backbone(
            posteriorgram,
            embedding,
            confidence,
            gradient,
            raw_shape
        )
        return self.head(fused_features)


def normalize_architecture_name(name: str | None) -> str:
    architecture = (name or "yolox").lower()
    aliases = {
        "crepe-yolo": "yolo",
        "crepe_yolo": "yolo",
        "crepe-yolox": "yolox",
        "crepe_yolox": "yolox",
    }
    architecture = aliases.get(architecture, architecture)
    if architecture not in {"yolo", "yolox"}:
        raise ValueError(f"Unsupported model architecture '{name}'. Use 'yolo' or 'yolox'.")
    return architecture


def build_model(model_cfg: dict | None = None) -> nn.Module:
    model_cfg = model_cfg or {}
    architecture = normalize_architecture_name(model_cfg.get("architecture", "yolox"))
    use_raw_shape = model_cfg.get("use_raw_shape", True)
    if architecture == "yolo":
        yolo_cfg = model_cfg.get("yolo", {})
        num_anchors = model_cfg.get("num_anchors", yolo_cfg.get("num_anchors", 3))
        return DualStreamMusicYOLO(num_anchors=num_anchors, use_raw_shape=use_raw_shape)
    return DualStreamMusicYOLOX(use_raw_shape=use_raw_shape)
