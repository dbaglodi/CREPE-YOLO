import torch
import torch.nn as nn

class VisualBranch(nn.Module):
    """
    Branch A: 2D CNN processing the visual pitch posteriorgram + CoordiConv.
    Downsamples spatial and temporal dimensions by a factor of 32.
    """
    def __init__(self, in_channels=2, out_channels=512):
        super().__init__()
        # Input: (B, 2, 360, T) -> 2 channels (Posteriorgram + Y-Coord)
        
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.SiLU(),
                nn.MaxPool2d(2) # Halves Height and Time
            )

        self.net = nn.Sequential(
            conv_block(in_channels, 32),   # Output: (B, 32, 180, T/2)
            conv_block(32, 64),            # Output: (B, 64, 90, T/4)
            conv_block(64, 128),           # Output: (B, 128, 45, T/8)
            conv_block(128, 256),          # Output: (B, 256, 22, T/16)
            conv_block(256, out_channels)  # Output: (B, 512, 11, T/32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SemanticBranch(nn.Module):
    """
    Branch B: CRNN processing the 1D semantic state.
    Downsamples Time by 32, applies BiLSTM, and forces a 64-channel bottleneck.
    """
    def __init__(self, in_dim=2051, hidden_dim=128, out_channels=64): 
        super().__init__()
        # Input: (B, 2050, T)
        
        def conv1d_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(out_c),
                nn.SiLU(),
                nn.MaxPool1d(2) # Halves Time only
            )

        # 1. Compress dimensionality & downsample time
        self.cnn = nn.Sequential(
            conv1d_block(in_dim, 512),  # T/2
            conv1d_block(512, 256),     # T/4
            conv1d_block(256, 128),     # T/8
            conv1d_block(128, 128),     # T/16
            conv1d_block(128, hidden_dim) # T/32, shape: (B, 128, T/32)
        )
        
        # 2. Temporal Smoothing (BiLSTM)
        # Bidirectional means output features will be 128 * 2 = 256
        self.lstm = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=hidden_dim, 
            batch_first=True, 
            bidirectional=True
        )
        
        # 3. The Bottleneck: Force down to 64 channels
        self.bottleneck = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.SiLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, 2050, T)
        x = self.cnn(x) # -> (B, 128, T/32)
        
        # PyTorch LSTM expects (Batch, Time, Features), so permute
        x = x.permute(0, 2, 1) # -> (B, T/32, 128)
        x, _ = self.lstm(x)    # -> (B, T/32, 256)
        x = x.permute(0, 2, 1) # -> (B, 256, T/32)
        
        x = self.bottleneck(x) # -> (B, 64, T/32)
        
        # Add the dummy Height dimension for tiling later
        return x.unsqueeze(-2) # -> (B, 64, 1, T/32)


class MusicYOLOHead(nn.Module):
    """
    Detection head for predicting note bounding boxes.
    """
    def __init__(self, in_channels=577, num_anchors=3):
        super().__init__()
        # Note: We output 5 values per anchor (obj_conf, x, y, w, h). 
        # If you add multi-instrument classification later, this becomes num_anchors * (5 + num_classes)
        self.num_anchors = num_anchors
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.Conv2d(256, num_anchors * 5, kernel_size=1) # Linear activation for final coordinates
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (B, 576, 11, T/32)
        # Output: (B, anchors * 5, 11, T/32)
        return self.conv(x)


class DualStreamMusicYOLO(nn.Module):
    """The complete end-to-end network."""
    def __init__(self, num_anchors=3):
        super().__init__()
        self.visual_branch = VisualBranch()
        self.semantic_branch = SemanticBranch(in_dim=2050)
        self.head = MusicYOLOHead(in_channels=577, num_anchors=num_anchors)

    def forward(self, posteriorgram, embedding, confidence, gradient, raw_shape):
        """
        Inputs:
            posteriorgram: (B, 1, 360, T)
            embedding: (B, 2048, T)
            confidence: (B, 1, T)
            gradient: (B, 1, T)
        """
        B, _, H, T = posteriorgram.shape
        device = posteriorgram.device

        # 1. Generate CoordiConv for Branch A
        # Create normalized Y-coordinates from 0.0 (bottom) to 1.0 (top)
        y_coords = torch.linspace(0, 1, steps=H, device=device).view(1, 1, H, 1)
        y_channel = y_coords.expand(B, 1, H, T) # Broadcast to batch and time
        
        # Concatenate posteriorgram and coordinates
        visual_input = torch.cat([posteriorgram, y_channel], dim=1) # -> (B, 2, 360, T)
        
        # 2. Process Visual Branch
        feat_a = self.visual_branch(visual_input) # -> (B, 512, 11, T/32)
        
        # 3. Prepare and Process Semantic Branch
        semantic_input = torch.cat([embedding, confidence, gradient], dim=1) # -> (B, 2051, T)
        feat_b = self.semantic_branch(semantic_input) # -> (B, 64, 1, T/32)
        
        # 4. Tiling & Fusion
        # Tile Branch B's output vertically to match Branch A's height (11)
        target_h = feat_a.shape[2]
        feat_b_tiled = feat_b.repeat(1, 1, target_h, 1) # -> (B, 64, 11, T/32)

        # Downsample raw_shape (T) to match the feature map time dimension (T/32)
        target_t = feat_a.shape[-1]
        raw_shape_downsampled = torch.nn.functional.adaptive_avg_pool1d(raw_shape, output_size=target_t) # -> (B, 1, T/32)

        # Tile it vertically to match the height (11)
        raw_shape_tiled = raw_shape_downsampled.unsqueeze(2).repeat(1, 1, target_h, 1) # -> (B, 1, 11, T/32)
        
        fused_features = torch.cat([feat_a, feat_b_tiled, raw_shape_tiled], dim=1) # -> (B, 577, 11, T/32)
        
        # 5. YOLO Detection Head
        predictions = self.head(fused_features) # -> (B, 15, 11, T/32)
        
        return predictions