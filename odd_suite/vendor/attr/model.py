"""YOLOv8n CSPDarknet in PyTorch; compatible with the existing MMYOLO weights.

Architecture adapted from OpenMMLab MMYOLO (Apache-2.0); see THIRD_PARTY.md.
Only the backbone is reused. Global pooling and classification heads are new.
"""
import torch
from torch import nn


class Conv(nn.Module):
    def __init__(self, cin, cout, k=1, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(cout, eps=0.001, momentum=0.03)
        self.activate = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.activate(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = Conv(channels, channels, 3)
        self.conv2 = Conv(channels, channels, 3)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class C2f(nn.Module):
    def __init__(self, channels, n):
        super().__init__()
        self.main_conv = Conv(channels, channels)
        self.final_conv = Conv((2 + n) * (channels // 2), channels)
        self.blocks = nn.ModuleList(Bottleneck(channels // 2) for _ in range(n))

    def forward(self, x):
        parts = list(self.main_conv(x).chunk(2, dim=1))
        for block in self.blocks:
            parts.append(block(parts[-1]))
        return self.final_conv(torch.cat(parts, dim=1))


class SPPF(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = Conv(channels, channels // 2)
        self.conv2 = Conv(2 * channels, channels)
        self.pool = nn.MaxPool2d(5, 1, 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        return self.conv2(torch.cat([x, y1, y2, self.pool(y2)], dim=1))


class YOLOv8nBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = Conv(3, 16, 3, 2)
        for i, (cin, cout, n) in enumerate([(16, 32, 1), (32, 64, 2),
                                           (64, 128, 2), (128, 256, 1)], 1):
            layers = [Conv(cin, cout, 3, 2), C2f(cout, n)]
            if i == 4:
                layers.append(SPPF(cout))
            setattr(self, f"stage{i}", nn.Sequential(*layers))

    def forward(self, x):
        x = self.stem(x)
        for i in range(1, 5):
            x = getattr(self, f"stage{i}")(x)
        return x


class MultiTaskYOLO(nn.Module):
    def __init__(self, classes, checkpoint=None):
        super().__init__()
        self.backbone = YOLOv8nBackbone()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.heads = nn.ModuleDict({k: nn.Linear(256, len(v)) for k, v in classes.items()})
        self.pretrained_report = None
        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state = state.get("state_dict", state)
            prefix = "backbone."
            weights = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            # Fail if ANY backbone tensor is missing, extra, or has a different shape.
            self.backbone.load_state_dict(weights, strict=True)
            self.pretrained_report = {"path": str(checkpoint), "state_entries": len(weights),
                                      "tensor_elements": sum(v.numel() for v in weights.values()),
                                      "strict": True}

    def forward(self, x):
        features = self.pool(self.backbone(x)).flatten(1)
        return {k: head(features) for k, head in self.heads.items()}
