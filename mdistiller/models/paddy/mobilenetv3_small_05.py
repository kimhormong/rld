import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class MobileNetV3Small05(nn.Module):
  """Unrolled, layer-by-layer trainable wrapper for MobileNetV3-Small (width_mult=0.5).
  Extracts features at resolution changes without block grouping.
  """

  def __init__(self):
    super().__init__()
    # Keep the original submodules registered so all parameters train normally
    model = models.mobilenet_v3_small(weights=None, num_classes=10, dropout_probability=0, width_mult=0.5)
    self.features = model.features
    self.classifier = model.classifier

    # Infer channel counts dynamically for ReviewKD configs
    self.stage_channels = self._get_channels()

  def _get_channels(self):
    self.eval()
    with torch.no_grad():
      _, feats = self.forward(torch.zeros(1, 3, 224, 224))
      channels = [f.shape[1] for f in feats["preact_feats"]]
    self.train()
    return channels

  def get_bn_before_relu(self):
    """Directly returns the trailing BatchNorm layer for each stage."""
    return [
        self.features[1].block[-1][1],  # Stage 1 terminal BN (56x56)
        self.features[3].block[-1][1],  # Stage 2 terminal BN (28x28)
        self.features[8].block[-1][1],  # Stage 3 terminal BN (14x14)
        self.features[12][1],  # Stage 4 terminal BN (7x7)
    ]

  def get_stage_channels(self):
    return self.stage_channels

  def forward(self, x):
    # --- Stem (224x224 -> 112x112) ---
    x = self.features[0](x)

    # --- Stage 1 (112x112 -> 56x56) ---
    x = self.features[1](x)
    f_56 = x  # Tap Level 1: (B, C1, 56, 56)

    # --- Stage 2 (56x56 -> 28x28) ---
    x = self.features[2](x)  # stride=2 downsample
    x = self.features[3](x)
    f_28 = x  # Tap Level 2: (B, C2, 28, 28)

    # --- Stage 3 (28x28 -> 14x14) ---
    x = self.features[4](x)  # stride=2 downsample
    x = self.features[5](x)
    x = self.features[6](x)
    x = self.features[7](x)
    x = self.features[8](x)
    f_14 = x  # Tap Level 3: (B, C3, 14, 14)

    # --- Stage 4 (14x14 -> 7x7) ---
    x = self.features[9](x)  # stride=2 downsample
    x = self.features[10](x)
    x = self.features[11](x)

    # Final expansion block: features[12] = Conv2d(0) -> BatchNorm(1) -> Hardswish(2)
    x = self.features[12][0](x)  # 1x1 Conv
    f_7 = self.features[12][1](x)  # Tap Level 4: Post-BN, Pre-Hardswish (7x7)
    x = self.features[12][2](f_7)  # Hardswish

    # --- Stage 5 (Global Representation: 1x1) ---
    avg_pool = F.adaptive_avg_pool2d(x, (1, 1))
    pooled = avg_pool.flatten(1) # Flatten to (B, C4) for classifier input

    # Standard classification head
    out = self.classifier(pooled)

    feats = {
        # Ordered apex-to-base: [f_56, f_28, f_14, f_7]
        "preact_feats": [f_56, f_28, f_14, f_7],
        "pooled_feat": pooled,
    }

    return out, feats