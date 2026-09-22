import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseLayer(nn.Module):
  """Single bottleneck dense layer: (1x1 conv -> 3x3 conv)."""

  def __init__(self, in_channels, growth_rate=32, bn_size=4):
    super(DenseLayer, self).__init__()
    # 1x1 convolution for dimensionality reduction
    self.dimension_reduction = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(
            in_channels,
            bn_size * growth_rate,
            kernel_size=1,
            stride=1,
            bias=False,
        ),
    )

    # 3x3 convolution for feature extraction
    self.feature_extraction = nn.Sequential(
        nn.BatchNorm2d(bn_size * growth_rate),
        nn.ReLU(inplace=True),
        nn.Conv2d(
            bn_size * growth_rate,
            growth_rate,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        ),
    )

  def forward(self, x):
    new_features = self.dimension_reduction(x)
    new_features = self.feature_extraction(new_features)
    return torch.cat([x, new_features], dim=1)


class DenseBlock(nn.Module):
  """Sequence of DenseLayer modules where each layer concatenates its output."""

  def __init__(self, num_layers, in_channels, growth_rate=32, bn_size=4):
    super(DenseBlock, self).__init__()
    self.layers = nn.ModuleList([
        DenseLayer(in_channels + i * growth_rate, growth_rate, bn_size)
        for i in range(num_layers)
    ])

  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
    return x


class TransitionLayer(nn.Module):
  """Expanded Transition Layer exposing individual components.

  Allows ReviewKD to tap pre-ReLU normalized features before pooling.
  """

  def __init__(self, in_channels, compression_factor=0.5):
    super(TransitionLayer, self).__init__()
    out_channels = int(in_channels * compression_factor)

    self.norm = nn.BatchNorm2d(in_channels)  # Tap point for ReviewKD
    self.relu = nn.ReLU(inplace=True)
    self.conv = nn.Conv2d(
        in_channels, out_channels, kernel_size=1, stride=1, bias=False
    )
    self.pool = nn.AvgPool2d(kernel_size=2, stride=2)  # Downsampling step

  def forward(self, x):
    pre_relu = self.norm(x)
    x = self.relu(pre_relu)
    x = self.conv(x)
    out = self.pool(x)
    return pre_relu, out


class DenseNet(nn.Module):
  """Expanded, layer-by-layer DenseNet-201 implementation for ReviewKD distillation.

  Block configuration for 201 layers: (6, 12, 48, 32).
  """

  def __init__(
      self,
      growth_rate=32,
      block_config=(6, 12, 48, 32),  # DenseNet-201 configuration
      num_init_features=64,
      bn_size=4,
      compression_factor=0.5,
      num_classes=1000,
  ):
    super(DenseNet, self).__init__()

    # ---------------- Stem (224x224 -> 56x56) ----------------
    self.conv0 = nn.Conv2d(
        3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False
    )
    self.norm0 = nn.BatchNorm2d(num_init_features)
    self.relu0 = nn.ReLU(inplace=True)
    self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    # ---------------- Stage 1 (56x56) ----------------
    num_features = num_init_features
    self.dense_block1 = DenseBlock(
        block_config[0], num_features, growth_rate, bn_size
    )
    num_features = num_features + block_config[0] * growth_rate  # 256
    self.transition1 = TransitionLayer(num_features, compression_factor)
    num_features = int(num_features * compression_factor)  # 128

    # ---------------- Stage 2 (28x28) ----------------
    self.dense_block2 = DenseBlock(
        block_config[1], num_features, growth_rate, bn_size
    )
    num_features = num_features + block_config[1] * growth_rate  # 512
    self.transition2 = TransitionLayer(num_features, compression_factor)
    num_features = int(num_features * compression_factor)  # 256

    # ---------------- Stage 3 (14x14) ----------------
    self.dense_block3 = DenseBlock(
        block_config[2], num_features, growth_rate, bn_size
    )
    num_features = num_features + block_config[2] * growth_rate  # 1792
    self.transition3 = TransitionLayer(num_features, compression_factor)
    num_features = int(num_features * compression_factor)  # 896

    # ---------------- Stage 4 (7x7) ----------------
    self.dense_block4 = DenseBlock(
        block_config[3], num_features, growth_rate, bn_size
    )
    num_features = num_features + block_config[3] * growth_rate  # 1920

    # Final normalization before classifier
    self.norm5 = nn.BatchNorm2d(num_features)
    self.relu5 = nn.ReLU(inplace=True)
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    # ---------------- Classifier ----------------
    self.classifier = nn.Linear(num_features, num_classes)

    # Channels for ReviewKD config: [1x1, 7x7, 14x14, 28x28, 56x56]
    self.stage_channels = [1920, 1920, 1792, 512, 256]

  def get_bn_before_relu(self):
    """Returns the BatchNorm layers preceding each stage's ReLU activation."""
    return [
        self.transition1.norm,
        self.transition2.norm,
        self.transition3.norm,
        self.norm5,
    ]

  def get_stage_channels(self):
    return self.stage_channels

  def forward(self, x):
    # Stem: 224x224 -> 112x112 -> 56x56
    x = self.conv0(x)
    x = self.norm0(x)
    stem = x
    x = self.relu0(x)
    x = self.pool0(x)

    # ---------------- Stage 1 (56x56x64) ----------------
    earlier_feat1 = x  # 56x56x64
    x = self.dense_block1(x)  # out: 56x56x256
    f1_pre, x = self.transition1(x)  # f1_pre = 56x56x256 pre-ReLU, out: pooled = 28x28x128,


    # ---------------- Stage 2 (28x28x128) ----------------
    earlier_feat2 = x  # 28x28x128
    x = self.dense_block2(x)  # out: 28x28x512
    f2_pre, x = self.transition2(x)  # f2_pre = 28x28x512 pre-ReLU, out: pooled = 14x14x256, 
    
    # ---------------- Stage 3 (14x14x256) ----------------
    earlier_feat3 = x  # 14x14
    x = self.dense_block3(x)  # out: 14x14x1792
    f3_pre, x = self.transition3(x)  #f3_pre = 14x14x1792 pre-ReLU, out: pooled: 7x7x896

    # ---------------- Stage 4 (7x7x896) ----------------
    earlier_feat4 = x  # 7x7x896
    x = self.dense_block4(x) # out: 7x7x1920
    f4_pre = self.norm5(x)  # Pre-ReLU final norm = 7x7x1920
    x = self.relu5(f4_pre)

    # ---------------- Stage 5 (1x1 GAP) ----------------
    f5_gap = self.avgpool(x)  # (B, 1920, 1, 1)
    pooled = torch.flatten(f5_gap, 1)
    out = self.classifier(pooled)

    feats = {
        # Raw un-normalized DenseBlock concatenated features
        "earlier_feats": [earlier_feat1, earlier_feat2, earlier_feat3, earlier_feat4],
        "preact_feats": [stem, f1_pre, f2_pre, f3_pre, f4_pre],
        "pooled_feat": pooled,
    }

    return out, feats


def densenet201(num_classes=10):
    return DenseNet(num_classes=num_classes)