import torch
import torch.nn as nn
from torchvision import models
import pandas as pd


# MobileNetV2-based CSRNet
class MobileNetCSRNet(nn.Module):
    def __init__(self, load_weights=True):
        super(MobileNetCSRNet, self).__init__()

        weights = models.MobileNet_V2_Weights.DEFAULT if load_weights else None
        mobilenet = models.mobilenet_v2(weights=weights)

        # FRONTEND (MobileNet feature extractor)
        self.frontend = nn.Sequential(*list(mobilenet.features.children())[:7])

        # BACKEND (dilated CNN)
        self.backend = nn.Sequential(
            nn.Conv2d(32, 512, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=2, dilation=2), nn.ReLU()
        )

        # OUTPUT
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        # init backend + output layers
        self._initialize_weights()

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return self.output_layer(x)

    def _initialize_weights(self):
        for m in self.backend.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        nn.init.normal_(self.output_layer.weight, std=0.01)
        nn.init.constant_(self.output_layer.bias, 0)


# Save model architecture as CSV
def save_model_architecture(model, out_csv):
    rows = []

    for name, module in model.named_modules():
        if name == "":
            continue
        param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)

        rows.append({
            "layer_name": name,
            "layer_type": module.__class__.__name__,
            "params": param_count
        })

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"✔ Saved MobileNet CSRNet architecture → {out_csv}")
