import torch
import torch.nn as nn
from torchvision import models
import pandas as pd


# Build CSRNet Architecture
class CSRNet(nn.Module):
    def __init__(self, load_weights=True):
        super(CSRNet, self).__init__()

        self.frontend_cfg = [64, 64, 'M',
                             128, 128, 'M',
                             256, 256, 256, 'M',
                             512, 512, 512]

        self.backend_cfg = [512, 512, 512, 256, 128, 64]

        self.frontend = make_layers(self.frontend_cfg)
        self.backend  = make_layers(self.backend_cfg, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        self._initialize_weights()

        # Load VGG16 weights
        if load_weights:
            try:
                vgg = models.vgg16(pretrained=True)
                vgg_layers = list(vgg.features.children())

                i = 0
                for layer in self.frontend:
                    if isinstance(layer, nn.Conv2d):
                        while not isinstance(vgg_layers[i], nn.Conv2d):
                            i += 1
                        layer.weight.data = vgg_layers[i].weight.data.clone()
                        layer.bias.data   = vgg_layers[i].bias.data.clone()
                        i += 1
            except Exception as e:
                print("Warning: pretrained VGG16 weight load failed →", e)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return self.output_layer(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None: nn.init.constant_(m.bias, 0)


# Utility: Make CNN layers
def make_layers(cfg, in_channels=3, dilation=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(2))
        else:
            conv = nn.Conv2d(
                in_channels,
                v,
                kernel_size=3,
                padding=2 if dilation else 1,
                dilation=2 if dilation else 1
            )
            layers += [conv, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


# Save Model Architecture as CSV
def save_model_architecture(model, out_csv):
    rows = []
    for name, module in model.named_modules():
        if name == "":
            continue
        param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
        rows.append({
            "layer_name": name,
            "layer_type": module.__class__.__name__,
            "trainable_params": param_count
        })

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"✔ Saved model architecture → {out_csv}")
