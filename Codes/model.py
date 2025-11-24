import torch.nn as nn
import torch
from torchvision import models

class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()
        
        # --- PART 1: THE FRONTEND ---
        # We use VGG16 (a famous pre-trained model) to understand basic shapes/features.
        # We only need the first few layers, so we load it and slice it.
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        features = list(vgg.features.children())
        
        # Take the first 23 layers of VGG16
        self.frontend_feat = nn.Sequential(*features[0:23])
        
        # --- PART 2: THE BACKEND ---
        # This part uses "Dilated Convolutions" (dilation=2).
        # This expands the receptive field, helping the AI count dense crowds better.
        self.backend_feat = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True)
        )
        
        # --- PART 3: OUTPUT ---
        # A 1x1 convolution that produces the final "Density Map".
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Pass the image through the frontend (VGG16)
        x = self.frontend_feat(x)
        
        # Pass through the backend (Dilated Layers)
        x = self.backend_feat(x)
        
        # Generate final map
        x = self.output_layer(x)
        return x

# --- TEST CODE ---
# This block runs only if you execute this file directly.
# It checks if the model builds without errors.
if __name__ == "__main__":
    try:
        model = CSRNet()
        # Move to GPU if available, else CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        print("✅ SUCCESS: CSRNet model built successfully!")
        print(f"   Running on: {device}")
        
        # Create a dummy image (1 image, 3 channels, 256x256 size) to test the flow
        dummy_input = torch.randn(1, 3, 256, 256).to(device)
        output = model(dummy_input)
        
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Output shape: {output.shape}")
        print("   (Note: Output height/width is usually 1/8th of input size)")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")