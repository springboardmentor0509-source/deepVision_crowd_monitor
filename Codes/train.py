import torch
import torch.nn as nn
import torch.optim as optim
from model import CSRNet
from dataset import CrowdDataset  # Make sure you have created dataset.py!
import os

def train_model(train_path, epochs=30, learning_rate=1e-5):
    # 1. Setup Device (GPU is faster, CPU is okay for testing)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Starting training on: {device}")

    # 2. Initialize Model, Loss, and Optimizer
    model = CSRNet().to(device)
    criterion = nn.MSELoss(size_average=False).to(device) # Mean Squared Error
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.95)

    # 3. Load Data
    # Note: For now, this assumes you have the dataset folder set up
    if not os.path.exists(train_path):
        print(f"❌ Error: Dataset path '{train_path}' not found!")
        return
        
    dataset = CrowdDataset(train_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    print(f"✅ Data loaded: {len(dataset)} images found.")

    # 4. Training Loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for i, (img, name) in enumerate(dataloader):
            img = img.to(device)
            
            # --- IMPORTANT: TARGET HANDLING ---
            # In a real run, we need "Ground Truth" density maps here.
            # For testing the code structure only, we will fake a target.
            # DELETE THIS LINE when you have real ground truth (.h5 files)
            target = torch.zeros(img.shape[0], 1, img.shape[2]//8, img.shape[3]//8).to(device)
            
            # Forward pass
            output = model(img)
            
            # Calculate loss (Difference between Guess and Real)
            loss = criterion(output, target)
            epoch_loss += loss.item()
            
            # Backward pass (Update weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f}")
        
        # Save the model every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_name = f"csrnet_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), save_name)
            print(f"💾 Saved model to: {save_name}")

if __name__ == "__main__":
    # Update this path to where your training images are!
    # If you are just testing the code, point it to your 'frames' folder.
    train_path = r"E:\DeepVision Crowd Monitor\Codes\dataCollections\frames"
    
    train_model(train_path, epochs=5)