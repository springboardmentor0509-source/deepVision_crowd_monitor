import torch
from model import CSRNet
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm as c
import os

def predict_people(image_path):
    # 1. Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Load the Model Architecture
    model = CSRNet().to(device)
    
    # 3. Load Trained Weights (Optional for now)
    # If you had a trained file, you would uncomment this:
    # checkpoint = torch.load('csrnet_epoch_10.pth', map_location=device)
    # model.load_state_dict(checkpoint)
    
    model.eval() # Set to evaluation mode (no training)

    # 4. Prepare the Image
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found at {image_path}")
        return

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Preprocessing (Same as training)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img).unsqueeze(0) # Add batch dimension (1, 3, H, W)
    img_tensor = img_tensor.to(device)

    # 5. Predict
    with torch.no_grad():
        output = model(img_tensor)
    
    # The output is a "Density Map". The sum of the map is the count!
    count = int(output.data.sum().item())
    
    print("\n" + "="*30)
    print(f"🧐 AI PREDICTION")
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Estimated People Count: {count}")
    print("="*30 + "\n")

    # 6. Visualization (Optional - Shows the heatmap)
    # We display the density map to see WHERE the AI thinks people are
    output = output.squeeze().cpu().numpy()
    plt.imshow(output, cmap=c.jet)
    plt.title(f"Density Map (Count: {count})")
    plt.show()

if __name__ == "__main__":
    # Test on one of your extracted frames
    # Make sure this path points to a real image inside your 'frames' folder
    test_image = r"E:\DeepVision Crowd Monitor\Codes\dataCollections\frames\frame_0.jpg"
    
    predict_people(test_image)