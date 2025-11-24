import os
import cv2
import matplotlib.pyplot as plt

# PATH CONFIGURATION based on your structure
# We are in 'Codes', so we look into 'dataCollections/frames'
frames_path = os.path.join("dataCollections", "frames")

def run_eda():
    # 1. Check if frames exist
    if not os.path.exists(frames_path):
        print(f"❌ Error: Could not find folder: {frames_path}")
        print("Make sure you ran extract_frames.py first!")
        return

    image_files = [f for f in os.listdir(frames_path) if f.endswith('.jpg')]
    print(f"📊 Analyzing {len(image_files)} images...")

    if len(image_files) == 0:
        print("No images found to analyze.")
        return

    # 2. brightness Analysis
    brightness_values = []

    for img_name in image_files[:100]: # Analyze first 100 images to be fast
        full_path = os.path.join(frames_path, img_name)
        img = cv2.imread(full_path)

        if img is not None:
            # Convert to HSV (Hue, Saturation, Value) to get Brightness (Value)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            avg_brightness = hsv[:,:,2].mean()
            brightness_values.append(avg_brightness)

    # 3. Plot the Graph
    plt.figure(figsize=(10, 5))
    plt.hist(brightness_values, bins=20, color='#3498db', edgecolor='black')
    plt.title("Brightness Distribution (Are images too dark/bright?)")
    plt.xlabel("Brightness Level (0=Black, 255=White)")
    plt.ylabel("Frame Count")
    plt.grid(axis='y', alpha=0.5)
    plt.show()
    print("✅ EDA Graph Generated!")

if __name__ == "__main__":
    run_eda()