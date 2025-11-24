import cv2
import os

def extract_frames(video_path, output_dir, fps_interval=1):
    # 1. Create the output folder if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created folder: {output_dir}")

    # 2. Load the video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"✅ Video loaded. FPS: {fps}")

    frame_count = 0
    saved_count = 0
    
    # Calculate how many frames to skip to get the desired interval (e.g., 1 frame per second)
    frames_to_skip = int(fps * fps_interval)

    print("Extracting frames... this might take a moment.")

    while True:
        success, frame = cap.read()
        
        if not success:
            break  # Video ended

        # Save frame only if it matches the interval
        if frame_count % frames_to_skip == 0:
            # Create a filename like "frame_0.jpg", "frame_1.jpg"
            frame_name = os.path.join(output_dir, f"frame_{saved_count}.jpg")
            cv2.imwrite(frame_name, frame)
            saved_count += 1
            print(f"Saved: {frame_name}")

        frame_count += 1

    cap.release()
    print(f"\nDone! Extracted {saved_count} frames to the '{output_dir}' folder.")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # The working path you confirmed
    video_path = r"E:\DeepVision Crowd Monitor\Codes\sampleDashboard\video.mp4"
    
    # The folder where images will be saved
    output_dir = "frames" 
    
    # Run the function (save 1 frame every 1 second)
    extract_frames(video_path, output_dir, fps_interval=1)