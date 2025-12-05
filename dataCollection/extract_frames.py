import cv2
import os

def extract_frames(video_path, output_dir, fps_interval=1):
    """
    Extract frames from video file at regular intervals.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f" Video: {video_path}")
    print(f"FPS: {fps}, Total Frames: {total_frames}, Duration: {duration:.2f}s")

    count = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % (fps * fps_interval) == 0:
            frame_resized = cv2.resize(frame, (640, 480))
            filename = os.path.join(output_dir, f"frame_{count:05d}.jpg")
            cv2.imwrite(filename, frame_resized)
            count += 1

        frame_count += 1

    cap.release()
    print(f" Extraction complete! {count} frames saved in '{output_dir}'.")

if __name__ == "__main__":
    video_path = "crowd_video.mp4"
    output_dir = "frames"
    extract_frames(video_path, output_dir, fps_interval=1)
