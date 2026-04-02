import cv2
import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Get bounding box from the first frame of a video.")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video file")
    args = parser.parse_args()

    video_path = args.video
    if not os.path.exists(video_path):
        print(f"Error: Path not found at {video_path}")
        sys.exit(1)

    frame = None
    if os.path.isdir(video_path):
        # Read the first image in the directory
        valid_exts = ('.jpg', '.png', '.jpeg')
        files = sorted([f for f in os.listdir(video_path) if f.lower().endswith(valid_exts)])
        if not files:
            print(f"Error: No image files found in directory {video_path}")
            sys.exit(1)
        first_frame_path = os.path.join(video_path, files[0])
        frame = cv2.imread(first_frame_path)
    else:
        # Treat as video file
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

    if frame is None:
        print("Error: Could not read the image or video.")
        sys.exit(1)

    print("\n--- INSTRUCTIONS ---")
    print("1. Click and drag your mouse over the object you want to remove.")
    print("2. Press ENTER or SPACE to confirm the box.")
    print("3. Press C to cancel and try again.")
    print("--------------------\n")

    bbox = cv2.selectROI("Select Object to Remove (Press ENTER to confirm)", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    
    x_min, y_min, w, h = bbox
    
    if w == 0 or h == 0:
        print("No box selected.")
        sys.exit(0)
        
    box_prompt = [int(x_min), int(y_min), int(x_min + w), int(y_min + h)]
    
    print(f"\n======================================")
    print(f"✅ Bounding Box 提取成功！")
    print(f"请复制下面的数组到你的 configs/part2_sam2_propainter.yaml 中的 prompt -> box 字段:")
    print(f"\nbox: {box_prompt}\n")
    print(f"======================================\n")

if __name__ == "__main__":
    main()
