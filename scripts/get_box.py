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
        print(f"Error: Video file not found at {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read the first frame of the video.")
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
