import argparse
import sys
import os

# Ensure src is in the python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.cv_project.pipeline.part2_pipeline import Part2Pipeline

def main():
    parser = argparse.ArgumentParser(description="Run Part 2 Pipeline: SAM 2 + ProPainter")
    parser.add_argument('--config', type=str, default='configs/part2_sam2_propainter.yaml', help='Path to configuration file')
    parser.add_argument('--video', type=str, default=None, help='Path to input video (overrides config)')
    args = parser.parse_args()

    print("=== Starting Part 2 Pipeline ===")
    pipeline = Part2Pipeline(args.config)
    pipeline.run(override_video_path=args.video)
    print("=== Pipeline Execution Finished ===")

if __name__ == "__main__":
    main()
