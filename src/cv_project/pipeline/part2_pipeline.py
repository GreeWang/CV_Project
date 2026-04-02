import os
import yaml
import shutil
import cv2

from src.cv_project.segmentation.sam2_tracker import SAM2VideoTracker
from src.cv_project.inpainting.propainter_inpaint import ProPainterInpainter

class Part2Pipeline:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.video_path = self.config['data'].get('input_video')
        self.output_dir = self.config['data'].get('output_dir', 'data/outputs/part2')
        
        # We need an intermediate directory for masks
        video_name = os.path.basename(self.video_path).split('.')[0]
        self.mask_dir = os.path.join(self.output_dir, f"{video_name}_masks")
        self.frames_dir = os.path.join(self.output_dir, f"{video_name}_frames")
        
        # Init tracker
        sam2_cfg = self.config['sam2']
        self.tracker = SAM2VideoTracker(
            checkpoint_path=sam2_cfg['checkpoint'],
            model_cfg=sam2_cfg['model_cfg']
        )
        self.prompt_cfg = sam2_cfg['prompt']

        # Init inpainter
        pp_cfg = self.config['propainter']
        self.inpainter = ProPainterInpainter(
            propainter_repo_path=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "ProPainter"),
            checkpoint_dir=pp_cfg['checkpoint_dir'],
            resize_ratio=pp_cfg.get('resize_ratio', 1.0),
            neighbor_stride=pp_cfg.get('neighbor_stride', 10),
            subvideo_length=pp_cfg.get('subvideo_length', 80)
        )

    def extract_frames(self, video_path, out_dir):
        print(f"[Pipeline] Extracting frames to {out_dir}")
        os.makedirs(out_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(os.path.join(out_dir, f"{frame_idx:05d}.jpg"), frame)
            frame_idx += 1
        cap.release()
        return out_dir

    def run(self, override_video_path=None):
        if override_video_path:
            self.video_path = override_video_path
            video_name = os.path.basename(self.video_path).split('.')[0]
            self.mask_dir = os.path.join(self.output_dir, f"{video_name}_masks")
            self.frames_dir = os.path.join(self.output_dir, f"{video_name}_frames")

        # 1. Prepare frames
        if os.path.isdir(self.video_path):
            input_frames_dir = self.video_path
        else:
            input_frames_dir = self.extract_frames(self.video_path, self.frames_dir)

        # 2. Run Segmentation (SAM 2)
        box = self.prompt_cfg.get('box', None)
        points = self.prompt_cfg.get('points', None)
        labels = self.prompt_cfg.get('labels', None)
        
        if self.prompt_cfg['type'] == 'box' and (not box or len(box) == 0):
            print("WARNING: Using dummy bounding box [100, 100, 300, 300] for testing.")
            box = [100, 100, 300, 300]

        _ = self.tracker.track(
            video_dir=input_frames_dir,
            box_prompt=box if self.prompt_cfg['type'] == 'box' else None,
            point_prompt=points if self.prompt_cfg['type'] == 'point' else None,
            point_label=labels if self.prompt_cfg['type'] == 'point' else None,
            out_mask_dir=self.mask_dir
        )

        # 3. Post-process masks (Dilation is internal to propainter usually, but we could do it here)
        # 4. ProPainter Inpainting
        self.inpainter.inpaint(
            video_path_or_dir=input_frames_dir,
            mask_dir=self.mask_dir,
            output_dir=self.output_dir
        )
        print(f"[Pipeline] Part 2 complete for {self.video_path}")
