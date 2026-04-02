import os
import sys
import torch
import numpy as np
import cv2

# Add the cloned SAM 2 repository to path
SAM2_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "sam2")
if SAM2_PATH not in sys.path:
    sys.path.append(SAM2_PATH)

try:
    from sam2.build_sam import build_sam2_video_predictor
except ImportError:
    print("WARNING: SAM 2 not found or not properly installed. Ensure you cloned sam2 into src/.")
    build_sam2_video_predictor = None

class SAM2VideoTracker:
    def __init__(self, checkpoint_path, model_cfg="sam2_hiera_s.yaml", device=None):
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
            
        print(f"[SAM2Tracker] Initializing on {self.device}")
        if build_sam2_video_predictor is None:
            raise RuntimeError("SAM 2 could not be loaded.")
            
        self.predictor = build_sam2_video_predictor(model_cfg, checkpoint_path, device=self.device)

    def track(self, video_dir, box_prompt=None, point_prompt=None, point_label=None, out_mask_dir=None):
        """
        Track object in video_dir (containing JPEG frames) and optionally save masks.
        """
        print(f"[SAM2Tracker] Initializing state for video frames in {video_dir}")
        inference_state = self.predictor.init_state(video_path=video_dir)
        self.predictor.reset_state(inference_state)
        
        ann_frame_idx = 0
        ann_obj_id = 1
        
        if box_prompt is not None and len(box_prompt) == 4:
            box = np.array(box_prompt, dtype=np.float32)
            self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                box=box,
            )
        elif point_prompt is not None and len(point_prompt) > 0:
            points = np.array(point_prompt, dtype=np.float32)
            labels = np.array(point_label, dtype=np.int32)
            self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                points=points,
                labels=labels,
            )
        else:
            raise ValueError("Must provide a valid box_prompt or point_prompt.")

        print(f"[SAM2Tracker] Propagating...")
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
            mask = (out_mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8) * 255
            # mask shape is usually [1, H, W] or [H, W], squeeze to [H,W]
            if mask.ndim == 3:
                mask = mask.squeeze(0)
            video_segments[out_frame_idx] = mask
            
            if out_mask_dir is not None:
                os.makedirs(out_mask_dir, exist_ok=True)
                frame_name = f"{out_frame_idx:05d}.png"
                cv2.imwrite(os.path.join(out_mask_dir, frame_name), mask)
                
        print(f"[SAM2Tracker] Tracking completed. {len(video_segments)} frames processed.")
        return video_segments
