import os
import subprocess
import sys

class ProPainterInpainter:
    def __init__(self, propainter_repo_path, checkpoint_dir="checkpoints/propainter", resize_ratio=1.0, neighbor_stride=10, subvideo_length=80):
        self.propainter_dir = propainter_repo_path
        self.inference_script = os.path.join(self.propainter_dir, "inference_propainter.py")
        self.resize_ratio = resize_ratio
        self.neighbor_stride = neighbor_stride
        self.subvideo_length = subvideo_length
        # Note: We rely on the user having the checkpoints in checkpoints/propainter

        if not os.path.exists(self.inference_script):
            print(f"WARNING: ProPainter inference script not found at {self.inference_script}")

    def inpaint(self, video_path_or_dir, mask_dir, output_dir):
        """
        Runs ProPainter inference using a subprocess call.
        """
        print(f"[ProPainter] Starting inpainting process...")
        os.makedirs(output_dir, exist_ok=True)
        
        cmd = [
            sys.executable,
            self.inference_script,
            "--video", video_path_or_dir,
            "--mask", mask_dir,
            "--output", output_dir,
            "--resize_ratio", str(self.resize_ratio),
            "--neighbor_length", str(self.neighbor_stride),
            "--subvideo_length", str(self.subvideo_length)
        ]
        
        print(f"[ProPainter] CMD: {' '.join(cmd)}")
        
        # We need to set PYTHONPATH so that ProPainter run correctly if it expects to be root
        env = os.environ.copy()
        env["PYTHONPATH"] = self.propainter_dir + (f":{env['PYTHONPATH']}" if "PYTHONPATH" in env else "")
        
        try:
            result = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
            print("[ProPainter] Success! Output printed to console (or captured).")
            # You can print result.stdout if debugging is needed
        except subprocess.CalledProcessError as e:
            print(f"[ProPainter] FAILED. Error:\n{e.stderr}")
            raise e
        
        print(f"[ProPainter] Done. Saved to {output_dir}")
        return output_dir
