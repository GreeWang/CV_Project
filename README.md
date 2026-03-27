# CV Project Part 1 Baseline

This repository contains a submission-ready Part 1 baseline for video object removal and inpainting.

The baseline pipeline is:

1. Extract or load ordered video frames
2. Run `YOLOv8-Seg` on each frame
3. Keep only dynamic-object classes relevant to the assignment
4. Use Lucas-Kanade sparse optical flow to reject likely static detections
5. Merge dynamic instance masks into frame-level masks
6. Post-process masks with hole filling, small-component removal, dilation, and temporal smoothing
7. Restore masked regions with temporal background copy first and `cv2.inpaint` fallback
8. Export masks, overlays, restored frames, comparison panels, final video, and a run summary JSON

## Environment

Recommended environment:

1. Windows
2. Python 3.10+
3. Local CUDA-capable GPU

Install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If the default PyTorch wheel does not match your CUDA runtime, install the correct CUDA build manually first.

## Run

```powershell
python scripts\run_part1.py --config configs\part1.yaml
```

Optional overrides:

```powershell
python scripts\run_part1.py --config configs\part1.yaml --set input.video_path="data\\raw\\bmx-trees\\input.mp4" --set output.dataset_name="bmx-trees"
```

## Output

Each run is saved under:

```text
results/part1/<dataset_name>/<timestamp>/
```

The run directory contains:

1. normalized frames
2. raw segmentation overlays
3. raw dynamic masks
4. final inpainting masks
5. temporal-fill intermediate frames
6. final inpainted frames
7. comparison panels
8. selected report figures
9. final restored video
10. `run_summary.json`

## Known Limitations

This baseline intentionally does not implement global camera-motion compensation. It can degrade on videos with strong camera motion or background shake.
