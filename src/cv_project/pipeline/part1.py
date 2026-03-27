from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from cv_project.data.io import (
    ensure_dir,
    extract_frames_from_video,
    list_frame_paths,
    normalize_frame_size,
    resolve_path,
    save_json,
    timestamp_now,
    write_video,
)
from cv_project.inpainting.restoration import spatial_inpaint, temporal_background_fill
from cv_project.motion.dynamic_filter import OpticalFlowDynamicFilter
from cv_project.pipeline.types import FrameRecord
from cv_project.segmentation.yolo_segmenter import YoloSegmenter
from cv_project.utils.mask_ops import merge_instance_masks, postprocess_masks
from cv_project.utils.visualization import create_comparison_panel, overlay_detections, overlay_mask_contours, save_report_frames


def run_part1_pipeline(config: dict, project_root: Path) -> dict:
    started_at = time.time()
    dataset_name = config["output"]["dataset_name"]
    run_dir = ensure_dir(resolve_path(config["output"]["root_dir"], project_root) / dataset_name / timestamp_now())
    frames_dir = ensure_dir(run_dir / "frames")
    overlays_dir = ensure_dir(run_dir / "raw_overlays")
    raw_masks_dir = ensure_dir(run_dir / "raw_dynamic_masks")
    final_masks_dir = ensure_dir(run_dir / "final_masks")
    contour_dir = ensure_dir(run_dir / "mask_contours")
    temporal_fill_dir = ensure_dir(run_dir / "temporal_fill")
    restored_dir = ensure_dir(run_dir / "restored_frames")
    panels_dir = ensure_dir(run_dir / "comparison_panels")
    figures_dir = ensure_dir(run_dir / "report_frames")

    input_video_path = resolve_path(config["input"].get("video_path"), project_root)
    input_frames_dir = resolve_path(config["input"].get("frames_dir"), project_root)
    frame_width = int(config["output"]["frame_name_width"])
    max_long_side = int(config["input"]["max_long_side"])

    if input_frames_dir is not None:
        source_frame_paths = list_frame_paths(input_frames_dir, config["input"]["image_extensions"])
        fps = float(config["output"].get("save_fps") or 24.0)
    elif input_video_path is not None:
        source_frame_paths, fps = extract_frames_from_video(input_video_path, frames_dir, max_long_side, frame_width)
    else:
        raise ValueError("Provide either input.video_path or input.frames_dir in the config.")

    frame_records: list[FrameRecord] = []
    loaded_frames: list[np.ndarray] = []
    if input_frames_dir is not None:
        for index, source_path in enumerate(source_frame_paths):
            image = cv2.imread(str(source_path))
            if image is None:
                raise RuntimeError(f"Unable to read frame image: {source_path}")
            image = normalize_frame_size(image, max_long_side)
            normalized_path = frames_dir / f"{index:0{frame_width}d}.png"
            cv2.imwrite(str(normalized_path), image)
            frame_records.append(FrameRecord(frame_index=index, image=image, source_path=str(normalized_path)))
            loaded_frames.append(image)
    else:
        for index, source_path in enumerate(source_frame_paths):
            image = cv2.imread(str(source_path))
            if image is None:
                raise RuntimeError(f"Unable to read frame image: {source_path}")
            frame_records.append(FrameRecord(frame_index=index, image=image, source_path=str(source_path)))
            loaded_frames.append(image)

    segmenter = YoloSegmenter(
        model_name=config["segmentation"]["model_name"],
        device=config["segmentation"]["device"],
        confidence_threshold=float(config["segmentation"]["confidence_threshold"]),
        iou_threshold=float(config["segmentation"]["iou_threshold"]),
        dynamic_classes=list(config["segmentation"]["dynamic_classes"]),
    )

    for record in tqdm(frame_records, desc="Segmentation"):
        record.raw_detections = segmenter.predict(record.image, record.frame_index)
        raw_overlay = overlay_detections(record.image, record.raw_detections, alpha=float(config["visualization"]["mask_alpha"]))
        cv2.imwrite(str(overlays_dir / f"{record.frame_index:06d}.png"), raw_overlay)

    dynamic_filter = OpticalFlowDynamicFilter(config["motion"])
    filtered_detections = dynamic_filter.apply(loaded_frames, [record.raw_detections for record in frame_records])

    raw_dynamic_masks: list[np.ndarray] = []
    for record, filtered in zip(frame_records, filtered_detections):
        record.filtered_detections = filtered
        raw_mask = merge_instance_masks([det.mask for det in filtered], record.image.shape[:2])
        record.raw_dynamic_mask = raw_mask
        raw_dynamic_masks.append(raw_mask)
        cv2.imwrite(str(raw_masks_dir / f"{record.frame_index:06d}.png"), raw_mask)

    final_masks = postprocess_masks(raw_dynamic_masks, config["mask_postprocess"])
    for record, final_mask in zip(frame_records, final_masks):
        record.final_mask = final_mask
        cv2.imwrite(str(final_masks_dir / f"{record.frame_index:06d}.png"), final_mask)
        contour_overlay = overlay_mask_contours(record.image, final_mask)
        cv2.imwrite(str(contour_dir / f"{record.frame_index:06d}.png"), contour_overlay)

    panel_paths: list[Path] = []
    restored_frame_paths: list[Path] = []
    total_temporal_filled_pixels = 0
    final_mask_list = [record.final_mask for record in frame_records]
    fallback_method = str(config["inpainting"]["fallback_method"])
    inpaint_radius = int(config["inpainting"]["inpaint_radius"])

    for record in tqdm(frame_records, desc="Restoration"):
        if bool(config["inpainting"].get("use_temporal_copy", True)):
            temporal_fill, residual_mask = temporal_background_fill(
                frame_index=record.frame_index,
                frames=loaded_frames,
                masks=final_mask_list,
                temporal_radius=int(config["inpainting"]["temporal_radius"]),
                use_temporal_median=bool(config["inpainting"]["use_temporal_median"]),
            )
        else:
            temporal_fill = record.image.copy()
            residual_mask = record.final_mask.copy()
        record.temporal_fill = temporal_fill
        filled_pixels = int(np.sum((record.final_mask > 0) & (residual_mask == 0)))
        total_temporal_filled_pixels += filled_pixels
        cv2.imwrite(str(temporal_fill_dir / f"{record.frame_index:06d}.png"), temporal_fill)

        restored = spatial_inpaint(temporal_fill, residual_mask, fallback_method, inpaint_radius)
        record.restored_image = restored
        restored_path = restored_dir / f"{record.frame_index:06d}.png"
        cv2.imwrite(str(restored_path), restored)
        restored_frame_paths.append(restored_path)

        raw_overlay = cv2.imread(str(overlays_dir / f"{record.frame_index:06d}.png"))
        panel = create_comparison_panel(
            original=record.image,
            raw_overlay=raw_overlay if raw_overlay is not None else record.image,
            dynamic_mask=record.raw_dynamic_mask,
            final_mask=record.final_mask,
            restored=restored,
            font_scale=float(config["visualization"]["panel_font_scale"]),
            font_thickness=int(config["visualization"]["panel_font_thickness"]),
        )
        panel_path = panels_dir / f"{record.frame_index:06d}.png"
        cv2.imwrite(str(panel_path), panel)
        panel_paths.append(panel_path)

    output_fps = float(config["output"]["save_fps"] or fps or 24.0)
    restored_video_path = run_dir / "restored.mp4"
    write_video(restored_frame_paths, restored_video_path, output_fps)

    report_frames = save_report_frames(
        panel_paths=panel_paths,
        output_dir=figures_dir,
        requested_count=int(config["visualization"]["save_report_frames"]),
        requested_indices=list(config["visualization"].get("report_frame_indices", [])),
    )

    summary = {
        "dataset_name": dataset_name,
        "run_dir": str(run_dir),
        "config": config,
        "input": {
            "video_path": str(input_video_path) if input_video_path else None,
            "frames_dir": str(input_frames_dir) if input_frames_dir else None,
        },
        "artifacts": {
            "frames_dir": str(frames_dir),
            "raw_overlays_dir": str(overlays_dir),
            "raw_dynamic_masks_dir": str(raw_masks_dir),
            "final_masks_dir": str(final_masks_dir),
            "mask_contours_dir": str(contour_dir),
            "temporal_fill_dir": str(temporal_fill_dir),
            "restored_frames_dir": str(restored_dir),
            "comparison_panels_dir": str(panels_dir),
            "report_frames_dir": str(figures_dir),
            "restored_video": str(restored_video_path),
        },
        "stats": {
            "num_frames": len(frame_records),
            "fps": output_fps,
            "detector_model": config["segmentation"]["model_name"],
            "confidence_threshold": config["segmentation"]["confidence_threshold"],
            "motion_displacement_threshold": config["motion"]["displacement_threshold"],
            "temporal_filled_pixels": total_temporal_filled_pixels,
            "elapsed_seconds": round(time.time() - started_at, 3),
        },
        "report_frames": report_frames,
    }
    save_json(summary, run_dir / "run_summary.json")
    return summary
