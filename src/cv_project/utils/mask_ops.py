from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage


def merge_instance_masks(masks: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    merged = np.zeros(shape, dtype=np.uint8)
    for mask in masks:
        merged = np.maximum(merged, (mask > 0).astype(np.uint8) * 255)
    return merged


def fill_holes(mask: np.ndarray) -> np.ndarray:
    filled = ndimage.binary_fill_holes(mask > 0)
    return filled.astype(np.uint8) * 255


def remove_small_components(mask: np.ndarray, min_area_ratio: float) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    min_area = int(round(mask.shape[0] * mask.shape[1] * min_area_ratio))
    output = np.zeros_like(binary)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            output[labels == label] = 1
    return output.astype(np.uint8) * 255


def dilate_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask, kernel, iterations=1)


def temporal_majority_vote(masks: list[np.ndarray], window: int, votes: int) -> list[np.ndarray]:
    if window <= 1:
        return [mask.copy() for mask in masks]
    half_window = window // 2
    binary_masks = [(mask > 0).astype(np.uint8) for mask in masks]
    smoothed: list[np.ndarray] = []
    for index in range(len(binary_masks)):
        start = max(0, index - half_window)
        end = min(len(binary_masks), index + half_window + 1)
        stacked = np.stack(binary_masks[start:end], axis=0)
        current = (stacked.sum(axis=0) >= votes).astype(np.uint8) * 255
        smoothed.append(current)
    return smoothed


def postprocess_masks(raw_masks: list[np.ndarray], config: dict) -> list[np.ndarray]:
    processed: list[np.ndarray] = []
    for mask in raw_masks:
        current = mask.copy()
        if config.get("fill_holes", True):
            current = fill_holes(current)
        current = remove_small_components(current, float(config["min_component_area_ratio"]))
        current = dilate_mask(current, int(config["dilation_kernel_size"]))
        processed.append(current)
    if config.get("temporal_smoothing", True):
        processed = temporal_majority_vote(
            processed,
            int(config["temporal_window"]),
            int(config["temporal_votes"]),
        )
    return processed
