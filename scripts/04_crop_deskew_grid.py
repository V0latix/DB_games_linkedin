#!/usr/bin/env python3
"""Crop and deskew the selected puzzle frame to produce a clean grid image."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from config import (
    APPROX_EPSILON_RATIO,
    CANNY_HIGH,
    CANNY_LOW,
    MORPH_CLOSE_KERNEL,
    QUAD_MIN_AREA_RATIO,
    TARGET_WIDTH,
)
from pipeline_utils import dump_json, ensure_dir, load_json, relative_to_cwd, utc_now_iso


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop and deskew selected puzzle frames")
    parser.add_argument("--selection-metadata", default="zip_archive/metadata/selected_frames.json")
    parser.add_argument("--output-metadata", default="zip_archive/metadata/grid_results.json")
    parser.add_argument("--grids-dir", default="zip_archive/grids")
    parser.add_argument("--target-width", type=int, default=TARGET_WIDTH)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only-video", default=None, help="Process only a specific video basename")
    parser.add_argument(
        "--override-frame",
        default=None,
        help="Manual frame path to use (requires --only-video)",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    sums = pts.sum(axis=1)
    rect[0] = pts[np.argmin(sums)]
    rect[2] = pts[np.argmax(sums)]

    diffs = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diffs)]
    rect[3] = pts[np.argmax(diffs)]
    return rect


def find_candidate_quad(image: np.ndarray) -> tuple[np.ndarray | None, str, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_CLOSE_KERNEL)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, "no_contours", 0.0

    image_area = image.shape[0] * image.shape[1]
    min_area = image_area * QUAD_MIN_AREA_RATIO

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    best_quad: np.ndarray | None = None
    best_area = 0.0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, APPROX_EPSILON_RATIO * perimeter, True)

        if len(approx) == 4 and area > best_area:
            best_quad = approx.reshape(4, 2).astype("float32")
            best_area = float(area)

    if best_quad is not None:
        return best_quad, "quad", best_area / image_area

    # Fallback for imperfect captures: keep the largest contour bounding box.
    largest = contours[0]
    x, y, w, h = cv2.boundingRect(largest)
    rect_pts = np.array(
        [
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h],
        ],
        dtype="float32",
    )
    ratio = float(cv2.contourArea(largest)) / image_area
    return rect_pts, "bounding_rect", ratio


def warp_from_quad(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    rect = order_points(quad)
    tl, tr, br, bl = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    max_width = max(max_width, 2)
    max_height = max(max_height, 2)

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    return warped


def resize_to_width(image: np.ndarray, target_width: int) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= 0 or h <= 0:
        return image
    if w == target_width:
        return image
    ratio = target_width / float(w)
    target_height = max(2, int(h * ratio))
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_CUBIC)


def main() -> int:
    args = parse_args()

    if args.override_frame and not args.only_video:
        raise SystemExit("--override-frame requires --only-video")

    selection_metadata = Path(args.selection_metadata)
    output_metadata = Path(args.output_metadata)
    grids_dir = Path(args.grids_dir)

    ensure_dir(grids_dir)
    ensure_dir(output_metadata.parent)

    selection = load_json(selection_metadata, default={})
    videos = selection.get("videos", [])

    if args.only_video:
        videos = [entry for entry in videos if entry.get("video_basename") == args.only_video]

    if args.limit is not None:
        videos = videos[: args.limit]

    previous = load_json(output_metadata, default={})
    previous_by_basename = {
        entry.get("video_basename"): entry
        for entry in previous.get("videos", [])
        if entry.get("video_basename")
    }

    output_entries: list[dict] = []

    for idx, entry in enumerate(videos, start=1):
        basename = entry.get("video_basename")
        if not basename:
            continue

        grid_path = grids_dir / f"{basename}_grid.png"
        previous_entry = previous_by_basename.get(basename)

        if grid_path.exists() and not args.force and previous_entry:
            output_entries.append(previous_entry)
            print(f"[{idx}/{len(videos)}] {basename}: skip (grid already exists)")
            continue

        frame_path = Path(entry.get("chosen_frame", ""))
        if args.override_frame and args.only_video == basename:
            frame_path = Path(args.override_frame)

        if not frame_path.is_absolute():
            frame_path = Path.cwd() / frame_path

        print(f"[{idx}/{len(videos)}] processing {basename}")

        if not frame_path.exists():
            output_entries.append(
                {
                    "video_basename": basename,
                    "chosen_frame": relative_to_cwd(frame_path),
                    "grid_image": relative_to_cwd(grid_path),
                    "status": "needs_review",
                    "error": "chosen frame does not exist",
                }
            )
            print("    -> missing frame")
            continue

        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            output_entries.append(
                {
                    "video_basename": basename,
                    "chosen_frame": relative_to_cwd(frame_path),
                    "grid_image": relative_to_cwd(grid_path),
                    "status": "needs_review",
                    "error": "unable to read frame",
                }
            )
            print("    -> unable to read frame")
            continue

        quad, method, area_ratio = find_candidate_quad(image)

        if quad is None:
            output_entries.append(
                {
                    "video_basename": basename,
                    "chosen_frame": relative_to_cwd(frame_path),
                    "grid_image": relative_to_cwd(grid_path),
                    "status": "needs_review",
                    "method": method,
                    "area_ratio": area_ratio,
                    "error": "grid contour not found",
                }
            )
            print("    -> no contour candidate")
            continue

        warped = warp_from_quad(image, quad)
        normalized = resize_to_width(warped, args.target_width)
        ensure_dir(grid_path.parent)
        cv2.imwrite(str(grid_path), normalized)

        status = "ok" if method == "quad" and area_ratio >= QUAD_MIN_AREA_RATIO else "needs_review"

        output_entry = {
            "video_basename": basename,
            "video_id": entry.get("video_id"),
            "title": entry.get("title"),
            "playlist_index": entry.get("playlist_index"),
            "source_url": entry.get("source_url"),
            "upload_date": entry.get("upload_date"),
            "chosen_frame": relative_to_cwd(frame_path),
            "grid_image": relative_to_cwd(grid_path),
            "status": status,
            "method": method,
            "area_ratio": round(float(area_ratio), 5),
            "error": None,
        }
        output_entries.append(output_entry)

        if args.verbose:
            print(f"    -> method={method} area_ratio={area_ratio:.3f} status={status}")

    payload = {
        "generated_at": utc_now_iso(),
        "settings": {
            "target_width": args.target_width,
            "limit": args.limit,
            "only_video": args.only_video,
            "override_frame": args.override_frame,
        },
        "videos": sorted(
            output_entries,
            key=lambda item: (
                item.get("playlist_index") is None,
                item.get("playlist_index") or 10**9,
                item.get("video_basename", ""),
            ),
        ),
    }

    dump_json(output_metadata, payload)
    print(f"Wrote grid metadata: {output_metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
