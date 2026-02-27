#!/usr/bin/env python3
"""Pick the best puzzle frame per video using image quality/grid heuristics."""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
from config import (
    BALANCE_BONUS_WEIGHT,
    BLUR_PENALTY_WEIGHT,
    BLUR_THRESHOLD,
    BLOCKINESS_PENALTY_WEIGHT,
    CANNY_HIGH,
    CANNY_LOW,
    DEFAULT_TOP_K,
    ENTROPY_MIN,
    ENTROPY_PENALTY_WEIGHT,
    GRID_SCORE_MIN,
    HOUGH_MAX_LINE_GAP,
    HOUGH_MIN_LINE_LENGTH,
    HOUGH_THRESHOLD,
)
from pipeline_utils import dump_json, ensure_dir, format_seconds, load_json, relative_to_cwd, utc_now_iso


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select best frame for each video")
    parser.add_argument("--frames-dir", default="zip_archive/frames")
    parser.add_argument("--metadata-in", default="zip_archive/metadata/frames_manifest.json")
    parser.add_argument("--metadata-out", default="zip_archive/metadata/selected_frames.json")
    parser.add_argument("--candidates-dir", default="zip_archive/candidates")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--grid-score-min", type=float, default=GRID_SCORE_MIN)
    parser.add_argument(
        "--head-seconds",
        type=float,
        default=5.0,
        help="Only evaluate frames from the first N seconds of each video",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def parse_frame_index(frame_path: Path) -> int:
    stem = frame_path.stem
    number = stem.split("_")[-1]
    try:
        return int(number)
    except ValueError:
        return 0


def image_entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    total = float(hist.sum())
    if total <= 0:
        return 0.0
    probs = hist / total
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def blockiness(gray: np.ndarray, block: int = 8) -> float:
    gray_f = gray.astype(np.float32)
    if gray_f.shape[0] < block + 1 or gray_f.shape[1] < block + 1:
        return 0.0

    diff_h = np.abs(np.diff(gray_f, axis=1))
    diff_v = np.abs(np.diff(gray_f, axis=0))

    boundary_cols = np.arange(block - 1, diff_h.shape[1], block)
    boundary_rows = np.arange(block - 1, diff_v.shape[0], block)

    if boundary_cols.size == 0 or boundary_rows.size == 0:
        return 0.0

    boundary_mean = (
        float(diff_h[:, boundary_cols].mean()) + float(diff_v[boundary_rows, :].mean())
    ) / 2.0

    non_boundary_h = np.delete(diff_h, boundary_cols, axis=1)
    non_boundary_v = np.delete(diff_v, boundary_rows, axis=0)

    non_boundary_mean = (
        float(non_boundary_h.mean()) if non_boundary_h.size else 0.0
    ) + (float(non_boundary_v.mean()) if non_boundary_v.size else 0.0)
    non_boundary_mean /= 2.0

    return max(0.0, boundary_mean - non_boundary_mean)


def detect_grid_lines(gray: np.ndarray) -> tuple[int, int]:
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP,
    )

    if lines is None:
        return 0, 0

    vertical = 0
    horizontal = 0

    for line in lines[:, 0]:
        x1, y1, x2, y2 = line
        angle = abs(math.degrees(math.atan2((y2 - y1), (x2 - x1))))
        angle = min(angle, 180.0 - angle)

        if angle <= 12:
            horizontal += 1
        elif angle >= 78:
            vertical += 1

    return vertical, horizontal


def score_frame(frame_path: Path) -> dict | None:
    image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    entropy = image_entropy(gray)
    block = blockiness(gray)
    vertical_count, horizontal_count = detect_grid_lines(gray)

    # Grid-likeness core signal: many vertical/horizontal lines with balanced counts.
    line_score = float(vertical_count + horizontal_count)
    balance_bonus = BALANCE_BONUS_WEIGHT * float(min(vertical_count, horizontal_count))

    # Penalize blurry and heavily compressed frames to avoid unstable puzzle captures.
    blur_penalty = max(0.0, (BLUR_THRESHOLD - lap_var) / BLUR_THRESHOLD) * BLUR_PENALTY_WEIGHT
    entropy_penalty = max(0.0, ENTROPY_MIN - entropy) * ENTROPY_PENALTY_WEIGHT
    compression_penalty = block * BLOCKINESS_PENALTY_WEIGHT

    total_score = line_score + balance_bonus - blur_penalty - entropy_penalty - compression_penalty

    return {
        "frame_path": str(frame_path),
        "score": round(float(total_score), 5),
        "line_score": round(float(line_score), 5),
        "vertical_lines": vertical_count,
        "horizontal_lines": horizontal_count,
        "laplacian_var": round(lap_var, 5),
        "entropy": round(entropy, 5),
        "blockiness": round(block, 5),
        "penalties": {
            "blur": round(float(blur_penalty), 5),
            "entropy": round(float(entropy_penalty), 5),
            "compression": round(float(compression_penalty), 5),
        },
    }


def remove_previous_candidates(candidate_dir: Path) -> None:
    if not candidate_dir.exists():
        return
    for path in candidate_dir.glob("*.png"):
        path.unlink()


def main() -> int:
    args = parse_args()

    if args.top_k <= 0:
        raise SystemExit("--top-k must be > 0")
    if args.head_seconds < 0:
        raise SystemExit("--head-seconds must be >= 0")

    frames_dir = Path(args.frames_dir)
    metadata_in = Path(args.metadata_in)
    metadata_out = Path(args.metadata_out)
    candidates_dir = Path(args.candidates_dir)

    ensure_dir(frames_dir)
    ensure_dir(candidates_dir)
    ensure_dir(metadata_out.parent)

    manifest = load_json(metadata_in, default={})
    videos = [entry for entry in manifest.get("videos", []) if entry.get("status") == "ok"]
    if args.limit is not None:
        videos = videos[: args.limit]

    previous = load_json(metadata_out, default={})
    previous_by_basename = {
        entry.get("video_basename"): entry
        for entry in previous.get("videos", [])
        if entry.get("video_basename")
    }

    output_entries: list[dict] = []

    for idx, video_entry in enumerate(videos, start=1):
        basename = video_entry["video_basename"]
        frame_subdir = Path(video_entry["frames_dir"])
        if not frame_subdir.is_absolute():
            frame_subdir = Path.cwd() / frame_subdir

        all_frames = sorted(frame_subdir.glob("frame_*.png"))
        fps_for_video = float(video_entry.get("fps") or 1.0)
        max_frame_index = int(args.head_seconds * max(0.001, fps_for_video)) + 1
        if args.head_seconds == 0:
            max_frame_index = 1
        frames = [
            frame for frame in all_frames if parse_frame_index(frame) <= max_frame_index
        ]
        if not frames:
            frames = all_frames[: max(1, min(5, len(all_frames)))]
        if not frames:
            output_entries.append(
                {
                    "video_basename": basename,
                    "status": "needs_review",
                    "error": "no frames found",
                }
            )
            print(f"[{idx}/{len(videos)}] {basename}: no frames found")
            continue

        previous_entry = previous_by_basename.get(basename)
        chosen_from_prev = previous_entry.get("chosen_frame") if previous_entry else None
        if (
            previous_entry
            and chosen_from_prev
            and Path(chosen_from_prev).exists()
            and not args.force
        ):
            output_entries.append(previous_entry)
            print(f"[{idx}/{len(videos)}] {basename}: skip (already selected)")
            continue

        print(f"[{idx}/{len(videos)}] scoring {basename} ({len(frames)} frames)")
        scored: list[dict] = []
        for frame in frames:
            metrics = score_frame(frame)
            if metrics is not None:
                scored.append(metrics)

        if not scored:
            output_entries.append(
                {
                    "video_basename": basename,
                    "status": "needs_review",
                    "error": "unable to score frames",
                }
            )
            print(f"    -> scoring failed")
            continue

        scored.sort(key=lambda item: item["score"], reverse=True)
        best = scored[0]
        top_candidates = scored[: args.top_k]

        candidate_subdir = candidates_dir / basename
        ensure_dir(candidate_subdir)
        if args.force:
            remove_previous_candidates(candidate_subdir)

        chosen_path = candidate_subdir / "chosen_frame.png"
        shutil.copy2(best["frame_path"], chosen_path)

        copied_candidates = []
        for rank, candidate in enumerate(top_candidates, start=1):
            original = Path(candidate["frame_path"])
            destination = candidate_subdir / f"candidate_{rank:02d}_{original.name}"
            shutil.copy2(original, destination)
            frame_index = parse_frame_index(original)
            fps = float(video_entry.get("fps") or 1.0)
            start_offset = float(video_entry.get("start_offset") or 0.0)
            timestamp_seconds = start_offset + max(0, frame_index - 1) / max(0.001, fps)
            copied_candidates.append(
                {
                    **candidate,
                    "frame_path": relative_to_cwd(destination),
                    "frame_index": frame_index,
                    "frame_timestamp": format_seconds(timestamp_seconds),
                }
            )

        best_index = parse_frame_index(Path(best["frame_path"]))
        fps = float(video_entry.get("fps") or 1.0)
        start_offset = float(video_entry.get("start_offset") or 0.0)
        best_timestamp_seconds = start_offset + max(0, best_index - 1) / max(0.001, fps)
        best_timestamp = format_seconds(best_timestamp_seconds)

        status = "ok" if float(best["score"]) >= args.grid_score_min else "needs_review"

        output_entry = {
            "video_basename": basename,
            "video_id": video_entry.get("video_id"),
            "title": video_entry.get("title"),
            "playlist_index": video_entry.get("playlist_index"),
            "source_url": video_entry.get("source_url"),
            "upload_date": video_entry.get("upload_date"),
            "raw_video_path": video_entry.get("raw_video_path"),
            "frames_dir": video_entry.get("frames_dir"),
            "chosen_frame": relative_to_cwd(chosen_path),
            "frame_index": best_index,
            "frame_timestamp": best_timestamp,
            "best_score": best["score"],
            "status": status,
            "top_candidates": copied_candidates,
            "score_threshold": args.grid_score_min,
            "head_seconds": args.head_seconds,
        }

        output_entries.append(output_entry)

        if args.verbose:
            print(f"    -> best_score={best['score']:.3f} status={status}")

    payload = {
        "generated_at": utc_now_iso(),
        "settings": {
            "top_k": args.top_k,
            "grid_score_min": args.grid_score_min,
            "head_seconds": args.head_seconds,
            "limit": args.limit,
        },
        "videos": sorted(
            output_entries,
            key=lambda entry: (
                entry.get("playlist_index") is None,
                entry.get("playlist_index") or 10**9,
                entry.get("video_basename", ""),
            ),
        ),
    }

    dump_json(metadata_out, payload)
    print(f"Wrote selection metadata: {metadata_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
