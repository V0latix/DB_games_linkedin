#!/usr/bin/env python3
"""Extract frames from downloaded playlist videos using ffmpeg."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from config import DEFAULT_FPS
from pipeline_utils import (
    discover_videos,
    dump_json,
    ensure_dir,
    load_json,
    relative_to_cwd,
    run_command,
    utc_now_iso,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames from raw playlist videos")
    parser.add_argument("--playlist-url", default=None, help="Optional source playlist URL")
    parser.add_argument("--raw-videos-dir", default="zip_archive/raw_videos")
    parser.add_argument("--frames-dir", default="zip_archive/frames")
    parser.add_argument("--metadata-path", default="zip_archive/metadata/frames_manifest.json")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="Frame extraction rate")
    parser.add_argument("--start_offset", type=float, default=0.0, help="Seconds to skip at start")
    parser.add_argument("--end_offset", type=float, default=0.0, help="Seconds to skip at end")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Re-extract even if frames exist")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def get_duration_seconds(video_path: Path, verbose: bool = False) -> float:
    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        verbose=verbose,
        check=True,
        capture_output=True,
    )
    return float(result.stdout.strip())


def clear_existing_frames(frame_dir: Path) -> None:
    for frame_path in frame_dir.glob("frame_*.png"):
        frame_path.unlink()


def count_frames(frame_dir: Path) -> int:
    return len(list(frame_dir.glob("frame_*.png")))


def extract_frames_for_video(
    video_path: Path,
    frame_dir: Path,
    fps: float,
    start_offset: float,
    end_offset: float,
    duration: float,
    *,
    verbose: bool,
) -> tuple[bool, str | None]:
    if duration <= 0:
        return False, "duration is zero"

    start = max(0.0, start_offset)
    stop = duration - max(0.0, end_offset)
    if stop <= start:
        return False, f"invalid offsets ({start_offset=} {end_offset=}, duration={duration:.2f})"

    ensure_dir(frame_dir)

    log_level = "info" if verbose else "error"
    output_pattern = frame_dir / "frame_%06d.png"
    ffmpeg_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        log_level,
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{stop:.3f}",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        "-vsync",
        "vfr",
        str(output_pattern),
    ]

    if verbose:
        print("$", " ".join(ffmpeg_cmd))

    try:
        subprocess.run(ffmpeg_cmd, check=True)
        return True, None
    except subprocess.CalledProcessError as exc:
        return False, f"ffmpeg failed with code {exc.returncode}"


def main() -> int:
    args = parse_args()

    if args.fps <= 0:
        raise SystemExit("--fps must be > 0")
    if args.start_offset < 0 or args.end_offset < 0:
        raise SystemExit("--start_offset and --end_offset must be >= 0")

    raw_videos_dir = Path(args.raw_videos_dir)
    frames_dir = Path(args.frames_dir)
    metadata_path = Path(args.metadata_path)

    ensure_dir(raw_videos_dir)
    ensure_dir(frames_dir)
    ensure_dir(metadata_path.parent)

    videos = discover_videos(raw_videos_dir, limit=args.limit)
    if not videos:
        print(f"No .mp4 videos found in {raw_videos_dir}")
        return 0

    previous = load_json(metadata_path, default={})
    prev_by_basename = {
        entry.get("video_basename"): entry for entry in previous.get("videos", []) if "video_basename" in entry
    }

    results: list[dict] = []

    for idx, video in enumerate(videos, start=1):
        frame_subdir = frames_dir / video.basename
        existing_count = count_frames(frame_subdir) if frame_subdir.exists() else 0

        if existing_count > 0 and not args.force:
            print(
                f"[{idx}/{len(videos)}] skip {video.basename} "
                f"({existing_count} frames already present)"
            )
            previous_entry = prev_by_basename.get(video.basename, {})
            merged = {
                **previous_entry,
                "video_basename": video.basename,
                "video_id": video.video_id,
                "title": video.title,
                "playlist_index": video.playlist_index,
                "source_url": video.source_url,
                "upload_date": video.upload_date,
                "raw_video_path": relative_to_cwd(video.video_path),
                "frames_dir": relative_to_cwd(frame_subdir),
                "frame_count": existing_count,
                "status": previous_entry.get("status", "ok"),
                "fps": previous_entry.get("fps", args.fps),
                "start_offset": previous_entry.get("start_offset", args.start_offset),
                "end_offset": previous_entry.get("end_offset", args.end_offset),
            }
            results.append(merged)
            continue

        ensure_dir(frame_subdir)
        if args.force:
            clear_existing_frames(frame_subdir)

        print(f"[{idx}/{len(videos)}] extracting {video.basename}")

        status = "ok"
        error_message = None
        duration = None

        try:
            duration = get_duration_seconds(video.video_path, verbose=args.verbose)
            ok, error_message = extract_frames_for_video(
                video.video_path,
                frame_subdir,
                fps=args.fps,
                start_offset=args.start_offset,
                end_offset=args.end_offset,
                duration=duration,
                verbose=args.verbose,
            )
            if not ok:
                status = "error"
        except Exception as exc:  # pragma: no cover - defensive
            status = "error"
            error_message = str(exc)

        frame_count = count_frames(frame_subdir)
        if frame_count == 0 and status == "ok":
            status = "error"
            error_message = "no frames extracted"

        print(
            f"    -> status={status} frames={frame_count}"
            + (f" error={error_message}" if error_message else "")
        )

        results.append(
            {
                "video_basename": video.basename,
                "video_id": video.video_id,
                "title": video.title,
                "playlist_index": video.playlist_index,
                "source_url": video.source_url,
                "upload_date": video.upload_date,
                "raw_video_path": relative_to_cwd(video.video_path),
                "frames_dir": relative_to_cwd(frame_subdir),
                "duration_seconds": duration,
                "fps": args.fps,
                "start_offset": args.start_offset,
                "end_offset": args.end_offset,
                "frame_count": frame_count,
                "status": status,
                "error": error_message,
            }
        )

    payload = {
        "generated_at": utc_now_iso(),
        "playlist_url": args.playlist_url,
        "settings": {
            "fps": args.fps,
            "start_offset": args.start_offset,
            "end_offset": args.end_offset,
            "limit": args.limit,
        },
        "videos": sorted(
            results,
            key=lambda entry: (
                entry.get("playlist_index") is None,
                entry.get("playlist_index") or 10**9,
                entry.get("video_basename", ""),
            ),
        ),
    }

    dump_json(metadata_path, payload)
    print(f"Wrote manifest: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
