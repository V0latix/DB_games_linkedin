#!/usr/bin/env python3
"""Export final archive metadata and optional review gallery/bundle."""

from __future__ import annotations

import argparse
import html
import os
import zipfile
from pathlib import Path

from pipeline_utils import (
    dump_json,
    ensure_dir,
    extract_puzzle_number_from_title,
    extract_puzzle_date_from_title,
    infer_decreasing_dates_by_index,
    load_json,
    parse_upload_date,
    relative_to_cwd,
    utc_now_iso,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Zip archive index and optional review files")
    parser.add_argument("--frames-metadata", default="zip_archive/metadata/frames_manifest.json")
    parser.add_argument("--selection-metadata", default="zip_archive/metadata/selected_frames.json")
    parser.add_argument("--grid-metadata", default="zip_archive/metadata/grid_results.json")
    parser.add_argument("--index-out", default="zip_archive/metadata/index.json")
    parser.add_argument("--gallery-out", default="zip_archive/metadata/gallery.html")
    parser.add_argument("--bundle-out", default="zip_archive/metadata/zip_archive_bundle.zip")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-gallery", action="store_true")
    parser.add_argument("--skip-bundle", action="store_true")
    parser.add_argument("--include-raw-in-bundle", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def map_by_basename(items: list[dict]) -> dict[str, dict]:
    output: dict[str, dict] = {}
    for item in items:
        basename = item.get("video_basename")
        if basename:
            output[basename] = item
    return output


def resolve_existing(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def derive_status(selection_status: str | None, grid_status: str | None) -> str:
    if selection_status == "ok" and grid_status == "ok":
        return "ok"
    return "needs_review"


def build_entry(basename: str, frame: dict, selected: dict, grid: dict) -> dict:
    raw_video_path = frame.get("raw_video_path")
    chosen_frame = selected.get("chosen_frame")
    grid_image = grid.get("grid_image")

    title = frame.get("title") or selected.get("title") or grid.get("title")
    upload_date = frame.get("upload_date") or selected.get("upload_date") or grid.get("upload_date")
    puzzle_date = extract_puzzle_date_from_title(title) or parse_upload_date(upload_date)
    puzzle_number = extract_puzzle_number_from_title(title)

    entry = {
        "video_basename": basename,
        "video_id": frame.get("video_id") or selected.get("video_id") or grid.get("video_id"),
        "title": title,
        "puzzle_number": puzzle_number,
        "puzzle_date": puzzle_date,
        "upload_date": upload_date,
        "playlist_index": frame.get("playlist_index")
        if frame.get("playlist_index") is not None
        else selected.get("playlist_index"),
        "source_url": frame.get("source_url") or selected.get("source_url") or grid.get("source_url"),
        "frame_timestamp": selected.get("frame_timestamp"),
        "paths": {
            "raw_video": raw_video_path,
            "chosen_frame": chosen_frame,
            "grid_image": grid_image,
        },
        "status": derive_status(selected.get("status"), grid.get("status")),
        "selection_score": selected.get("best_score"),
        "crop_method": grid.get("method"),
        "notes": [
            note
            for note in [selected.get("error"), grid.get("error")]
            if note
        ],
    }
    return entry


def _gallery_href(output_path: Path, raw_path: str | None) -> str | None:
    if not raw_path:
        return None
    target = resolve_existing(raw_path)
    if target is None:
        return None
    rel = os.path.relpath(target, output_path.parent.resolve())
    return rel


def write_gallery(entries: list[dict], output_path: Path) -> None:
    rows = []
    for entry in entries:
        title = html.escape(str(entry.get("title") or entry.get("video_basename") or "(untitled)"))
        status = html.escape(str(entry.get("status", "unknown")))
        timestamp = html.escape(str(entry.get("frame_timestamp") or ""))
        grid_href = _gallery_href(output_path, entry.get("paths", {}).get("grid_image"))
        chosen_href = _gallery_href(output_path, entry.get("paths", {}).get("chosen_frame"))

        grid_cell = f'<a href="{html.escape(grid_href)}">grid</a>' if grid_href else "-"
        chosen_cell = f'<a href="{html.escape(chosen_href)}">chosen frame</a>' if chosen_href else "-"

        rows.append(
            "<tr>"
            f"<td>{title}</td>"
            f"<td>{status}</td>"
            f"<td>{timestamp}</td>"
            f"<td>{grid_cell}</td>"
            f"<td>{chosen_cell}</td>"
            "</tr>"
        )

    html_doc = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Zip Archive Review</title>
  <style>
    body { font-family: ui-sans-serif, system-ui, -apple-system, sans-serif; margin: 24px; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
    th { background: #f5f5f5; }
    tr:nth-child(even) { background: #fafafa; }
    .ok { color: #0a7f3f; }
    .needs_review { color: #9a6700; }
  </style>
</head>
<body>
  <h1>Zip Archive Review</h1>
  <p>Generated by scripts/05_export_archive.py</p>
  <table>
    <thead>
      <tr>
        <th>Title</th>
        <th>Status</th>
        <th>Frame Timestamp</th>
        <th>Grid</th>
        <th>Chosen Frame</th>
      </tr>
    </thead>
    <tbody>
"""
    html_doc += "\n".join(rows)
    html_doc += """
    </tbody>
  </table>
</body>
</html>
"""

    ensure_dir(output_path.parent)
    output_path.write_text(html_doc, encoding="utf-8")


def infer_missing_puzzle_numbers(entries: list[dict]) -> None:
    by_date: dict[str, list[int]] = {}
    for entry in entries:
        num = entry.get("puzzle_number")
        date = entry.get("puzzle_date")
        if isinstance(num, int) and isinstance(date, str) and date:
            by_date.setdefault(date, []).append(num)

    sorted_entries = sorted(
        entries,
        key=lambda item: (
            item.get("playlist_index") is None,
            item.get("playlist_index") if item.get("playlist_index") is not None else 10**9,
        ),
    )

    for i, entry in enumerate(sorted_entries):
        if isinstance(entry.get("puzzle_number"), int):
            continue

        date = entry.get("puzzle_date")
        if isinstance(date, str) and date in by_date and by_date[date]:
            # Weekly bonus entries often share date with a numbered daily puzzle.
            entry["puzzle_number"] = min(by_date[date])
            continue

        prev_num = None
        next_num = None
        for j in range(i - 1, -1, -1):
            v = sorted_entries[j].get("puzzle_number")
            if isinstance(v, int):
                prev_num = v
                break
        for j in range(i + 1, len(sorted_entries)):
            v = sorted_entries[j].get("puzzle_number")
            if isinstance(v, int):
                next_num = v
                break

        if prev_num is not None and next_num is not None:
            if prev_num == next_num + 1:
                entry["puzzle_number"] = prev_num
            else:
                entry["puzzle_number"] = max(prev_num, next_num)
        elif prev_num is not None:
            entry["puzzle_number"] = prev_num
        elif next_num is not None:
            entry["puzzle_number"] = next_num


def write_bundle(
    entries: list[dict],
    index_path: Path,
    gallery_path: Path | None,
    bundle_path: Path,
    include_raw: bool,
) -> None:
    ensure_dir(bundle_path.parent)

    with zipfile.ZipFile(bundle_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(index_path, arcname="metadata/index.json")

        if gallery_path and gallery_path.exists():
            zf.write(gallery_path, arcname="metadata/gallery.html")

        for entry in entries:
            chosen = resolve_existing(entry.get("paths", {}).get("chosen_frame"))
            grid = resolve_existing(entry.get("paths", {}).get("grid_image"))
            raw = resolve_existing(entry.get("paths", {}).get("raw_video"))
            basename = str(entry.get("video_basename") or "unknown")
            date_part = str(entry.get("puzzle_date") or "date_unknown")
            puzzle_number = entry.get("puzzle_number")
            if isinstance(puzzle_number, int):
                number_part = f"#{puzzle_number:03d}"
            else:
                number_part = "#unknown"
            stem = f"{number_part}_{date_part}__{basename}"

            if chosen and chosen.exists():
                zf.write(chosen, arcname=f"candidates/{stem}__chosen_frame.png")
            if grid and grid.exists():
                zf.write(grid, arcname=f"grids/{stem}__grid.png")
            if include_raw and raw and raw.exists():
                zf.write(raw, arcname=f"raw_videos/{stem}{raw.suffix}")


def main() -> int:
    args = parse_args()

    frames_meta = load_json(Path(args.frames_metadata), default={})
    selected_meta = load_json(Path(args.selection_metadata), default={})
    grid_meta = load_json(Path(args.grid_metadata), default={})

    frames_by = map_by_basename(frames_meta.get("videos", []))
    selected_by = map_by_basename(selected_meta.get("videos", []))
    grid_by = map_by_basename(grid_meta.get("videos", []))

    all_keys = sorted(
        set(frames_by.keys()) | set(selected_by.keys()) | set(grid_by.keys()),
        key=lambda key: (
            (frames_by.get(key, {}).get("playlist_index") is None),
            frames_by.get(key, {}).get("playlist_index")
            if frames_by.get(key, {}).get("playlist_index") is not None
            else 10**9,
            key,
        ),
    )

    if args.limit is not None:
        all_keys = all_keys[: args.limit]

    entries = [
        build_entry(
            key,
            frame=frames_by.get(key, {}),
            selected=selected_by.get(key, {}),
            grid=grid_by.get(key, {}),
        )
        for key in all_keys
    ]
    infer_decreasing_dates_by_index(entries)
    infer_missing_puzzle_numbers(entries)

    index_path = Path(args.index_out)
    ensure_dir(index_path.parent)

    payload = {
        "generated_at": utc_now_iso(),
        "count": len(entries),
        "entries": entries,
    }
    dump_json(index_path, payload)
    print(f"Wrote index: {index_path}")

    gallery_path = Path(args.gallery_out)
    if not args.skip_gallery:
        if gallery_path.exists() and not args.force:
            print(f"Gallery exists, skip: {gallery_path} (use --force to overwrite)")
        else:
            write_gallery(entries, gallery_path)
            print(f"Wrote gallery: {gallery_path}")

    bundle_path = Path(args.bundle_out)
    if not args.skip_bundle:
        if bundle_path.exists() and not args.force:
            print(f"Bundle exists, skip: {bundle_path} (use --force to overwrite)")
        else:
            write_bundle(
                entries,
                index_path=index_path,
                gallery_path=None if args.skip_gallery else gallery_path,
                bundle_path=bundle_path,
                include_raw=args.include_raw_in_bundle,
            )
            print(f"Wrote archive bundle: {bundle_path}")

    if args.verbose:
        ok_count = sum(1 for item in entries if item.get("status") == "ok")
        needs_review_count = len(entries) - ok_count
        print(f"Status summary -> ok={ok_count} needs_review={needs_review_count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
