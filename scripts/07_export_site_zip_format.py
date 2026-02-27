#!/usr/bin/env python3
"""Export extracted Zip puzzles into the exact site dataset format.

Output schema:
{
  "game": "zip",
  "version": 1,
  "puzzles": [
    {"id": 1, "source": "...", "n": 6, "numbers": [...], "walls": [...]}
  ]
}
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from pipeline_utils import dump_json, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export site-compatible Zip dataset JSON")
    parser.add_argument("--manifest", default="zip_archive/metadata/puzzles_zip_manifest.json")
    parser.add_argument("--out-json", default="zip_archive/metadata/zip_site_format.json")
    parser.add_argument("--out-zip", default="zip_archive/metadata/zip_site_format_bundle.zip")
    parser.add_argument("--source", default="youtube_extracted")
    parser.add_argument("--include-needs-review", action="store_true")
    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def is_valid_puzzle_payload(payload: dict) -> bool:
    if payload.get("game") != "zip":
        return False
    if not isinstance(payload.get("n"), int):
        return False
    if not isinstance(payload.get("numbers"), list):
        return False
    if not isinstance(payload.get("walls"), list):
        return False
    return True


def main() -> int:
    args = parse_args()

    manifest_path = Path(args.manifest)
    out_json_path = Path(args.out_json)
    out_zip_path = Path(args.out_zip)

    manifest = load_json(manifest_path, default={})
    entries = manifest.get("entries", [])

    selected: list[dict] = []
    for entry in entries:
        status = entry.get("status")
        if status != "ok" and not args.include_needs_review:
            continue

        confidence = entry.get("recognition_confidence")
        if args.min_confidence is not None and isinstance(confidence, (float, int)):
            if float(confidence) < args.min_confidence:
                continue

        json_path_value = entry.get("json_path")
        if not json_path_value:
            continue

        json_path = Path(json_path_value)
        if not json_path.exists():
            continue

        payload = load_json(json_path, default={})
        if not is_valid_puzzle_payload(payload):
            continue

        selected.append(
            {
                "playlist_index": entry.get("playlist_index"),
                "json_path": json_path,
                "payload": payload,
            }
        )

    selected.sort(
        key=lambda item: (
            item.get("playlist_index") is None,
            item.get("playlist_index") if item.get("playlist_index") is not None else 10**9,
            str(item.get("json_path")),
        )
    )

    puzzles: list[dict] = []
    for idx, item in enumerate(selected, start=1):
        payload = item["payload"]
        puzzles.append(
            {
                "id": idx,
                "source": args.source,
                "n": int(payload["n"]),
                "numbers": payload["numbers"],
                "walls": payload["walls"],
            }
        )

    out_payload = {"game": "zip", "version": 1, "puzzles": puzzles}
    dump_json(out_json_path, out_payload)

    with zipfile.ZipFile(out_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(out_json_path, arcname="data/zip_unique.json")

    if args.verbose:
        print(f"Manifest entries: {len(entries)}")
        print(f"Selected puzzles: {len(puzzles)}")

    print(f"Wrote site-format JSON: {out_json_path}")
    print(f"Wrote site-format ZIP: {out_zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
