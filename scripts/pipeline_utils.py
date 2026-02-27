"""Shared helpers for the Zip extraction pipeline scripts."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


@dataclass
class VideoAsset:
    basename: str
    video_path: Path
    info_path: Path | None
    video_id: str | None
    title: str | None
    playlist_index: int | None
    source_url: str | None
    upload_date: str | None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def run_command(
    cmd: list[str],
    *,
    verbose: bool = False,
    check: bool = True,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    if verbose:
        print("$", " ".join(cmd))
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        capture_output=capture_output,
    )


def format_seconds(total_seconds: float) -> str:
    total_seconds = max(0.0, float(total_seconds))
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def _read_video_info(info_path: Path) -> dict[str, Any]:
    if not info_path.exists():
        return {}
    with info_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def discover_videos(raw_videos_dir: Path, limit: int | None = None) -> list[VideoAsset]:
    videos = sorted(raw_videos_dir.glob("*.mp4"))
    assets: list[VideoAsset] = []

    for video_path in videos:
        basename = video_path.stem
        info_path = video_path.with_suffix(".info.json")
        info = _read_video_info(info_path) if info_path.exists() else {}

        playlist_index = _to_int(info.get("playlist_index"))
        if playlist_index is None:
            prefix = basename.split("_", 1)[0]
            playlist_index = _to_int(prefix)

        asset = VideoAsset(
            basename=basename,
            video_path=video_path,
            info_path=info_path if info_path.exists() else None,
            video_id=info.get("id"),
            title=info.get("title"),
            playlist_index=playlist_index,
            source_url=(
                info.get("webpage_url")
                or info.get("original_url")
                or info.get("url")
                or info.get("uploader_url")
            ),
            upload_date=info.get("upload_date"),
        )
        assets.append(asset)

    assets.sort(key=lambda item: (item.playlist_index is None, item.playlist_index or 10**9, item.basename))

    if limit is not None:
        return assets[:limit]
    return assets


def relative_to_cwd(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


_MONTHS = {
    "jan": 1,
    "january": 1,
    "janv": 1,
    "janvier": 1,
    "feb": 2,
    "february": 2,
    "fev": 2,
    "fevr": 2,
    "fevrier": 2,
    "mar": 3,
    "march": 3,
    "mars": 3,
    "apr": 4,
    "april": 4,
    "avr": 4,
    "avril": 4,
    "may": 5,
    "mai": 5,
    "jun": 6,
    "june": 6,
    "juin": 6,
    "jul": 7,
    "july": 7,
    "juillet": 7,
    "aug": 8,
    "august": 8,
    "aout": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "septembre": 9,
    "oct": 10,
    "october": 10,
    "octobre": 10,
    "nov": 11,
    "november": 11,
    "novembre": 11,
    "dec": 12,
    "december": 12,
    "decembre": 12,
}


def _safe_date(year: int, month: int, day: int) -> str | None:
    try:
        return datetime(year=year, month=month, day=day).date().isoformat()
    except ValueError:
        return None


def extract_puzzle_date_from_title(title: str | None) -> str | None:
    if not title:
        return None

    clean = title.strip()
    if not clean:
        return None
    lower = clean.lower()

    patterns = [
        r"(20\d{2})[-_/\.](0?[1-9]|1[0-2])[-_/\.](0?[1-9]|[12]\d|3[01])\s*$",
        r"(0?[1-9]|[12]\d|3[01])[-_/\.](0?[1-9]|1[0-2])[-_/\.](20\d{2})\s*$",
    ]

    for idx, pattern in enumerate(patterns):
        match = re.search(pattern, lower)
        if not match:
            continue
        if idx == 0:
            year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
        else:
            day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
        value = _safe_date(year, month, day)
        if value:
            return value

    month_keys = sorted(_MONTHS.keys(), key=len, reverse=True)
    month_group = "|".join(re.escape(key) for key in month_keys)
    month_name_patterns = [
        rf"({month_group})\.?\s+(0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?[,]?\s+(20\d{{2}})\s*$",
        rf"(0?[1-9]|[12]\d|3[01])\s+({month_group})\.?[,]?\s+(20\d{{2}})\s*$",
    ]

    for idx, pattern in enumerate(month_name_patterns):
        match = re.search(pattern, lower)
        if not match:
            continue
        if idx == 0:
            month_name, day, year = match.group(1), int(match.group(2)), int(match.group(3))
        else:
            day, month_name, year = int(match.group(1)), match.group(2), int(match.group(3))
        month = _MONTHS.get(month_name.strip("."))
        if month is None:
            continue
        value = _safe_date(year, month, day)
        if value:
            return value

    return None


def parse_upload_date(upload_date: str | None) -> str | None:
    if not upload_date:
        return None
    value = str(upload_date).strip()
    if len(value) != 8 or not value.isdigit():
        return None
    year = int(value[0:4])
    month = int(value[4:6])
    day = int(value[6:8])
    return _safe_date(year, month, day)


def extract_puzzle_number_from_title(title: str | None) -> int | None:
    if not title:
        return None
    text = title.strip()
    if not text:
        return None

    patterns = [
        r"puzzle\s*#\s*(\d+)",
        r"zip\s*#\s*(\d+)",
        r"#\s*(\d+)",
    ]
    lower = text.lower()
    for pattern in patterns:
        match = re.search(pattern, lower)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                continue
    return None


def infer_decreasing_dates_by_index(entries: list[dict]) -> None:
    indexed: list[tuple[int, int, dict]] = []
    for pos, entry in enumerate(entries):
        idx = entry.get("playlist_index")
        if isinstance(idx, int):
            indexed.append((idx, pos, entry))
    indexed.sort(key=lambda item: item[0])

    known: list[tuple[int, datetime]] = []
    for idx, _, entry in indexed:
        value = entry.get("puzzle_date")
        if not value:
            continue
        try:
            known.append((idx, datetime.strptime(value, "%Y-%m-%d")))
        except ValueError:
            continue

    if not known:
        return

    known.sort(key=lambda item: item[0])

    for idx, _, entry in indexed:
        if entry.get("puzzle_date"):
            continue

        left = None
        right = None
        for anchor_idx, anchor_date in known:
            if anchor_idx < idx:
                left = (anchor_idx, anchor_date)
            elif anchor_idx > idx:
                right = (anchor_idx, anchor_date)
                break

        inferred = None
        if left is not None:
            inferred = left[1] - timedelta(days=(idx - left[0]))
        elif right is not None:
            inferred = right[1] + timedelta(days=(right[0] - idx))

        if inferred is not None:
            entry["puzzle_date"] = inferred.date().isoformat()
