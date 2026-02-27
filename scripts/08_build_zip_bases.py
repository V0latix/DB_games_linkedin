#!/usr/bin/env python3
"""Build `data/zip.json` and `data/zip_solution.json`.

`zip.json` keeps the same schema as `zip_unique.json` with one extra field:
  - `name`: `Zip #XXX - YYYY-MM-DD`

`zip_solution.json` adds:
  - `solution`: canonical solved path
  - `solution_solver`: solver used for the canonical solution
  - `solver_runs`: per-solver status/metrics
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from linkedin_game_solver.games.zip.parser import parse_puzzle_dict  # noqa: E402
from linkedin_game_solver.games.zip.solvers import get_solver  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build zip and zip_solution datasets")
    parser.add_argument("--zip-unique", default="data/zip_unique.json")
    parser.add_argument("--index-json", default="zip_archive/metadata/index.json")
    parser.add_argument("--manifest-json", default="zip_archive/metadata/puzzles_zip_manifest.json")
    parser.add_argument("--out-zip", default="data/zip.json")
    parser.add_argument("--out-zip-solution", default="data/zip_solution.json")
    parser.add_argument("--out-zip-archive", default="data/zip_bundle.zip")
    parser.add_argument("--out-zip-solution-archive", default="data/zip_solution_bundle.zip")
    parser.add_argument(
        "--solvers",
        default="articulation,forced,heuristic,heuristic_nolcv,baseline",
        help="Comma-separated list",
    )
    parser.add_argument("--time-limit-s", type=float, default=1.5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def is_valid_zip_payload(payload: dict[str, Any]) -> bool:
    if payload.get("game") != "zip":
        return False
    if not isinstance(payload.get("n"), int):
        return False
    if not isinstance(payload.get("numbers"), list):
        return False
    if not isinstance(payload.get("walls"), list):
        return False
    return True


_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


def extract_number_from_title(title: str | None) -> int | None:
    if not title:
        return None
    match = re.search(r"(?:puzzle|zip)\s*#?\s*(\d+)", title, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"#\s*(\d+)", title)
    if match:
        return int(match.group(1))
    return None


def _safe_date(year: int, month: int, day: int) -> str | None:
    try:
        return datetime(year=year, month=month, day=day).date().isoformat()
    except ValueError:
        return None


def extract_date_from_title(title: str | None) -> str | None:
    if not title:
        return None
    text = title.strip()
    if not text:
        return None
    low = text.lower()

    ymd = re.search(r"(20\d{2})[-_/\.](0?[1-9]|1[0-2])[-_/\.](0?[1-9]|[12]\d|3[01])", low)
    if ymd:
        return _safe_date(int(ymd.group(1)), int(ymd.group(2)), int(ymd.group(3)))

    dmy = re.search(r"(0?[1-9]|[12]\d|3[01])[-_/\.](0?[1-9]|1[0-2])[-_/\.](20\d{2})", low)
    if dmy:
        return _safe_date(int(dmy.group(3)), int(dmy.group(2)), int(dmy.group(1)))

    month_names = "|".join(sorted(_MONTHS.keys(), key=len, reverse=True))
    md1 = re.search(
        rf"({month_names})\.?\s+(0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?[,]?\s+(20\d{{2}})",
        low,
    )
    if md1:
        month = _MONTHS[md1.group(1).strip(".")]
        return _safe_date(int(md1.group(3)), month, int(md1.group(2)))

    md2 = re.search(rf"(0?[1-9]|[12]\d|3[01])\s+({month_names})\.?[,]?\s+(20\d{{2}})", low)
    if md2:
        month = _MONTHS[md2.group(2).strip(".")]
        return _safe_date(int(md2.group(3)), month, int(md2.group(1)))

    return None


def parse_upload_date(value: str | None) -> str | None:
    if not value:
        return None
    value = str(value).strip()
    if len(value) != 8 or not value.isdigit():
        return None
    return _safe_date(int(value[0:4]), int(value[4:6]), int(value[6:8]))


def load_labels(index_json: Path, manifest_json: Path, expected_count: int) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []

    if not index_json.exists():
        return labels

    index_payload = load_json(index_json)
    index_entries = index_payload.get("entries", []) if isinstance(index_payload, dict) else []
    index_by_basename = {
        entry.get("video_basename"): entry for entry in index_entries if entry.get("video_basename")
    }

    if manifest_json.exists():
        manifest_payload = load_json(manifest_json)
        manifest_entries = (
            manifest_payload.get("entries", []) if isinstance(manifest_payload, dict) else []
        )
        selected: list[dict[str, Any]] = []
        for entry in manifest_entries:
            json_path_value = entry.get("json_path")
            if not json_path_value:
                continue
            json_path = Path(json_path_value)
            if not json_path.is_absolute():
                json_path = Path.cwd() / json_path
            if not json_path.exists():
                continue
            payload = load_json(json_path)
            if not is_valid_zip_payload(payload):
                continue
            selected.append(entry)

        selected.sort(
            key=lambda item: (
                item.get("playlist_index") is None,
                item.get("playlist_index") if item.get("playlist_index") is not None else 10**9,
                str(item.get("json_path") or ""),
            )
        )
        if selected:
            for selected_entry in selected:
                basename = selected_entry.get("video_basename")
                idx_entry = index_by_basename.get(basename, {})
                title = idx_entry.get("title") or selected_entry.get("title")
                number = idx_entry.get("puzzle_number")
                if not isinstance(number, int):
                    number = extract_number_from_title(title)
                date = idx_entry.get("puzzle_date")
                if not isinstance(date, str):
                    date = extract_date_from_title(title) or parse_upload_date(idx_entry.get("upload_date"))
                labels.append(
                    {
                        "video_basename": basename,
                        "playlist_index": idx_entry.get("playlist_index")
                        if idx_entry
                        else selected_entry.get("playlist_index"),
                        "title": title,
                        "puzzle_number": number,
                        "puzzle_date": date,
                    }
                )

    if not labels:
        fallback = sorted(
            index_entries,
            key=lambda item: (
                item.get("playlist_index") is None,
                item.get("playlist_index") if item.get("playlist_index") is not None else 10**9,
                item.get("video_basename") or "",
            ),
        )
        for entry in fallback:
            title = entry.get("title")
            number = entry.get("puzzle_number")
            if not isinstance(number, int):
                number = extract_number_from_title(title)
            date = entry.get("puzzle_date")
            if not isinstance(date, str):
                date = extract_date_from_title(title) or parse_upload_date(entry.get("upload_date"))
            labels.append(
                {
                    "video_basename": entry.get("video_basename"),
                    "playlist_index": entry.get("playlist_index"),
                    "title": title,
                    "puzzle_number": number,
                    "puzzle_date": date,
                }
            )

    if len(labels) > expected_count:
        labels = labels[:expected_count]

    while len(labels) < expected_count:
        labels.append({})

    return labels


def puzzle_name(index: int, meta: dict[str, Any]) -> str:
    number = meta.get("puzzle_number")
    date = meta.get("puzzle_date")
    if isinstance(number, int) and isinstance(date, str) and date:
        return f"Zip #{number:03d} - {date}"
    if isinstance(number, int):
        return f"Zip #{number:03d}"
    if isinstance(date, str) and date:
        return f"Zip - {date}"
    return f"Zip #{index:03d}"


def as_cell_path(path: list[tuple[int, int]]) -> list[list[int]]:
    return [[int(r), int(c)] for r, c in path]


def run_solvers(
    puzzle_payload: dict[str, Any],
    solver_names: list[str],
    time_limit_s: float,
) -> tuple[dict[str, Any] | None, str | None, dict[str, Any]]:
    try:
        puzzle = parse_puzzle_dict(
            {
                "game": "zip",
                "n": puzzle_payload["n"],
                "numbers": puzzle_payload["numbers"],
                "walls": puzzle_payload["walls"],
            }
        )
    except Exception as exc:
        parse_error = f"parse_error: {exc}"
        runs = {
            solver_name: {
                "solved": False,
                "time_ms": 0.0,
                "nodes": 0,
                "backtracks": 0,
                "error": parse_error,
            }
            for solver_name in solver_names
        }
        return None, None, runs

    best_solution: dict[str, Any] | None = None
    best_solver: str | None = None
    best_time = float("inf")
    runs: dict[str, Any] = {}

    for solver_name in solver_names:
        solver = get_solver(solver_name)
        result = solver(puzzle, time_limit_s=time_limit_s)
        solved = bool(result.solved and result.solution is not None)
        path = as_cell_path(result.solution.path) if solved else None
        run_info = {
            "solved": solved,
            "time_ms": round(float(result.metrics.time_ms), 4),
            "nodes": int(result.metrics.nodes),
            "backtracks": int(result.metrics.backtracks),
            "error": result.error,
        }
        runs[solver_name] = run_info

        if solved and path is not None:
            if run_info["time_ms"] < best_time:
                best_time = run_info["time_ms"]
                best_solver = solver_name
                best_solution = {"path": path}

    return best_solution, best_solver, runs


def write_zip_bundle(zip_path: Path, json_path: Path, arcname: str) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(json_path, arcname=arcname)


def main() -> int:
    args = parse_args()
    zip_unique_path = Path(args.zip_unique)
    index_json = Path(args.index_json)
    manifest_json = Path(args.manifest_json)
    out_zip = Path(args.out_zip)
    out_zip_solution = Path(args.out_zip_solution)
    out_zip_archive = Path(args.out_zip_archive)
    out_zip_solution_archive = Path(args.out_zip_solution_archive)

    if not zip_unique_path.exists():
        raise SystemExit(f"Missing file: {zip_unique_path}")

    base_payload = load_json(zip_unique_path)
    base_puzzles = base_payload.get("puzzles", [])
    if not isinstance(base_puzzles, list):
        raise SystemExit(f"Invalid format in {zip_unique_path}: expected puzzles[]")

    if args.limit is not None:
        base_puzzles = base_puzzles[: args.limit]

    labels = load_labels(index_json=index_json, manifest_json=manifest_json, expected_count=len(base_puzzles))
    solver_names = [name.strip() for name in args.solvers.split(",") if name.strip()]
    if not solver_names:
        raise SystemExit("No solvers configured")

    zip_puzzles: list[dict[str, Any]] = []
    zip_solution_puzzles: list[dict[str, Any]] = []

    for idx, puzzle in enumerate(base_puzzles, start=1):
        meta = labels[idx - 1] if idx - 1 < len(labels) else {}
        name = puzzle_name(idx, meta)

        zip_item = {
            "id": int(puzzle["id"]),
            "source": puzzle["source"],
            "n": int(puzzle["n"]),
            "numbers": puzzle["numbers"],
            "walls": puzzle["walls"],
            "name": name,
        }
        zip_puzzles.append(zip_item)

        best_solution, best_solver, solver_runs = run_solvers(
            puzzle_payload=zip_item,
            solver_names=solver_names,
            time_limit_s=args.time_limit_s,
        )

        zip_solution_item = dict(zip_item)
        zip_solution_item["solution"] = best_solution["path"] if best_solution else None
        zip_solution_item["solution_solver"] = best_solver
        zip_solution_item["solver_runs"] = solver_runs
        zip_solution_puzzles.append(zip_solution_item)

        if args.verbose and idx % 20 == 0:
            solved = sum(1 for p in zip_solution_puzzles if p.get("solution") is not None)
            print(f"[{idx}/{len(base_puzzles)}] solved={solved}")

    zip_payload = {
        "game": "zip",
        "version": int(base_payload.get("version", 1)),
        "puzzles": zip_puzzles,
    }
    zip_solution_payload = {
        "game": "zip",
        "version": int(base_payload.get("version", 1)),
        "puzzles": zip_solution_puzzles,
    }

    dump_json(out_zip, zip_payload)
    dump_json(out_zip_solution, zip_solution_payload)
    write_zip_bundle(out_zip_archive, out_zip, arcname="data/zip.json")
    write_zip_bundle(out_zip_solution_archive, out_zip_solution, arcname="data/zip_solution.json")

    solved_count = sum(1 for puzzle in zip_solution_puzzles if puzzle.get("solution") is not None)
    print(f"Wrote {out_zip} ({len(zip_puzzles)} puzzles)")
    print(f"Wrote {out_zip_solution} ({len(zip_solution_puzzles)} puzzles, solved={solved_count})")
    print(f"Wrote {out_zip_archive}")
    print(f"Wrote {out_zip_solution_archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
