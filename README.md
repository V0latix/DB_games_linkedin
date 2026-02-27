# DB_games_linkedin

Standalone Zip dataset repository built from YouTube video frames (no LinkedIn scraping).

## Final datasets

- `data/zip_unique.json`: site-compatible base (`game`, `version`, `puzzles`).
- `data/zip.json`: same structure as `zip_unique`, plus puzzle naming:
  - `name`: `Zip #XXX - YYYY-MM-DD`
- `data/zip_solution.json`: same as `zip.json`, plus:
  - `solution`: solved path (if found)
  - `solution_solver`: solver chosen for the canonical solution
  - `solver_runs`: per-solver status + metrics
- `data/zip_bundle.zip`: archive containing `data/zip.json`
- `data/zip_solution_bundle.zip`: archive containing `data/zip_solution.json`

## Repository structure

- `scripts/00_setup_check.py` to `scripts/07_export_site_zip_format.py`: YouTube -> frames -> grids -> extracted puzzle JSON pipeline.
- `scripts/08_build_zip_bases.py`: builds `zip.json` and `zip_solution.json`.
- `src/linkedin_game_solver/...`: local Zip solver subset used by script `08`.
- `requirements.txt`: Python dependencies for CV extraction pipeline.

## Setup (macOS/Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install external tools:

```bash
brew install yt-dlp ffmpeg
```

## Full extraction pipeline (from YouTube playlist)

```bash
python3 scripts/00_setup_check.py
export PLAYLIST_URL="https://www.youtube.com/playlist?list=..."

bash scripts/01_download_playlist.sh --playlist-url "$PLAYLIST_URL"
python3 scripts/02_extract_frames.py --playlist-url "$PLAYLIST_URL" --fps 0.5
python3 scripts/03_pick_best_frames.py --head-seconds 5
python3 scripts/04_crop_deskew_grid.py
python3 scripts/05_export_archive.py
python3 scripts/06_grids_to_zip_puzzles.py
python3 scripts/07_export_site_zip_format.py --include-needs-review
```

## Build `zip` and `zip_solution`

If you already have `zip_archive/metadata/index.json` and `zip_archive/metadata/puzzles_zip_manifest.json` in the current workspace:

```bash
python3 scripts/08_build_zip_bases.py
```

If metadata is in another workspace, pass absolute paths:

```bash
python3 scripts/08_build_zip_bases.py \
  --index-json /absolute/path/to/zip_archive/metadata/index.json \
  --manifest-json /absolute/path/to/zip_archive/metadata/puzzles_zip_manifest.json
```

Optional tuning:

- `--time-limit-s 1.5` per solver per puzzle.
- `--solvers articulation,forced,heuristic,heuristic_nolcv,baseline`
- `--limit N` for a fast test run.

## Notes

- Intended usage: personal research / solver work.
- Do not republish extracted frames or complete puzzle images.
