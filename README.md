# DB_games_linkedin

Base finale des puzzles Zip extraits depuis des vidéos YouTube LinkedIn Games, avec les scripts pour régénérer la base.

## Contenu

- `data/zip_unique.json`: dataset final au format exact du site (`game`, `version`, `puzzles`).
- `scripts/`: pipeline complet YouTube -> frames -> grilles -> JSON site format.
- `requirements.txt`: dépendances Python minimales pour la partie vision.

## Pipeline

Ordre des scripts:

1. `scripts/00_setup_check.py`
2. `scripts/01_download_playlist.sh`
3. `scripts/02_extract_frames.py`
4. `scripts/03_pick_best_frames.py`
5. `scripts/04_crop_deskew_grid.py`
6. `scripts/05_export_archive.py`
7. `scripts/06_grids_to_zip_puzzles.py`
8. `scripts/07_export_site_zip_format.py`

## Re-générer la base finale

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/00_setup_check.py
export PLAYLIST_URL="https://www.youtube.com/playlist?list=..."

bash scripts/01_download_playlist.sh --playlist-url "$PLAYLIST_URL"
python scripts/02_extract_frames.py --playlist-url "$PLAYLIST_URL" --fps 0.5
python scripts/03_pick_best_frames.py --head-seconds 5
python scripts/04_crop_deskew_grid.py
python scripts/05_export_archive.py
python scripts/06_grids_to_zip_puzzles.py
python scripts/07_export_site_zip_format.py
```

Sortie finale site format:

- `zip_archive/metadata/zip_site_format.json`
- `zip_archive/metadata/zip_site_format_bundle.zip`

## Notes

- Source: extraction vidéo YouTube (pas de scraping LinkedIn).
- Les puzzles issus de vision sont fournis avec un statut qualité (`ok` / `needs_review`) dans les manifests intermédiaires.
