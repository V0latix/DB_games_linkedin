#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Download a YouTube playlist into zip_archive/raw_videos.

Usage:
  scripts/01_download_playlist.sh --playlist-url URL [--limit N] [--max-height 720] [--force]

Options:
  --playlist-url URL   YouTube playlist URL (or export PLAYLIST_URL first)
  --limit N            Download only first N videos from playlist
  --max-height H       Max video height (default: 720)
  --force              Overwrite existing outputs
  -h, --help           Show this help
EOF
}

PLAYLIST_URL="${PLAYLIST_URL:-}"
LIMIT=""
MAX_HEIGHT="720"
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --playlist-url)
      PLAYLIST_URL="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --max-height)
      MAX_HEIGHT="$2"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [[ -z "$PLAYLIST_URL" ]]; then
        PLAYLIST_URL="$1"
        shift
      else
        echo "Unknown argument: $1" >&2
        usage
        exit 1
      fi
      ;;
  esac
done

if [[ -z "$PLAYLIST_URL" ]]; then
  echo "Missing playlist URL. Use --playlist-url or export PLAYLIST_URL." >&2
  usage
  exit 1
fi

RAW_DIR="zip_archive/raw_videos"
META_DIR="zip_archive/metadata"
ARCHIVE_FILE="${META_DIR}/downloaded_ids.txt"
mkdir -p "$RAW_DIR" "$META_DIR"

OUTPUT_TEMPLATE="${RAW_DIR}/%(playlist_index)03d_%(id)s_%(title).120B.%(ext)s"
FORMAT_EXPR="bv*[height<=${MAX_HEIGHT}][ext=mp4]+ba[ext=m4a]/b[height<=${MAX_HEIGHT}][ext=mp4]/b[ext=mp4]/b"

YT_CMD=(
  yt-dlp
  --yes-playlist
  --ignore-errors
  --no-abort-on-error
  --newline
  --restrict-filenames
  --merge-output-format mp4
  --remux-video mp4
  --write-info-json
  --format "$FORMAT_EXPR"
  --output "$OUTPUT_TEMPLATE"
)

if [[ "$FORCE" -eq 1 ]]; then
  YT_CMD+=(--force-overwrites)
else
  YT_CMD+=(--download-archive "$ARCHIVE_FILE")
fi

if [[ -n "$LIMIT" ]]; then
  YT_CMD+=(--playlist-items "1-${LIMIT}")
fi

YT_CMD+=("$PLAYLIST_URL")

echo "Downloading playlist: $PLAYLIST_URL"
echo "Output: $RAW_DIR"
"${YT_CMD[@]}"

echo "Done. Raw videos are in $RAW_DIR"
