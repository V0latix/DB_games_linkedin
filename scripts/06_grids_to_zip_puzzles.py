#!/usr/bin/env python3
"""Convert extracted Zip grid PNGs into JSON puzzles compatible with the repo Zip format."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from pipeline_utils import dump_json, ensure_dir, load_json, relative_to_cwd, utc_now_iso


@dataclass
class Circle:
    x: float
    y: float
    r: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert grid PNGs to Zip JSON puzzles")
    parser.add_argument("--index", default="zip_archive/metadata/index.json")
    parser.add_argument("--out-dir", default="zip_archive/puzzles_zip")
    parser.add_argument("--manifest", default="zip_archive/metadata/puzzles_zip_manifest.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def cluster_positions(values: list[float], eps: float = 8.0) -> list[float]:
    if not values:
        return []
    values = sorted(values)
    clusters: list[list[float]] = []
    for value in values:
        if not clusters or abs(value - clusters[-1][-1]) > eps:
            clusters.append([value])
        else:
            clusters[-1].append(value)
    return [float(sum(cluster) / len(cluster)) for cluster in clusters]


def detect_grid_lines(gray: np.ndarray) -> tuple[list[float], list[float]]:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 30, 100)

    h, w = gray.shape[:2]
    all_h: list[float] = []
    all_v: list[float] = []

    for threshold in (25, 35, 45, 55):
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=threshold,
            minLineLength=60,
            maxLineGap=4,
        )
        if lines is None:
            continue

        for line in lines[:, 0]:
            x1, y1, x2, y2 = line
            angle = abs(np.degrees(np.arctan2((y2 - y1), (x2 - x1))))
            angle = min(angle, 180.0 - angle)
            if angle < 8:
                all_h.append((y1 + y2) / 2.0)
            elif angle > 82:
                all_v.append((x1 + x2) / 2.0)

    ys = cluster_positions(all_h, eps=7.0)
    xs = cluster_positions(all_v, eps=7.0)

    xs = regularize_line_spacing(xs, axis_limit=w)
    ys = regularize_line_spacing(ys, axis_limit=h)

    return xs, ys


def regularize_line_spacing(lines: list[float], axis_limit: int) -> list[float]:
    if len(lines) < 3:
        return lines

    lines = sorted(lines)
    gaps = [b - a for a, b in zip(lines[:-1], lines[1:]) if 35 <= (b - a) <= 260]
    if not gaps:
        return lines

    step = float(np.median(np.array(gaps, dtype=np.float32)))
    if step <= 10:
        return lines

    # Build a lattice with the dominant step and keep the observed support.
    offset = float(np.median([value % step for value in lines]))
    start = offset
    while start - step >= -0.5 * step:
        start -= step

    lattice: list[float] = []
    value = start
    while value <= axis_limit + 0.5 * step:
        lattice.append(value)
        value += step

    supported = [
        value for value in lattice if any(abs(obs - value) <= 0.35 * step for obs in lines)
    ]
    if len(supported) < 3:
        return lines

    first, last = min(supported), max(supported)
    dense: list[float] = []
    value = first
    while value <= last + 1e-6:
        dense.append(value)
        value += step

    return dense


def detect_circles(gray: np.ndarray) -> list[Circle]:
    # Black circular checkpoints in the puzzle are very dark compared to the board.
    dark = (gray < 55).astype(np.uint8) * 255
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circles: list[Circle] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 700 or area > 18000:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.62:
            continue

        (x, y), r = cv2.minEnclosingCircle(contour)
        if r < 12 or r > 80:
            continue
        circles.append(Circle(x=float(x), y=float(y), r=float(r)))

    circles.sort(key=lambda c: (c.y, c.x))
    return circles


def build_digit_knn() -> cv2.ml_KNearest:
    samples: list[np.ndarray] = []
    labels: list[int] = []

    for digit in range(10):
        text = str(digit)
        for font_scale in (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3):
            for thickness in (1, 2, 3):
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        img = np.zeros((32, 24), dtype=np.uint8)
                        (tw, th), _ = cv2.getTextSize(
                            text,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale,
                            thickness,
                        )
                        x = max(0, (24 - tw) // 2 + dx)
                        y = max(th + 1, min(31, (32 + th) // 2 + dy))
                        cv2.putText(
                            img,
                            text,
                            (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale,
                            255,
                            thickness,
                            cv2.LINE_AA,
                        )

                        for blur in (False, True):
                            variant = cv2.GaussianBlur(img, (3, 3), 0) if blur else img.copy()
                            _, variant = cv2.threshold(variant, 100, 255, cv2.THRESH_BINARY)
                            variant = cv2.resize(variant, (20, 28), interpolation=cv2.INTER_AREA)
                            samples.append(variant.reshape(-1).astype(np.float32) / 255.0)
                            labels.append(digit)

    knn = cv2.ml.KNearest_create()
    knn.train(np.array(samples, np.float32), cv2.ml.ROW_SAMPLE, np.array(labels, np.int32))
    return knn


def text_patch_from_circle(gray: np.ndarray, circle: Circle) -> np.ndarray:
    x0 = max(0, int(circle.x - circle.r * 0.82))
    x1 = min(gray.shape[1], int(circle.x + circle.r * 0.82))
    y0 = max(0, int(circle.y - circle.r * 0.82))
    y1 = min(gray.shape[0], int(circle.y + circle.r * 0.82))
    roi = gray[y0:y1, x0:x1]

    _, text_mask = cv2.threshold(roi, 155, 255, cv2.THRESH_BINARY)

    h, w = text_mask.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    cx, cy = w / 2.0, h / 2.0
    rr = min(h, w) * 0.49
    circle_mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= rr * rr
    text_mask = text_mask * circle_mask.astype(np.uint8)

    points = cv2.findNonZero(text_mask)
    out = np.zeros((80, 80), dtype=np.uint8)
    if points is None:
        return out

    x, y, w2, h2 = cv2.boundingRect(points)
    crop = text_mask[y : y + h2, x : x + w2]

    scale = min(70 / max(1, w2), 70 / max(1, h2))
    nw, nh = max(1, int(w2 * scale)), max(1, int(h2 * scale))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)

    ox, oy = (80 - nw) // 2, (80 - nh) // 2
    out[oy : oy + nh, ox : ox + nw] = resized
    return out


def predict_digits_with_knn(text_patch: np.ndarray, knn: cv2.ml_KNearest) -> tuple[int | None, float]:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(text_patch, connectivity=8)

    boxes: list[tuple[int, int, int, int]] = []
    for idx in range(1, n_labels):
        x, y, w, h, area = stats[idx]
        if area < 20 or h < 8 or w < 3:
            continue
        boxes.append((x, y, w, h))

    if not boxes:
        return None, 0.0

    boxes.sort(key=lambda item: item[0])
    digits: list[str] = []
    confidences: list[float] = []

    for x, y, w, h in boxes:
        patch = text_patch[y : y + h, x : x + w]
        canvas = np.zeros((h + 8, w + 8), dtype=np.uint8)
        canvas[4 : 4 + h, 4 : 4 + w] = patch
        sample_img = cv2.resize(canvas, (20, 28), interpolation=cv2.INTER_AREA)
        sample = sample_img.reshape(1, -1).astype(np.float32) / 255.0

        _ret, result, neighbors, _dist = knn.findNearest(sample, k=5)
        pred = int(result[0, 0])

        neighbor_vals = neighbors.flatten().astype(int).tolist()
        confidence = neighbor_vals.count(pred) / max(1, len(neighbor_vals))

        digits.append(str(pred))
        confidences.append(float(confidence))

    value = int("".join(digits)) if digits else None
    avg_conf = float(sum(confidences) / len(confidences)) if confidences else 0.0
    return value, avg_conf


def number_template(number: int) -> list[np.ndarray]:
    text = str(number)
    templates: list[np.ndarray] = []

    for scale in (0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5):
        for thickness in (2, 3):
            img = np.zeros((80, 80), dtype=np.uint8)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
            if tw > 74 or th > 74:
                continue
            x = (80 - tw) // 2
            y = (80 + th) // 2
            cv2.putText(
                img,
                text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                255,
                thickness,
                cv2.LINE_AA,
            )
            _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            templates.append(img)

    return templates


def similarity_to_number(text_patch: np.ndarray, number: int, cache: dict[int, list[np.ndarray]]) -> float:
    templates = cache.setdefault(number, number_template(number))
    if not templates:
        return -1.0
    best = -1.0
    for tmpl in templates:
        score = float(cv2.matchTemplate(text_patch, tmpl, cv2.TM_CCOEFF_NORMED)[0, 0])
        if score > best:
            best = score
    return best


def assign_unique_labels(
    text_patches: list[np.ndarray],
    knn_guesses: list[int | None],
    knn_confidences: list[float],
) -> tuple[list[int], float]:
    k = len(text_patches)
    available = set(range(1, k + 1))
    assigned: list[int | None] = [None] * k

    cache: dict[int, list[np.ndarray]] = {}
    score_table: list[list[float]] = []

    for i, patch in enumerate(text_patches):
        row: list[float] = []
        for label in range(1, k + 1):
            score = similarity_to_number(patch, label, cache)
            guess = knn_guesses[i]
            conf = knn_confidences[i]
            if guess is not None and label == guess:
                score += 0.20 * conf
            row.append(score)
        score_table.append(row)

    picked_scores: list[float] = []

    while available:
        best: tuple[float, int, int] | None = None
        for i in range(k):
            if assigned[i] is not None:
                continue
            for label in sorted(available):
                score = score_table[i][label - 1]
                if best is None or score > best[0]:
                    best = (score, i, label)
        assert best is not None
        score, row_idx, label = best
        assigned[row_idx] = label
        picked_scores.append(score)
        available.remove(label)

    final = [int(value) for value in assigned if value is not None]
    confidence = float(sum(picked_scores) / len(picked_scores)) if picked_scores else 0.0
    return final, confidence


def infer_cell_index(value: float, lines: list[float], n: int) -> int:
    if not lines:
        return 0
    idx = int(np.searchsorted(np.array(lines, dtype=np.float32), value, side="right") - 1)
    return max(0, min(n - 1, idx))


def output_filename(entry: dict, fallback_idx: int) -> str:
    puzzle_number = entry.get("puzzle_number")
    if isinstance(puzzle_number, int):
        num_part = f"#{puzzle_number:03d}"
    else:
        num_part = "#unknown"
    date_part = str(entry.get("puzzle_date") or "date_unknown")
    basename = str(entry.get("video_basename") or f"video_{fallback_idx:03d}")
    return f"{num_part}_{date_part}__{basename}.json"


def convert_one(entry: dict, out_dir: Path, knn: cv2.ml_KNearest, force: bool = False) -> dict:
    grid_path_value = entry.get("paths", {}).get("grid_image")
    if not grid_path_value:
        return {
            "video_basename": entry.get("video_basename"),
            "status": "needs_review",
            "error": "missing grid path",
        }

    grid_path = Path(grid_path_value)
    if not grid_path.is_absolute():
        grid_path = Path.cwd() / grid_path

    file_name = output_filename(entry, fallback_idx=int(entry.get("playlist_index") or 0))
    out_path = out_dir / file_name

    if out_path.exists() and not force:
        return {
            "video_basename": entry.get("video_basename"),
            "playlist_index": entry.get("playlist_index"),
            "grid_image": relative_to_cwd(grid_path),
            "json_path": relative_to_cwd(out_path),
            "status": "ok",
            "skipped": True,
        }

    image = cv2.imread(str(grid_path), cv2.IMREAD_COLOR)
    if image is None:
        return {
            "video_basename": entry.get("video_basename"),
            "playlist_index": entry.get("playlist_index"),
            "grid_image": relative_to_cwd(grid_path),
            "status": "needs_review",
            "error": "cannot read grid image",
        }

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    xs, ys = detect_grid_lines(gray)

    n = min(len(xs), len(ys)) - 1
    if n < 2:
        return {
            "video_basename": entry.get("video_basename"),
            "playlist_index": entry.get("playlist_index"),
            "grid_image": relative_to_cwd(grid_path),
            "status": "needs_review",
            "error": f"unable to infer grid size ({len(xs)=}, {len(ys)=})",
        }

    circles = detect_circles(gray)
    if not circles:
        return {
            "video_basename": entry.get("video_basename"),
            "playlist_index": entry.get("playlist_index"),
            "grid_image": relative_to_cwd(grid_path),
            "status": "needs_review",
            "error": "no number circles detected",
        }

    patches: list[np.ndarray] = []
    knn_guesses: list[int | None] = []
    knn_confs: list[float] = []

    for circle in circles:
        patch = text_patch_from_circle(gray, circle)
        guess, conf = predict_digits_with_knn(patch, knn)
        patches.append(patch)
        knn_guesses.append(guess)
        knn_confs.append(conf)

    labels, assign_conf = assign_unique_labels(patches, knn_guesses, knn_confs)

    seen_cells: set[tuple[int, int]] = set()
    numbers: list[dict[str, int]] = []
    duplicate_cell = False

    for circle, label in zip(circles, labels):
        row = infer_cell_index(circle.y, ys, n)
        col = infer_cell_index(circle.x, xs, n)
        cell = (row, col)
        if cell in seen_cells:
            duplicate_cell = True
        seen_cells.add(cell)
        numbers.append({"k": int(label), "r": int(row), "c": int(col)})

    numbers.sort(key=lambda item: item["k"])

    payload = {
        "game": "zip",
        "n": int(n),
        "numbers": numbers,
        "walls": [],
        "meta": {
            "video_id": entry.get("video_id"),
            "playlist_index": entry.get("playlist_index"),
            "puzzle_number": entry.get("puzzle_number"),
            "puzzle_date": entry.get("puzzle_date"),
            "source_url": entry.get("source_url"),
            "frame_timestamp": entry.get("frame_timestamp"),
            "grid_image": relative_to_cwd(grid_path),
            "conversion": "png_to_zip_v1",
        },
    }

    ensure_dir(out_path.parent)
    dump_json(out_path, payload)

    status = "ok"
    reasons: list[str] = []
    if duplicate_cell:
        status = "needs_review"
        reasons.append("duplicate checkpoint cell detected")
    if assign_conf < 0.18:
        status = "needs_review"
        reasons.append(f"low number recognition confidence ({assign_conf:.3f})")

    return {
        "video_basename": entry.get("video_basename"),
        "playlist_index": entry.get("playlist_index"),
        "video_id": entry.get("video_id"),
        "puzzle_number": entry.get("puzzle_number"),
        "puzzle_date": entry.get("puzzle_date"),
        "grid_image": relative_to_cwd(grid_path),
        "json_path": relative_to_cwd(out_path),
        "n": int(n),
        "checkpoint_count": len(numbers),
        "recognition_confidence": round(assign_conf, 5),
        "status": status,
        "notes": reasons,
    }


def main() -> int:
    args = parse_args()

    index_path = Path(args.index)
    out_dir = Path(args.out_dir)
    manifest_path = Path(args.manifest)

    ensure_dir(out_dir)
    ensure_dir(manifest_path.parent)

    index_payload = load_json(index_path, default={})
    entries = list(index_payload.get("entries", []))
    if args.limit is not None:
        entries = entries[: args.limit]

    if not entries:
        print(f"No entries in {index_path}")
        return 0

    knn = build_digit_knn()

    results: list[dict] = []
    for idx, entry in enumerate(entries, start=1):
        result = convert_one(entry, out_dir=out_dir, knn=knn, force=args.force)
        results.append(result)

        if args.verbose:
            print(
                f"[{idx}/{len(entries)}] {entry.get('video_basename')} -> "
                f"{result.get('status')} ({result.get('json_path', 'no output')})"
            )

    ok_count = sum(1 for item in results if item.get("status") == "ok")
    review_count = len(results) - ok_count

    payload = {
        "generated_at": utc_now_iso(),
        "source_index": str(index_path),
        "count": len(results),
        "ok": ok_count,
        "needs_review": review_count,
        "entries": sorted(
            results,
            key=lambda item: (
                item.get("playlist_index") is None,
                item.get("playlist_index") or 10**9,
                item.get("video_basename", ""),
            ),
        ),
    }

    dump_json(manifest_path, payload)
    print(f"Wrote puzzles manifest: {manifest_path} (ok={ok_count}, needs_review={review_count})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
