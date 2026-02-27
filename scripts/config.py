"""Central configuration for the Zip video extraction pipeline."""

from __future__ import annotations

# Geometry / output
TARGET_WIDTH = 1024

# Edge detection
CANNY_LOW = 60
CANNY_HIGH = 180

# Hough transform
HOUGH_THRESHOLD = 80
HOUGH_MIN_LINE_LENGTH = 45
HOUGH_MAX_LINE_GAP = 12

# Quality thresholds
BLUR_THRESHOLD = 90.0
GRID_SCORE_MIN = 18.0
ENTROPY_MIN = 4.2

# Morphology / contour extraction
MORPH_CLOSE_KERNEL = (7, 7)
MORPH_OPEN_KERNEL = (3, 3)
APPROX_EPSILON_RATIO = 0.02
QUAD_MIN_AREA_RATIO = 0.08

# Heuristic weights (frame ranking)
BALANCE_BONUS_WEIGHT = 0.5
BLUR_PENALTY_WEIGHT = 10.0
ENTROPY_PENALTY_WEIGHT = 4.0
BLOCKINESS_PENALTY_WEIGHT = 0.08

# Defaults
DEFAULT_FPS = 0.5
DEFAULT_TOP_K = 5
