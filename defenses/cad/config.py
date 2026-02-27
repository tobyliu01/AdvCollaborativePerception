from __future__ import annotations

import os

# Debug-only settings.
DEBUG_CASE_IDS = {0}
DEBUG_PAIR_IDS = {0}
DEBUG_MODE = False

TOTAL_FRAMES = 10
CONFLICT_AREA_THRESHOLD = 0.6
ATTACK_MATCH_DISTANCE_THRESHOLD = 2.0
MAX_RANGE = 50.0
LOAD_CACHED_OCCUPANCY_MAP = False

CAD_ROOT = "/workspace/hdd/datasets/yutongl/AdvCollaborativePerception/result/defense/cad"
CAD_RESULT_ROOT = os.path.join(CAD_ROOT, "result")
CAD_VISUALIZATION_ROOT = os.path.join(CAD_ROOT, "visualization")
CAD_OCCUPANCY_MAP_ROOT = os.path.join(CAD_ROOT, "occupancy_map")
