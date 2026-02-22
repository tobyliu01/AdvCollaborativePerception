from __future__ import annotations

# Debug-only settings.
DEBUG_CASE_IDS = {0, 1, 11, 21, 31, 41, 51, 61, 71, 81}
DEBUG_PAIR_IDS = {0, 1, 2, 3, 4}
DEBUG_MODE = False

# Paths and settings.
OPENCOOD_ROOT = "/workspace/hdd/datasets/yutongl/AdvCollaborativePerception/models/OpenCOOD"
FOV_POLYGON_MODE = "fast"  # fast / slow / both
FOV_VISUALIZATION_ROOT = "/workspace/hdd/datasets/yutongl/AdvCollaborativePerception/defense/mate/visualization"
TRUST_SCORE_ROOT = "/workspace/hdd/datasets/yutongl/AdvCollaborativePerception/result/defense/mate"
UNMATCHED_GROUND_TRUTH_VISIBILITY_METHOD = "point_count"  # polygon / point_count
UNMATCHED_GROUND_TRUTH_POINT_THRESHOLD = 60
ATTACK_OBJECT_ONLY_MODE = True
