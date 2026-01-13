"""
Summoner's Rift map data - lane paths for minion detection

All coordinates are normalized to 0-100 range (matching detector output)
Origin (0,0) is top-left, (100,100) is bottom-right
"""

from typing import Tuple


# Lane paths - used to assign detected minions to specific lanes
TOP_LANE_PATH = [
    (10, 15), (20, 12), (40, 10), (60, 10), (80, 12), (90, 15)
]

MID_LANE_PATH = [
    (15, 15), (30, 30), (50, 50), (70, 70), (85, 85)
]

BOT_LANE_PATH = [
    (15, 90), (25, 88), (45, 85), (65, 88), (85, 90)
]

LANE_PATHS = {
    "top": TOP_LANE_PATH,
    "mid": MID_LANE_PATH,
    "bot": BOT_LANE_PATH,
}


def is_point_in_bounds(x: float, y: float) -> bool:
    """Simple bounds check - is point within 0-100 range"""
    return 0 <= x <= 100 and 0 <= y <= 100
