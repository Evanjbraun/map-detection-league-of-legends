"""Jungle camp detection using template matching + color detection"""

from typing import List
import cv2
import numpy as np
from loguru import logger

from detection.base import BaseDetector
from api.schemas import JungleCamp, Position
from config import settings


class JungleCampDetector(BaseDetector):
    """
    Detects jungle camp status (alive/cleared/respawning)

    Strategy:
    1. Use known camp positions (camps are at fixed locations on minimap)
    2. Check color at each position (gold dot = alive, no dot = cleared)
    3. Optional: Template matching for more accuracy
    4. Optional: OCR for respawn timers (when available)

    Camp locations (normalized 0-100):
    ORDER side (bottom-left):
    - Blue Buff: ~20, 80
    - Gromp: ~15, 75
    - Wolves: ~25, 70
    - Red Buff: ~40, 85
    - Raptors: ~35, 75
    - Krugs: ~45, 90

    CHAOS side (top-right):
    - Blue Buff: ~80, 20
    - Gromp: ~85, 25
    - Wolves: ~75, 30
    - Red Buff: ~60, 15
    - Raptors: ~65, 25
    - Krugs: ~55, 10
    """

    def __init__(self):
        super().__init__()

        # Known camp positions (x, y) in normalized coordinates (0-100)
        # These are approximate and may need calibration per resolution
        self.camp_positions = {
            # ORDER side
            "blue_buff_order": (20, 80, "ORDER"),
            "gromp_order": (15, 75, "ORDER"),
            "wolves_order": (25, 70, "ORDER"),
            "red_buff_order": (40, 85, "ORDER"),
            "raptors_order": (35, 75, "ORDER"),
            "krugs_order": (45, 90, "ORDER"),

            # CHAOS side
            "blue_buff_chaos": (80, 20, "CHAOS"),
            "gromp_chaos": (85, 25, "CHAOS"),
            "wolves_chaos": (75, 30, "CHAOS"),
            "red_buff_chaos": (60, 15, "CHAOS"),
            "raptors_chaos": (65, 25, "CHAOS"),
            "krugs_chaos": (55, 10, "CHAOS"),
        }

        # HSV color range for jungle camp indicators (gold/yellow dots)
        self.camp_alive_lower = np.array([20, 100, 150])  # Yellow/gold color
        self.camp_alive_upper = np.array([30, 255, 255])

        # Detection radius around camp position (pixels)
        self.search_radius = 15

    def initialize(self) -> None:
        """Initialize jungle camp detector"""
        logger.info("Initializing JungleCampDetector...")

        # TODO: Load camp position calibration from config
        # TODO: Load template images for camp icons
        # TODO: Initialize OCR for timers

        super().initialize()
        logger.info("JungleCampDetector ready")

    def detect(self, minimap: np.ndarray) -> List[JungleCamp]:
        """
        Detect all jungle camp statuses

        Args:
            minimap: OpenCV BGR image of minimap

        Returns:
            List of JungleCamp objects
        """
        if minimap is None or minimap.size == 0:
            logger.warning("JungleCampDetector received empty minimap")
            return []

        height, width = minimap.shape[:2]
        camps = []

        try:
            # Convert to HSV for color detection
            hsv = self.convert_to_hsv(minimap)

            # Check each camp position
            for camp_name, (norm_x, norm_y, side) in self.camp_positions.items():
                # Convert normalized position to pixels
                pixel_x = int((norm_x / 100.0) * width)
                pixel_y = int((norm_y / 100.0) * height)

                # Detect camp status
                status = self._detect_camp_status(
                    hsv, pixel_x, pixel_y, width, height
                )

                # Extract camp type from name
                camp_type = camp_name.rsplit('_', 1)[0]  # Remove _order or _chaos suffix

                camps.append(JungleCamp(
                    type=camp_type,
                    position=Position(x=norm_x, y=norm_y),
                    side=side,
                    status=status,
                    respawnTimer=None,  # TODO: Implement OCR for timers
                    confidence=0.7  # Fixed confidence for now
                ))

            logger.debug(f"Detected {len(camps)} jungle camps")

        except Exception as e:
            logger.error(f"JungleCampDetector error: {e}")
            return []

        return camps

    def _detect_camp_status(
        self,
        hsv: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int
    ) -> str:
        """
        Detect if a camp is alive/cleared/respawning at given position

        Args:
            hsv: Minimap in HSV color space
            x, y: Pixel coordinates of camp center
            width, height: Minimap dimensions

        Returns:
            "alive", "cleared", or "respawning"
        """
        # Extract region around camp position
        x1 = max(0, x - self.search_radius)
        y1 = max(0, y - self.search_radius)
        x2 = min(width, x + self.search_radius)
        y2 = min(height, y + self.search_radius)

        roi = hsv[y1:y2, x1:x2]

        if roi.size == 0:
            return "cleared"

        # Check for gold/yellow color (alive indicator)
        mask = self.apply_color_mask(roi, self.camp_alive_lower, self.camp_alive_upper)

        # Count pixels in range
        alive_pixels = cv2.countNonZero(mask)

        # Threshold: if enough yellow pixels, camp is alive
        if alive_pixels > 5:  # Arbitrary threshold, may need tuning
            return "alive"

        # TODO: Check for timer text (respawning)
        # For now, assume cleared if not alive
        return "cleared"
