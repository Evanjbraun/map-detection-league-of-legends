"""Jungle camp detection using template matching"""

from typing import List
import cv2
import numpy as np
from pathlib import Path
from loguru import logger

from detection.base import BaseDetector
from api.schemas import JungleCamp, Position
from config import settings


class JungleCampDetector(BaseDetector):
    """
    Detects jungle camps using template matching

    Simple approach:
    1. Load camp template from models/templates/camps/
    2. Use template matching to find camps on minimap
    3. Return detected positions
    """

    def __init__(self):
        super().__init__()
        self.template = None
        self.match_threshold = 0.70  # Start with stricter threshold

    def initialize(self) -> None:
        """Load the jungle camp template"""
        logger.info("Initializing JungleCampDetector...")
        super().initialize()

        from config import PROJECT_ROOT

        # Load the jungleMob template
        template_path = PROJECT_ROOT / "models" / "templates" / "camps" / "jungleMob.jpg"

        if template_path.exists():
            img = cv2.imread(str(template_path))
            if img is not None:
                # Convert to grayscale for template matching
                self.template = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                logger.info(f"Loaded jungle template: {template_path.name} ({img.shape[1]}x{img.shape[0]} px)")
            else:
                logger.error(f"Failed to load template: {template_path}")
        else:
            logger.error(f"Template not found: {template_path}")

    def detect(self, minimap: np.ndarray) -> List[JungleCamp]:
        """
        Detect jungle camps on the minimap

        Args:
            minimap: BGR image of the minimap

        Returns:
            List of detected JungleCamp objects
        """
        if minimap is None or minimap.size == 0:
            return []

        if self.template is None:
            logger.warning("No template loaded, skipping jungle camp detection")
            return []

        camps = []

        try:
            height, width = minimap.shape[:2]

            # Convert minimap to grayscale for template matching
            minimap_gray = cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY)

            # Also convert to HSV for color validation
            minimap_hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)

            # Perform template matching
            result = cv2.matchTemplate(minimap_gray, self.template, cv2.TM_CCOEFF_NORMED)

            # Find matches above threshold
            locations = np.where(result >= self.match_threshold)

            logger.debug(f"Found {len(locations[0])} raw template matches")

            # Collect all detections with color validation
            detections = []
            template_h, template_w = self.template.shape

            for pt in zip(*locations[::-1]):  # Switch x and y
                x, y = pt
                confidence = result[y, x]

                # Get center of detection
                center_x = x + template_w // 2
                center_y = y + template_h // 2

                # COLOR VALIDATION: Check if this detection has orange color
                # Jungle camps are orange on the minimap
                check_radius = 4  # Check slightly larger area
                x1 = max(0, center_x - check_radius)
                y1 = max(0, center_y - check_radius)
                x2 = min(width, center_x + check_radius)
                y2 = min(height, center_y + check_radius)

                roi_hsv = minimap_hsv[y1:y2, x1:x2]

                # Orange/gold color range for jungle camps
                # H: 10-25 (orange), S: 120-255 (saturated), V: 120-255 (bright)
                lower_orange = np.array([10, 120, 120])
                upper_orange = np.array([25, 255, 255])
                orange_mask = cv2.inRange(roi_hsv, lower_orange, upper_orange)
                orange_pixels = cv2.countNonZero(orange_mask)

                # Require at least 3 orange pixels to be a valid jungle camp
                if orange_pixels >= 3:
                    detections.append((center_x, center_y, confidence))
                    logger.debug(f"  Accepted: ({center_x}, {center_y}) conf={confidence:.3f} orange_px={orange_pixels}")
                else:
                    logger.debug(f"  Rejected: ({center_x}, {center_y}) - only {orange_pixels} orange pixels (need 3+)")

            # Apply non-maximum suppression to remove duplicates
            detections = self._non_max_suppression(detections, threshold=20)

            logger.info(f"After NMS: {len(detections)} jungle camps detected")

            # Convert to normalized coordinates
            normalized_detections = []
            for x, y, conf in detections:
                x_norm = (x / width) * 100
                y_norm = (y / height) * 100
                normalized_detections.append((x_norm, y_norm, conf))

            # Classify camps by position
            camps = self._classify_camps(normalized_detections)

        except Exception as e:
            logger.error(f"Error in jungle camp detection: {e}")
            import traceback
            traceback.print_exc()

        return camps

    def _classify_camps(self, detections: List[tuple]) -> List[JungleCamp]:
        """
        Classify jungle camps by their position

        Coordinate system (0-100 normalized):
        - (0, 0) = top-left corner
        - (100, 100) = bottom-right corner
        - ORDER base: bottom-left at (0, 100)
        - CHAOS base: top-right at (100, 0)

        Main diagonal: from top-left (0,0) to bottom-right (100,100), equation: y = x
        - ORDER side: below/left of diagonal (y > x means higher Y value = lower on screen = ORDER)
        - CHAOS side: above/right of diagonal (y < x means lower Y value = higher on screen = CHAOS)

        Args:
            detections: List of (x_norm, y_norm, confidence) tuples

        Returns:
            List of classified JungleCamp objects
        """
        camps = []
        order_camps = []
        chaos_camps = []

        # Separate by team using diagonal line (y = x)
        # In OpenCV coords: Y increases downward
        # ORDER (bottom-left): y > x (further down and left)
        # CHAOS (top-right): y < x (further up and right)
        for x_norm, y_norm, conf in detections:
            if y_norm > x_norm:  # Below diagonal = ORDER (bottom-left quadrant)
                order_camps.append((x_norm, y_norm, conf))
            else:  # Above diagonal = CHAOS (top-right quadrant)
                chaos_camps.append((x_norm, y_norm, conf))

        # Sort ORDER camps by Y coordinate (top to bottom)
        order_camps.sort(key=lambda c: c[1])
        # ORDER camp order (top to bottom): blue_buff, gromp, wolves, red_buff, raptors, krugs
        order_types = ["blue_buff", "gromp", "wolves", "red_buff", "raptors", "krugs"]

        for i, (x_norm, y_norm, conf) in enumerate(order_camps):
            camp_type = order_types[i] if i < len(order_types) else "gromp"
            camps.append(JungleCamp(
                position=Position(x=x_norm, y=y_norm),
                type=camp_type,
                side="ORDER",
                status="alive",
                respawnTimer=None,
                confidence=float(conf)
            ))
            logger.debug(f"  ORDER camp #{i}: {camp_type} at ({x_norm:.1f}, {y_norm:.1f})")

        # Sort CHAOS camps by Y coordinate (top to bottom)
        chaos_camps.sort(key=lambda c: c[1])
        # CHAOS camp order (top to bottom): krugs, raptors, red_buff, wolves, gromp, blue_buff
        chaos_types = ["krugs", "raptors", "red_buff", "wolves", "gromp", "blue_buff"]

        for i, (x_norm, y_norm, conf) in enumerate(chaos_camps):
            camp_type = chaos_types[i] if i < len(chaos_types) else "gromp"
            camps.append(JungleCamp(
                position=Position(x=x_norm, y=y_norm),
                type=camp_type,
                side="CHAOS",
                status="alive",
                respawnTimer=None,
                confidence=float(conf)
            ))
            logger.debug(f"  CHAOS camp #{i}: {camp_type} at ({x_norm:.1f}, {y_norm:.1f})")

        logger.info(f"Classified {len(order_camps)} ORDER camps, {len(chaos_camps)} CHAOS camps")

        return camps

    def _non_max_suppression(self, detections: List[tuple], threshold: int = 20) -> List[tuple]:
        """
        Remove duplicate detections that are too close together

        Args:
            detections: List of (x, y, confidence) tuples
            threshold: Minimum distance between detections in pixels

        Returns:
            Filtered list of detections
        """
        if not detections:
            return []

        # Sort by confidence (highest first)
        detections = sorted(detections, key=lambda d: d[2], reverse=True)

        filtered = []

        for x, y, conf in detections:
            # Check if too close to any already-accepted detection
            too_close = False
            for fx, fy, _ in filtered:
                distance = np.sqrt((x - fx)**2 + (y - fy)**2)
                if distance < threshold:
                    too_close = True
                    break

            if not too_close:
                filtered.append((x, y, conf))

        return filtered
