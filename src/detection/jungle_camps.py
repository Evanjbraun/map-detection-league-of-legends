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
        self.jungle_template = None  # Regular camps (diamond icon)
        self.buff_template = None     # Buff camps (dragon head icon)
        self.match_threshold = 0.70  # Start with stricter threshold

    def initialize(self) -> None:
        """Load both jungle camp templates"""
        logger.info("Initializing JungleCampDetector...")
        super().initialize()

        from config import PROJECT_ROOT

        templates_dir = PROJECT_ROOT / "models" / "templates" / "camps"

        # Load the jungleMob template (regular camps - diamond icon)
        jungle_path = templates_dir / "jungleMob.jpg"
        if jungle_path.exists():
            img = cv2.imread(str(jungle_path))
            if img is not None:
                self.jungle_template = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                logger.info(f"Loaded jungle template: {jungle_path.name} ({img.shape[1]}x{img.shape[0]} px)")
            else:
                logger.error(f"Failed to load jungle template: {jungle_path}")
        else:
            logger.error(f"Jungle template not found: {jungle_path}")

        # Load the buffMob template (buff camps - dragon head icon)
        buff_path = templates_dir / "buffMob.jpg"
        if buff_path.exists():
            img = cv2.imread(str(buff_path))
            if img is not None:
                self.buff_template = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                logger.info(f"Loaded buff template: {buff_path.name} ({img.shape[1]}x{img.shape[0]} px)")
            else:
                logger.error(f"Failed to load buff template: {buff_path}")
        else:
            logger.error(f"Buff template not found: {buff_path}")

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

        if self.jungle_template is None and self.buff_template is None:
            logger.warning("No templates loaded, skipping jungle camp detection")
            return []

        camps = []

        try:
            height, width = minimap.shape[:2]

            # Convert minimap to grayscale for template matching
            minimap_gray = cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY)

            # Also convert to HSV for color validation
            minimap_hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)

            # Collect all detections from both templates
            all_detections = []

            # Match regular jungle camps (diamond icon)
            if self.jungle_template is not None:
                jungle_detections = self._match_template(minimap_gray, minimap_hsv, self.jungle_template, "jungle")
                all_detections.extend(jungle_detections)

            # Match buff camps (dragon head icon)
            if self.buff_template is not None:
                buff_detections = self._match_template(minimap_gray, minimap_hsv, self.buff_template, "buff")
                all_detections.extend(buff_detections)

            # Apply NMS across all detections (both templates)
            all_detections = self._non_max_suppression(all_detections, threshold=20)

            # Convert to normalized coordinates
            normalized_detections = []
            for x, y, conf in all_detections:
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

    def _match_template(self, minimap_gray: np.ndarray, minimap_hsv: np.ndarray,
                       template: np.ndarray, template_name: str) -> List[tuple]:
        """
        Match a single template against the minimap with color validation

        Args:
            minimap_gray: Grayscale minimap
            minimap_hsv: HSV minimap for color validation
            template: Grayscale template to match
            template_name: Name for logging ("jungle" or "buff")

        Returns:
            List of (x, y, confidence) detections
        """
        height, width = minimap_gray.shape[:2]
        detections = []

        # Perform template matching
        result = cv2.matchTemplate(minimap_gray, template, cv2.TM_CCOEFF_NORMED)

        # Find matches above threshold
        locations = np.where(result >= self.match_threshold)

        # Collect all detections with color validation
        template_h, template_w = template.shape

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

        return detections

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
        scuttle_camps = []

        # Separate by team using diagonal line (y = x)
        # River is along the diagonal, so camps near the diagonal are scuttles
        # In OpenCV coords: Y increases downward
        # ORDER (bottom-left): y > x (further down and left)
        # CHAOS (top-right): y < x (further up and right)
        for x_norm, y_norm, conf in detections:
            # Calculate distance from diagonal line (y = x)
            diagonal_distance = abs(y_norm - x_norm)

            # If near diagonal (river), it's a scuttle crab
            # Scuttles spawn in river, which is along the diagonal
            if diagonal_distance < 8:  # Within 8 units of diagonal = river/scuttle
                # Determine which river based on position along diagonal
                # Top-right river vs bottom-left river
                if x_norm + y_norm < 100:  # Upper river (top-left half)
                    scuttle_camps.append((x_norm, y_norm, conf, "TOP_RIVER"))
                else:  # Lower river (bottom-right half)
                    scuttle_camps.append((x_norm, y_norm, conf, "BOT_RIVER"))
            elif y_norm > x_norm:  # Below diagonal = ORDER (bottom-left quadrant)
                order_camps.append((x_norm, y_norm, conf))
            else:  # Above diagonal = CHAOS (top-right quadrant)
                chaos_camps.append((x_norm, y_norm, conf))

        # Process scuttle crabs first
        for x_norm, y_norm, conf, river_side in scuttle_camps:
            camps.append(JungleCamp(
                position=Position(x=x_norm, y=y_norm),
                type="scuttle",
                side=river_side,
                status="alive",
                respawnTimer=None,
                confidence=float(conf)
            ))

        # Sort ORDER camps by Y coordinate (top to bottom)
        order_camps.sort(key=lambda c: c[1])
        # ORDER camp order (top to bottom): gromp, blue_buff, wolves, raptors, red_buff, krugs
        order_types = ["gromp", "blue_buff", "wolves", "raptors", "red_buff", "krugs"]

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

        # Sort CHAOS camps by Y coordinate (top to bottom)
        chaos_camps.sort(key=lambda c: c[1])
        # CHAOS camp order (top to bottom): krugs, red_buff, raptors, wolves, blue_buff, gromp
        chaos_types = ["krugs", "red_buff", "raptors", "wolves", "blue_buff", "gromp"]

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
