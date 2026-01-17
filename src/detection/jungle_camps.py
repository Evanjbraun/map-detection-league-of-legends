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

            # Log raw detections for position mapping
            logger.debug(f"🔍 Raw camp detections ({len(normalized_detections)} total):")
            for x_norm, y_norm, conf in sorted(normalized_detections, key=lambda d: (d[1], d[0])):
                logger.debug(f"   ({x_norm:.1f}, {y_norm:.1f}) conf={conf:.2f}")

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
        Classify jungle camps by matching detections to known fixed positions.

        Each camp has a unique, fixed position on the map. We match each detection
        to the closest expected position. This prevents classification shifting
        when camps are dead/missing.

        Args:
            detections: List of (x_norm, y_norm, confidence) tuples

        Returns:
            List of classified JungleCamp objects
        """
        # Fixed camp positions from actual game data
        # Format: "SIDE_type": (x, y)
        EXPECTED_POSITIONS = {
            # ORDER side camps (top to bottom by Y)
            "ORDER_gromp": (13.8, 43.8),
            "ORDER_blue_buff": (24.3, 47.1),
            "ORDER_wolves": (24.8, 56.4),
            "ORDER_raptors": (46.0, 64.0),
            "ORDER_red_buff": (50.7, 73.3),
            "ORDER_krugs": (55.0, 81.9),

            # CHAOS side camps (top to bottom by Y)
            "CHAOS_krugs": (42.4, 18.6),
            "CHAOS_red_buff": (46.0, 27.4),
            "CHAOS_raptors": (51.2, 36.2),
            "CHAOS_wolves": (72.4, 44.3),
            "CHAOS_blue_buff": (72.4, 53.8),
            "CHAOS_gromp": (83.1, 56.9),

            # Scuttle crabs
            "TOP_RIVER_scuttle": (28.6, 36.0),
            "BOT_RIVER_scuttle": (69.0, 65.2),
        }

        camps = []
        matched_camps = set()  # Track which expected camps have been matched

        # Match each detection to the closest expected camp position
        for x_norm, y_norm, conf in detections:
            best_match = None
            best_distance = float('inf')

            for camp_key, (exp_x, exp_y) in EXPECTED_POSITIONS.items():
                if camp_key in matched_camps:
                    continue  # Already matched this camp

                distance = np.sqrt((x_norm - exp_x)**2 + (y_norm - exp_y)**2)
                if distance < best_distance:
                    best_distance = distance
                    best_match = camp_key

            # Only accept match if within tolerance (5 units)
            if best_match and best_distance < 5:
                matched_camps.add(best_match)

                # Parse camp key to get side and type
                parts = best_match.split("_")
                if "scuttle" in best_match:
                    side = parts[0] + "_" + parts[1]  # "TOP_RIVER" or "BOT_RIVER"
                    camp_type = "scuttle"
                else:
                    side = parts[0]  # "ORDER" or "CHAOS"
                    camp_type = "_".join(parts[1:])  # "gromp", "blue_buff", etc.

                camps.append(JungleCamp(
                    position=Position(x=x_norm, y=y_norm),
                    type=camp_type,
                    side=side,
                    status="alive",
                    respawnTimer=None,
                    confidence=float(conf)
                ))
            else:
                # Detection doesn't match any known camp - log it
                logger.debug(f"   Unmatched detection at ({x_norm:.1f}, {y_norm:.1f}) - closest camp distance: {best_distance:.1f}")

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
