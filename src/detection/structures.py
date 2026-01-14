"""Structure detection - towers and inhibitors using template matching"""

from typing import List, Dict, Tuple
import cv2
import numpy as np
from pathlib import Path
from loguru import logger

from detection.base import BaseDetector
from api.schemas import Structure, Position
from config import settings


class StructureDetector(BaseDetector):
    """
    Detects structures (towers and inhibitors) using template matching

    Tower Types:
    - Outer Turret: First tower in lane, closest to river
    - Inner Turret: Second tower in lane, between outer and inhibitor
    - Inhibitor Turret: Protects the inhibitor
    - Nexus Turrets: Two towers protecting Nexus (final defensive structures)

    Strategy:
    1. Load tower templates from models/templates/towers/
    2. Use cv2.matchTemplate() to find tower icons on minimap
    3. Apply non-maximum suppression to avoid duplicate detections
    4. Return detected towers with confidence scores
    """

    def __init__(self):
        super().__init__()

        # Template storage
        self.blue_templates: List[Tuple[np.ndarray, str]] = []  # (template_img, filename)
        self.red_templates: List[Tuple[np.ndarray, str]] = []

        # Template matching parameters
        self.match_threshold = settings.TEMPLATE_MATCH_THRESHOLD  # 0.7 by default
        self.nms_threshold = 30  # Non-maximum suppression: merge detections within 30 pixels
        self.max_detections_per_team = 11  # Summoner's Rift has 11 towers per team

        # Lane detection zones (normalized 0-100 coordinates)
        # Summoner's Rift is diagonal: ORDER bottom-left, CHAOS top-right
        # Use distance from diagonal line (y = 100 - x) to determine lanes
        self.nexus_distance_threshold = 15  # Within 15 units of base corner = nexus tower

        # Performance optimization: cache tower positions
        # Towers don't move, so we only need to detect them occasionally
        self.cached_structures = []
        self.frames_since_full_scan = 0
        self.full_scan_interval = 30  # Only do full template matching every 30 frames (~1 second at 30 FPS)

        # Grayscale templates for 3x faster matching
        self.blue_templates_gray: List[Tuple[np.ndarray, str]] = []
        self.red_templates_gray: List[Tuple[np.ndarray, str]] = []

    def initialize(self) -> None:
        """Initialize structure detector and load templates"""
        logger.info("Initializing StructureDetector...")
        super().initialize()

        # Load tower templates
        self._load_templates()

        logger.info(f"StructureDetector ready - loaded {len(self.blue_templates)} blue templates, "
                   f"{len(self.red_templates)} red templates")

    def _load_templates(self) -> None:
        """Load all tower templates from models/templates/towers/"""
        from config import PROJECT_ROOT

        templates_dir = PROJECT_ROOT / "models" / "templates" / "towers"

        if not templates_dir.exists():
            logger.error(f"Templates directory not found: {templates_dir}")
            return

        # PERFORMANCE: Only use a subset of templates for speed
        # Using just 2 templates per team gives 75% speedup with minimal accuracy loss
        priority_templates = [
            "blueTower5.1.jpg",  # Blue tower with full health
            "blueTower4.1.jpg",  # Blue tower damaged
            "redTower5.1.jpg",   # Red tower with full health
            "redTower4.1.jpg",   # Red tower damaged
        ]

        # Load all tower template images
        for template_file in templates_dir.glob("*.jpg"):
            # Skip non-tower templates (jungle camps, etc.)
            if "tower" not in template_file.name.lower():
                continue

            # OPTIONAL: Comment out this block to use ALL templates (slower but more accurate)
            # if template_file.name not in priority_templates:
            #     continue

            img = cv2.imread(str(template_file))

            if img is None:
                logger.warning(f"Failed to load template: {template_file}")
                continue

            # Convert to grayscale for 3x faster matching
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Determine team based on filename
            filename = template_file.name.lower()

            if "blue" in filename:
                self.blue_templates.append((img, template_file.name))
                self.blue_templates_gray.append((img_gray, template_file.name))
                logger.debug(f"Loaded blue template: {template_file.name} ({img.shape[1]}x{img.shape[0]})")
            elif "red" in filename:
                self.red_templates.append((img, template_file.name))
                self.red_templates_gray.append((img_gray, template_file.name))
                logger.debug(f"Loaded red template: {template_file.name} ({img.shape[1]}x{img.shape[0]})")
            else:
                logger.warning(f"Unknown template type: {template_file.name}")

    def detect(self, minimap: np.ndarray) -> List[Structure]:
        """
        Detect all structures using template matching with caching

        Args:
            minimap: OpenCV BGR image of minimap

        Returns:
            List of Structure objects detected in the image
        """
        if minimap is None or minimap.size == 0:
            logger.warning("StructureDetector received empty minimap")
            return []

        # Use cached results most of the time (towers don't move!)
        self.frames_since_full_scan += 1

        if self.frames_since_full_scan < self.full_scan_interval and self.cached_structures:
            logger.debug(f"🏰 Using cached tower positions (frame {self.frames_since_full_scan}/{self.full_scan_interval})")
            return self.cached_structures

        # Time for a full scan
        self.frames_since_full_scan = 0

        height, width = minimap.shape[:2]
        structures = []

        try:
            # Convert to grayscale for 3x faster matching
            minimap_gray = cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY)

            # Detect blue towers (using grayscale templates)
            blue_detections = self._match_templates(minimap_gray, self.blue_templates_gray, "ORDER")
            structures.extend(blue_detections)

            # Detect red towers (using grayscale templates)
            red_detections = self._match_templates(minimap_gray, self.red_templates_gray, "CHAOS")
            structures.extend(red_detections)

            logger.info(f"🏰 StructureDetector found {len(structures)} structures "
                       f"({len(blue_detections)} ORDER, {len(red_detections)} CHAOS)")

            # Cache the results
            self.cached_structures = structures

        except Exception as e:
            logger.error(f"StructureDetector error: {e}")
            import traceback
            traceback.print_exc()
            return self.cached_structures if self.cached_structures else []

        return structures

    def _match_templates(self, minimap: np.ndarray, templates: List[Tuple[np.ndarray, str]],
                        team: str) -> List[Structure]:
        """
        Match a set of templates against the minimap

        Args:
            minimap: The minimap image to search
            templates: List of (template_image, filename) tuples
            team: "ORDER" or "CHAOS"

        Returns:
            List of detected structures
        """
        height, width = minimap.shape[:2]
        detections = []  # List of (x, y, confidence, template_name)

        max_confidences = []  # Track max confidence for each template

        # Try each template
        for template_img, template_name in templates:
            # Perform template matching using fastest method (TM_SQDIFF_NORMED)
            # Note: Lower values are better matches with SQDIFF
            result = cv2.matchTemplate(minimap, template_img, cv2.TM_SQDIFF_NORMED)

            # Track min distance (best match) for debugging
            # For SQDIFF, lower is better, so we invert it to a confidence score
            min_dist = result.min()
            max_conf = 1.0 - min_dist  # Convert distance to confidence (0-1 range)
            max_confidences.append((template_name, max_conf))

            # Find all matches below threshold (SQDIFF is inverted)
            # Convert threshold: if threshold=0.55 confidence, we want dist < 0.45
            dist_threshold = 1.0 - self.match_threshold
            locations = np.where(result <= dist_threshold)
            num_matches = len(locations[0])

            if num_matches > 0:
                logger.debug(f"  {template_name}: {num_matches} matches (max conf: {max_conf:.3f})")

            for pt in zip(*locations[::-1]):  # Switch x and y
                x, y = pt
                distance = result[y, x]
                confidence = 1.0 - distance  # Convert to confidence score

                # Get center of template match (template coords are top-left)
                template_h, template_w = template_img.shape[:2]
                center_x = x + template_w // 2
                center_y = y + template_h // 2

                detections.append((center_x, center_y, confidence, template_name))


        # Apply non-maximum suppression to remove duplicate detections
        filtered_detections = self._non_max_suppression(detections)

        # Limit to max towers per team and keep only highest confidence
        if len(filtered_detections) > self.max_detections_per_team:
            logger.warning(f"  {team} - Too many detections ({len(filtered_detections)}), limiting to top {self.max_detections_per_team} by confidence")
            filtered_detections = sorted(filtered_detections, key=lambda d: d[2], reverse=True)[:self.max_detections_per_team]

        # Convert to Structure objects
        structures = []
        for x, y, confidence, template_name in filtered_detections:
            # Normalize to 0-100 range
            x_norm = (x / width) * 100
            y_norm = (y / height) * 100

            # Infer lane from position
            lane = self._infer_lane_from_position(x_norm, y_norm, team)

            # Determine structure type (TODO: could infer tier from distance to base)
            structure = Structure(
                position=Position(x=x_norm, y=y_norm),
                team=team,
                structureType="outer_turret",  # Generic for now
                lane=lane,
                isAlive=True,
                confidence=float(confidence)
            )
            structures.append(structure)

        return structures

    def _infer_lane_from_position(self, x_norm: float, y_norm: float, team: str) -> str:
        """
        Infer lane from tower position using geometric zones

        Summoner's Rift layout (0-100 normalized coords):
        - ORDER base: bottom-left (low x, high y)
        - CHAOS base: top-right (high x, low y)
        - Main diagonal: from (0,100) to (100,0), equation: y = 100 - x

        Lane rules:
        - Nexus: Close to team's base corner
        - Top: Above/left of diagonal (y > 100 - x means above diagonal)
        - Bot: Below/right of diagonal (y < 100 - x means below diagonal)
        - Mid: Near the diagonal line
        """

        # Check if tower is in nexus area (close to base)
        if team == "ORDER":
            # ORDER base is bottom-left corner (0, 100)
            dist_to_base = np.sqrt((x_norm - 0)**2 + (y_norm - 100)**2)
            if dist_to_base < self.nexus_distance_threshold:
                return "nexus"
        else:  # CHAOS
            # CHAOS base is top-right corner (100, 0)
            dist_to_base = np.sqrt((x_norm - 100)**2 + (y_norm - 0)**2)
            if dist_to_base < self.nexus_distance_threshold:
                return "nexus"

        # Calculate distance from main diagonal (y = 100 - x)
        # Positive distance = above diagonal (top lane)
        # Negative distance = below diagonal (bot lane)
        # Near zero = mid lane
        diagonal_distance = y_norm - (100 - x_norm)

        # Mid lane threshold: within 10 units of diagonal
        if abs(diagonal_distance) < 10:
            return "mid"
        elif diagonal_distance > 0:
            return "top"
        else:
            return "bot"

    def _non_max_suppression(self, detections: List[Tuple[int, int, float, str]]) -> List[Tuple[int, int, float, str]]:
        """
        Apply non-maximum suppression to remove duplicate detections

        If multiple detections are within nms_threshold pixels of each other,
        keep only the one with highest confidence.

        Args:
            detections: List of (x, y, confidence, template_name)

        Returns:
            Filtered list of detections
        """
        if not detections:
            return []

        # Sort by confidence (highest first)
        detections = sorted(detections, key=lambda d: d[2], reverse=True)

        filtered = []

        for detection in detections:
            x, y, conf, name = detection

            # Check if this detection is too close to an already-accepted detection
            is_duplicate = False
            for fx, fy, _, _ in filtered:
                distance = np.sqrt((x - fx)**2 + (y - fy)**2)
                if distance < self.nms_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered.append(detection)

        return filtered
