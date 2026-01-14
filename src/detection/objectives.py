"""Objective detection using template matching"""

from typing import List, Dict, Tuple
import cv2
import numpy as np
from pathlib import Path
from loguru import logger

from detection.base import BaseDetector
from api.schemas import Objective, Position
from config import settings


class ObjectiveDetector(BaseDetector):
    """
    Detects major objectives (Dragon, Baron, Herald, Grubs) using template matching

    Simple approach:
    1. Load objective templates from models/templates/obj/
    2. Use template matching to find objectives on minimap
    3. Return detected positions and types
    """

    def __init__(self):
        super().__init__()
        self.templates: Dict[str, np.ndarray] = {}  # template_name -> grayscale image
        self.match_threshold = 0.65  # Slightly lower than jungle camps to catch objectives

    def initialize(self) -> None:
        """Load all objective templates"""
        logger.info("Initializing ObjectiveDetector...")
        super().initialize()

        from config import PROJECT_ROOT

        templates_dir = PROJECT_ROOT / "models" / "templates" / "obj"

        if not templates_dir.exists():
            logger.error(f"Objectives template directory not found: {templates_dir}")
            return

        # Load all objective templates
        for template_file in templates_dir.glob("*.jpg"):
            template_name = template_file.stem  # Filename without extension
            img = cv2.imread(str(template_file))

            if img is not None:
                # Convert to grayscale for template matching
                self.templates[template_name] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                logger.info(f"Loaded objective template: {template_name} ({img.shape[1]}x{img.shape[0]} px)")
            else:
                logger.error(f"Failed to load objective template: {template_file}")

        logger.info(f"ObjectiveDetector ready - loaded {len(self.templates)} templates")

    def detect(self, minimap: np.ndarray) -> List[Objective]:
        """
        Detect objectives on the minimap

        Args:
            minimap: BGR image of the minimap

        Returns:
            List of detected Objective objects
        """
        if minimap is None or minimap.size == 0:
            return []

        if not self.templates:
            logger.warning("No templates loaded, skipping objective detection")
            return []

        objectives = []

        try:
            height, width = minimap.shape[:2]

            # Convert minimap to grayscale
            minimap_gray = cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY)

            # Try to match each template
            all_detections = []

            for template_name, template in self.templates.items():
                detections = self._match_template(minimap_gray, template, template_name)
                all_detections.extend([(x, y, conf, template_name) for x, y, conf in detections])

            # Apply NMS across all detections
            all_detections = self._non_max_suppression(all_detections, threshold=30)

            # Convert to Objective objects with normalized coordinates
            for x, y, conf, template_name in all_detections:
                x_norm = (x / width) * 100
                y_norm = (y / height) * 100

                # Determine objective type from template name
                obj_type = self._classify_objective_type(template_name)

                objectives.append(Objective(
                    position=Position(x=x_norm, y=y_norm),
                    type=obj_type,
                    status="alive",  # TODO: detect status (alive/dead/respawning)
                    respawnTimer=None,
                    confidence=float(conf)
                ))


        except Exception as e:
            logger.error(f"Error in objective detection: {e}")
            import traceback
            traceback.print_exc()

        return objectives

    def _match_template(self, minimap_gray: np.ndarray, template: np.ndarray,
                       template_name: str) -> List[Tuple[int, int, float]]:
        """
        Match a single template against the minimap

        Args:
            minimap_gray: Grayscale minimap
            template: Grayscale template to match
            template_name: Name for logging

        Returns:
            List of (x, y, confidence) detections
        """
        detections = []

        # Perform template matching
        result = cv2.matchTemplate(minimap_gray, template, cv2.TM_CCOEFF_NORMED)

        # Find matches above threshold
        locations = np.where(result >= self.match_threshold)

        template_h, template_w = template.shape

        for pt in zip(*locations[::-1]):  # Switch x and y
            x, y = pt
            confidence = result[y, x]

            # Get center of detection
            center_x = x + template_w // 2
            center_y = y + template_h // 2

            detections.append((center_x, center_y, confidence))

        return detections

    def _classify_objective_type(self, template_name: str) -> str:
        """
        Determine the objective type from template name

        Args:
            template_name: Name of the matched template (e.g., "oceandrake", "baron")

        Returns:
            Objective type: "dragon", "baron", or "herald"
        """
        template_lower = template_name.lower()

        if "baron" in template_lower:
            return "baron"
        elif "herald" in template_lower:
            return "herald"
        elif "drake" in template_lower or "dragon" in template_lower:
            return "dragon"
        elif "grub" in template_lower:
            return "dragon"  # Grubs are in dragon pit area, classify as dragon for now
        else:
            return "dragon"  # Default to dragon

    def _non_max_suppression(self, detections: List[Tuple[int, int, float, str]],
                            threshold: int = 30) -> List[Tuple[int, int, float, str]]:
        """
        Remove duplicate detections that are too close together

        Args:
            detections: List of (x, y, confidence, template_name) tuples
            threshold: Minimum distance between detections in pixels

        Returns:
            Filtered list of detections
        """
        if not detections:
            return []

        # Sort by confidence (highest first)
        detections = sorted(detections, key=lambda d: d[2], reverse=True)

        filtered = []

        for x, y, conf, name in detections:
            # Check if too close to any already-accepted detection
            too_close = False
            for fx, fy, _, _ in filtered:
                distance = np.sqrt((x - fx)**2 + (y - fy)**2)
                if distance < threshold:
                    too_close = True
                    break

            if not too_close:
                filtered.append((x, y, conf, name))

        return filtered
