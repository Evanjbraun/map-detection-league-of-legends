"""Objective detection (Dragon, Baron, Herald)"""

from typing import List
import cv2
import numpy as np
from loguru import logger

from detection.base import BaseDetector
from api.schemas import Objective, Position
from config import settings


class ObjectiveDetector(BaseDetector):
    """
    Detects major objectives (Dragon, Baron, Herald) and their status

    Strategy:
    1. Use known objective positions (fixed locations on minimap)
    2. Color detection or template matching
    3. OCR for respawn timers
    """

    def __init__(self):
        super().__init__()

        # Known objective positions (normalized 0-100)
        self.objective_positions = {
            "dragon": (30, 60),   # Bottom river
            "baron": (70, 40),    # Top river
            "herald": (70, 40),   # Same as baron (early game)
        }

    def initialize(self) -> None:
        """Initialize objective detector"""
        logger.info("Initializing ObjectiveDetector...")
        super().initialize()
        logger.info("ObjectiveDetector ready")

    def detect(self, minimap: np.ndarray) -> List[Objective]:
        """
        Detect all objectives and their status

        Args:
            minimap: OpenCV BGR image of minimap

        Returns:
            List of Objective objects
        """
        if minimap is None or minimap.size == 0:
            logger.warning("ObjectiveDetector received empty minimap")
            return []

        objectives = []

        try:
            # TODO: Implement objective detection
            # For now, return empty list
            logger.debug("Objective detection not yet implemented")

        except Exception as e:
            logger.error(f"ObjectiveDetector error: {e}")
            return []

        return objectives
