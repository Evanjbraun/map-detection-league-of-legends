"""Champion position detection using color + blob detection"""

from typing import List
import cv2
import numpy as np
from loguru import logger

from detection.base import BaseDetector
from api.schemas import ChampionSighting, Position
from config import settings


class ChampionDetector(BaseDetector):
    """
    Detects champion positions on minimap using color-based blob detection

    Strategy:
    1. Convert minimap to HSV color space
    2. Apply color masks for ORDER (blue) and CHAOS (red) teams
    3. Find blobs (connected regions) in each mask
    4. Identify player champion (larger icon or special highlight)
    5. Return ChampionSighting objects with team and confidence
    """

    def __init__(self):
        super().__init__()

        # HSV color ranges for team detection
        # These may need tuning based on graphics settings
        # ORDER team (blue) - typically cyan/blue icons
        self.order_lower = np.array([90, 100, 100])   # H: 90-130, S: 100-255, V: 100-255
        self.order_upper = np.array([130, 255, 255])

        # CHAOS team (red) - typically red/magenta icons
        self.chaos_lower = np.array([0, 100, 100])    # H: 0-10 and 170-180, S: 100-255, V: 100-255
        self.chaos_upper = np.array([10, 255, 255])
        self.chaos_lower2 = np.array([170, 100, 100])  # Red wraps around in HSV
        self.chaos_upper2 = np.array([180, 255, 255])

        # Player champion detection (often has green highlight)
        self.player_lower = np.array([40, 100, 100])   # H: 40-80 (green), S: 100-255, V: 100-255
        self.player_upper = np.array([80, 255, 255])

        # Blob size thresholds (in pixels)
        self.min_champion_area = 5.0      # Minimum area for champion icon
        self.max_champion_area = 100.0    # Maximum area (filter out large false positives)
        self.player_min_area = 8.0        # Player icon is often slightly larger

    def initialize(self) -> None:
        """Initialize champion detector"""
        logger.info("Initializing ChampionDetector...")

        # Load custom color ranges from config if available
        # TODO: Add config options for HSV tuning
        # self.order_lower = settings.CHAMPION_ORDER_HSV_LOWER
        # etc.

        super().initialize()
        logger.info("ChampionDetector ready")

    def detect(self, minimap: np.ndarray) -> List[ChampionSighting]:
        """
        Detect all champions visible on minimap

        Args:
            minimap: OpenCV BGR image of minimap

        Returns:
            List of ChampionSighting objects
        """
        if minimap is None or minimap.size == 0:
            logger.warning("ChampionDetector received empty minimap")
            return []

        height, width = minimap.shape[:2]
        champions = []

        try:
            # Convert to HSV for color detection
            hsv = self.convert_to_hsv(minimap)

            # Detect player champion first (to mark as isPlayer=True)
            player_position = self._detect_player(hsv, width, height)

            # Detect ORDER team (blue)
            order_champions = self._detect_team(
                hsv, width, height,
                team="ORDER",
                lower=self.order_lower,
                upper=self.order_upper,
                player_position=player_position
            )
            champions.extend(order_champions)

            # Detect CHAOS team (red) - handle HSV wrap-around
            chaos_champions = self._detect_team_red(
                hsv, width, height,
                player_position=player_position
            )
            champions.extend(chaos_champions)

            logger.debug(f"Detected {len(champions)} champions ({len(order_champions)} ORDER, {len(chaos_champions)} CHAOS)")

        except Exception as e:
            logger.error(f"ChampionDetector error: {e}")
            return []

        return champions

    def _detect_player(self, hsv: np.ndarray, width: int, height: int) -> tuple | None:
        """
        Detect player champion position (often has green highlight)

        Returns:
            (x, y) tuple if found, None otherwise
        """
        mask = self.apply_color_mask(hsv, self.player_lower, self.player_upper)
        blobs = self.find_blobs(mask, min_area=self.player_min_area)

        if blobs:
            # Take largest blob as player
            largest = max(blobs, key=lambda b: b["area"])
            return largest["center"]

        return None

    def _detect_team(
        self,
        hsv: np.ndarray,
        width: int,
        height: int,
        team: str,
        lower: np.ndarray,
        upper: np.ndarray,
        player_position: tuple | None
    ) -> List[ChampionSighting]:
        """
        Detect champions for a specific team

        Args:
            hsv: Minimap in HSV color space
            width, height: Minimap dimensions
            team: "ORDER" or "CHAOS"
            lower, upper: HSV color range bounds
            player_position: Player position to mark isPlayer=True

        Returns:
            List of ChampionSighting objects
        """
        mask = self.apply_color_mask(hsv, lower, upper)
        blobs = self.find_blobs(mask, min_area=self.min_champion_area)

        champions = []
        for blob in blobs:
            area = blob["area"]

            # Filter by size
            if area > self.max_champion_area:
                continue

            x, y = blob["center"]

            # Check if this is the player
            is_player = False
            if player_position:
                px, py = player_position
                distance = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
                if distance < 10:  # Within 10 pixels
                    is_player = True

            # Normalize coordinates to 0-100 range
            x_norm, y_norm = self.scale_coordinates(x, y, width, height)

            # Calculate confidence based on blob properties
            confidence = self._calculate_confidence(area)

            champions.append(ChampionSighting(
                position=Position(x=x_norm, y=y_norm),
                team=team,
                isPlayer=is_player,
                confidence=confidence
            ))

        return champions

    def _detect_team_red(
        self,
        hsv: np.ndarray,
        width: int,
        height: int,
        player_position: tuple | None
    ) -> List[ChampionSighting]:
        """
        Detect CHAOS team (red) - special handling for HSV wrap-around

        Red color wraps around in HSV (0-10 and 170-180)
        """
        # Detect both red ranges
        mask1 = self.apply_color_mask(hsv, self.chaos_lower, self.chaos_upper)
        mask2 = self.apply_color_mask(hsv, self.chaos_lower2, self.chaos_upper2)

        # Combine masks
        combined_mask = cv2.bitwise_or(mask1, mask2)

        blobs = self.find_blobs(combined_mask, min_area=self.min_champion_area)

        champions = []
        for blob in blobs:
            area = blob["area"]

            # Filter by size
            if area > self.max_champion_area:
                continue

            x, y = blob["center"]

            # Check if this is the player
            is_player = False
            if player_position:
                px, py = player_position
                distance = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
                if distance < 10:
                    is_player = True

            # Normalize coordinates
            x_norm, y_norm = self.scale_coordinates(x, y, width, height)

            # Calculate confidence
            confidence = self._calculate_confidence(area)

            champions.append(ChampionSighting(
                position=Position(x=x_norm, y=y_norm),
                team="CHAOS",
                isPlayer=is_player,
                confidence=confidence
            ))

        return champions

    def _calculate_confidence(self, area: float) -> float:
        """
        Calculate confidence score based on blob properties

        Args:
            area: Blob area in pixels

        Returns:
            Confidence score 0.0-1.0
        """
        # Confidence based on how close area is to expected champion icon size
        # Typical champion icon: 10-30 pixels area
        ideal_area = 20.0

        # Distance from ideal
        distance = abs(area - ideal_area)

        # Normalize to 0-1 (closer to ideal = higher confidence)
        confidence = max(0.0, 1.0 - (distance / ideal_area))

        # Clamp between 0.5 and 1.0 (we detected something, so at least 50% confident)
        confidence = max(0.5, min(1.0, confidence))

        return round(confidence, 2)
