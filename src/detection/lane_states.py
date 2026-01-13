"""Lane state detection (minion wave positions)"""

from typing import List, Dict, Tuple
import cv2
import numpy as np
from loguru import logger

from detection.base import BaseDetector
from api.schemas import LaneState, Position, MinionCount, Minion
from config import settings


class LaneStateDetector(BaseDetector):
    """
    Detects minion wave positions and push direction

    Strategy:
    1. Detect blue (ORDER) and red (CHAOS) minion dots using color masks
    2. Filter by size (minions are smaller than champions: 2-10 pixels)
    3. Group nearby minions into waves
    4. Determine which lane each wave belongs to
    5. Calculate wave center position and push direction
    """

    def __init__(self):
        super().__init__()

        # HSV color ranges for minion detection
        # Based on empirical sampling from live game:
        # Blue minions: RGB #4C98D8 → HSV (207°, 65%, 85%)
        # Red minions:  RGB #D9392F → HSV (4°, 78%, 85%)

        # ORDER minions (cyan/light blue dots)
        # Hue 207° in OpenCV = 207/2 = 103.5
        # Tightened: ±10° tolerance = 93-113, higher saturation to avoid river
        self.order_minion_lower = np.array([93, 80, 100])    # Hue 93-113, Sat 80+, Val 100+
        self.order_minion_upper = np.array([113, 255, 255])

        # CHAOS minions (red dots)
        # Hue 4° in OpenCV = 4/2 = 2
        # Tightened: ±6° tolerance = 0-8, higher saturation/value for more vibrant reds only
        self.chaos_minion_lower1 = np.array([0, 100, 120])   # Red low range (0-8°)
        self.chaos_minion_upper1 = np.array([8, 255, 255])
        self.chaos_minion_lower2 = np.array([172, 100, 120]) # Red high range (172-180°)
        self.chaos_minion_upper2 = np.array([180, 255, 255])

        # Size thresholds - strict to avoid false positives
        self.min_minion_area = 1.0      # Very small dots (single minion)
        self.max_minion_area = 12.0     # Single minion maximum
        self.max_wave_area = 40.0       # Clustered wave maximum (reduced from 84 to exclude towers)

        # Wave clustering distance (pixels)
        self.wave_cluster_distance = 30  # Minions within 30px are considered same wave

        # Tower positions (normalized 0-100 coordinates) - exclusion zones
        # Towers are static and always in these locations on Summoner's Rift
        self.tower_positions = [
            # Blue side (ORDER) towers
            (18, 82), (30, 75), (42, 68),  # Bot lane
            (18, 18), (25, 25), (32, 32),  # Top lane
            (40, 50), (50, 50), (60, 50),  # Mid lane

            # Red side (CHAOS) towers
            (82, 18), (75, 30), (68, 42),  # Top lane
            (82, 82), (75, 75), (68, 68),  # Bot lane
            (50, 40), (50, 50), (50, 60),  # Mid lane
        ]
        self.tower_exclusion_radius = 3.0  # Exclude detections within 3 units of tower positions

        # Import accurate lane paths from map data
        from data.map_data import LANE_PATHS, is_point_in_bounds
        self.lane_paths = LANE_PATHS
        self.is_point_in_bounds = is_point_in_bounds

    def initialize(self) -> None:
        """Initialize lane state detector"""
        logger.info("Initializing LaneStateDetector...")
        super().initialize()
        logger.info("LaneStateDetector ready")

    def detect(self, minimap: np.ndarray) -> List[LaneState]:
        """
        Detect all lane states

        Args:
            minimap: OpenCV BGR image of minimap

        Returns:
            List of LaneState objects for each active lane
        """
        if minimap is None or minimap.size == 0:
            logger.warning("LaneStateDetector received empty minimap")
            return []

        height, width = minimap.shape[:2]
        lane_states = []

        try:
            # Convert to HSV
            hsv = self.convert_to_hsv(minimap)

            # Detect ORDER minions (blue)
            order_minions = self._detect_minions(
                hsv, width, height,
                "ORDER",
                self.order_minion_lower,
                self.order_minion_upper
            )

            # Detect CHAOS minions (red - handle HSV wraparound)
            chaos_minions = self._detect_minions_red(hsv, width, height)

            logger.debug(f"Detected {len(order_minions)} ORDER minions, {len(chaos_minions)} CHAOS minions")

            # Group minions into waves
            order_waves = self._cluster_into_waves(order_minions)
            chaos_waves = self._cluster_into_waves(chaos_minions)

            logger.debug(f"Found {len(order_waves)} ORDER waves, {len(chaos_waves)} CHAOS waves")

            # Analyze each lane
            for lane_name in ["top", "mid", "bot"]:
                # Find waves in this lane
                lane_order_minions = self._get_lane_minions(order_minions, lane_name)
                lane_chaos_minions = self._get_lane_minions(chaos_minions, lane_name)

                # Skip if no minions in this lane
                if not lane_order_minions and not lane_chaos_minions:
                    continue

                # Calculate wave center position
                all_lane_minions = lane_order_minions + lane_chaos_minions
                if not all_lane_minions:
                    continue

                center_x = sum(m[0] for m in all_lane_minions) / len(all_lane_minions)
                center_y = sum(m[1] for m in all_lane_minions) / len(all_lane_minions)

                # Determine push direction
                push_direction = self._calculate_push_direction(
                    lane_name,
                    len(lane_order_minions),
                    len(lane_chaos_minions),
                    center_x,
                    center_y
                )

                # Create minion objects for each detected minion
                minion_objects = []

                # Add ORDER minions
                for mx, my in lane_order_minions:
                    minion_objects.append(Minion(
                        position=Position(x=mx, y=my),
                        team="ORDER",
                        confidence=0.8
                    ))

                # Add CHAOS minions
                for mx, my in lane_chaos_minions:
                    minion_objects.append(Minion(
                        position=Position(x=mx, y=my),
                        team="CHAOS",
                        confidence=0.8
                    ))

                # Create lane state
                lane_state = LaneState(
                    lane=lane_name,
                    wavePosition=Position(x=center_x, y=center_y),
                    pushDirection=push_direction,
                    minionCount=MinionCount(
                        ORDER=len(lane_order_minions),
                        CHAOS=len(lane_chaos_minions)
                    ),
                    minions=minion_objects,  # Include individual minions
                    confidence=0.8  # TODO: Calculate based on detection quality
                )

                lane_states.append(lane_state)

            logger.debug(f"Detected {len(lane_states)} active lanes")

        except Exception as e:
            logger.error(f"LaneStateDetector error: {e}")
            import traceback
            traceback.print_exc()
            return []

        return lane_states

    def _separate_clustered_minions(self, mask: np.ndarray, min_distance: int = 3) -> List[Tuple[int, int]]:
        """
        Separate clustered/overlapping minions using distance transform + peak finding

        Args:
            mask: Binary mask of minion pixels
            min_distance: Minimum distance between minion centers (pixels)

        Returns:
            List of (x, y) pixel coordinates for individual minion centers
        """
        # Clean up noise with morphological opening
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Apply distance transform - calculates distance from each white pixel to nearest black pixel
        dist_transform = cv2.distanceTransform(mask_clean, cv2.DIST_L2, 5)

        # Find local maxima using dilation
        # A pixel is a local maximum if it equals the dilated version
        kernel_peak = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.dilate(dist_transform, kernel_peak)
        local_max = (dist_transform == dilated)

        # Filter peaks by minimum distance value (must be at least 1.5 pixels from edge)
        # This filters out noise and small artifacts
        local_max = local_max & (dist_transform > 1.5)

        # Get coordinates of peaks
        minion_centers = []
        y_coords, x_coords = np.where(local_max)

        for x, y in zip(x_coords, y_coords):
            minion_centers.append((int(x), int(y)))

        return minion_centers

    def _detect_minions(
        self,
        hsv: np.ndarray,
        width: int,
        height: int,
        team: str,
        lower: np.ndarray,
        upper: np.ndarray
    ) -> List[Tuple[float, float]]:
        """
        Detect minions for a specific team, handling clustered minions

        Returns:
            List of (x, y) coordinates in normalized 0-100 range
        """
        mask = self.apply_color_mask(hsv, lower, upper)

        # Clean up mask - remove single-pixel noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Filter by blob size BEFORE distance transform to remove large objects (champions, river, structures, towers)
        # Find all connected components
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a filtered mask containing only appropriately-sized blobs
        filtered_mask = np.zeros_like(mask)

        for contour in contours:
            area = cv2.contourArea(contour)
            # Keep blobs that are minion-sized OR wave-sized (but not huge like champions/structures/towers)
            if self.min_minion_area <= area <= self.max_wave_area:
                cv2.drawContours(filtered_mask, [contour], -1, 255, -1)

        # Use distance transform to separate clustered minions
        minion_centers = self._separate_clustered_minions(filtered_mask)

        minions = []
        filtered_out = {"out_of_bounds": 0, "near_tower": 0}

        for x, y in minion_centers:
            # Normalize coordinates
            x_norm, y_norm = self.scale_coordinates(x, y, width, height)

            # Filter out-of-bounds detections
            if not self.is_point_in_bounds(x_norm, y_norm):
                filtered_out["out_of_bounds"] += 1
                continue

            # Filter out detections near tower positions (towers are static, minions move)
            is_near_tower = False
            for tower_x, tower_y in self.tower_positions:
                distance = np.sqrt((x_norm - tower_x)**2 + (y_norm - tower_y)**2)
                if distance < self.tower_exclusion_radius:
                    is_near_tower = True
                    break

            if is_near_tower:
                filtered_out["near_tower"] += 1
                continue

            minions.append((x_norm, y_norm))

        # Log filtering stats
        if filtered_out["out_of_bounds"] > 0 or filtered_out["near_tower"] > 0:
            logger.debug(f"{team} filtering: {filtered_out['out_of_bounds']} out of bounds, {filtered_out['near_tower']} near towers")

        return minions

    def _detect_minions_red(
        self,
        hsv: np.ndarray,
        width: int,
        height: int
    ) -> List[Tuple[float, float]]:
        """
        Detect CHAOS minions (red) - handle HSV wraparound, separate clustered minions
        """
        # Detect both red ranges
        mask1 = self.apply_color_mask(hsv, self.chaos_minion_lower1, self.chaos_minion_upper1)
        mask2 = self.apply_color_mask(hsv, self.chaos_minion_lower2, self.chaos_minion_upper2)

        # Combine masks
        combined_mask = cv2.bitwise_or(mask1, mask2)

        # Clean up mask - remove single-pixel noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

        # Filter by blob size BEFORE distance transform to remove large objects (towers, champions, structures)
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a filtered mask containing only appropriately-sized blobs
        filtered_mask = np.zeros_like(combined_mask)

        for contour in contours:
            area = cv2.contourArea(contour)
            # Keep blobs that are minion-sized OR wave-sized (but not huge like champions/structures/towers)
            if self.min_minion_area <= area <= self.max_wave_area:
                cv2.drawContours(filtered_mask, [contour], -1, 255, -1)

        # Use distance transform to separate clustered minions
        minion_centers = self._separate_clustered_minions(filtered_mask)

        minions = []
        filtered_out = {"out_of_bounds": 0, "near_tower": 0}

        for x, y in minion_centers:
            # Normalize coordinates
            x_norm, y_norm = self.scale_coordinates(x, y, width, height)

            # Filter out-of-bounds detections
            if not self.is_point_in_bounds(x_norm, y_norm):
                filtered_out["out_of_bounds"] += 1
                continue

            # Filter out detections near tower positions
            is_near_tower = False
            for tower_x, tower_y in self.tower_positions:
                distance = np.sqrt((x_norm - tower_x)**2 + (y_norm - tower_y)**2)
                if distance < self.tower_exclusion_radius:
                    is_near_tower = True
                    break

            if is_near_tower:
                filtered_out["near_tower"] += 1
                continue

            minions.append((x_norm, y_norm))

        # Log filtering stats
        if filtered_out["out_of_bounds"] > 0 or filtered_out["near_tower"] > 0:
            logger.debug(f"CHAOS filtering: {filtered_out['out_of_bounds']} out of bounds, {filtered_out['near_tower']} near towers")

        return minions

    def _cluster_into_waves(self, minions: List[Tuple[float, float]]) -> List[List[Tuple[float, float]]]:
        """
        Group nearby minions into waves

        Args:
            minions: List of (x, y) coordinates

        Returns:
            List of waves, where each wave is a list of minion positions
        """
        if not minions:
            return []

        waves = []
        remaining = minions.copy()

        while remaining:
            # Start new wave with first minion
            wave = [remaining.pop(0)]

            # Find all minions close to this wave
            i = 0
            while i < len(remaining):
                minion = remaining[i]

                # Check distance to any minion in current wave
                min_dist = min(
                    self._distance(minion, wave_minion)
                    for wave_minion in wave
                )

                if min_dist < self.wave_cluster_distance:
                    wave.append(remaining.pop(i))
                else:
                    i += 1

            waves.append(wave)

        return waves

    def _get_lane_minions(
        self,
        minions: List[Tuple[float, float]],
        lane_name: str
    ) -> List[Tuple[float, float]]:
        """
        Filter minions that belong to a specific lane

        Uses distance to lane path to determine if minion is in that lane
        """
        lane_path = self.lane_paths[lane_name]
        lane_minions = []

        for minion_pos in minions:
            # Calculate minimum distance from minion to any segment of the lane path
            min_dist = float('inf')

            for i in range(len(lane_path) - 1):
                segment_start = lane_path[i]
                segment_end = lane_path[i + 1]
                dist = self._point_to_line_distance(minion_pos, segment_start, segment_end)
                min_dist = min(min_dist, dist)

            # If close enough to lane path, include it
            if min_dist < 20:  # Within 20 units of lane path
                lane_minions.append(minion_pos)

        return lane_minions

    def _calculate_push_direction(
        self,
        lane_name: str,
        order_count: int,
        chaos_count: int,
        center_x: float,
        center_y: float
    ) -> str:
        """
        Determine which way the wave is pushing

        Logic:
        - More ORDER minions = pushing toward CHAOS
        - More CHAOS minions = pushing toward ORDER
        - Equal = neutral
        - Also consider wave position on the map
        """
        # Simple logic based on minion counts
        if order_count > chaos_count + 2:
            return "toward_CHAOS"
        elif chaos_count > order_count + 2:
            return "toward_ORDER"
        else:
            return "neutral"

    def _distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points"""
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    def _point_to_line_distance(
        self,
        point: Tuple[float, float],
        line_start: Tuple[float, float],
        line_end: Tuple[float, float]
    ) -> float:
        """
        Calculate perpendicular distance from point to line segment
        """
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end

        # Vector from line_start to line_end
        dx = x2 - x1
        dy = y2 - y1

        # Avoid division by zero
        if dx == 0 and dy == 0:
            return self._distance(point, line_start)

        # Calculate parameter t for closest point on line
        t = max(0, min(1, ((x0 - x1) * dx + (y0 - y1) * dy) / (dx * dx + dy * dy)))

        # Closest point on line segment
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy

        # Distance to closest point
        return self._distance(point, (closest_x, closest_y))
