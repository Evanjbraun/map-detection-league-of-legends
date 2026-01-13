"""Base detector class for all CV detectors"""

from abc import ABC, abstractmethod
from typing import Any, List
import cv2
import numpy as np
from loguru import logger


class BaseDetector(ABC):
    """Abstract base class for all detectors"""

    def __init__(self):
        self.is_initialized = False

    def initialize(self) -> None:
        """
        Initialize detector (load templates, models, calibrate, etc.)
        Override this method if detector needs initialization
        """
        self.is_initialized = True
        logger.debug(f"{self.__class__.__name__} initialized")

    @abstractmethod
    def detect(self, minimap: np.ndarray) -> Any:
        """
        Run detection on minimap image

        Args:
            minimap: OpenCV image (BGR numpy array)

        Returns:
            Detection results (type varies by detector)
        """
        pass

    # Utility methods shared by all detectors

    @staticmethod
    def convert_to_hsv(image: np.ndarray) -> np.ndarray:
        """Convert BGR image to HSV color space"""
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    @staticmethod
    def apply_color_mask(hsv_image: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        """
        Apply color range mask to HSV image

        Args:
            hsv_image: Image in HSV color space
            lower: Lower bound HSV values [H, S, V]
            upper: Upper bound HSV values [H, S, V]

        Returns:
            Binary mask (white = in range, black = out of range)
        """
        return cv2.inRange(hsv_image, lower, upper)

    @staticmethod
    def find_blobs(mask: np.ndarray, min_area: float = 5.0) -> List[dict]:
        """
        Find connected components (blobs) in binary mask

        Args:
            mask: Binary mask image
            min_area: Minimum blob area to consider

        Returns:
            List of blob info dicts with keys: center (x, y), area, contour
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        blobs = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                # Calculate centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    blobs.append({
                        "center": (cx, cy),
                        "area": area,
                        "contour": contour
                    })

        return blobs

    @staticmethod
    def scale_coordinates(x: float, y: float, source_width: int, source_height: int) -> tuple:
        """
        Scale pixel coordinates to normalized 0-100 range

        Args:
            x, y: Pixel coordinates
            source_width, source_height: Image dimensions

        Returns:
            (x_normalized, y_normalized) in range [0, 100]
        """
        x_norm = (x / source_width) * 100.0
        y_norm = (y / source_height) * 100.0
        return (x_norm, y_norm)
