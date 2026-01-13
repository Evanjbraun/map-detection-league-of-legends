"""
Automatic minimap detection using edge/shape-based computer vision
Robust to GPU settings changes (saturation, contrast, gamma)
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from loguru import logger


def detect_minimap_region(screenshot: np.ndarray,
                          expected_size_range: Tuple[int, int] = (200, 600)) -> Optional[Tuple[int, int, int, int]]:
    """
    Auto-detect minimap region using edge detection and contour analysis

    This approach is color-agnostic (robust to GPU settings) and looks for:
    1. Dark border around minimap
    2. Rectangular/square shape
    3. Size in expected range (typically 200-600px)
    4. Position in bottom-right quadrant
    5. Color variation inside (not solid color)

    Args:
        screenshot: Full screen capture (BGR format from OpenCV)
        expected_size_range: (min_size, max_size) in pixels

    Returns:
        (x, y, width, height) tuple or None if detection fails
    """
    if screenshot is None or screenshot.size == 0:
        logger.error("Invalid screenshot for auto-detection")
        return None

    height, width = screenshot.shape[:2]
    logger.debug(f"Screenshot size: {width}x{height}")

    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edge detection
    # These thresholds work well for detecting dark borders
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    logger.debug(f"Found {len(contours)} contours")

    # Filter candidates
    candidates = []

    for contour in contours:
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)

        # Filter by size
        min_size, max_size = expected_size_range
        if w < min_size or h < min_size or w > max_size or h > max_size:
            continue

        # Filter by aspect ratio (should be roughly square)
        # League minimap is approximately 1:1 aspect ratio
        aspect_ratio = w / h
        if aspect_ratio < 0.8 or aspect_ratio > 1.25:
            continue

        # Filter by position (bottom-right quadrant)
        # Minimap is typically in the bottom 60% and right 60% of screen
        if x < width * 0.4 or y < height * 0.4:
            continue

        # Calculate contour area vs bounding box area
        # A good rectangular border should have high fill ratio
        contour_area = cv2.contourArea(contour)
        bbox_area = w * h
        fill_ratio = contour_area / bbox_area if bbox_area > 0 else 0

        # Minimap border should have reasonable fill ratio
        if fill_ratio < 0.5:
            continue

        # Check color variation inside the region
        roi = screenshot[y:y+h, x:x+w]
        if not _has_color_variation(roi):
            continue

        # Passed all filters - add as candidate
        candidates.append({
            'region': (x, y, w, h),
            'score': _score_candidate(x, y, w, h, width, height, fill_ratio)
        })

        logger.debug(f"Candidate: ({x}, {y}) {w}x{h}, fill={fill_ratio:.2f}")

    if not candidates:
        logger.warning("No minimap candidates found")
        return None

    # Sort by score (highest first)
    candidates.sort(key=lambda c: c['score'], reverse=True)

    # Return best candidate
    best = candidates[0]
    x, y, w, h = best['region']
    logger.success(f"Detected minimap: ({x}, {y}) {w}x{h} (score: {best['score']:.2f})")

    return (x, y, w, h)


def _has_color_variation(roi: np.ndarray, threshold: float = 20.0) -> bool:
    """
    Check if region has color variation (not solid color)

    Args:
        roi: Region of interest (BGR image)
        threshold: Minimum standard deviation to consider varied

    Returns:
        True if region has sufficient color variation
    """
    if roi.size == 0:
        return False

    # Calculate standard deviation across all channels
    std_dev = np.std(roi)

    # Minimap should have varied colors (terrain, champions, icons)
    # Solid color regions will have very low std dev
    return std_dev > threshold


def _score_candidate(x: int, y: int, w: int, h: int,
                     screen_width: int, screen_height: int,
                     fill_ratio: float) -> float:
    """
    Score a candidate region based on typical minimap characteristics

    Higher score = more likely to be the minimap

    Scoring factors:
    - Position (prefer bottom-right corner)
    - Size (prefer ~250-500px range)
    - Aspect ratio (prefer square)
    - Fill ratio (prefer ~0.8-0.95)

    Args:
        x, y, w, h: Candidate region
        screen_width, screen_height: Full screen dimensions
        fill_ratio: Contour area / bounding box area

    Returns:
        Score value (higher is better)
    """
    score = 0.0

    # Position score (bottom-right is best)
    # Minimap is typically at the very bottom-right
    right_edge = x + w
    bottom_edge = y + h

    # Distance from bottom-right corner (normalized)
    dist_from_corner = (
        ((screen_width - right_edge) ** 2 + (screen_height - bottom_edge) ** 2) ** 0.5
    ) / (screen_width + screen_height)

    # Closer to corner = higher score
    position_score = 1.0 - min(dist_from_corner, 1.0)
    score += position_score * 40  # Weight: 40 points

    # Size score (prefer ~250-500px)
    ideal_size = 400
    size_diff = abs(w - ideal_size) + abs(h - ideal_size)
    size_score = max(0, 1.0 - (size_diff / 500))
    score += size_score * 30  # Weight: 30 points

    # Aspect ratio score (prefer square)
    aspect_ratio = w / h
    aspect_diff = abs(aspect_ratio - 1.0)
    aspect_score = max(0, 1.0 - aspect_diff)
    score += aspect_score * 20  # Weight: 20 points

    # Fill ratio score (prefer ~0.85)
    fill_diff = abs(fill_ratio - 0.85)
    fill_score = max(0, 1.0 - (fill_diff * 2))
    score += fill_score * 10  # Weight: 10 points

    return score


def test_detection_on_screenshot(screenshot_path: str, debug_output: bool = False) -> Optional[Tuple[int, int, int, int]]:
    """
    Test auto-detection on a saved screenshot

    Args:
        screenshot_path: Path to screenshot file
        debug_output: If True, save annotated image showing detection

    Returns:
        Detected region or None
    """
    # Read screenshot
    img = cv2.imread(screenshot_path)
    if img is None:
        logger.error(f"Failed to load screenshot: {screenshot_path}")
        return None

    logger.info(f"Testing detection on {screenshot_path}")

    # Run detection
    region = detect_minimap_region(img)

    # Save debug output if requested
    if debug_output and region is not None:
        x, y, w, h = region

        # Draw rectangle on image
        annotated = img.copy()
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 3)

        # Add text
        text = f"Minimap: ({x}, {y}) {w}x{h}"
        cv2.putText(annotated, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (0, 255, 0), 2)

        # Save
        output_path = screenshot_path.replace('.', '_detected.')
        cv2.imwrite(output_path, annotated)
        logger.info(f"Saved annotated image to {output_path}")

    return region
