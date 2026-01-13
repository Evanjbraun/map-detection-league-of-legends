"""
Ultra-fast screen capture using MSS (100+ FPS capable)
"""

import asyncio
from typing import Optional, Tuple
from pathlib import Path

import mss
import numpy as np
from PIL import Image
from loguru import logger

from config import settings


class ScreenCapture:
    """High-performance screen capture"""

    def __init__(self):
        self.sct: Optional[mss.mss] = None
        self.monitor_info: Optional[dict] = None

    async def initialize(self):
        """Initialize screen capture"""
        self.sct = mss.mss()

        # Get primary monitor info
        self.monitor_info = self.sct.monitors[1]  # 0 = all monitors, 1 = primary
        logger.info(f"Primary monitor: {self.monitor_info['width']}x{self.monitor_info['height']}")

    async def cleanup(self):
        """Cleanup resources"""
        if self.sct:
            self.sct.close()

    async def capture_region(self, x: int, y: int, width: int, height: int) -> Optional[Image.Image]:
        """
        Capture a specific screen region

        Args:
            x, y: Top-left corner coordinates
            width, height: Region dimensions

        Returns:
            PIL Image or None if capture fails
        """
        if not self.sct:
            return None

        try:
            # Define capture region
            monitor = {
                "top": y,
                "left": x,
                "width": width,
                "height": height
            }

            # Capture (this is extremely fast with MSS)
            screenshot = self.sct.grab(monitor)

            # Convert to PIL Image
            img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")

            return img

        except Exception as e:
            logger.error(f"Screen capture failed: {e}")
            return None

    async def capture_full_screen(self) -> Optional[Image.Image]:
        """Capture entire primary monitor"""
        if not self.sct or not self.monitor_info:
            return None

        return await self.capture_region(
            self.monitor_info["left"],
            self.monitor_info["top"],
            self.monitor_info["width"],
            self.monitor_info["height"]
        )

    @staticmethod
    def needs_calibration() -> bool:
        """
        Check if minimap calibration is needed

        Returns True if:
        - Coordinates are default (0, 0)
        - Size is too small
        """
        return (
            settings.MINIMAP_X == 0 and
            settings.MINIMAP_Y == 0
        ) or settings.MINIMAP_WIDTH < 100 or settings.MINIMAP_HEIGHT < 100

    @staticmethod
    def calibrate_minimap() -> Tuple[int, int, int, int]:
        """
        Run calibration UI and save coordinates to .env

        Returns:
            (x, y, width, height) tuple
        """
        from capture.calibration import calibrate_minimap_region

        logger.info("🎯 Minimap calibration required")

        # Run calibration
        x, y, width, height = calibrate_minimap_region(
            settings.MINIMAP_X,
            settings.MINIMAP_Y,
            settings.MINIMAP_WIDTH,
            settings.MINIMAP_HEIGHT
        )

        # Save to .env
        env_path = Path(__file__).parent.parent / '.env'

        try:
            if env_path.exists():
                with open(env_path, 'r') as f:
                    lines = f.readlines()
            else:
                lines = []

            # Update values
            updated = {'MINIMAP_X': False, 'MINIMAP_Y': False, 'MINIMAP_WIDTH': False, 'MINIMAP_HEIGHT': False}

            for i, line in enumerate(lines):
                if line.startswith('MINIMAP_X='):
                    lines[i] = f'MINIMAP_X={x}\n'
                    updated['MINIMAP_X'] = True
                elif line.startswith('MINIMAP_Y='):
                    lines[i] = f'MINIMAP_Y={y}\n'
                    updated['MINIMAP_Y'] = True
                elif line.startswith('MINIMAP_WIDTH='):
                    lines[i] = f'MINIMAP_WIDTH={width}\n'
                    updated['MINIMAP_WIDTH'] = True
                elif line.startswith('MINIMAP_HEIGHT='):
                    lines[i] = f'MINIMAP_HEIGHT={height}\n'
                    updated['MINIMAP_HEIGHT'] = True

            # Add missing lines
            if not updated['MINIMAP_X']:
                lines.append(f'MINIMAP_X={x}\n')
            if not updated['MINIMAP_Y']:
                lines.append(f'MINIMAP_Y={y}\n')
            if not updated['MINIMAP_WIDTH']:
                lines.append(f'MINIMAP_WIDTH={width}\n')
            if not updated['MINIMAP_HEIGHT']:
                lines.append(f'MINIMAP_HEIGHT={height}\n')

            with open(env_path, 'w') as f:
                f.writelines(lines)

            logger.success(f"✅ Minimap coordinates saved: ({x}, {y}) {width}x{height}")

        except Exception as e:
            logger.error(f"Failed to save calibration: {e}")

        return (x, y, width, height)
