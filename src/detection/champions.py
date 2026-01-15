"""Champion detection using Riot API + template matching"""

from typing import List, Dict, Optional
import cv2
import numpy as np
from pathlib import Path
from loguru import logger

# Optional imports for Riot API (graceful fallback if not installed)
try:
    import requests
    import urllib3
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests library not available - champion detection will be disabled")

from detection.base import BaseDetector
from api.schemas import ChampionSighting, Position
from config import settings


class ChampionDetector(BaseDetector):
    """
    Detects champions on the minimap using optimized template matching

    Strategy:
    1. Fetch champion list from Riot in-game API (local, no auth needed)
    2. Load only the 10 champion templates for current game (16x faster!)
    3. Use template matching to find champions on minimap
    4. Classify team based on position (diagonal split)

    Optimization: Instead of matching 160+ champions, we only match the 10 in the game
    """

    def __init__(self):
        super().__init__()
        self.templates: Dict[str, np.ndarray] = {}  # champion_name -> grayscale template
        self.champion_names: List[str] = []
        self.champion_teams: Dict[str, str] = {}  # champion_name -> "ORDER" or "CHAOS"
        self.match_threshold = 0.75  # Very strict threshold - champions must match well to avoid false positives

        # Riot API state
        self.api_fetch_attempted = False
        self.api_available = False

    def initialize(self) -> None:
        """Initialize champion detector and load templates"""
        logger.info("Initializing ChampionDetector...")
        super().initialize()

        # Try to fetch champions from Riot in-game API
        self.champion_names = self._fetch_champions_from_riot_api()

        if self.champion_names:
            logger.info(f"✅ Fetched {len(self.champion_names)} champions from Riot API")
            logger.info(f"📋 Champions in game: {', '.join(self.champion_names)}")
            self.api_available = True
        else:
            logger.warning("❌ Riot API unavailable - will use fallback strategy")
            logger.warning("⚠️  This will load ALL champion templates (slow performance!)")
            self.api_available = False

        # Load champion templates
        if self.champion_names:
            logger.info(f"🎯 Loading ONLY {len(self.champion_names)} champion templates (optimized)")
            self._load_champion_templates(self.champion_names)
        else:
            # DISABLED: Don't load all templates - it's too slow
            # Instead, skip champion detection until Riot API is available
            logger.error("❌ Champion detection DISABLED - Riot API unavailable")
            logger.error("💡 Start a League of Legends game to enable champion detection")
            logger.error("💡 The Riot API (127.0.0.1:2999) is only available during active games")
            return  # Exit early, no templates loaded

        logger.info(f"✅ ChampionDetector ready - loaded {len(self.templates)} templates in memory")
        if len(self.templates) > 15:
            logger.error(f"🚨 WARNING: Loaded {len(self.templates)} templates! Expected only 10. This will cause performance issues!")
            logger.error("🚨 Make sure League of Legends is running with a game in progress for Riot API to work")

    def _fetch_champions_from_riot_api(self) -> List[str]:
        """
        Fetch champion names and teams from Riot's local in-game API

        The League client exposes game data at https://127.0.0.1:2999/liveclientdata/
        This is only available when a game is running.

        Returns:
            List of champion names in the current game (e.g., ["Yasuo", "Ahri", ...])
            Also populates self.champion_teams mapping
        """
        if self.api_fetch_attempted:
            return []

        self.api_fetch_attempted = True

        # Check if requests library is available
        if not REQUESTS_AVAILABLE:
            logger.error("❌ Riot API failed: 'requests' library not installed in venv")
            logger.error("   Install with: pip install requests urllib3")
            return []

        try:
            # Disable SSL warnings (Riot uses self-signed cert)
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

            url = "https://127.0.0.1:2999/liveclientdata/playerlist"
            logger.debug(f"Attempting to fetch champions from Riot API: {url}")

            response = requests.get(url, verify=False, timeout=3)

            if response.status_code == 200:
                data = response.json()
                champions = []

                # Extract champion names and teams from all players
                for player in data:
                    champ_name = player.get('championName', '')
                    team = player.get('team', '')  # "ORDER" or "CHAOS"

                    if champ_name:
                        champions.append(champ_name)
                        # Store team mapping for later use
                        if team in ["ORDER", "CHAOS"]:
                            self.champion_teams[champ_name] = team

                logger.info(f"✅ Riot API success: fetched {len(champions)} champions")
                logger.debug(f"Team mappings: {self.champion_teams}")
                return champions
            else:
                logger.warning(f"❌ Riot API returned HTTP {response.status_code}")
                logger.warning(f"   Make sure a League of Legends game is running")
                return []

        except requests.exceptions.ConnectionError as e:
            logger.warning(f"❌ Riot API connection failed: Unable to connect to 127.0.0.1:2999")
            logger.warning(f"   This is normal if no League game is running")
            logger.debug(f"   Connection error details: {e}")
            return []
        except requests.exceptions.Timeout:
            logger.warning(f"❌ Riot API timeout: No response from 127.0.0.1:2999 after 3 seconds")
            logger.warning(f"   Make sure League of Legends is running with an active game")
            return []
        except Exception as e:
            logger.error(f"❌ Riot API unexpected error: {type(e).__name__}: {e}")
            logger.debug(f"   Full error details:", exc_info=True)
            return []

    def _normalize_champion_name(self, champ_name: str) -> str:
        """
        Normalize champion name for template filename matching

        Removes spaces, apostrophes, and converts to lowercase
        Examples:
        - "Kai'Sa" -> "kaisa"
        - "Cho'Gath" -> "chogath"
        - "Lee Sin" -> "leesin"
        - "Twisted Fate" -> "twistedfate"

        Args:
            champ_name: Original champion name from Riot API

        Returns:
            Normalized filename-safe champion name
        """
        return champ_name.replace("'", "").replace(" ", "").lower()

    def _process_champion_template(self, img: np.ndarray, champ_name: str) -> Optional[np.ndarray]:
        """
        Process champion portrait template for minimap matching

        Champion portraits are circular with decorative frames. The minimap shows
        these as small circular icons. We crop to the center circle and resize.

        Args:
            img: BGR image of champion portrait
            champ_name: Champion name for logging

        Returns:
            Grayscale, cropped, resized template ready for matching
        """
        try:
            h, w = img.shape[:2]

            # Crop to center 70% (removes outer frame/border)
            crop_percent = 0.70
            crop_size = int(min(h, w) * crop_percent)
            center_x, center_y = w // 2, h // 2
            half_crop = crop_size // 2

            x1 = max(0, center_x - half_crop)
            y1 = max(0, center_y - half_crop)
            x2 = min(w, center_x + half_crop)
            y2 = min(h, center_y + half_crop)

            cropped = img[y1:y2, x1:x2]

            # Resize to minimap icon size (experiment with 20, 25, 30 pixels)
            # Start with 25px as a good middle ground
            target_size = 25
            resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_AREA)

            # Convert to grayscale for template matching
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

            logger.debug(f"  Processed {champ_name}: {w}x{h} → crop to {crop_size}x{crop_size} → resize to {target_size}x{target_size}")

            return gray

        except Exception as e:
            logger.error(f"Failed to process template for {champ_name}: {e}")
            return None

    def _load_champion_templates(self, champion_names: List[str]) -> None:
        """
        Load only specific champion templates (optimized for performance)

        Args:
            champion_names: List of champion names to load (e.g., ["Yasuo", "Ahri"])
        """
        from config import PROJECT_ROOT

        templates_dir = PROJECT_ROOT / "models" / "templates" / "champions"

        if not templates_dir.exists():
            logger.error(f"Champions template directory not found: {templates_dir}")
            return

        for champ_name in champion_names:
            # Normalize the champion name for filename matching
            normalized_name = self._normalize_champion_name(champ_name)

            # Try different file extensions with normalized name
            possible_names = [
                f"{normalized_name}.jpg",   # "kaisa.jpg"
                f"{normalized_name}.png",   # "kaisa.png"
            ]

            for filename in possible_names:
                template_path = templates_dir / filename
                if template_path.exists():
                    img = cv2.imread(str(template_path))
                    if img is not None:
                        # Process template: crop to center circle and resize to minimap size
                        processed_template = self._process_champion_template(img, champ_name)
                        if processed_template is not None:
                            self.templates[champ_name] = processed_template
                            logger.info(f"Loaded champion template: {champ_name} -> {normalized_name} ({img.shape[1]}x{img.shape[0]} → {processed_template.shape[1]}x{processed_template.shape[0]} px)")
                        break
            else:
                logger.warning(f"Template not found for champion: {champ_name} (tried: {normalized_name}.jpg/.png)")

    def _load_all_champion_templates(self) -> None:
        """
        Load all champion templates as fallback (slower but always works)

        Used when Riot API is unavailable (not in game yet)
        """
        from config import PROJECT_ROOT

        templates_dir = PROJECT_ROOT / "models" / "templates" / "champions"

        if not templates_dir.exists():
            logger.error(f"Champions template directory not found: {templates_dir}")
            return

        # Load all .jpg and .png files
        for template_file in templates_dir.glob("*"):
            if template_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue

            champ_name = template_file.stem
            img = cv2.imread(str(template_file))

            if img is not None:
                self.templates[champ_name] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                logger.warning(f"Failed to load template: {template_file}")

        logger.info(f"Loaded {len(self.templates)} champion templates (all available)")

    def detect(self, minimap: np.ndarray) -> List[ChampionSighting]:
        """
        Detect champions on the minimap

        Args:
            minimap: BGR image of the minimap

        Returns:
            List of detected ChampionSighting objects
        """
        if minimap is None or minimap.size == 0:
            return []

        if not self.templates:
            logger.warning("No templates loaded, skipping champion detection")
            return []

        champions = []

        try:
            height, width = minimap.shape[:2]

            # Convert to grayscale for template matching
            minimap_gray = cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY)

            # Collect all detections from all templates
            all_detections = []

            for champ_name, template in self.templates.items():
                detections = self._match_template(minimap_gray, template, champ_name)
                all_detections.extend([(x, y, conf, champ_name) for x, y, conf in detections])

            # Apply NMS across all detections to remove overlapping duplicates
            all_detections = self._non_max_suppression(all_detections, threshold=20)

            # CRITICAL: Keep only the best detection for each champion
            # A champion can only be in one location at a time
            best_per_champion = {}
            for x, y, conf, champ_name in all_detections:
                if champ_name not in best_per_champion or conf > best_per_champion[champ_name][2]:
                    best_per_champion[champ_name] = (x, y, conf, champ_name)

            # Convert back to list
            all_detections = list(best_per_champion.values())

            # Convert to ChampionSighting objects with normalized coordinates
            for x, y, conf, champ_name in all_detections:
                x_norm = (x / width) * 100
                y_norm = (y / height) * 100

                # Get team from Riot API mapping, fallback to diagonal classification
                team = self.champion_teams.get(champ_name, None)
                if team is None:
                    # Fallback to diagonal classification if API data unavailable
                    team = self._classify_team(x_norm, y_norm)

                # TODO: Detect which champion is the player
                is_player = False

                champions.append(ChampionSighting(
                    championName=champ_name,
                    position=Position(x=x_norm, y=y_norm),
                    team=team,
                    isPlayer=is_player,
                    confidence=float(conf)
                ))

            if champions:
                logger.debug(f"✅ Returning {len(champions)} champion sightings")
                for champ in champions:
                    logger.debug(f"   - {champ.championName} at ({champ.position.x:.1f}, {champ.position.y:.1f}) team={champ.team} conf={champ.confidence:.2f}")

        except Exception as e:
            logger.error(f"Error in champion detection: {e}")
            import traceback
            traceback.print_exc()

        return champions

    def _match_template(self, minimap_gray: np.ndarray, template: np.ndarray,
                       champ_name: str) -> List[tuple]:
        """
        Match a single champion template against the minimap

        Args:
            minimap_gray: Grayscale minimap
            template: Grayscale template to match
            champ_name: Champion name for logging

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

    def _classify_team(self, x_norm: float, y_norm: float) -> str:
        """
        Classify team based on position using diagonal split

        Same logic as jungle camps:
        - ORDER (blue team): bottom-left side (y > x)
        - CHAOS (red team): top-right side (y < x)

        Args:
            x_norm: Normalized X coordinate (0-100)
            y_norm: Normalized Y coordinate (0-100)

        Returns:
            "ORDER", "CHAOS", or "UNKNOWN"
        """
        # Calculate distance from diagonal line (y = x)
        diagonal_distance = abs(y_norm - x_norm)

        # If very close to diagonal, hard to determine team
        if diagonal_distance < 5:
            return "UNKNOWN"

        # ORDER (blue): bottom-left (y > x)
        # CHAOS (red): top-right (y < x)
        if y_norm > x_norm:
            return "ORDER"
        else:
            return "CHAOS"

    def _non_max_suppression(self, detections: List[tuple], threshold: int = 20) -> List[tuple]:
        """
        Remove duplicate detections that are too close together

        Args:
            detections: List of (x, y, confidence, champ_name) tuples
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
