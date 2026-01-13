"""Test template matching to debug tower detection"""

import cv2
import numpy as np
from pathlib import Path

# Load a sample minimap (you'll need to capture one first)
PROJECT_ROOT = Path(__file__).parent

# Try to load the minimap.png if it exists
minimap_path = PROJECT_ROOT / "minimap.png"

if not minimap_path.exists():
    print(f"ERROR: Please capture a minimap image and save it as minimap.png in the project root")
    exit(1)

minimap = cv2.imread(str(minimap_path))
print(f"Minimap loaded: {minimap.shape}")

# Load templates
templates_dir = PROJECT_ROOT / "models" / "templates" / "towers"

blue_templates = []
red_templates = []

for template_file in templates_dir.glob("*.jpg"):
    img = cv2.imread(str(template_file))
    if img is None:
        continue

    filename = template_file.name.lower()
    if "blue" in filename:
        blue_templates.append((img, template_file.name))
    elif "red" in filename:
        red_templates.append((img, template_file.name))

print(f"Loaded {len(blue_templates)} blue templates, {len(red_templates)} red templates")

# Test template matching with different thresholds
thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

for threshold in thresholds:
    print(f"\n=== Testing with threshold {threshold} ===")

    total_matches = 0

    # Test blue templates
    for template_img, template_name in blue_templates[:3]:  # Test first 3
        result = cv2.matchTemplate(minimap, template_img, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= threshold)
        matches = len(locations[0])

        if matches > 0:
            max_conf = result.max()
            print(f"  {template_name}: {matches} matches (max confidence: {max_conf:.3f})")
            total_matches += matches

    # Test red templates
    for template_img, template_name in red_templates[:3]:  # Test first 3
        result = cv2.matchTemplate(minimap, template_img, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= threshold)
        matches = len(locations[0])

        if matches > 0:
            max_conf = result.max()
            print(f"  {template_name}: {matches} matches (max confidence: {max_conf:.3f})")
            total_matches += matches

    print(f"  Total matches: {total_matches}")

print("\n=== Visualizing best matches (threshold 0.6) ===")

# Create visualization
output = minimap.copy()

for template_img, template_name in (blue_templates + red_templates):
    result = cv2.matchTemplate(minimap, template_img, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= 0.6)

    template_h, template_w = template_img.shape[:2]

    for pt in zip(*locations[::-1]):
        x, y = pt
        confidence = result[y, x]

        # Draw rectangle
        color = (255, 0, 0) if "blue" in template_name.lower() else (0, 0, 255)
        cv2.rectangle(output, pt, (x + template_w, y + template_h), color, 2)

        # Draw center point
        center_x = x + template_w // 2
        center_y = y + template_h // 2
        cv2.circle(output, (center_x, center_y), 3, (0, 255, 0), -1)

# Save visualization
output_path = PROJECT_ROOT / "template_matching_debug.png"
cv2.imwrite(str(output_path), output)
print(f"\nVisualization saved to: {output_path}")
