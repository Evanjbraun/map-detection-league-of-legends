"""Detection module - exports all CV detectors"""

from detection.base import BaseDetector
from detection.champions import ChampionDetector
from detection.jungle_camps import JungleCampDetector
from detection.objectives import ObjectiveDetector
from detection.lane_states import LaneStateDetector
from detection.structures import StructureDetector

__all__ = [
    "BaseDetector",
    "ChampionDetector",
    "JungleCampDetector",
    "ObjectiveDetector",
    "LaneStateDetector",
    "StructureDetector",
]
