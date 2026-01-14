"""Pydantic models matching the JSON contract"""

from typing import List, Optional, Literal, Dict
from pydantic import BaseModel, Field


class Position(BaseModel):
    x: float = Field(..., ge=0, le=100)
    y: float = Field(..., ge=0, le=100)


class ChampionSighting(BaseModel):
    position: Position
    team: Literal["ORDER", "CHAOS", "UNKNOWN"]
    isPlayer: bool
    confidence: float = Field(..., ge=0.0, le=1.0)


class JungleCamp(BaseModel):
    type: Literal["blue_buff", "gromp", "wolves", "red_buff", "raptors", "krugs", "scuttle"]
    position: Position
    side: Literal["ORDER", "CHAOS", "TOP_RIVER", "BOT_RIVER"]
    status: Literal["alive", "cleared", "respawning"]
    respawnTimer: Optional[float] = None
    confidence: float = Field(..., ge=0.0, le=1.0)


class Objective(BaseModel):
    type: Literal["dragon", "baron", "herald"]
    position: Position
    status: Literal["alive", "dead", "respawning"]
    respawnTimer: Optional[float] = None
    confidence: float = Field(..., ge=0.0, le=1.0)


class MinionCount(BaseModel):
    ORDER: int = Field(..., ge=0)
    CHAOS: int = Field(..., ge=0)


class Minion(BaseModel):
    position: Position
    team: Literal["ORDER", "CHAOS"]
    confidence: float = Field(..., ge=0.0, le=1.0)


class LaneState(BaseModel):
    lane: Literal["top", "mid", "bot"]
    wavePosition: Position
    pushDirection: Literal["toward_ORDER", "toward_CHAOS", "neutral"]
    minionCount: MinionCount
    minions: List[Minion] = Field(default_factory=list)  # Individual minion positions
    confidence: float = Field(..., ge=0.0, le=1.0)


class TowerSummary(BaseModel):
    """Compact tower state summary for LLM consumption"""
    ORDER: Dict[str, int] = Field(default_factory=dict)  # {"top": 3, "mid": 2, "bot": 3, "nexus": 2}
    CHAOS: Dict[str, int] = Field(default_factory=dict)

class Tower(BaseModel):
    """Deprecated: Use structures field instead"""
    position: Position
    team: Literal["ORDER", "CHAOS"]
    lane: Literal["top", "mid", "bot"]
    tier: Literal["outer", "inner", "inhibitor", "nexus"]
    status: Literal["alive", "destroyed"]
    confidence: float = Field(..., ge=0.0, le=1.0)


class Structure(BaseModel):
    """Structures include towers and inhibitors"""
    position: Position
    team: Literal["ORDER", "CHAOS"]
    structureType: Literal["outer_turret", "inner_turret", "inhibitor_turret", "nexus_turret", "inhibitor"]
    lane: Literal["top", "mid", "bot", "nexus"]  # nexus for nexus turrets
    isAlive: bool
    confidence: float = Field(..., ge=0.0, le=1.0)


class MinimapResolution(BaseModel):
    width: int
    height: int


class Metadata(BaseModel):
    minimapResolution: MinimapResolution
    detectionErrors: List[str] = Field(default_factory=list)


class CVAnalysisResponse(BaseModel):
    """Main response matching JSON contract"""
    timestamp: int
    processingTimeMs: float
    playerPosition: Optional[Position] = None
    champions: List[ChampionSighting] = Field(default_factory=list)
    jungleCamps: List[JungleCamp] = Field(default_factory=list)
    objectives: List[Objective] = Field(default_factory=list)
    laneStates: List[LaneState] = Field(default_factory=list)
    towers: TowerSummary = Field(default_factory=TowerSummary)  # Compact summary for LLM
    structures: List[Structure] = Field(default_factory=list)  # Detailed tower data
    metadata: Metadata
