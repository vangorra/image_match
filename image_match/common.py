from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from image_match.const import (
    DEFAULT_CROP_HEIGHT,
    DEFAULT_CROP_WIDTH,
    DEFAULT_CROP_X,
    DEFAULT_CROP_Y,
    DEFAULT_MATCH_CONFIDENCE,
    DEFAULT_MATCH_MODE,
    MatchMode,
)


@dataclass(frozen=True)
class TransformConfig:
    x: float = DEFAULT_CROP_X
    y: float = DEFAULT_CROP_Y
    width: float = DEFAULT_CROP_WIDTH
    height: float = DEFAULT_CROP_HEIGHT


@dataclass(frozen=True)
class MatchConfig:
    reference_dir: Path
    sample_url: str
    transform_config: TransformConfig = TransformConfig()
    match_mode: MatchMode = DEFAULT_MATCH_MODE
    dump_dir: Optional[Path] = None
    match_confidence: float = DEFAULT_MATCH_CONFIDENCE


@dataclass(frozen=True)
class DoMatchResult:
    is_match: bool
    run_duration: float
    get_image_duration: float
    check_count: int
    reference_image_path: Optional[str]


@dataclass(frozen=True)
class ServeConfig:
    """
    Configuration for starting a REST API.
    """

    debug: bool = False
    match_configs: Dict[str, MatchConfig] = field(default_factory=dict)
