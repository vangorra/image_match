from dataclasses import dataclass, field
import logging
from pathlib import Path
import sys
from typing import Dict, Optional

from image_match.const import (
    DEFAULT_CROP_HEIGHT,
    DEFAULT_CROP_WIDTH,
    DEFAULT_CROP_X,
    DEFAULT_CROP_Y,
    DEFAULT_DUMP_MODE,
    DEFAULT_MATCH_CONFIDENCE,
    DEFAULT_MATCH_MODE,
    LOGLEVEL_TRACE,
    DumpMode,
    LogLevel,
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
    dump_mode: DumpMode = DEFAULT_DUMP_MODE


@dataclass(frozen=True)
class ServeConfig:
    """
    Configuration for starting a REST API.
    """

    debug: bool = False
    match_configs: Dict[str, MatchConfig] = field(default_factory=dict)


def logging_trace(message: str) -> None:
    logging.log(LOGLEVEL_TRACE, message)


def maybe_set_log_level(log_level_str_value: str) -> None:
    for entry in LogLevel:
        if entry.value.str_value == log_level_str_value:
            logging.addLevelName(LOGLEVEL_TRACE, "TRACE")
            logging.basicConfig(
                format="%(asctime)s [%(levelname)s] %(message)s",
                level=entry.value.int_value,
                datefmt="%Y-%m-%d %H:%M:%S",
                stream=sys.stderr,
            )
            return
