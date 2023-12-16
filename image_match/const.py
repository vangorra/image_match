from dataclasses import dataclass
from enum import Enum
import logging

ARG_SAMPLE_URL = "sample-url"
ARG_OUTPUT_FILE = "output-file"
ARG_CROP_X = "crop-x"
ARG_CROP_Y = "crop-y"
ARG_CROP_WIDTH = "crop-width"
ARG_CROP_HEIGHT = "crop-height"
ARG_REFERENCE_DIR = "reference-dir"
ARG_MATCH_MODE = "match-mode"
ARG_DUMP_DIR = "dump-dir"
ARG_MATCH_CONFIDENCE = "match-confidence"
ARG_CONFIG = "config"
ARG_LOG_LEVEL = "log-level"
ARG_DUMP_MODE = "dump-mode"

MATCH_MODE_FLANN_VALUE = "flann"
MATCH_MODE_BRUTE_FORCE_VALUE = "brute_force"


class MatchMode(str, Enum):
    FLANN = MATCH_MODE_FLANN_VALUE
    BRUTE_FORCE = MATCH_MODE_BRUTE_FORCE_VALUE


@dataclass(frozen=True)
class LoggingLevelMap:
    str_value: str
    int_value: int


LOGLEVEL_TRACE = 9


class LogLevel(Enum):
    CRITICAL = LoggingLevelMap(str_value="critical", int_value=logging.CRITICAL)
    ERROR = LoggingLevelMap(str_value="error", int_value=logging.ERROR)
    WARN = LoggingLevelMap(str_value="warn", int_value=logging.WARN)
    INFO = LoggingLevelMap(str_value="info", int_value=logging.INFO)
    DEBUG = LoggingLevelMap(str_value="debug", int_value=logging.DEBUG)
    TRACE = LoggingLevelMap(str_value="trace", int_value=LOGLEVEL_TRACE)


class DumpMode(str, Enum):
    NO_MATCH = "no-match"
    MATCH = "match"
    BOTH = "both"


DEFAULT_MATCH_MODE = MatchMode.FLANN
DEFAULT_DUMP_DIR = None
DEFAULT_MATCH_CONFIDENCE = 0.9
DEFAULT_CROP_X = 0.0
DEFAULT_CROP_Y = 0.0
DEFAULT_CROP_WIDTH = 1.0
DEFAULT_CROP_HEIGHT = 1.0
DEFAULT_LOG_LEVEL = LogLevel.INFO
DEFAULT_DUMP_MODE = DumpMode.BOTH

IMAGE_EXTENSIONS = tuple([".jpg", ".png", ".webp"])
