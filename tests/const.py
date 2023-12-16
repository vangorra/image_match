import re
from pathlib import Path

TEMP_DIR = Path("./build/tmp").absolute()
REFERENCE_DIR = Path("./reference/chicken_door/").absolute()
REFERENCE_SOURCE_SAMPLE_DIR = REFERENCE_DIR.joinpath("sample").absolute()
REFERENCE_SAMPLE_CLOSED_DAY = REFERENCE_SOURCE_SAMPLE_DIR.joinpath("closed_day.png")
REFERENCE_SAMPLE_CLOSED_NIGHT = REFERENCE_SOURCE_SAMPLE_DIR.joinpath("closed_night.png")
REFERENCE_SAMPLE_OPEN_DAY = REFERENCE_SOURCE_SAMPLE_DIR.joinpath("open_day.png")
REFERENCE_SAMPLE_OPEN_NIGHT = REFERENCE_SOURCE_SAMPLE_DIR.joinpath("open_night.png")

PATTERN_MATCH = re.compile('"is_match": true')
PATTERN_NO_MATCH = re.compile('"is_match": false')

DUMP_DIR = TEMP_DIR.joinpath("dump")

DUMP_FILE_NAMES_MATCH = (
    "debug.png",
    "reference_gray_blur.png",
    "reference_gray_blur_threshold.png",
    "reference_gray.png",
    "reference.png",
    "reference_result.png",
    "sample_cropped.png",
    "sample_gray_blur.png",
    "sample_gray_blur_threshold.png",
    "sample_gray.png",
    "sample.png",
    "sample_result.png",
)

DUMP_FILE_NAMES_NO_MATCH = (
    "sample_cropped.png",
    "sample_gray_blur.png",
    "sample_gray_blur_threshold.png",
    "sample_gray.png",
    "sample.png",
    "sample_result.png",
)
