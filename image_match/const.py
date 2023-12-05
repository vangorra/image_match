from enum import Enum

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

MATCH_MODE_FLANN_VALUE = "flann"
MATCH_MODE_BRUTE_FORCE_VALUE = "brute_force"


class MatchMode(str, Enum):
    FLANN = MATCH_MODE_FLANN_VALUE
    BRUTE_FORCE = MATCH_MODE_BRUTE_FORCE_VALUE


DEFAULT_MATCH_MODE = MatchMode.FLANN
DEFAULT_DUMP_DIR = None
DEFAULT_MATCH_CONFIDENCE = 0.9
DEFAULT_CROP_X = 0.0
DEFAULT_CROP_Y = 0.0
DEFAULT_CROP_WIDTH = 1.0
DEFAULT_CROP_HEIGHT = 1.0

IMAGE_EXTENSIONS = tuple([".jpg", ".png", ".webp"])

FILE_SAMPLE_IMAGE = "a1_sample_image.png"
FILE_SAMPLE_IMAGE_CROPPED = "b1_sample_image_cropped.png"
FILE_REFERENCE_IMAGE = "c1_reference_image.png"

FILE_SAMPLE_IMAGE_GRAY = "d1_sample_image_gray.png"
FILE_SAMPLE_IMAGE_BLUR = "d2_sample_image_blur.png"
FILE_SAMPLE_IMAGE_THRESHOLD = "d3_sample_image_threshold.png"

FILE_REFERENCE_IMAGE_GRAY = "e1_reference_image_gray.png"
FILE_REFERENCE_IMAGE_BLUR = "e2_reference_image_blur.png"
FILE_REFERENCE_IMAGE_THRESHOLD = "e3_reference_image_threshold.png"

FILE_RESULT_DEBUG_IMAGE = "f1_result_debug_image.png"

DUMP_FILE_NAMES = [
    FILE_SAMPLE_IMAGE,
    FILE_SAMPLE_IMAGE_CROPPED,
    FILE_REFERENCE_IMAGE,
    FILE_SAMPLE_IMAGE_GRAY,
    FILE_SAMPLE_IMAGE_BLUR,
    FILE_SAMPLE_IMAGE_THRESHOLD,
    FILE_REFERENCE_IMAGE_GRAY,
    FILE_REFERENCE_IMAGE_BLUR,
    FILE_REFERENCE_IMAGE_THRESHOLD,
    FILE_RESULT_DEBUG_IMAGE,
]
