import abc
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, cast

import cv2
from numpy import uint8
from numpy.typing import NDArray

from image_match.const import (
    DEFAULT_CROP_HEIGHT,
    DEFAULT_CROP_WIDTH,
    DEFAULT_CROP_X,
    DEFAULT_CROP_Y,
    DEFAULT_MATCH_CONFIDENCE,
    DEFAULT_MATCH_MODE,
    DUMP_FILE_NAMES,
    FILE_REFERENCE_IMAGE,
    FILE_REFERENCE_IMAGE_BLUR,
    FILE_REFERENCE_IMAGE_GRAY,
    FILE_REFERENCE_IMAGE_THRESHOLD,
    FILE_RESULT_DEBUG_IMAGE,
    FILE_SAMPLE_IMAGE,
    FILE_SAMPLE_IMAGE_BLUR,
    FILE_SAMPLE_IMAGE_CROPPED,
    FILE_SAMPLE_IMAGE_GRAY,
    FILE_SAMPLE_IMAGE_THRESHOLD,
    IMAGE_EXTENSIONS,
    MatchMode,
)

Cv2Image = NDArray[uint8]


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
class AbsDoMatchResult(abc.ABC):
    sample_image: Cv2Image
    sample_image_cropped: Cv2Image
    is_match: bool
    run_duration: float
    get_image_duration: float
    check_count: int


@dataclass(frozen=True)
class DoMatchPositiveResult(AbsDoMatchResult):
    reference_image: Cv2Image
    reference_image_path: Path


@dataclass(frozen=True)
class DoMatchNegativeResult(AbsDoMatchResult):
    pass


class Scanner(abc.ABC):
    """
    Threshold code sourced from: https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
    """

    def __init__(self, options: MatchConfig) -> None:
        super().__init__()
        self._options = options

    def _get_image(self) -> Cv2Image:
        return fetch_image(self._options.sample_url)

    def do_match(self) -> AbsDoMatchResult:
        start_time = time.perf_counter()
        reference_dir = self._options.reference_dir
        reference_file_paths: List[Path] = [
            reference_dir.joinpath(file_name) for file_name in os.listdir(reference_dir)
        ]
        reference_image_paths = [
            path
            for path in reference_file_paths
            if path.is_file() and path.name.lower().endswith(IMAGE_EXTENSIONS)
        ]

        if self._options.dump_dir:
            dump_file_paths = [
                file_path
                for file_path in [
                    self._get_dump_image_path(file_name)
                    for file_name in DUMP_FILE_NAMES
                ]
                if file_path.exists()
            ]

            for file_path in dump_file_paths:
                os.remove(file_path)

        sample_image = self._get_image()
        get_image_duration = time.perf_counter() - start_time
        self._maybe_dump_image(sample_image, FILE_SAMPLE_IMAGE)

        sample_image = crop_image(sample_image, self._options.transform_config)
        self._maybe_dump_image(sample_image, FILE_SAMPLE_IMAGE_CROPPED)

        prepared_sample_image = self._do_prepare_sample_image(sample_image)

        check_count = 0
        for reference_image_path in reference_image_paths:
            check_count += 1
            reference_image = cv2.imread(str(reference_image_path))
            self._maybe_dump_image(reference_image, FILE_REFERENCE_IMAGE)

            prepared_reference_image = self._do_prepare_reference_image(reference_image)

            if self._options.match_mode == MatchMode.FLANN:
                is_match = self._do_match_flann(
                    prepared_reference_image, prepared_sample_image
                )
            else:
                is_match = self._do_match_brute_force(
                    prepared_reference_image, prepared_sample_image
                )

            if is_match:
                return DoMatchPositiveResult(
                    reference_image=reference_image,
                    reference_image_path=reference_image_path,
                    sample_image=sample_image,
                    sample_image_cropped=sample_image,
                    is_match=True,
                    run_duration=(time.perf_counter() - start_time),
                    get_image_duration=get_image_duration,
                    check_count=check_count,
                )

        return DoMatchNegativeResult(
            sample_image=sample_image,
            sample_image_cropped=sample_image,
            is_match=False,
            run_duration=(time.perf_counter() - start_time),
            get_image_duration=get_image_duration,
            check_count=check_count,
        )

    def _do_prepare_image(
        self, image: Cv2Image, gray_name: str, blur_name: str, threshold_name: str
    ) -> Cv2Image:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # convert to gray
        self._maybe_dump_image(image, gray_name)
        image = cv2.GaussianBlur(image, (5, 5), 0)
        self._maybe_dump_image(image, blur_name)
        _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self._maybe_dump_image(image, threshold_name)

        return image

    def _do_prepare_sample_image(self, image: Cv2Image) -> Cv2Image:
        return self._do_prepare_image(
            image=image,
            gray_name=FILE_SAMPLE_IMAGE_GRAY,
            blur_name=FILE_SAMPLE_IMAGE_BLUR,
            threshold_name=FILE_SAMPLE_IMAGE_THRESHOLD,
        )

    def _do_prepare_reference_image(self, image: Cv2Image) -> Cv2Image:
        return self._do_prepare_image(
            image=image,
            gray_name=FILE_REFERENCE_IMAGE_GRAY,
            blur_name=FILE_REFERENCE_IMAGE_BLUR,
            threshold_name=FILE_REFERENCE_IMAGE_THRESHOLD,
        )

    def _do_match_flann(
        self,
        prepared_reference_image: Cv2Image,
        prepared_sample_image: Cv2Image,
    ) -> bool:
        # Initiate SIFT detector
        sift = cv2.SIFT_create()  # type: ignore [attr-defined]
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(prepared_sample_image, None)
        kp2, des2 = sift.detectAndCompute(prepared_reference_image, None)
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        # Need to draw only good matches, so create a mask
        matchesMask = [[0, 0] for i in range(len(matches))]
        match_count = 0
        # ratio test as per Lowe's paper
        inverse_confidence = 1.0 - self._options.match_confidence
        for i, (m, n) in enumerate(matches):
            if m.distance < inverse_confidence * n.distance:
                matchesMask[i] = [1, 0]
                match_count += 1
        draw_params = dict(
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            matchesMask=matchesMask,
            flags=cv2.DrawMatchesFlags_DEFAULT,
        )
        debug_image = cv2.drawMatchesKnn(  # type: ignore [call-overload]
            prepared_sample_image,
            kp1,
            prepared_reference_image,
            kp2,
            matches,
            None,
            **draw_params,
        )
        self._maybe_dump_image(debug_image, FILE_RESULT_DEBUG_IMAGE)

        return match_count > 0

    def _do_match_brute_force(
        self,
        prepared_reference_image: Cv2Image,
        prepared_sample_image: Cv2Image,
    ) -> bool:
        # Initiate ORB detector
        orb = cv2.ORB_create()  # type: ignore [attr-defined]
        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(prepared_sample_image, None)
        kp2, des2 = orb.detectAndCompute(prepared_reference_image, None)

        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply ratio test
        good = []
        match_count = 0
        inverse_confidence = 1.0 - self._options.match_confidence
        for m, n in matches:
            if m.distance < inverse_confidence * n.distance:
                good.append([m])
                match_count += 1
        # cv2.drawMatchesKnn expects list of lists as matches.
        debug_image = cv2.drawMatchesKnn(
            prepared_sample_image,
            kp1,
            prepared_reference_image,
            kp2,
            good,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        self._maybe_dump_image(debug_image, FILE_RESULT_DEBUG_IMAGE)

        return match_count > 0

    def _get_dump_image_path(self, image_name: str) -> Path:
        assert self._options.dump_dir, "dump_dir is not set"
        return self._options.dump_dir.joinpath(image_name)

    def _maybe_dump_image(self, image: Cv2Image, image_name: str) -> None:
        if not self._options.dump_dir:
            return

        full_path = self._get_dump_image_path(image_name)
        os.makedirs(full_path.parent, exist_ok=True)
        write_image(image, full_path)


def fetch_image(url: str) -> Cv2Image:
    if url.startswith("rtsp://"):
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            raise CannotConnectToStream()

        ret, frame = cap.read()
        if not ret:
            raise FailedToReadFrame()

        return cast(Cv2Image, frame)

    return cast(Cv2Image, cv2.imread(url))


def crop_image(image: Cv2Image, transform_options: TransformConfig) -> Cv2Image:
    height, width = image.shape[:2]

    crop_x = int(transform_options.x * width)
    crop_y = int(transform_options.y * height)
    crop_width = int(transform_options.width * width)
    crop_height = int(transform_options.height * height)

    return image[crop_y : (crop_y + crop_height), crop_x : (crop_x + crop_width)]


def write_image(image: Cv2Image, image_path: Path) -> None:
    cv2.imwrite(str(image_path), image)


class CannotConnectToStream(Exception):
    pass


class FailedToReadFrame(Exception):
    pass
