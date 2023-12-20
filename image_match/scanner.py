import abc
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
import os
import time
from pathlib import Path
from typing import Callable, List, Optional, cast
from typing_extensions import Self
from functools import lru_cache
import multiprocessing
import hashlib
import logging
from uuid import uuid4

import cv2
from cv2.typing import MatLike
from image_match.common import MatchConfig, TransformConfig, logging_trace

from image_match.const import (
    IMAGE_EXTENSIONS,
    DumpMode,
    MatchMode,
)


def crop_image(image: MatLike, transform_options: TransformConfig) -> MatLike:
    height, width = image.shape[:2]

    crop_x = int(transform_options.x * width)
    crop_y = int(transform_options.y * height)
    crop_width = int(transform_options.width * width)
    crop_height = int(transform_options.height * height)

    return image[crop_y : (crop_y + crop_height), crop_x : (crop_x + crop_width)]


def write_image(image: MatLike, image_path: Path) -> None:
    cv2.imwrite(str(image_path), image)


class CannotConnectToStream(Exception):
    pass


class FailedToReadFrame(Exception):
    pass


@dataclass(frozen=True)
class AbsImagePreparerResult(abc.ABC):
    filename_prefix: str
    result: MatLike

    def dump_to_directory(self, path: Path) -> None:
        os.makedirs(path, exist_ok=True)
        write_image(
            self.result,
            path.joinpath(f"{self.filename_prefix}_result.png"),
        )
        self._dump_to_directory(path)

    @abc.abstractmethod
    def _dump_to_directory(self, path: Path) -> None:
        raise NotImplementedError()


@dataclass(frozen=True)
class AbsImagePreparer(abc.ABC):
    filename_prefix: str

    @abc.abstractmethod
    def prepare_image(self, image: MatLike) -> AbsImagePreparerResult:
        raise NotImplementedError()


@dataclass(frozen=True)
class AutoThresholdImagePreparerResult(AbsImagePreparerResult):
    filename_prefix: str
    gray_image: MatLike
    gray_blur_image: MatLike
    gray_blur_threshold_image: MatLike

    def _dump_to_directory(self, path: Path) -> None:
        write_image(
            self.gray_image,
            path.joinpath(f"{self.filename_prefix}_gray.png"),
        )
        write_image(
            self.gray_blur_image,
            path.joinpath(f"{self.filename_prefix}_gray_blur.png"),
        )
        write_image(
            self.gray_blur_threshold_image,
            path.joinpath(f"{self.filename_prefix}_gray_blur_threshold.png"),
        )


class AutoThresholdImagePreparer(AbsImagePreparer):
    def prepare_image(self, image: MatLike) -> AutoThresholdImagePreparerResult:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        _, threshold_image = cv2.threshold(
            blur_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        return AutoThresholdImagePreparerResult(
            filename_prefix=self.filename_prefix,
            gray_image=gray_image,
            gray_blur_image=blur_image,
            gray_blur_threshold_image=threshold_image,
            result=threshold_image,
        )


@dataclass(frozen=True)
class BaseImageMatcherOptions:
    match_confidence: float


@dataclass(frozen=True)
class AbsImageMatcherResult(abc.ABC):
    is_match: bool

    def dump_to_directory(self, path: Path) -> None:
        os.makedirs(path, exist_ok=True)
        self._dump_to_directory(path)

    def _dump_to_directory(self, path: Path) -> None:
        raise NotImplementedError()


@dataclass(frozen=True)
class ImageMatcherResult(AbsImageMatcherResult):
    debug_image: MatLike

    def _dump_to_directory(self, path: Path) -> None:
        os.makedirs(path, exist_ok=True)

        write_image(
            self.debug_image,
            path.joinpath("debug.png"),
        )


class AbsImageMatcher(abc.ABC):
    def __init__(self, options: BaseImageMatcherOptions) -> None:
        self._options = options

    @abc.abstractmethod
    def match(
        self, prepared_sample_image: MatLike, prepared_reference_image: MatLike
    ) -> AbsImageMatcherResult:
        raise NotImplementedError()


class FlannImageMatcher(AbsImageMatcher):
    def match(
        self, prepared_sample_image: MatLike, prepared_reference_image: MatLike
    ) -> ImageMatcherResult:
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

        return ImageMatcherResult(
            is_match=match_count > 0,
            debug_image=debug_image,
        )


class BruteForceImageMatcher(AbsImageMatcher):
    def match(
        self, prepared_sample_image: MatLike, prepared_reference_image: MatLike
    ) -> ImageMatcherResult:
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

        return ImageMatcherResult(
            is_match=match_count > 0,
            debug_image=debug_image,
        )


class ImageMatchers:
    @staticmethod
    def get_by_mode(
        match_mode: MatchMode, options: BaseImageMatcherOptions
    ) -> AbsImageMatcher:
        if match_mode == MatchMode.FLANN:
            return FlannImageMatcher(options)
        elif match_mode == MatchMode.BRUTE_FORCE:
            return BruteForceImageMatcher(options)
        else:
            raise ValueError("Provided match_mode is invalid.")


class Timer:
    _start_time: Optional[float] = None
    _end_time: Optional[float] = None
    _duration: Optional[float] = None

    def start(self) -> Self:
        self._duration = None
        self._end_time = None
        self._start_time = time.perf_counter()
        return self

    def stop(self) -> Self:
        assert self._start_time, "Timer must have been started."
        self._end_time = time.perf_counter()
        self._duration = self._end_time - self._start_time
        return self

    def duration(self) -> float:
        assert self._duration, "Timer must have been started and stopped."
        return self._duration


@dataclass(frozen=True)
class FetchImageResult:
    filename_prefix: str
    duration: float
    image: MatLike

    def dump_to_directory(self, path: Path) -> None:
        os.makedirs(path, exist_ok=True)

        write_image(self.image, path.joinpath(f"{self.filename_prefix}.png"))


@dataclass(frozen=True)
class FetchPreparedReferenceImageResult:
    fetch_result: FetchImageResult
    prepare_result: AbsImagePreparerResult

    def dump_to_directory(self, path: Path) -> None:
        os.makedirs(path, exist_ok=True)

        self.fetch_result.dump_to_directory(path)
        self.prepare_result.dump_to_directory(path)


@dataclass(frozen=True)
class ProcessResult:
    fetch_prepared_reference_image_result: FetchPreparedReferenceImageResult
    matcher_result: AbsImageMatcherResult
    reference_image_path: Path
    match_duration: float
    fetch_prepared_image_duration: float
    total_duration: float

    def dump_to_directory(self, path: Path) -> None:
        os.makedirs(path, exist_ok=True)

        self.fetch_prepared_reference_image_result.dump_to_directory(path)
        self.matcher_result.dump_to_directory(path)


@dataclass(frozen=True)
class Durations:
    match_all: float
    total: float
    sample_fetch: float
    sample_prepare: float

    reference_match: float
    reference_fetch_prepared_image: float
    reference_match_total: float


@dataclass(frozen=True)
class DoMatchResultSerializable:
    is_match: bool
    reference_image_path: Optional[str]
    match_attempt_count: int
    durations: Durations


@dataclass(frozen=True)
class DoMatchResult:
    match_config: MatchConfig
    is_match: bool
    process_result: Optional[ProcessResult]
    sample_image_fetch_result: FetchImageResult
    cropped_sample_image: MatLike
    prepared_cropped_sample_image_result: AbsImagePreparerResult
    match_all_duration: float
    total_duration: float
    sample_prepare_duration: float
    match_attempt_count: int

    def maybe_dump_to_directory(self) -> Optional[Path]:
        if not self.match_config.dump_dir:
            return None

        should_dump = (
            (self.match_config.dump_mode == DumpMode.BOTH)
            or (self.match_config.dump_mode == DumpMode.MATCH and self.is_match)
            or (self.match_config.dump_mode == DumpMode.NO_MATCH and not self.is_match)
        )

        if not should_dump:
            return None

        return self.dump_to_directory(self.match_config.dump_dir)

    def dump_to_directory(self, path: Path) -> Path:
        date_time_prefix = str(datetime.now())
        random_suffix = str(uuid4())[-4:]
        match_str = "match" if self.is_match else "nomatch"
        dump_dir_name = f"{date_time_prefix}_{match_str}_{random_suffix}"
        dir_path = path.joinpath(dump_dir_name)

        logging.debug(f"Dumping results to '{str(dir_path)}'.")
        os.makedirs(dir_path, exist_ok=True)

        if self.process_result:
            self.process_result.dump_to_directory(dir_path)

        self.prepared_cropped_sample_image_result.dump_to_directory(dir_path)

        self.sample_image_fetch_result.dump_to_directory(dir_path)

        write_image(
            self.cropped_sample_image,
            dir_path.joinpath("sample_cropped.png"),
        )

        return dir_path

    def to_serializable(self) -> DoMatchResultSerializable:
        return DoMatchResultSerializable(
            is_match=self.is_match,
            reference_image_path=str(self.process_result.reference_image_path)
            if self.process_result
            else None,
            match_attempt_count=self.match_attempt_count,
            durations=Durations(
                match_all=self.match_all_duration,
                total=self.total_duration,
                sample_fetch=self.sample_image_fetch_result.duration,
                sample_prepare=self.sample_prepare_duration,
                reference_match=self.process_result.match_duration
                if self.process_result and self.process_result.match_duration
                else -1,
                reference_fetch_prepared_image=self.process_result.fetch_prepared_image_duration
                if self.process_result
                and self.process_result.fetch_prepared_image_duration
                else -1,
                reference_match_total=self.process_result.total_duration
                if self.process_result and self.process_result.total_duration
                else -1,
            ),
        )


class AbsImageFetcher(abc.ABC):
    _image_url: str

    def __init__(self, image_url: str) -> None:
        self._image_url = image_url

    def fetch(self, dump_filename_prefix: str) -> FetchImageResult:
        timer = Timer().start()

        image = self._fetch_image()

        return FetchImageResult(
            filename_prefix=dump_filename_prefix,
            image=image,
            duration=timer.stop().duration(),
        )

    @abc.abstractmethod
    def _fetch_image(self) -> MatLike:
        raise NotImplementedError()


def noop() -> None:
    """
    Do nothing. Used to help code coverage determine if lines were covered in
    areas where CPython cannot detect (like continue in loops).
    https://github.com/nedbat/coveragepy/issues/198
    """


class RtspImageFetcher(AbsImageFetcher):
    _cap: Optional[cv2.VideoCapture] = None

    def _fetch_image(self) -> MatLike:
        max_retries = 2
        for attempt_num in range(1, max_retries + 1):
            is_last_attempt = attempt_num == max_retries
            logging.debug(
                f"Attempt {attempt_num}/{max_retries} read lastest from RTSP stream."
            )

            if not self._cap:
                logging.debug(f"Starting new RTSP capture for url '{self._image_url}'")
                self._cap = cv2.VideoCapture(self._image_url)

            if not self._cap.isOpened():
                self._cap = None

                if is_last_attempt:
                    raise CannotConnectToStream()

                noop()
                continue

            ret, frame = self._cap.read()
            if not ret:
                self._cap = None

                if is_last_attempt:
                    raise FailedToReadFrame()

                noop()
                continue

            return cast(MatLike, frame)

        raise ValueError("This is bug and code should have never reached here.")


class StaticImageFetch(AbsImageFetcher):
    def _fetch_image(self) -> MatLike:
        return cast(MatLike, cv2.imread(self._image_url))


class ImageFetchers:
    @staticmethod
    def get_by_url(url: str) -> AbsImageFetcher:
        if url.startswith("rtsp:"):
            return RtspImageFetcher(url)
        else:
            return StaticImageFetch(url)


class MatchOrchestrator:
    _config: MatchConfig
    _cached_fetch_prepared_reference_image: Callable[
        [Path], FetchPreparedReferenceImageResult
    ]
    _sample_image_preparer: AbsImagePreparer
    _reference_image_preparer: AbsImagePreparer
    _image_matcher: AbsImageMatcher
    _sample_fetcher: AbsImageFetcher
    _fetch_cached_hash: str = ""

    def __init__(
        self,
        match_config: MatchConfig,
        sample_fetcher: AbsImageFetcher,
        image_matcher: AbsImageMatcher,
        sample_image_preparer: AbsImagePreparer,
        reference_image_preparer: AbsImagePreparer,
    ) -> None:
        self._config = match_config
        self._sample_fetcher = sample_fetcher
        self._image_matcher = image_matcher
        self._sample_image_preparer = sample_image_preparer
        self._reference_image_preparer = reference_image_preparer

    def _fetch_prepared_reference_image(
        self, path: Path
    ) -> FetchPreparedReferenceImageResult:
        fetch_result = StaticImageFetch(str(path)).fetch("reference")
        prepare_result = self._reference_image_preparer.prepare_image(
            fetch_result.image
        )
        return FetchPreparedReferenceImageResult(
            fetch_result=fetch_result,
            prepare_result=prepare_result,
        )

    def _get_reference_image_paths(self) -> List[Path]:
        reference_dir = self._config.reference_dir
        reference_file_paths: List[Path] = [
            reference_dir.joinpath(file_name) for file_name in os.listdir(reference_dir)
        ]
        image_file_paths = [
            path
            for path in reference_file_paths
            if path.is_file() and path.name.lower().endswith(IMAGE_EXTENSIONS)
        ]
        image_file_paths.sort()
        return image_file_paths

    def do_match(self) -> DoMatchResult:
        total_timer = Timer().start()

        logging.debug(f"Fetch sample image from '{self._config.sample_url}'.")
        sample_image_fetch_result = self._sample_fetcher.fetch("sample")
        sample_image = sample_image_fetch_result.image

        prepare_sample_timer = Timer().start()
        logging.debug("Crop sample image.")
        cropped_sample_image = crop_image(sample_image, self._config.transform_config)
        logging.debug("Prepare sample image.")
        prepared_cropped_sample_image_result = (
            self._sample_image_preparer.prepare_image(cropped_sample_image)
        )
        prepare_sample_timer.stop()

        process_result: Optional[ProcessResult] = None

        match_all_timer = Timer().start()
        reference_image_paths = self._get_reference_image_paths()
        max_worker_count = max(multiprocessing.cpu_count() - 2, 1)
        logging.info(
            f"Using {max_worker_count} threads to match {len(reference_image_paths)} reference images."
        )
        with ThreadPoolExecutor(max_workers=max_worker_count) as executor:
            futures: List[Future] = []

            def on_future_done(future: Future) -> None:
                nonlocal futures
                nonlocal process_result

                if future.cancelled():
                    return

                result: ProcessResult = future.result()
                if not result.matcher_result.is_match or process_result:
                    return
                process_result = result

                # Cancel remaining futures.
                for future in futures:
                    future.cancel()

            # Recreate reference cache function (if needed).
            reference_image_paths_str = ",".join(
                [
                    str(reference_image_path)
                    for reference_image_path in reference_image_paths
                ]
            ).encode("utf-8")
            fetch_cached_hash = hashlib.md5(reference_image_paths_str).hexdigest()
            if fetch_cached_hash != self._fetch_cached_hash:
                cache_limit = len(reference_image_paths)
                logging.info(
                    f"Reference files changed, regenerate _cached_fetch_prepared_reference_image() with limit {cache_limit}."
                )
                self._fetch_cached_hash = fetch_cached_hash
                self._cached_fetch_prepared_reference_image = lru_cache(cache_limit)(
                    self._fetch_prepared_reference_image
                )

            # Schedule processing tasks.
            for reference_image_path in reference_image_paths:
                future: Future = executor.submit(
                    self._process_image,
                    prepared_cropped_sample_image_result,
                    reference_image_path,
                )
                futures.append(future)
                future.add_done_callback(on_future_done)

        logging_trace("Wait for futures to complete.")
        wait(futures)
        match_all_timer.stop()
        logging.debug(f"Match completed in {match_all_timer.duration()}s.")

        return DoMatchResult(
            match_config=self._config,
            is_match=process_result.matcher_result.is_match
            if process_result
            else False,
            process_result=process_result,
            sample_image_fetch_result=sample_image_fetch_result,
            cropped_sample_image=cropped_sample_image,
            prepared_cropped_sample_image_result=prepared_cropped_sample_image_result,
            match_all_duration=match_all_timer.duration(),
            total_duration=total_timer.stop().duration(),
            sample_prepare_duration=prepare_sample_timer.duration(),
            match_attempt_count=len(
                [
                    future
                    for future in futures
                    if future.done() and not future.cancelled()
                ]
            ),
        )

    def _process_image(
        self,
        prepared_sample_image_result: AbsImagePreparerResult,
        reference_image_path: Path,
    ) -> ProcessResult:
        logging_trace(f"_process_image {reference_image_path}")

        total_timer = Timer().start()
        fetch_preapred_image_timer = Timer().start()
        fetch_prepared_reference_image_result = (
            self._cached_fetch_prepared_reference_image(reference_image_path)
        )
        fetch_preapred_image_timer.stop()

        match_timer = Timer().start()
        matcher_result = self._image_matcher.match(
            prepared_sample_image_result.result,
            fetch_prepared_reference_image_result.prepare_result.result,
        )
        match_timer.stop()

        return ProcessResult(
            fetch_prepared_reference_image_result=fetch_prepared_reference_image_result,
            matcher_result=matcher_result,
            reference_image_path=reference_image_path,
            match_duration=match_timer.duration(),
            fetch_prepared_image_duration=fetch_preapred_image_timer.duration(),
            total_duration=total_timer.stop().duration(),
        )

    @staticmethod
    def from_config(config: MatchConfig) -> "MatchOrchestrator":
        sample_image_preparer = AutoThresholdImagePreparer("sample")
        reference_image_preparer = AutoThresholdImagePreparer("reference")

        image_matcher_options = BaseImageMatcherOptions(
            match_confidence=config.match_confidence
        )

        image_matcher = ImageMatchers.get_by_mode(
            config.match_mode, image_matcher_options
        )
        sample_fetcher = ImageFetchers.get_by_url(config.sample_url)

        return MatchOrchestrator(
            config,
            sample_fetcher=sample_fetcher,
            image_matcher=image_matcher,
            sample_image_preparer=sample_image_preparer,
            reference_image_preparer=reference_image_preparer,
        )
