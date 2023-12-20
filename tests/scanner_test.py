from typing import Any, cast
from unittest.mock import Mock, patch
from image_match.scanner import (
    CannotConnectToStream,
    FailedToReadFrame,
    ImageFetchers,
    ImageMatchers,
)
import pytest


class MockVideoCapture:
    def isOpened(self) -> bool:
        return True

    def read(self) -> Any:
        pass

    def open(self, url: str) -> None:
        pass


@patch("image_match.scanner.cv2.VideoCapture")
def test_fetch_image_rstp(video_capture_mock: Mock) -> None:
    obj = Mock(MockVideoCapture)
    obj.isOpened.return_value = True
    obj.read.return_value = True, "THE FRAME!"

    video_capture_mock.return_value = obj

    res = ImageFetchers.get_by_url("rtsp://localhost/blah").fetch("test")

    assert res.image == "THE FRAME!"


def test_image_matchers_get_by_mode_invalid() -> None:
    with pytest.raises(ValueError):
        ImageMatchers.get_by_mode(cast(Any, "AA"), cast(Any, None))


@patch("cv2.VideoCapture")
def test_rtsp_image_fetch_use_existing_connection(video_capture_mock: Mock) -> None:
    video_capture = video_capture_mock.return_value
    video_capture.read.return_value = (1, "AAA")
    video_capture.isOpened.return_value = True

    fetcher = ImageFetchers.get_by_url("rtsp://localhost/blah")
    assert fetcher._fetch_image() == "AAA"
    assert fetcher._fetch_image() == "AAA"


@patch("cv2.VideoCapture")
def test_rtsp_image_fetch_is_opened_never_true(video_capture_mock: Mock) -> None:
    video_capture = video_capture_mock.return_value
    video_capture.read.return_value = (1, "AAA")
    video_capture.isOpened.return_value = False

    fetcher = ImageFetchers.get_by_url("rtsp://localhost/blah")
    with pytest.raises(CannotConnectToStream):
        fetcher._fetch_image()


@patch("cv2.VideoCapture")
def test_rtsp_image_fetch_read_frame_never_succes(video_capture_mock: Mock) -> None:
    video_capture = video_capture_mock.return_value
    video_capture.read.return_value = (False, "AAA")
    video_capture.isOpened.return_value = True

    fetcher = ImageFetchers.get_by_url("rtsp://localhost/blah")
    with pytest.raises(FailedToReadFrame):
        fetcher._fetch_image()
