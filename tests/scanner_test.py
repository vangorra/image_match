from typing import Any
from unittest.mock import Mock, patch
from image_match.scanner import (
    CannotConnectToStream,
    FailedToReadFrame,
    RtspImageFetcher,
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

    res = RtspImageFetcher("rtsp://localhost/blah").fetch("test")

    assert res.image == "THE FRAME!"


@patch("image_match.scanner.cv2.VideoCapture")
def test_fetch_image_rstp_opne_closed(video_capture_mock: Mock) -> None:
    obj = Mock(MockVideoCapture)
    obj.isOpened.side_effect = [False, True]
    obj.read.return_value = True, "THE FRAME!"

    video_capture_mock.return_value = obj

    res = RtspImageFetcher("rtsp://localhost/blah").fetch("test")

    assert res.image == "THE FRAME!"


@patch("image_match.scanner.cv2.VideoCapture")
def test_fetch_image_rstp_not_opened(video_capture_mock: Mock) -> None:
    obj = Mock(MockVideoCapture)
    obj.isOpened.side_effect = [False, False]
    obj.read.return_value = True, "THE FRAME!"

    video_capture_mock.return_value = obj

    with pytest.raises(CannotConnectToStream):
        RtspImageFetcher("rtsp://localhost/blah").fetch("test")


@patch("image_match.scanner.cv2.VideoCapture")
def test_fetch_image_rstp_no_frame(video_capture_mock: Mock) -> None:
    obj = Mock(MockVideoCapture)
    obj.isOpened.return_value = True
    obj.read.return_value = False, None

    video_capture_mock.return_value = obj

    with pytest.raises(FailedToReadFrame):
        RtspImageFetcher("rtsp://localhost/blah").fetch("test")
