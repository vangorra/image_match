from typing import Any
from unittest.mock import Mock, patch
from image_match.scanner import (
    CannotConnectToStream,
    FailedToReadFrame,
    MatchOrchestrator,
)
import pytest


class MockVideoCapture:
    def isOpened(self) -> bool:
        return True

    def read(self) -> Any:
        pass


@patch("image_match.scanner.cv2.VideoCapture")
def test_fetch_image_rstp(video_capture_mock: Mock) -> None:
    obj = Mock(MockVideoCapture)
    obj.isOpened.return_value = True
    obj.read.return_value = True, "THE FRAME!"

    video_capture_mock.return_value = obj

    res = MatchOrchestrator.fetch_image("rtsp://localhost/blah", "test")

    assert res.image == "THE FRAME!"


@patch("image_match.scanner.cv2.VideoCapture")
def test_fetch_image_rstp_not_opened(video_capture_mock: Mock) -> None:
    obj = Mock(MockVideoCapture)
    obj.isOpened.return_value = False
    obj.read.return_value = True, "THE FRAME!"

    video_capture_mock.return_value = obj

    with pytest.raises(CannotConnectToStream):
        MatchOrchestrator.fetch_image("rtsp://localhost/blah", "test")


@patch("image_match.scanner.cv2.VideoCapture")
def test_fetch_image_rstp_no_frame(video_capture_mock: Mock) -> None:
    obj = Mock(MockVideoCapture)
    obj.isOpened.return_value = True
    obj.read.return_value = False, None

    video_capture_mock.return_value = obj

    with pytest.raises(FailedToReadFrame):
        MatchOrchestrator.fetch_image("rtsp://localhost/blah", "test")
