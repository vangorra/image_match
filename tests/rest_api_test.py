from image_match.rest_api import ServeConfig, new_rest_api_app
from image_match.scanner import MatchConfig, TransformConfig
from tests.const import (
    REFERENCE_DIR,
    REFERENCE_SAMPLE_CLOSED_DAY,
    REFERENCE_SAMPLE_OPEN_DAY,
)

CLOSED_NAME = "chicken_door_closed"
OPEN_NAME = "chicken_door_open"

SERVE_CONFIG = ServeConfig(
    match_configs={
        CLOSED_NAME: MatchConfig(
            reference_dir=REFERENCE_DIR,
            sample_url=str(REFERENCE_SAMPLE_CLOSED_DAY),
            transform_config=TransformConfig(
                x=0.0,
                y=0.6,
                width=0.05,
                height=0.15,
            ),
        ),
        OPEN_NAME: MatchConfig(
            reference_dir=REFERENCE_DIR,
            sample_url=str(REFERENCE_SAMPLE_OPEN_DAY),
            transform_config=TransformConfig(
                x=0.0,
                y=0.6,
                width=0.05,
                height=0.15,
            ),
        ),
    }
)


def test_new_rest_api_app_door_closed() -> None:
    app = new_rest_api_app(SERVE_CONFIG)
    client = app.test_client()
    response = client.get(f"/match/{CLOSED_NAME}")
    assert response.status == "200 OK"
    assert response.json
    assert response.json["is_match"]


def test_new_rest_api_app_door_open() -> None:
    app = new_rest_api_app(SERVE_CONFIG)
    client = app.test_client()
    response = client.get(f"/match/{OPEN_NAME}")
    assert response.status == "200 OK"
    assert response.json
    assert not response.json["is_match"]


def test_new_rest_api_app_invalid_config() -> None:
    app = new_rest_api_app(SERVE_CONFIG)
    client = app.test_client()
    response = client.get("/match/unknown_name")
    assert response.status == "404 NOT FOUND"
    assert not response.json
