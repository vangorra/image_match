from dataclasses import asdict
import os
from pathlib import Path
from datetime import datetime
from typing import Any
from uuid import uuid4

import click
from pydantic_yaml import parse_yaml_raw_as
import json
import logging

from image_match import scanner
from image_match.const import (
    ARG_CONFIG,
    ARG_CROP_HEIGHT,
    ARG_CROP_WIDTH,
    ARG_CROP_X,
    ARG_CROP_Y,
    ARG_DUMP_DIR,
    ARG_LOG_LEVEL,
    ARG_MATCH_CONFIDENCE,
    ARG_MATCH_MODE,
    ARG_OUTPUT_FILE,
    ARG_REFERENCE_DIR,
    ARG_SAMPLE_URL,
    DEFAULT_CROP_HEIGHT,
    DEFAULT_CROP_WIDTH,
    DEFAULT_CROP_X,
    DEFAULT_CROP_Y,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MATCH_CONFIDENCE,
    DEFAULT_MATCH_MODE,
    LogLevel,
    MatchMode,
)
from image_match.common import maybe_set_log_level
from image_match.rest_api import ServeConfig, new_rest_api_app

CLI_ARG_REFERENCE_DIR = f"--{ARG_REFERENCE_DIR}"
CLI_ARG_OUTPUT_FILE = f"--{ARG_OUTPUT_FILE}"
CLI_ARG_SAMPLE_URL = f"--{ARG_SAMPLE_URL}"
CLI_ARG_CROP_X = f"--{ARG_CROP_X}"
CLI_ARG_CROP_Y = f"--{ARG_CROP_Y}"
CLI_ARG_CROP_WIDTH = f"--{ARG_CROP_WIDTH}"
CLI_ARG_CROP_HEIGHT = f"--{ARG_CROP_HEIGHT}"
CLI_ARG_CONFIG = f"--{ARG_CONFIG}"
CLI_ARG_MATCH_CONFIDENCE = f"--{ARG_MATCH_CONFIDENCE}"
CLI_ARG_MATCH_MODE = f"--{ARG_MATCH_MODE}"
CLI_ARG_DUMP_DIR = f"--{ARG_DUMP_DIR}"
CLI_ARG_LOG_LEVEL = f"--{ARG_LOG_LEVEL}"

log_level_option = click.option(
    CLI_ARG_LOG_LEVEL,
    type=click.Choice(
        [
            LogLevel.CRITICAL.value.str_value,
            LogLevel.ERROR.value.str_value,
            LogLevel.WARN.value.str_value,
            LogLevel.INFO.value.str_value,
            LogLevel.DEBUG.value.str_value,
            LogLevel.TRACE.value.str_value,
        ]
    ),
    default=DEFAULT_LOG_LEVEL.value.str_value,
)


def transform_options(function: Any) -> Any:
    function = click.option(
        CLI_ARG_CROP_X, type=click.FloatRange(0.0, 1.0), default=DEFAULT_CROP_X
    )(function)
    function = click.option(
        CLI_ARG_CROP_Y, type=click.FloatRange(0.0, 1.0), default=DEFAULT_CROP_Y
    )(function)
    function = click.option(
        CLI_ARG_CROP_WIDTH, type=click.FloatRange(0.0, 1.0), default=DEFAULT_CROP_WIDTH
    )(function)
    function = click.option(
        CLI_ARG_CROP_HEIGHT,
        type=click.FloatRange(0.0, 1.0),
        default=DEFAULT_CROP_HEIGHT,
    )(function)
    return function


sample_url_option = click.option(CLI_ARG_SAMPLE_URL, type=str, required=True)


@click.group()
def main() -> None:
    pass  # pragma: no cover


@main.command()
@sample_url_option
@click.option(
    CLI_ARG_OUTPUT_FILE,
    type=click.Path(file_okay=True, dir_okay=False, resolve_path=True, path_type=Path),
    required=True,
)
@transform_options
@log_level_option
def fetch(
    sample_url: str,
    output_file: Path,
    crop_x: float,
    crop_y: float,
    crop_width: float,
    crop_height: float,
    log_level: str,
) -> None:
    maybe_set_log_level(log_level)

    logging.info(f"Fetch {sample_url}")
    result = scanner.ImageFetchers.get_by_url(sample_url).fetch("sample")
    sample_image = result.image
    transform_options = scanner.TransformConfig(
        x=crop_x, y=crop_y, width=crop_width, height=crop_height
    )
    logging.info("Crop")
    cropped_image = scanner.crop_image(sample_image, transform_options)

    logging.info(f"Write to {output_file}")
    os.makedirs(output_file.parent, exist_ok=True)
    scanner.write_image(cropped_image, output_file)


@main.command()
@click.option(
    CLI_ARG_REFERENCE_DIR,
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        exists=True,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
    required=True,
)
@sample_url_option
@transform_options
@click.option(
    CLI_ARG_MATCH_MODE,
    type=click.Choice([MatchMode.BRUTE_FORCE.value, MatchMode.FLANN.value]),
    default=DEFAULT_MATCH_MODE.value,
)
@click.option(
    CLI_ARG_DUMP_DIR,
    type=click.Path(file_okay=False, dir_okay=True, resolve_path=True, path_type=Path),
)
@click.option(
    CLI_ARG_MATCH_CONFIDENCE,
    type=click.FloatRange(0.0, 1.0),
    default=DEFAULT_MATCH_CONFIDENCE,
)
@log_level_option
def match(
    reference_dir: Path,
    sample_url: str,
    crop_x: float,
    crop_y: float,
    crop_width: float,
    crop_height: float,
    match_mode: str,
    dump_dir: Path,
    match_confidence: float,
    log_level: str,
) -> None:
    maybe_set_log_level(log_level)

    orchestrator = scanner.MatchOrchestrator.from_config(
        scanner.MatchConfig(
            reference_dir=reference_dir,
            sample_url=sample_url,
            match_mode=MatchMode(match_mode),
            dump_dir=dump_dir,
            match_confidence=match_confidence,
            transform_config=scanner.TransformConfig(
                x=crop_x, y=crop_y, width=crop_width, height=crop_height
            ),
        )
    )

    result = orchestrator.do_match()
    if dump_dir:
        date_time_prefix = str(datetime.now())
        random_suffix = str(uuid4())[-4:]
        dump_dir_name = f"{date_time_prefix}_{random_suffix}"
        result.dump_to_directory(dump_dir.joinpath(dump_dir_name))

    print(json.dumps(asdict(result.to_serializable()), indent=2))


@main.command()
@click.option(
    CLI_ARG_CONFIG,
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        exists=True,
        readable=True,
        resolve_path=True,
        path_type=Path,
    ),
    required=True,
)
@log_level_option
def serve(config: Path, log_level: str) -> None:
    maybe_set_log_level(log_level)

    serve_config = parse_yaml_raw_as(ServeConfig, config.read_text())  # type: ignore [type-var]

    app = new_rest_api_app(serve_config)
    app.run(host="0.0.0.0", debug=serve_config.debug)


if __name__ == "__main__":
    main()  # pragma: no cover
