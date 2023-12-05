import os
from pathlib import Path
from typing import cast

import click
from pydantic_yaml import parse_yaml_raw_as

from image_match import scanner
from image_match.const import (
    ARG_CONFIG,
    ARG_CROP_HEIGHT,
    ARG_CROP_WIDTH,
    ARG_CROP_X,
    ARG_CROP_Y,
    ARG_DUMP_DIR,
    ARG_MATCH_CONFIDENCE,
    ARG_MATCH_MODE,
    ARG_OUTPUT_FILE,
    ARG_REFERENCE_DIR,
    ARG_SAMPLE_URL,
    DEFAULT_MATCH_MODE,
    MatchMode,
)
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


@click.group()
def main() -> None:
    pass  # pragma: no cover


@main.command()
@click.option(CLI_ARG_SAMPLE_URL, type=str, required=True)
@click.option(
    CLI_ARG_OUTPUT_FILE,
    type=click.Path(file_okay=True, dir_okay=False, resolve_path=True, path_type=Path),
    required=True,
)
@click.option(CLI_ARG_CROP_X, type=click.FloatRange(0.0, 1.0), default=0.0)
@click.option(CLI_ARG_CROP_Y, type=click.FloatRange(0.0, 1.0), default=0.0)
@click.option(CLI_ARG_CROP_WIDTH, type=click.FloatRange(0.0, 1.0), default=1.0)
@click.option(CLI_ARG_CROP_HEIGHT, type=click.FloatRange(0.0, 1.0), default=1.0)
def fetch(
    sample_url: str,
    output_file: Path,
    crop_x: float,
    crop_y: float,
    crop_width: float,
    crop_height: float,
) -> None:
    print("Fetch", sample_url)
    sample_image = scanner.fetch_image(sample_url)
    transform_options = scanner.TransformConfig(
        x=crop_x, y=crop_y, width=crop_width, height=crop_height
    )
    print("Crop")
    cropped_image = scanner.crop_image(sample_image, transform_options)

    print("Write to", output_file)
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
@click.option(CLI_ARG_SAMPLE_URL, type=str, required=True)
@click.option(CLI_ARG_CROP_X, type=click.FloatRange(0.0, 1.0), default=0.0)
@click.option(CLI_ARG_CROP_Y, type=click.FloatRange(0.0, 1.0), default=0.0)
@click.option(CLI_ARG_CROP_WIDTH, type=click.FloatRange(0.0, 1.0), default=1.0)
@click.option(CLI_ARG_CROP_HEIGHT, type=click.FloatRange(0.0, 1.0), default=1.0)
@click.option(
    CLI_ARG_MATCH_MODE,
    type=click.Choice([MatchMode.BRUTE_FORCE.value, MatchMode.FLANN.value]),
    default=DEFAULT_MATCH_MODE.value,
)
@click.option(
    CLI_ARG_DUMP_DIR,
    type=click.Path(file_okay=False, dir_okay=True, resolve_path=True, path_type=Path),
)
@click.option(CLI_ARG_MATCH_CONFIDENCE, type=click.FloatRange(0.0, 1.0), default=0.9)
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
) -> None:
    scanner_obj = scanner.Scanner(
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

    res = scanner_obj.do_match()

    if isinstance(res, scanner.DoMatchPositiveResult):
        res_pos = cast(scanner.DoMatchPositiveResult, res)
        print("Match", res_pos.reference_image_path)
    else:
        print("No match")


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
def serve(config: Path) -> None:
    serve_config = parse_yaml_raw_as(ServeConfig, config.read_text())  # type: ignore [type-var]

    app = new_rest_api_app(serve_config)
    app.run(host="localhost", port=serve_config.port, debug=serve_config.debug)


if __name__ == "__main__":
    main()  # pragma: no cover
