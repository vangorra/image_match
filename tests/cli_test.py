import os
import re
from pathlib import Path
from typing import Optional
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from image_match.cli import (
    CLI_ARG_CONFIG,
    CLI_ARG_CROP_HEIGHT,
    CLI_ARG_CROP_WIDTH,
    CLI_ARG_CROP_X,
    CLI_ARG_CROP_Y,
    CLI_ARG_DUMP_DIR,
    CLI_ARG_MATCH_CONFIDENCE,
    CLI_ARG_MATCH_MODE,
    CLI_ARG_OUTPUT_FILE,
    CLI_ARG_REFERENCE_DIR,
    CLI_ARG_SAMPLE_URL,
    fetch,
    main,
    match,
    serve,
)
from image_match.const import DUMP_FILE_NAMES
from image_match.scanner import MatchMode
from tests.const import (
    DUMP_DIR,
    PATTERN_MATCH,
    PATTERN_NO_MATCH,
    REFERENCE_DIR,
    REFERENCE_SAMPLE_CLOSED_DAY,
    REFERENCE_SAMPLE_CLOSED_NIGHT,
    REFERENCE_SAMPLE_OPEN_DAY,
    REFERENCE_SAMPLE_OPEN_NIGHT,
    TEMP_DIR,
)


def test_main() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "Usage: " in result.output
    assert (
        """Commands:
  fetch
  match
  serve"""
        in result.output
    )


def test_fetch() -> None:
    output_file_path = TEMP_DIR.joinpath("output.png")

    runner = CliRunner()
    result = runner.invoke(
        fetch,
        [
            CLI_ARG_SAMPLE_URL,
            str(REFERENCE_SAMPLE_CLOSED_DAY),
            CLI_ARG_OUTPUT_FILE,
            str(output_file_path),
        ],
    )

    assert result.exit_code == 0
    assert output_file_path.exists(), "Output file does not exist."


@pytest.mark.parametrize(
    "sample_url,output_pattern,match_mode,dump_enabled",
    [
        (REFERENCE_SAMPLE_CLOSED_DAY, PATTERN_MATCH, None, True),
        (REFERENCE_SAMPLE_CLOSED_NIGHT, PATTERN_MATCH, None, True),
        (REFERENCE_SAMPLE_OPEN_DAY, PATTERN_NO_MATCH, None, True),
        (REFERENCE_SAMPLE_OPEN_NIGHT, PATTERN_NO_MATCH, None, True),
        (
            REFERENCE_SAMPLE_CLOSED_DAY,
            PATTERN_MATCH,
            MatchMode.BRUTE_FORCE,
            False,
        ),
        (
            REFERENCE_SAMPLE_CLOSED_NIGHT,
            PATTERN_MATCH,
            MatchMode.BRUTE_FORCE,
            False,
        ),
        (
            REFERENCE_SAMPLE_OPEN_DAY,
            PATTERN_NO_MATCH,
            MatchMode.BRUTE_FORCE,
            False,
        ),
        (
            REFERENCE_SAMPLE_OPEN_NIGHT,
            PATTERN_NO_MATCH,
            MatchMode.BRUTE_FORCE,
            False,
        ),
        (
            REFERENCE_SAMPLE_CLOSED_DAY,
            PATTERN_MATCH,
            MatchMode.FLANN,
            False,
        ),
        (
            REFERENCE_SAMPLE_CLOSED_NIGHT,
            PATTERN_MATCH,
            MatchMode.FLANN,
            False,
        ),
        (
            REFERENCE_SAMPLE_OPEN_DAY,
            PATTERN_NO_MATCH,
            MatchMode.FLANN,
            False,
        ),
        (
            REFERENCE_SAMPLE_OPEN_NIGHT,
            PATTERN_NO_MATCH,
            MatchMode.FLANN,
            False,
        ),
    ],
)
def test_match(
    sample_url: Path,
    output_pattern: re.Pattern,
    match_mode: Optional[MatchMode],
    dump_enabled: bool,
) -> None:
    args = [
        CLI_ARG_REFERENCE_DIR,
        str(REFERENCE_DIR),
        CLI_ARG_SAMPLE_URL,
        str(sample_url),
        CLI_ARG_MATCH_CONFIDENCE,
        "0.9",
        CLI_ARG_CROP_X,
        "0.0",
        CLI_ARG_CROP_Y,
        "0.6",
        CLI_ARG_CROP_WIDTH,
        "0.05",
        CLI_ARG_CROP_HEIGHT,
        "0.15",
    ]

    if match_mode:
        args.append(CLI_ARG_MATCH_MODE)
        args.append(match_mode.value)

    if dump_enabled:
        args.append(CLI_ARG_DUMP_DIR)
        args.append(str(DUMP_DIR))

    runner = CliRunner()
    result = runner.invoke(match, args)

    assert result.exit_code == 0

    assert output_pattern.match(result.output)

    if dump_enabled:
        for file_name in DUMP_FILE_NAMES:
            assert DUMP_DIR.joinpath(file_name).exists(), "Dump file should exist."


@pytest.mark.parametrize(
    "port,debug",
    [
        (8123, True),
        (8080, False),
    ],
)
@patch("image_match.cli.new_rest_api_app")
def test_serve(new_rest_api_app_mock: Mock, port: int, debug: bool) -> None:
    config_file = TEMP_DIR.joinpath("config.yaml")
    os.makedirs(config_file.parent, exist_ok=True)

    with open(config_file, "w") as handle:
        handle.write(
            f"""---
port: {port}
debug: {debug}"""
        )

    class MockApp:
        def run(self) -> None:
            pass

    app_mock = Mock(MockApp)
    new_rest_api_app_mock.return_value = app_mock

    runner = CliRunner()
    result = runner.invoke(serve, [CLI_ARG_CONFIG, str(config_file)])

    assert result.exit_code == 0
    app_mock.run.assert_called_with(host="localhost", port=port, debug=debug)
