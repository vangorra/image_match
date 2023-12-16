import json
import os
import re
from pathlib import Path
import shutil
from typing import Generator, Optional, Tuple
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
from image_match.scanner import MatchMode, DoMatchResultSerializable
from tests.const import (
    DUMP_DIR,
    DUMP_FILE_NAMES_MATCH,
    DUMP_FILE_NAMES_NO_MATCH,
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


@pytest.fixture(autouse=True)
def run_around_tests() -> Generator:
    if DUMP_DIR.exists():
        shutil.rmtree(DUMP_DIR)
    yield


@pytest.mark.parametrize(
    "sample_url,assert_matches,match_mode,dump_enabled,expect_dump_files",
    [
        (REFERENCE_SAMPLE_CLOSED_DAY, True, None, True, DUMP_FILE_NAMES_MATCH),
        (REFERENCE_SAMPLE_CLOSED_NIGHT, True, None, True, DUMP_FILE_NAMES_MATCH),
        (REFERENCE_SAMPLE_OPEN_DAY, False, None, True, DUMP_FILE_NAMES_NO_MATCH),
        (REFERENCE_SAMPLE_OPEN_NIGHT, False, None, True, DUMP_FILE_NAMES_NO_MATCH),
        (
            REFERENCE_SAMPLE_CLOSED_DAY,
            True,
            MatchMode.BRUTE_FORCE,
            False,
            (),
        ),
        (
            REFERENCE_SAMPLE_CLOSED_NIGHT,
            True,
            MatchMode.BRUTE_FORCE,
            False,
            (),
        ),
        (
            REFERENCE_SAMPLE_OPEN_DAY,
            False,
            MatchMode.BRUTE_FORCE,
            False,
            (),
        ),
        (
            REFERENCE_SAMPLE_OPEN_NIGHT,
            False,
            MatchMode.BRUTE_FORCE,
            False,
            (),
        ),
        (
            REFERENCE_SAMPLE_CLOSED_DAY,
            True,
            MatchMode.FLANN,
            False,
            (),
        ),
        (
            REFERENCE_SAMPLE_CLOSED_NIGHT,
            True,
            MatchMode.FLANN,
            False,
            (),
        ),
        (
            REFERENCE_SAMPLE_OPEN_DAY,
            False,
            MatchMode.FLANN,
            False,
            (),
        ),
        (
            REFERENCE_SAMPLE_OPEN_NIGHT,
            False,
            MatchMode.FLANN,
            False,
            (),
        ),
    ],
)
def test_match(
    sample_url: Path,
    assert_matches: bool,
    match_mode: Optional[MatchMode],
    dump_enabled: bool,
    expect_dump_files: Tuple[str, ...],
) -> None:
    if DUMP_DIR.exists():
        shutil.rmtree(DUMP_DIR)

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

    runner = CliRunner(mix_stderr=False)
    result = runner.invoke(match, args)

    assert result.exit_code == 0

    output_obj = json.loads(result.stdout)
    match_result = DoMatchResultSerializable(**output_obj)
    assert (
        match_result.is_match == assert_matches
    ), f"Expect is_match property of output to be '{assert_matches}' but was '{match_result.is_match}'."

    if dump_enabled:
        dump_files = os.listdir(DUMP_DIR)
        assert dump_files, f"Expected files to exist in {DUMP_DIR}"
        assert (
            len(dump_files) == 1
        ), f"Expected only a single dump directory in '{DUMP_DIR}', found {len(dump_files)}."

        dump_dir_name = dump_files[0]
        dump_dir_path = DUMP_DIR.joinpath(dump_dir_name)
        assert re.compile(
            "[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\\.[0-9]{6}_[a-f0-9]{4}"
        ).match(
            dump_dir_name
        ), f"Dump directory name in '{dump_dir_path}' should look like '2023-12-05 13:42:19.773830_ae4f'."
        assert (
            dump_dir_path.is_dir()
        ), f"Expected path '{dump_dir_path}' to be a directory."

        for file_name in expect_dump_files:
            dump_file_path = dump_dir_path.joinpath(file_name)
            assert (
                dump_file_path.exists()
            ), f"Dump file should exist '{dump_file_path}'."


@pytest.mark.parametrize(
    "debug",
    [
        (True),
        (False),
    ],
)
@patch("image_match.cli.new_rest_api_app")
def test_serve(new_rest_api_app_mock: Mock, debug: bool) -> None:
    config_file = TEMP_DIR.joinpath("config.yaml")
    os.makedirs(config_file.parent, exist_ok=True)

    with open(config_file, "w") as handle:
        handle.write(
            f"""---
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
    app_mock.run.assert_called_with(host="0.0.0.0", debug=debug)
