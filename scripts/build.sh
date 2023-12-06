#!/usr/bin/env bash
set -euf -o pipefail

SELF_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$SELF_DIR/.."

LINT_PATHS="tests image_match"

echo ""
echo "=Remove build dir="
rm ./build -rf

echo ""
echo "=Install depdendencies="
pipx run poetry install

echo ""
echo "=Remove unused imports="
AUTOFLAKE_ARGS=""
if [[ "${CI:-}" = "1" ]]; then
  AUTOFLAKE_ARGS="--check"
fi
pipx run poetry run autoflake \
  $AUTOFLAKE_ARGS \
  --in-place \
  --recursive \
  --remove-all-unused-imports \
  $LINT_PATHS

echo ""
echo "=Format code="
BLACK_ARGS=""
if [[ "${CI:-}" = "1" ]]; then
  BLACK_ARGS="--check"
fi
pipx run poetry run black $BLACK_ARGS .

echo
echo "=Lint with flake8="
pipx run poetry run flake8

echo ""
echo "=Analyze code="
pipx run poetry run mypy
