"""
Module use for providing REST API functionality.
"""

from dataclasses import asdict
from typing import Any

from flask import Flask
from image_match import scanner
from image_match.common import ServeConfig


def new_rest_api_app(serve_config: ServeConfig) -> Flask:
    """
    Create a new REST API from a ServeConfig.
    """
    app = Flask(__name__)

    @app.route("/match/<name>", methods=["GET"])
    def match(name: str) -> Any:
        if name not in serve_config.match_configs:
            return "Not found", 404

        scanner_obj = scanner.Scanner(serve_config.match_configs[name])
        return asdict(scanner_obj.do_match())

    return app
