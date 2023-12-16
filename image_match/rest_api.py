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
    orchestrators = {
        name: scanner.MatchOrchestrator.from_config(match_config)
        for (name, match_config) in serve_config.match_configs.items()
    }

    @app.route("/match/<name>", methods=["GET"])
    def match(name: str) -> Any:
        if name not in orchestrators:
            return "Not found", 404

        orchestrator = orchestrators[name]

        result = orchestrator.do_match()
        result.maybe_dump_to_directory()

        return asdict(result.to_serializable())

    return app
