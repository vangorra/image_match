"""
Module use for providing REST API functionality.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, cast

from flask import Flask
from flask.typing import ResponseReturnValue
from image_match import scanner


@dataclass(frozen=True)
class ServeConfig:
    """
    Configuration for starting a REST API.
    """

    debug: bool = False
    match_configs: Dict[str, scanner.MatchConfig] = field(default_factory=dict)


def new_rest_api_app(serve_config: ServeConfig) -> Flask:
    """
    Create a new REST API from a ServeConfig.
    """
    app = Flask(__name__)

    @app.route("/match/<name>", methods=["GET"])
    def match(name: str) -> ResponseReturnValue:
        if name not in serve_config.match_configs:
            return "Not found", 404

        scanner_obj = scanner.Scanner(serve_config.match_configs[name])
        res = scanner_obj.do_match()
        response_obj: Dict[str, Any] = {
            "is_match": res.is_match,
            "run_time": res.run_duration,
            "get_image_duration": res.get_image_duration,
            "check_count": res.check_count,
        }

        if isinstance(res, scanner.DoMatchPositiveResult):
            res_pos = cast(scanner.DoMatchPositiveResult, res)
            response_obj = {
                **response_obj,
                **{"reference_image_path": str(res_pos.reference_image_path)},
            }

        return response_obj

    return app
