from http import HTTPStatus
from typing import Dict, List

import tornado
from bdilab_server.protocols.request_handler import (
    RequestHandler,
)  # pylint: disable=no-name-in-module


class UpdateParamsRequestHandler(RequestHandler):
    def __init__(self, request: Dict):  # pylint: disable=useless-super-delegation
        super().__init__(request)

    def validate(self):
        pass

    def extract_request(self) -> List:
        return self.request.get("drift_batch_size", None), \
               self.request.get("alpha", None), \
               self.request.get("x_ref", None)
