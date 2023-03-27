from http import HTTPStatus
from typing import Union, List, Dict

import tornado.web

from bdilab_server.protocols.request_handler import RequestHandler


class ServiceflowRequestHandler(RequestHandler):
    def __init__(self, request: Dict):
        super().__init__(request
                         )

    def validate(self):
        pass

    def extract_request(self) -> Union[List, Dict]:
        pass