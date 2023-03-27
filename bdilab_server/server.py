import json
import logging
import os
from http import HTTPStatus
from typing import Dict, Optional

import requests
import tornado.httpserver
import tornado.ioloop
import tornado.web
from bdilab_server.base import CEModel, ModelResponse
from bdilab_server.protocols.request_handler import RequestHandler
from bdilab_server.protocols.tensorflow_http import TensorflowRequestHandler
from bdilab_server.protocols.v2 import KFservingV2RequestHandler
from cloudevents.sdk import converters
from cloudevents.sdk import marshaller
from cloudevents.sdk.event import v1
from bdilab_server.protocols import Protocol
import uuid
# from bdilab_server.constants import DRIFT_BATCH_SIZE, ALPHA

DEFAULT_HTTP_PORT = 8080
CESERVER_LOGLEVEL = os.environ.get("CESERVER_LOGLEVEL", "INFO").upper()
logging.basicConfig(level=CESERVER_LOGLEVEL)

DEFAULT_LABELS = {
    "seldon_deployment_namespace": os.environ.get(
        "SELDON_DEPLOYMENT_NAMESPACE", "NOT_IMPLEMENTED"
    )
}

DRIFT_BATCH_SIZE = 100
ALPHA = 0.05

class CEServer(object):
    def __init__(
        self,
        protocol: Protocol,
        event_type: str,
        event_source: str,
        http_port: int = DEFAULT_HTTP_PORT,
        reply_url: str = None,
    ):
        """
        CloudEvents server

        Parameters
        ----------
        protocol
             wire protocol
        http_port
             http port to listen on
        reply_url
             reply url to send response event
        event_type
             type of event being handled (for req logging purposes)
        """
        self.registered_model: Optional[CEModel] = None
        self.http_port = http_port
        self.protocol = protocol
        self.reply_url = reply_url
        self._http_server: Optional[tornado.httpserver.HTTPServer] = None
        self.event_type = event_type
        self.event_source = event_source


    def create_application(self):
        return tornado.web.Application(
            [
                # Outlier detector
                (
                    r"/",
                    EventHandler,
                    dict(
                        protocol=self.protocol,
                        model=self.registered_model,
                        reply_url=self.reply_url,
                        event_type=self.event_type,
                        event_source=self.event_source,
                        # seldon_metrics=self.seldon_metrics,
                    ),
                ),
                (
                    r"/updateParams",
                    UpdateParamsHandler,
                    dict(
                        model=self.registered_model
                    )
                ),
                # Protocol Discovery API that returns the serving protocol supported by this server.
                (r"/protocol", ProtocolHandler, dict(protocol=self.protocol)),
                # # Prometheus Metrics API that returns metrics for model servers
                # (
                #     r"/v1/metrics",
                #     MetricsHandler,
                #     dict(seldon_metrics=self.seldon_metrics),
                # ),
            ]
        )

    def start(self, model: CEModel):
        """
        Start the server

        Parameters
        ----------
        model
             The model to load

        """
        self.register_model(model)

        self._http_server = tornado.httpserver.HTTPServer(
            self.create_application())

        logging.info("Listening on port %s" % self.http_port)
        self._http_server.bind(self.http_port)
        self._http_server.start(1)  # Single worker at present
        tornado.ioloop.IOLoop.current().start()

    def register_model(self, model: CEModel):
        if not model.name:
            raise Exception(
                "Failed to register model, model.name must be provided.")
        self.registered_model = model
        logging.info("Registering model:" + model.name)


def get_request_handler(protocol, request: Dict) -> RequestHandler:
    """
    Create a request handler for the data

    Parameters
    ----------
    protocol
         Protocol to use
    request
         The incoming request
    Returns
    -------
         A Request Handler for the desired protocol

    """
    if protocol == Protocol.tensorflow_http:
        return TensorflowRequestHandler(request)
    elif protocol == Protocol.kfserving_http:
        return KFservingV2RequestHandler(request)
    else:
        raise Exception(f"Unknown protocol {protocol}")


def sendCloudEvent(event: v1.Event, url: str):
    """
    Send CloudEvent

    Parameters
    ----------
    event
         CloudEvent to send
    url
         Url to send event

    """
    http_marshaller = marshaller.NewDefaultHTTPMarshaller()
    binary_headers, binary_data = http_marshaller.ToRequest(
        event, converters.TypeBinary, json.dumps
    )

    logging.info("binary CloudEvent")
    for k, v in binary_headers.items():
        logging.info("{0}: {1}\r\n".format(k, v))
    logging.info(binary_data)

    response = requests.post(url, headers=binary_headers, data=binary_data)
    response.raise_for_status()


class UpdateParamsHandler(tornado.web.RequestHandler):

    def initialize(
        self,
        model: CEModel,
    ):
        self.model = model

    def put(self):
        if not self.model.ready:
            self.model.load()
        try:
            body = json.loads(self.request.body)
        except json.decoder.JSONDecodeError as e:
            raise tornado.web.HTTPError(
                status_code=HTTPStatus.BAD_REQUEST,
                reason="Unrecognized request format: %s" % e,
            )
        if body.get("drift_batch_size"):
            # global DRIFT_BATCH_SIZE
            # DRIFT_BATCH_SIZE = body["drift_batch_size"]
            self.model.drift_batch_size = body["drift_batch_size"]
        if body.get("alpha"):
            # global ALPHA
            # ALPHA = body["alpha"]
            d_model = getattr(self.model, "model")
            if hasattr(d_model, "_detector"):
                d_model._detector.alpha = body["alpha"]
            else:
                d_model.alpha = body["alpha"]


class EventHandler(tornado.web.RequestHandler):
    def initialize(
        self,
        protocol: str,
        model: CEModel,
        reply_url: str,
        event_type: str,
        event_source: str,
    ):
        """
        Event Handler

        Parameters
        ----------
        protocol
             The protocol to expect
        model
             The model to use
        reply_url
             The reply url to send model responses
        event_type
             The CE event type to be sent
        event_source
             The CE event source
        """
        self.protocol = protocol
        self.model = model
        self.reply_url = reply_url
        self.event_type = event_type
        self.event_source = event_source

    def post(self):
        """
        Handle post request. Extract data. Call event handler and optionally send a reply event.

        """
        if not self.model.ready:
            self.model.load()

        # # 更新drift_batch_size
        # self.model.drift_batch_size = DRIFT_BATCH_SIZE
        #
        # # 更新alpha参数
        # d_model = getattr(self.model, "model")
        # if hasattr(d_model, "_detector"):
        #     d_model._detector.alpha = ALPHA
        # else:
        #     d_model.alpha = ALPHA

        # logging.info(self.model.drift_batch_size)
        # logging.info(getattr(self.model, "model").alpha)

        try:
            body = json.loads(self.request.body)
        except json.decoder.JSONDecodeError as e:
            raise tornado.web.HTTPError(
                status_code=HTTPStatus.BAD_REQUEST,
                reason="Unrecognized request format: %s" % e,
            )

        # Extract payload from request
        request_handler: RequestHandler = get_request_handler(
            self.protocol, body)
        request_handler.validate()
        request = request_handler.extract_request()

        # Create event from request body
        event = v1.Event()
        http_marshaller = marshaller.NewDefaultHTTPMarshaller()
        event = http_marshaller.FromRequest(
            event, self.request.headers, self.request.body, json.loads
        )
        logging.debug(json.dumps(event.Properties()))

        # Extract any desired request headers
        headers = {}

        for (key, val) in self.request.headers.get_all():
            headers[key] = val

        response: Optional[ModelResponse] = self.model.process_event(
            request, headers)

        if response is None:
            return

        # runtime_metrics = response.metrics
        # if runtime_metrics is not None:
        #     if validate_metrics(runtime_metrics):
        #         self.seldon_metrics.update(runtime_metrics, self.event_type)
        #     else:
        #         logging.error("Metrics returned are invalid: " + str(runtime_metrics))
        if response.data is not None:

            # Create event from response if reply_url is active
            if not self.reply_url == "":
                if event.EventID() is None or event.EventID() == "":
                    resp_event_id = uuid.uuid1().hex
                else:
                    resp_event_id = event.EventID()
                revent = (
                    v1.Event()
                    .SetContentType("application/json")
                    .SetData(response.data)
                    .SetEventID(resp_event_id)
                    .SetSource(self.event_source)
                    .SetEventType(self.event_type)
                    .SetExtensions(event.Extensions())
                )
                logging.debug(json.dumps(revent.Properties()))
                sendCloudEvent(revent, self.reply_url)
            self.write(json.dumps(response.data))


class LivenessHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Alive")


class ProtocolHandler(tornado.web.RequestHandler):
    def initialize(self, protocol: Protocol):
        self.protocol = protocol

    def get(self):
        self.write(str(self.protocol.value))


# class MetricsHandler(tornado.web.RequestHandler):
#     def initialize(self, seldon_metrics: SeldonMetrics):
#         self.seldon_metrics = seldon_metrics

#     def get(self):
#         metrics, mimetype = self.seldon_metrics.generate_metrics()
#         self.set_header("Content-Type", mimetype)
#         self.write(metrics)
