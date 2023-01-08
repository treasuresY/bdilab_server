import json
from typing import List, Dict, Optional, Union
import logging
import dill
import os
from bdilab_server.constants import (
    REQUEST_ID_HEADER_NAME,
    NAMESPACE_HEADER_NAME,
)

from bdilab_server.base import CEModel, ModelResponse
from bdilab_server.base.storage import download_model


def _load_class_module(module_path: str) -> str:
    components = module_path.split(".")
    mod = __import__(".".join(components[:-1]))
    for comp in components[1:]:
        print(mod, comp)
        mod = getattr(mod, comp)
    return mod


class CustomMetricsModel(CEModel):  # pylint:disable=c-extension-no-member
    def __init__(
        self, name: str, storage_uri: str, elasticsearch_uri: str = None, model=None
    ):
        """
        Custom Metrics Model

        Parameters
        ----------
        name
             The name of the model
        storage_uri
             The URI location of the model
        """
        super().__init__(name)
        self.name = name
        self.storage_uri = storage_uri
        self.model = model
        self.ready = False
        self.elasticsearch_client = None

    def load(self):
        """
        Load the model from storage

        """
        if "/" in self.storage_uri:
            model_folder = download_model(self.storage_uri)
            self.model = dill.load(
                open(os.path.join(model_folder, "meta.pickle"), "rb")
            )
        else:
            # Load from locally available models
            MetricsClass = _load_class_module(self.storage_uri)
            self.model = MetricsClass()

        self.ready = True

    def process_event(self, inputs: Union[List, Dict], headers: Dict) -> Optional[ModelResponse]:
        """
        Process the event and return Alibi Detect score

        Parameters
        ----------
        inputs
             Input data
        headers
             Header options

        Returns
        -------
             SeldonResponse response

        """
        logging.info("PROCESSING Feedback Event.")
        logging.info(str(headers))
        logging.info("----")

        metrics: List[Dict] = []
        output: Dict = {}
        truth = None
        response = None
        error = None

        if not isinstance(inputs, dict):
            raise Exception(f"Data is not a dict: {json.dumps(inputs)}")

        if "truth" not in inputs:
            raise Exception(
                f"No truth value provided in: {json.dumps(inputs)}")
        else:
            truth = inputs["truth"]

        # We automatically add any metrics provided in the incoming request
        if "metrics" in inputs:
            metrics.extend(inputs["metrics"])

        # If response is provided then we can perform a comparison
        if "response" in inputs:
            response = inputs["response"]

        else:
            error = "Neither response nor request Puid provided in headers"

        if error:
            raise Exception(error)

        logging.error(f"{truth}, {response}")
        metrics_transformed = self.model.transform(truth, response)

        metrics.extend(metrics_transformed.metrics)

        return ModelResponse(data={}, metrics=metrics)
