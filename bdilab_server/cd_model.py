import json
from typing import List, Dict, Optional, Union, cast
import logging
import numpy as np
import os
from .numpy_encoder import NumpyEncoder
from bdilab_server.base import BdilabDetectModel, ModelResponse
from bdilab_detect.utils.saving import load_detector
from bdilab_server.constants import ENV_DRIFT_TYPE_FEATURE
from bdilab_server.base.storage import download_model
from bdilab_server.base import Data


DRIFT_TYPE_FEATURE = os.environ.get(ENV_DRIFT_TYPE_FEATURE, "").upper() == "TRUE"


def _append_drift_metrcs(metrics, drift, name):
    metric_found = drift.get(name)

    # Assumes metric_found is always float/int or list/np.array when not none
    if metric_found is not None:
        if not isinstance(metric_found, (list, np.ndarray)):
            metric_found = [metric_found]

        for i, instance in enumerate(metric_found):
            metrics.append(
                {
                    "key": f"bdilab_metric_drift_{name}",
                    "value": instance,
                    "type": "GAUGE",
                    "tags": {"index": str(i)},
                }
            )


class BdilabDetectConceptDriftModel(
    BdilabDetectModel
):  # pylint:disable=c-extension-no-member
    def __init__(
        self,
        name: str,
        storage_uri: str,
        model: Optional[Data] = None,
        drift_batch_size: int = 1000,
        p_val: float = 0.05
    ):
        """
        Outlier Detection / Concept Drift Model

        Parameters
        ----------
        name
             The name of the model
        storage_uri
             The URI location of the model
        drift_batch_size
             The batch size to fill before checking for drift
        model
             Bdilab detect model
        """
        super().__init__(name, storage_uri, model)
        self.batch: Optional[np.ndarray] = None
        self.model: Data = model
        self.drift_batch_size = drift_batch_size
        self.p_val = p_val

    def load(self):
        """
        Load the model from storage

        """
        model_folder = download_model(self.storage_uri)
        self.model: Data = load_detector(model_folder)
        # self.model = load_detector("bdilab_server/bdilab_detect/saving/test/dill_file/")
        # 设置超参数 p_val
        if hasattr(self.model, "_detector"):
            self.model._detector.alpha = self.p_val
        else:
            self.alpha = self.p_val
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
             Bdilab Detect response

        """
        logging.info("PROCESSING EVENT.")
        logging.info(str(headers))
        logging.info("----")
        try:
            X = np.array(inputs)
        except Exception as e:
            raise Exception(
                "Failed to initialize NumPy array from inputs: %s, %s" % (e, inputs)
            )

        if self.batch is None:
            self.batch = X
        else:
            self.batch = np.concatenate((self.batch, X))

        self.batch = cast(np.ndarray, self.batch)

        if self.batch.shape[0] >= self.drift_batch_size:
            logging.info(
                "Running drift detection. Batch size is %d. Needed %d",
                self.batch.shape[0],
                self.drift_batch_size,
            )
            if DRIFT_TYPE_FEATURE:
                cd_preds = self.model.predict(self.batch, drift_type='feature')
            else:
                cd_preds = self.model.predict(self.batch)

            logging.info("Ran drift test")

            output = json.loads(json.dumps(cd_preds, cls=NumpyEncoder))
            # 添加批处理大小
            output["data"]["drift_batch_size"] = self.batch.shape[0]
            self.batch = None

            metrics: List[Dict] = []
            drift = output.get("data")

            if drift:
                _append_drift_metrcs(metrics, drift, "is_drift")
                _append_drift_metrcs(metrics, drift, "distance")
                _append_drift_metrcs(metrics, drift, "p_val")
                _append_drift_metrcs(metrics, drift, "threshold")

            return ModelResponse(data=output, metrics=metrics)
        else:
            logging.info(
                "Not running drift detection. Batch size is %d. Need %d",
                self.batch.shape[0],
                self.drift_batch_size,
            )
            return None
