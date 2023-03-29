import logging
import os
from multiprocessing import Manager
from typing import Dict, List, Tuple

import numpy as np
from prometheus_client import exposition
from prometheus_client.core import (
    CollectorRegistry,
    CounterMetricFamily,
    GaugeMetricFamily,
    HistogramMetricFamily,
)
from prometheus_client.utils import floatToGoString

from bdilab_server.prometheus_metrics.env_utils import (
    get_deployment_name,
    get_image_name,
    get_model_name,
    get_predictor_name,
    get_predictor_version,
)

logger = logging.getLogger(__name__)

FEEDBACK_KEY = "bdilab_api_model_feedback"
FEEDBACK_REWARD_KEY = "bdilab_api_model_feedback_reward"

COUNTER = "COUNTER"
GAUGE = "GAUGE"
TIMER = "TIMER"

# This sets the bins spread logarithmically between 0.001 and 30
BINS = [0] + list(np.logspace(-3, np.log10(30), 50)) + [np.inf]


def split_image_tag(tag: str) -> Tuple[str]:
    """
    Extract image name and version from an image tag.
    Parameters
    ----------
    tag
        Fully qualified docker image tag. Eg. treasures/sklearn-iris:0.1
    Returns
    -------
        Image name, image version tuple
    """
    *name_parts, version = tag.split(":")
    return ":".join(name_parts), version


# Development placeholder
image = get_image_name()
model_image, model_version = split_image_tag(image)
predictor_version = get_predictor_version()

legacy_mode = os.environ.get("BDILAB_EXECUTOR_ENABLED", "true").lower() == "false"

DEFAULT_LABELS = {
    "deployment_name": get_deployment_name(),
    "model_name": get_model_name(),
    "model_image": model_image,
    "model_version": model_version,
    "predictor_name": get_predictor_name(),
    "predictor_version": predictor_version,
}


DEFAULT_LABELS["bdilab_deployment_name"] = DEFAULT_LABELS["deployment_name"]
DEFAULT_LABELS["image_name"] = DEFAULT_LABELS["model_image"]
DEFAULT_LABELS["image_version"] = DEFAULT_LABELS["model_version"]

FEEDBACK_METRIC_METHOD_TAG = "feedback"
PREDICT_METRIC_METHOD_TAG = "predict"
INPUT_TRANSFORM_METRIC_METHOD_TAG = "inputtransform"
OUTPUT_TRANSFORM_METRIC_METHOD_TAG = "outputtransform"
ROUTER_METRIC_METHOD_TAG = "router"
AGGREGATE_METRIC_METHOD_TAG = "aggregate"
HEALTH_METRIC_METHOD_TAG = "health"


class BdilabMetrics:
    """Class to manage custom prometheus_metrics stored in shared memory."""

    def __init__(self, worker_id_func=os.getpid, extra_default_labels={}):
        # We keep reference to Manager so it does not get garbage collected
        self._manager = Manager()   # self._manager：一个 multiprocessing.Manager 对象，用于管理共享内存。这个引用确保 Manager 对象不会被垃圾回收。
        self._lock = self._manager.Lock()   # self._lock：一个由 Manager 创建的锁对象，用于确保对共享数据的访问是线程安全的。
        self.data = self._manager.dict()    # self.data：一个由 Manager 创建的字典对象，用于存储共享数据。这个字典将包含每个工作进程的度量数据。
        self.worker_id_func = worker_id_func    # self.worker_id_func：一个函数，用于获取当前工作进程的 ID。这个函数将在后续的度量更新中使用。
        self._extra_default_labels = extra_default_labels   # self._extra_default_labels：一个包含额外默认标签的字典，这些标签将添加到所有度量指标中。

    def __del__(self):
        self._manager.shutdown()

    def update_reward(self, reward: float):
        """Update prometheus_metrics key corresponding to feedback reward counter."""
        if not reward or legacy_mode:
            return
        self.update(
            [{"type": "COUNTER", "key": FEEDBACK_KEY, "value": 1}],
            FEEDBACK_METRIC_METHOD_TAG,
        )
        self.update(
            [{"type": "COUNTER", "key": FEEDBACK_REWARD_KEY, "value": reward}],
            FEEDBACK_METRIC_METHOD_TAG,
        )

    def update(self, custom_metrics: List[Dict], method: str):
        # Read a corresponding worker's metric data with lock as Proxy objects
        # are not thread-safe, see "Thread safety of proxies" here
        # https://docs.python.org/3.7/library/multiprocessing.html#programming-guidelines
        logger.debug("Updating prometheus_metrics: {}".format(custom_metrics))
        with self._lock:
            worker_data = self.data.get(self.worker_id_func(), {})
        logger.debug("Read current prometheus_metrics data from shared memory")

        for metrics in custom_metrics:
            metrics_type = metrics.get("type", "COUNTER")
            tags = metrics.get("tags", {})
            tags["method"] = method
            key = (metrics_type, metrics["key"], BdilabMetrics._generate_tags_key(tags))
            # Add tag that specifies which method added the prometheus_metrics
            if metrics_type == "COUNTER":
                value = worker_data.get(key, {}).get("value", 0)
                worker_data[key] = {"value": value + metrics["value"], "tags": tags}
            elif metrics_type == "TIMER":
                vals, sumv = worker_data.get(key, {}).get(
                    "value", (list(np.zeros(len(BINS) - 1)), 0)
                )
                # Dividing by 1000 because unit is milliseconds
                worker_data[key] = {
                    "value": self._update_hist(metrics["value"] / 1000, vals, sumv),
                    "tags": tags,
                }
            elif metrics_type == "GAUGE":
                worker_data[key] = {"value": metrics["value"], "tags": tags}
            else:
                logger.error(f"Unknown prometheus_metrics type: {metrics_type}")

        # Write worker's data with lock (again - Proxy objects are not thread-safe)
        with self._lock:
            self.data[self.worker_id_func()] = worker_data
        logger.debug("Updated prometheus_metrics in the shared memory.")

    def collect(self):
        # Read all workers prometheus_metrics with lock to avoid other processes / threads
        # writing to it at the same time. Casting to `dict` works like reading of data.
        logger.debug("BdilabMetrics.collect called")
        with self._lock:
            data = dict(self.data)
        logger.debug("Read current prometheus_metrics data from shared memory")

        for worker_id, worker_data in data.items():
            for (item_type, item_name, item_tags), item in worker_data.items():
                labels_keys, labels_values = self._merge_labels(
                    str(worker_id), item["tags"]
                )
                if item_type == "GAUGE":
                    yield self._expose_gauge(
                        item_name, item["value"], labels_keys, labels_values
                    )
                elif item_type == "COUNTER":
                    yield self._expose_counter(
                        item_name, item["value"], labels_keys, labels_values
                    )
                elif item_type == "TIMER":
                    yield self._expose_histogram(
                        item_name, item["value"], labels_keys, labels_values
                    )

    def generate_metrics(self):
        myregistry = CollectorRegistry()
        myregistry.register(self)
        return (
            exposition.generate_latest(myregistry).decode("utf-8"),
            exposition.CONTENT_TYPE_LATEST,
        )

    def clear(self):
        """
        Clear all prometheus_metrics from current worker.
        """
        worker_id = self.worker_id_func()
        logger.debug(f"Clearing prometheus_metrics from worker #{worker_id}")
        with self._lock:
            if worker_id in self.data:
                del self.data[worker_id]

    def _merge_labels(self, worker, tags):
        labels = {
            **tags,
            **DEFAULT_LABELS,
            **self._extra_default_labels,
            "worker_id": str(worker),
        }
        return list(labels.keys()), list(labels.values())

    @staticmethod
    def _generate_tags_key(tags):
        return "_".join(["-".join(i) for i in sorted(tags.items())])

    @staticmethod
    def _update_hist(x, vals, sumv):
        hist = np.histogram([x], BINS)[0]
        vals = list(np.array(vals) + hist)
        return vals, sumv + x

    @staticmethod
    def _expose_gauge(name, value, labels_keys, labels_values):
        metric = GaugeMetricFamily(name, "", labels=labels_keys)
        metric.add_metric(labels_values, value)
        return metric

    @staticmethod
    def _expose_counter(name, value, labels_keys, labels_values):
        metric = CounterMetricFamily(name, "", labels=labels_keys)
        metric.add_metric(labels_values, value)
        return metric

    @staticmethod
    def _expose_histogram(name, value, labels_keys, labels_values):
        vals, sumv = value
        buckets = [[floatToGoString(b), v] for v, b in zip(np.cumsum(vals), BINS[1:])]

        metric = HistogramMetricFamily(name, "", labels=labels_keys)
        metric.add_metric(labels_values, buckets, sum_value=sumv)
        return metric


def create_counter(key: str, value: float):
    """
    Utility method to create a counter metric
    Parameters
    ----------
    key
       Counter name
    value
       Counter value
    Returns
    -------
       Valid counter metric dict
    """
    test = value + 1
    return {"key": key, "type": COUNTER, "value": value}


def create_gauge(key: str, value: float) -> Dict:
    """
    Utility method to create a gauge metric
    Parameters
    ----------
    key
      Gauge name
    value
      Gauge value
    Returns
    -------
       Valid Gauge metric dict
    """
    test = value + 1
    return {"key": key, "type": GAUGE, "value": value}


def create_timer(key: str, value: float) -> Dict:
    """
    Utility mehtod to create a timer metric
    Parameters
    ----------
    key
      Name of metric
    value
      Value for metric
    Returns
    -------
       Valid timer metric dict
    """
    test = value + 1
    return {"key": key, "type": TIMER, "value": value}


def validate_metrics(metrics: List[Dict]) -> bool:
    """
    Validate a list of prometheus_metrics
    Parameters
    ----------
    metrics
       List of prometheus_metrics
    Returns
    -------
    """
    if isinstance(metrics, (list,)):
        for metric in metrics:
            if not ("key" in metric and "value" in metric and "type" in metric):
                return False
            if not (
                metric["type"] == COUNTER
                or metric["type"] == GAUGE
                or metric["type"] == TIMER
            ):
                return False
            try:
                metric["value"] + 1
            except TypeError:
                return False
    else:
        return False
    return True