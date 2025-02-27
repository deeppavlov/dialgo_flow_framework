import os
import importlib

import pytest

from tests.test_utils import get_path_from_tests_to_current_dir
from chatsky.ml.models.remote_api.google_dialogflow_model import dialogflow_available
from chatsky.ml.models.remote_api.rasa_model import rasa_available
from chatsky.ml.models.remote_api.hf_api_model import hf_api_available

from chatsky.utils.testing.common import check_happy_path
from tests.context_storages.test_dbs import ping_localhost

RASA_ACTIVE = ping_localhost(5005)

dot_path_to_addon = get_path_from_tests_to_current_dir(__file__, separator=".")


@pytest.mark.parametrize(
    ["example_module_name", "skip_condition"],
    [
        ("7_rasa", os.getenv("RASA_API_KEY") is None or not rasa_available or not RASA_ACTIVE),
        (
            "5_dialogflow",
            not (os.getenv("GDF_ACCOUNT_JSON") and os.path.exists(os.getenv("GDF_ACCOUNT_JSON")))
            or not dialogflow_available,
        ),
        ("6_hf_api", os.getenv("HF_API_KEY") is None or not hf_api_available),
    ],
)
@pytest.mark.rasa
@pytest.mark.dialogflow
@pytest.mark.huggingface
@pytest.mark.docker
def test_examples(example_module_name: str, skip_condition):
    if skip_condition:
        pytest.skip()
    module = importlib.import_module(f"tutorials.{dot_path_to_addon}.{example_module_name}")
    pipeline = getattr(module, "pipeline")
    happy_path = getattr(module, "happy_path")
    check_happy_path(pipeline, happy_path)
