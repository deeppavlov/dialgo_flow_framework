import pytest

from chatsky import Pipeline
from chatsky.ml.dataset import Dataset
from chatsky.utils.testing.toy_script import TOY_SCRIPT

from tests.test_utils import get_path_from_tests_to_current_dir


@pytest.fixture(scope="session")
def testing_pipeline():
    pipeline = Pipeline(
        TOY_SCRIPT, start_label=("greeting_flow", "start_node"), fallback_label=("greeting_flow", "fallback_node")
    )
    yield pipeline


@pytest.fixture(scope="session")
def testing_dataset():
    yield Dataset.parse_json(f"tutorials/{get_path_from_tests_to_current_dir(__file__)}/data/example.json")


@pytest.fixture(scope="session")
def hf_model_name():
    yield "obsei-ai/sell-buy-intent-classifier-bert-mini"


@pytest.fixture(scope="session")
def save_dir(tmpdir_factory):
    dir_name = tmpdir_factory.mktemp("testdir")
    yield dir_name


@pytest.fixture(scope="session")
def save_file(save_dir):
    file_name = save_dir.join("testfile")
    return str(file_name)
