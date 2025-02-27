import os
import pytest

from chatsky.ml.models.remote_api.google_dialogflow_model import (
    GoogleDialogFlowModel,
    GoogleDialogFlowModel,
    dialogflow_available,
)


@pytest.fixture(scope="session")
def testing_model():
    gdf_json = os.getenv("GDF_ACCOUNT_JSON")
    if gdf_json:
        yield GoogleDialogFlowModel.from_file(filename=gdf_json, namespace_key="dialogflow")
    else:
        yield None


@pytest.fixture(scope="session")
def testing_async_model():
    gdf_json = os.getenv("GDF_ACCOUNT_JSON")
    if gdf_json:
        yield GoogleDialogFlowModel.from_file(filename=gdf_json, namespace_key="gdf_async")
    else:
        yield None


@pytest.mark.skipif(not dialogflow_available, reason="Dialogflow deps missing.")
@pytest.mark.skipif(
    not os.getenv("GDF_ACCOUNT_JSON") or not os.path.exists(os.getenv("GDF_ACCOUNT_JSON")),
    reason="GDF_ACCOUNT_JSON missing.",
)
@pytest.mark.dialogflow
def test_predict(testing_model: GoogleDialogFlowModel):
    test_phrase = "I would like some food"  # no matching intent in test project
    result = testing_model.predict(test_phrase)
    assert isinstance(result, dict)
    assert len(result) == 1  # should only return the fallback intent
    test_phrase_2 = "Hello there"
    result_2 = testing_model.predict(test_phrase_2)  # has a matching intent
    assert isinstance(result_2, dict)
    assert len(result_2) >= 1  # expected to return 'hello' intent


@pytest.mark.skipif(not dialogflow_available, reason="Dialogflow deps missing.")
@pytest.mark.skipif(
    not os.getenv("GDF_ACCOUNT_JSON") or not os.path.exists(os.getenv("GDF_ACCOUNT_JSON")),
    reason="GDF_ACCOUNT_JSON missing.",
)
@pytest.mark.dialogflow
@pytest.mark.asyncio
async def test_async_predict(testing_async_model: GoogleDialogFlowModel):
    test_phrase = "I would like some food"  # no matching intent in test project
    result = await testing_async_model.predict(test_phrase)
    assert isinstance(result, dict)
    assert len(result) == 1  # should only return the fallback intent
    test_phrase_2 = "Hello there"
    result_2 = await testing_async_model.predict(test_phrase_2)  # has a matching intent
    assert isinstance(result_2, dict)
    assert len(result_2) >= 1  # expected to return 'hello' intent
