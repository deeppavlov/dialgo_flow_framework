import os
import pytest

from chatsky.ml.models.remote_api.rasa_model import RasaModel, rasa_available
from tests.context_storages.test_dbs import ping_localhost

RASA_ACTIVE = ping_localhost(5005)


@pytest.fixture(scope="session")
def testing_model():
    rasa_url, api_key = "http://localhost:5005", os.getenv("RASA_API_KEY")
    if rasa_url and api_key:
        yield RasaModel(model=rasa_url, api_key=api_key, namespace_key="rasa")
    else:
        yield None


@pytest.fixture(scope="session")
def testing_async_model():
    rasa_url, api_key = "http://localhost:5005", os.getenv("RASA_API_KEY")
    if rasa_url and api_key:
        yield RasaModel(model=rasa_url, api_key=api_key, namespace_key="rasa_async")
    else:
        yield None


@pytest.mark.skipif(not rasa_available, reason="Async deps missing.")
@pytest.mark.skipif(not RASA_ACTIVE, reason="RASA inactive.")
@pytest.mark.skipif(not os.getenv("RASA_API_KEY"), reason="No RASA API key.")
@pytest.mark.rasa
@pytest.mark.docker
@pytest.mark.asyncio
async def test_async_predict(testing_async_model: RasaModel):
    request = "Hello there"
    result = await testing_async_model.predict(request=request)
    assert isinstance(result, dict)
    assert len(result) > 0
    assert result["greet"] > 0.9  # testing on default intents that include 'greet'
