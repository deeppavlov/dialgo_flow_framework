"""
Rasa Model
---------------

This module provides an annotator model
that queries an external RASA NLU Server for utterance classification.
"""

import asyncio
from async_lru import alru_cache
import json
from urllib.parse import urljoin
from typing import Optional, List
from pydantic import BaseModel

try:
    import httpx
    import requests

    rasa_available = True
except ImportError:
    htppx = object()
    rasa_available = False

from http import HTTPStatus
from chatsky.ml.models.base_model import ExtrasBaseAPIModel

LABEL_KEY = "labels"


class RasaIntent(BaseModel):
    """Class for integration with Rasa NLU server HTTP API."""

    confidence: float
    name: str


class RasaEntity(BaseModel):
    """Class for integration with Rasa NLU server HTTP API."""

    start: int
    end: int
    confidence: Optional[float]
    value: str
    entity: str


class RasaResponse(BaseModel):
    """Class for integration with Rasa NLU server HTTP API."""

    text: str
    intent_ranking: Optional[List[RasaIntent]] = None
    intent: Optional[RasaIntent] = None
    entities: Optional[List[RasaEntity]] = None


class RasaModel(ExtrasBaseAPIModel):
    """
    This class implements an asynchronous connection to RASA NLU server for dialog annotation.
    In order to work with this class, you need to have a running instance of Rasa NLU Server
    with the model trained to recognize your intents.
    Please, refer to the `RASA docs <https://rasa.com/docs/rasa/nlu-only-server>`_ on how to
    develop a RASA project and launch an NLU-only server.

    :param model: Rasa model url.
    :param api_key: Rasa api key for request authorization. The exact authentification method can be retrieved
        from your Rasa Server config.
    :param jwt_token: Rasa jwt token for request authorization. The exact authentification method can be retrieved
        from your Rasa Server config.
    :param namespace_key: Name of the namespace in framework states that the model will be using.
    :param retries: The number of times requests will be repeated in case of failure.
    :param headers: Fill in this parameter, if you want to override the standard set of headers with custom headers.

    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        jwt_token: Optional[str] = None,
        namespace_key: Optional[str] = None,
        *,
        retries: int = 10,
        headers: Optional[dict] = None,
    ):
        if not rasa_available:
            raise ImportError("`httpx` package missing. Try `pip install chatsky[httpx]`")
        super().__init__()
        self.headers = headers or {"Content-Type": "application/json"}
        health_check = requests.get(model)
        health_check.raise_for_status()
        self.parse_url = urljoin(model, ("model/parse" + (f"?token={api_key}" if api_key else "")))
        self.train_url = urljoin(model, ("model/train" + (f"?token={api_key}" if api_key else "")))
        if jwt_token is not None:
            self.headers["Authorization"] = "Bearer " + jwt_token
        self.retries = retries

    @alru_cache(maxsize=10)
    async def predict(self, request: str) -> dict:
        client = httpx.AsyncClient()
        message = {"text": request}
        retries = 0
        while retries < self.retries:
            retries += 1
            response: httpx.Response = await client.post(self.parse_url, headers=self.headers, data=json.dumps(message))
            if response.status_code == HTTPStatus.SERVICE_UNAVAILABLE:  # Wait for model to warm up
                await asyncio.sleep(1)
            elif response.status_code == HTTPStatus.OK:
                break
            else:
                raise httpx.HTTPStatusError(str(response.status_code) + " " + response.text)

        json_response = response.json()
        parsed = RasaResponse.model_validate(json_response)
        result = {item.name: item.confidence for item in parsed.intent_ranking} if parsed.intent_ranking else dict()

        await client.aclose()
        return result
