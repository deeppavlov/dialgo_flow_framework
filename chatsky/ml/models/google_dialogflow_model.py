"""
Google Dialogflow Model
------------------------

The module allows you to use Google Dialogflow as a service
to gain insights about user intents.
"""

import uuid
import json
from pathlib import Path
from async_lru import alru_cache

from chatsky.ml.models.base_model import ExtrasBaseAPIModel

try:
    from google.cloud import dialogflow_v2
    from google.oauth2 import service_account

    dialogflow_available = True
except ImportError:
    dialogflow_v2 = None
    service_account = None
    dialogflow_available = False


class GoogleDialogFlowModel(ExtrasBaseAPIModel):
    """
    This class implements an asynchronous connection to Google Dialogflow for dialog annotation.
    Note, that before you use the class, you need to set up a Dialogflow project,
    create intents, and train a language model, which can be easily done
    using the Dialogflow web interface (see the official
    `instructions <https://cloud.google.com/dialogflow/es/docs/quick/build-agent>`_).
    After this is done, you should obtain a service account JSON file from Google
    and pass it to this class, using :py:meth:`~from_file` method.

    :param model: A parsed service account json for your dialogflow project.
        Calling json.load() on the file obtained from Google is sufficient to get the
        credentials object. Alternatively, use :py:meth:`~from_file` method.
    :param namespace_key: Name of the namespace in framework states that the model will be using.
    :param language: Language parameter is passed to the Dialogflow wrapper.
    """

    def __init__(
        self,
        model: dict,
        *,
        language: str = "en",
    ) -> None:
        if not dialogflow_available:
            raise ImportError("`google-cloud-dialogflow` package missing. Try `pip install chatsky[dialogflow]`.")
        super().__init__()
        self._language = language
        if isinstance(model, dict):
            info = model
        else:
            raise ValueError("Please, pass the service account credentials as dict.")

        self._credentials = service_account.Credentials.from_service_account_info(info)

    @classmethod
    def from_file(cls, filename: str, language: str = "en"):
        """
        :param filename: Path to the Dialogflow credentials saved as JSON.
        :param language: The language parameter is forwarded to the underlying
            Dialogflow wrapper.
        """
        assert Path(filename).exists(), f"Path {filename} does not exist."
        with open(filename, "r", encoding="utf-8") as file:
            info = json.load(file)
        return cls(model=info, language=language)

    @alru_cache(maxsize=10)
    async def predict(self, request: str) -> dict:
        session_id = uuid.uuid4()
        session_client = dialogflow_v2.SessionsAsyncClient(credentials=self._credentials)
        session_path = session_client.session_path(self._credentials.project_id, session_id)
        query_input = dialogflow_v2.QueryInput(text=dialogflow_v2.TextInput(text=request, language_code=self._language))
        request = dialogflow_v2.DetectIntentRequest(session=session_path, query_input=query_input)
        response = await session_client.detect_intent(request=request)
        result: dialogflow_v2.QueryResult = response.query_result
        if result.intent is not None:
            return {result.intent.display_name: result.intent_detection_confidence}
        return {}
