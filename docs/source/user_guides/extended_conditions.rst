Quick Start -- extended_conditions
-----------------------------------

Description
~~~~~~~~~~~

Dialog Flow Extended Conditions allow for integration of various machine learning models for Natural Language Understanding (NLU) into DFF scripting.
The former include the language modelling tools provided by ``Sklearn``, ``Gensim``, and ``HuggingFace``
as well as cloud-hosted services (``Rasa NLU``, ``Google Dialogflow``). 

Installation
~~~~~~~~~~~~~

.. code-block:: bash

    pip install dff[async] # required by RASA and HuggingFace iference API
    pip install dff[dialogflow]
    pip install dff[sklearn]
    pip install dff[gensim]
    pip install dff[huggingface] # for local deployment of hf models

Instantiate a model
~~~~~~~~~~~~~~~~~~~

The library provides a number of wrappers for different model types. All these classes implement a uniform interface.

 - ``namespace_key`` should be used in all types of models to save the annotation results to separate namespaces in the context. 
 - The ``model`` parameter is set either with the local model instance or with the address of an external annotator.

However, some of the parameters are class-specific.

 - The ``tokenizer`` parameter is only required for Sklearn, Gensim, and HuggingFace models. See the signature of the corresponding classes for more information.
 - ``device`` parameter is only required for Hugging Face models. Use ``torch.device("cpu")`` or ``torch.device("cuda")``.
 - ``dataset`` should be passed to all cosine matchers, so that they have a pre-defined set of labels and examples, against which user utterances will be compared.

Using the library, you can deploy models locally.

.. code-block:: python

    hf_model = HFClassifier(
        model=model,
        tokenizer=tokenizer,
        device=torch.device("cpu")
        namespace_key="HF"
    )

Another option is to employ remotely hosted annotators. For instance, you can connect to a running RASA service for NLU annotations.

.. code-block:: python

    rasa_model = RasaModel(
        model="http://my-rasa-server",
        namespace_key="rasa",
    )

Use the Model class in your Script graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Annotation should be performed at the ``PRE_TRANSITION_PROCESSING`` step.

.. code-block:: python

    script = {
        GLOBAL: {
            PRE_TRANSITION_PROCESSING: {
                "get_intents_from_rasa": rasa_model 
            }
        }
    }

Extracted values can be accessed by all functions that use the ``Context`` object.
We provide several such functions that can be leveraged as transition conditions.

.. code-block:: python

    from df_extended_conditions.conditions import has_cls_label

    script = {
        "root": {
            "start": {
                TRANSITIONS: {
                    ("next_flow", "next_node"): has_cls_label("user_happy", threshold=0.9, namespace="some_model")
                }
            }
        },
        ...
    }

Custom API Connector
~~~~~~~~~~~~~~~~~~~~

The following code snippets demonstrate how you can write a connector for an external web API.

.. code-block:: python

    import json
    from http import HTTPStatus

    import requests
    import httpx

    from dff.script.extras.conditions.models.remote_api.async_mixin import (
        AsyncMixin,
    )
    from dff.script.extras.conditions.models.base_model import ExtrasBaseModel


To create a synchronous connector to an API, we recommend you to inherit the class from ``ExtrasBaseModel``.
The only method that you have to override is the ``predict`` method.
It takes a request string and returns a {label: probability} dictionary.
In case the request has not been successful, an empty dictionary can be returned.

The same applies to asynchronous connectors,
although they should also inherit from ``AsyncMixin`` class
in order to make the ``__call__`` method asynchronous.
We use `httpx` as an asynchronous http client.


.. code-block:: python

    class CustomAPIConnector(ExtrasBaseModel):
        def __init__(self, url: str, namespace_key: str = "default") -> None:
            super().__init__(namespace_key)
            self.url = url

        def predict(self, request: str) -> dict:
            result = requests.post(self.url, data=json.dumps({"data": request}))
            if result.status_code != HTTPStatus.OK:
                return {}
            json_response = result.json()
            return {
                label: probability for label, probability in json_response.items()
            }


.. code-block:: python

    class AsyncCustomAPIConnector(AsyncMixin, CustomAPIConnector):
        async def predict(self, request: str) -> dict:
            client = httpx.AsyncClient()
            result = await client.post(self.url, data=json.dumps({"data": request}))
            await client.aclose()
            if result.status_code != HTTPStatus.OK:
                return {}
            json_response = result.json()
            return {
                label: probability for label, probability in json_response.items()
            }

Custom Classifier
~~~~~~~~~~~~~~~~~

In this section, we show the way you can adapt a classifier model to DFF's class system.

.. code-block:: python

    import pickle

    from dff.script.extras.conditions.models.base_model import ExtrasBaseModel


In order to create your own classifier, create a child class of the ``ExtrasBaseModel`` abstract type.

``ExtrasBaseModel`` only has one abstract method, ``predict``, that should necessarily be overridden.
The method takes a request string and returns a dictionary of class labels
and their respective probabilities.

You can override the rest of the methods, namely ``save``, ``load``, ``fit`` and ``transform``
at your own convenience, e.g. lack of those will not raise an error.

* ``fit`` should take a new dataset and retrain / update the underlying model.
* ``transform`` should take a request string and produce a vector.
* ``save`` and ``load`` are self-explanatory.

.. code-block:: python

    class MyCustomClassifier(ExtrasBaseModel):
        def __init__(
            self, swear_words: list, namespace_key: str = "default"
        ) -> None:
            self.swear_words = swear_words or ["hell", "damn", "curses"]
            super().__init__(namespace_key)

        def predict(self, request: str) -> dict:
            probs = {}

            if any([word in request for word in self.swear_words]):
                probs["swearing"] = 1.0

            return probs

        def save(self, filename: str):
            with open(filename, "wb+") as file:
                pickle.dump(self.swear_words, file)

        @classmethod
        def load(cls, filename: str, namespace_key: str):
            with open(filename, "rb") as file:
                swear_words = pickle.load(file)
            return cls(swear_words=swear_words, namespace_key=namespace_key)

Custom Matcher
~~~~~~~~~~~~~~

The following code snippets demonstrate the way in which a custom matcher can be implemented.


.. code-block:: python

    from dff.script.extras.conditions.models.base_model import ExtrasBaseModel
    from dff.script.extras.conditions.models.local.cosine_matchers.cosine_matcher_mixin import (
        CosineMatcherMixin,
    )

To build  your own cosine matcher, you should inherit
from the ``CosineMatcherMixin`` and from the ``ExtrasBaseModel``,
with the former taking precedence.
This requires the ``__init__`` method to take ``dataset`` argument.

In your class, override the ``transform`` method
that is used to obtain a two-dimensional vector (optimally, a Numpy array) from a string.

Unlike the classifier case, the ``predict`` method is already implemented for you,
so you don't have to tamper with it.

Those two steps should suffice to get your matcher up and running.
You can override the rest of the methods, namely ``save``, ``load``, and ``fit`` at your own convenience,
e.g. lack of those will not raise an error.

* ``fit`` should take a new dataset and retrain / update the underlying model.
* ``save`` and ``load`` are self-explanatory.
    You may use pickle, utils from joblib, or any other serializer.

.. code-block:: python

    class MyCustomMatcher(CosineMatcherMixin, ExtrasBaseModel):
        def __init__(self, model, dataset, namespace_key) -> None:
            CosineMatcherMixin.__init__(self, dataset)
            ExtrasBaseModel.__init__(self, namespace_key)
            self.model = model

        def transform(self, request: str):
            vector = self.model(request)
            return vector
