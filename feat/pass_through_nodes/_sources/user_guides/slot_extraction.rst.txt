Slot Extraction
---------------

Introduction
~~~~~~~~~~~~

Extracting and filling slots is an essential part of any conversational service
that comprises the inherent business logic. Like most frameworks, Chatsky
provides components that address this task as a part of its ``slots`` module.
These can be easily customized to leverage neural networks specifically designed
for slot extraction or any other logic you might want to integrate.

API overview
~~~~~~~~~~~~

Defining slots
==============

The basic building block of the API is the
`BaseSlot <../apiref/chatsky.slots.slots.html#chatsky.slots.slots.BaseSlot>`_ class
and its descendants that vary depending on the value extraction logic.
Each slot has a name by which it can be accessed and a method for extracting values.
Below, we demonstrate the most basic class that extracts values
from user utterances using a regular expression:
`RegexpSlot <../apiref/chatsky.slots.slots.html#chatsky.slots.types.RegexpSlot>`_.

.. code-block:: python

    from chatsky.slots import RegexpSlot
    ...
    email_slot = RegexpSlot(regexp=r"[a-z@\.A-Z]+")

The slots can implement arbitrary logic including requests to external services.
For instance, Deeppavlov library includes a number of models that may be of use for slot
extraction task. In particular, we will demonstrate the use of the following
`NER model <https://docs.deeppavlov.ai/en/master/features/models/NER.html>`_
that was trained and validated on the conll_2003 dataset.

.. code-block:: shell

    docker pull deeppavlov/deeppavlov:latest
    docker run -d --name=ner \
        -e CONFIG=ner_conll2003_bert \
        -p 5000:5000 \
        -v ~/.deeppavlov:/root/deeppavlov \
        -v ~/.cache:/root/cache \
        deeppavlov/deeppavlov:latest

Now that you have a Deeppavlov docker image running on port 5000, you can take the following steps to take
full advantage of its predictions.

.. code-block:: python

    import requests
    from chatsky.slots import FunctionSlot
    from chatsky import Message

    # we assume that there is a 'NER' service running on port 5000 
    async def extract_first_name(utterance: Message) -> str:
        """Return the first entity of type B-PER (first name) found in the utterance."""
        ner_request = requests.post(
            "http://localhost:5000/model",
            json={"x": [utterance.text]}
        )
        ner_tuple = ner_request.json()
        if "B-PER" not in ner_tuple[1][0]:
            return ""
        return ner_tuple[0][0][ner_tuple[1][0].index("B-PER")]

    name_slot = FunctionSlot(func=extract_first_name)

Individual slots can be grouped allowing the developer to access them together
as a namespace. This can be achieved using the
`GroupSlot <../apiref/chatsky.slots.slots.html#chatsky.slots.slots.GroupSlot>`_
component that is initialized with other slot instances as its children.
The group slots also allows for arbitrary nesting, i.e. it is possible to include
group slots in other group slots.

.. code-block:: python

    from chatsky.slots import GroupSlot

    profile_slot = GroupSlot(name=name_slot, email=email_slot)

After defining all your slots, pass ``GroupSlot`` as pipeline's `slots` argument.
That slot is a root slot: it contains all other group and value slots.

.. code-block:: python

    from chatsky import Pipeline

    pipeline = Pipeline(..., slots=profile_slot)

Slot names
==========

Any slot can be accessed by a slot name:
A dot-separated string that acts as a path from the root slot to the needed slot.

In the example above ``name_slot`` would have the name "name"
because that is the key used to store it in the ``profile_slot``.

Let's consider nested group slots:

.. code-block:: python

    from chatsky.slots import GroupSlot

    root_slot = GroupSlot(profile=GroupSlot(name=name_slot, email=email_slot))

In this example ``name_slot`` would be accessible by the "profile.name" name.

Using slots
===========

Slots can be extracted at the ``PRE_TRANSITION`` stage
using the `Extract <../apiref/chatsky.processing.slots.html#chatsky.processing.slots.Extract>`_
function from the `processing` submodule.
You can pass any number of names of the slots that you want to extract to this function.

.. code-block:: python

    from chatsky import proc

    PRE_TRANSITION: {"extract_first_name": proc.Extract("name", "email")}

The `conditions` submodule provides a function for checking if specific slots have been extracted.

.. code-block:: python
    
    from chatsky import cnd

    TRANSITIONS: [Tr(dst="all_information", cnd=cnd.SlotsExtracted("name", "email", mode="all"))]
    TRANSITIONS: [Tr(dst="partial_information", cnd=cnd.SlotsExtracted("name", "email", mode="any"))]

.. note::

    You can combine ``slots_extracted`` with the
    `Negation <../apiref/chatsky.conditions.standard.html#chatsky.conditions.standard.Negation>`_
    condition to make a transition to an extractor node if a slot has not been extracted yet.

Both `processing` and `response` submodules provide functions for filling templates with
extracted slot values.

- ``FillTemplate`` changes the response function of the current node to ``FilledTemplate`` at the ``PRE_RESPONSE`` level.
- ``FilledTemplate`` is a response function that fills slots after producing inner response
  (which can be ``BaseResponse``, ``Message`` or simply a string).

.. code-block:: python
    
    from chatsky import proc, rsp

    PRE_RESPONSE: {"fill_response_slots": proc.FillTemplate()}
    RESPONSE: "Your first name: {name}"


    RESPONSE: rsp.FilledTemplate("Your first name: {name}")

An example of script utilizing slot extraction can be found in the
`tutorials section <../tutorials/tutorials.slots.1_basic_example.html>`_.

Further reading
===============

All of the functions described in the previous sections call methods of the
`SlotManager <../apiref/chatsky.slots.slots.html#chatsky.slots.slots.SlotManager>`_
class under the hood.

An instance of this class can be accessed in runtime via ``ctx.framework_data.slot_manager``.

This class allows for more detailed access to the slots API.
For example, you can access exceptions that occurred during slot extraction:

.. code-block:: python

    slot_manager = ctx.framework_data.slot_manager
    extracted_value = slot_manager.get_extracted_slot("name")
    exception = extracted_value.extracted_value if not extracted_value.is_slot_extracted else None
