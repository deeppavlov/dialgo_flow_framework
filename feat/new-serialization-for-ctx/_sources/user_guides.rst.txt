User guides
-----------

:doc:`Basic concepts <./user_guides/basic_conceptions>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the ``basic concepts`` guide the basics of Chatsky are described,
those include but are not limited to: dialog graph creation, specifying start and fallback nodes,
setting transitions and conditions, using ``Context`` object in order to receive information
about current script execution.

:doc:`LLM Integration guide <./user_guides/llm_integration>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The LLM Integration guide covers functionality for incorporating LLMs into the script for
generating responses and checking conditions.

:doc:`Slot extraction <./user_guides/slot_extraction>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``slot extraction`` guide demonstrates the slot extraction functionality
currently integrated in the library. Chatsky only provides basic building blocks for this task,
which can be trivially extended to support any NLU engine or slot extraction model
of your liking.

:doc:`Context guide <./user_guides/context_guide>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``context guide`` walks you through the details of working with the
``Context`` object, the backbone of the Chatsky API, including most of the relevant fields and methods.

:doc:`Superset guide <./user_guides/superset_guide>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``superset guide`` highlights the usage of Superset visualization tool
for exploring the telemetry data collected from your conversational services.
We show how to plug in the telemetry collection and configure the pre-built
Superset dashboard shipped with Chatsky.

:doc:`Optimization guide <./user_guides/optimization_guide>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``optimization guide`` demonstrates various tools provided by the library
that you can use to profile your conversational service,
and to locate and remove performance bottlenecks.

:doc:`YAML import guide <./user_guides/pipeline_import>`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``yaml import guide`` shows another option for initializing ``Pipeline``
objects -- from yaml or json files.

.. toctree::
   :hidden:

   user_guides/basic_conceptions
   user_guides/llm_integration
   user_guides/slot_extraction
   user_guides/context_guide
   user_guides/superset_guide
   user_guides/optimization_guide
   user_guides/pipeline_import
