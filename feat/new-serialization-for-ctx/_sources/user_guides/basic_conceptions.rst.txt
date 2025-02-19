Basic Concepts
--------------

Introduction
~~~~~~~~~~~~

Chatsky is a modern tool for designing conversational services.

Chatsky introduces a specialized Domain-Specific Language (DSL) based on standard Python functions and data structures
which makes it very easy for developers with any level of expertise to design a script for user - bot interaction. 
The script comes in a form of a *dialog graph* where
each node equals a specific state of the dialog, i.e. a specific conversation turn.
The graph includes the majority of the conversation logic, and covers one or several user scenarios, all in a single Python dict.

In this tutorial, we describe the basics of Chatsky API,
and walk you through the process of creating and maintaining a conversational service with the help of Chatsky.


Creating Conversational Services with Chatsky
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Installation
============

To get started with Chatsky, you need to install its core dependencies, which can be done using the following command:

.. code-block:: shell

    pip3 install chatsky

Defining Dialogue Goals and User Scenarios
==========================================

To create a conversational service using Chatsky, you start by defining the overall dialogue goal
and breaking down the dialogue into smaller scenarios based on the user intents or actions that you want to cover.
Chatsky's Domain-Specific Language makes it easy to break down the dialog script into `flows`, i.e. named groups of nodes
unified by a specific purpose.

For instance, if one of the dialog options that we provide to the user is to play a game,
the bot can have a 'game' flow that contains dialog states related to this subject, while other flows
cover other topics, e.g. 'time' flow can include questions and answers related to telling the time,
'weather' to telling the weather, etc.

Creating Dialogue Flows for User Scenarios
==========================================

Once you have Chatsky installed, you can define dialog flows targeting various user scenarios
and combine them in a global script object. A flow consists of one or more nodes
that represent conversation turns.

.. note::

    In other words, the script object has 3 levels of nestedness:
    **script - flow - node**

Let's assume that the only user scenario of the service is the chat bot playing ping pong with the user.
The practical implementation of this is that the bot is supposed to reply 'pong' to messages that say 'ping'
and handle any other messages as exceptions. The pseudo-code for the said flow would be as follows:

.. code-block:: text

    1. User starts a conversation
    2. Respond with "Hi!"

    If user writes "Hello!":
        3. Respond with "Let's play ping-pong!"

        If user afterwards writes "Ping" or "ping" or "Ping!" or "ping!":
            4. Respond with "Pong!"
            Repeat this behaviour

    If user writes something else:
        5. Respond with "That was against the rules"
        Go to responding with "2" after user replies

This leaves us with a single dialog flow in the dialog graph that we lay down below, with the annotations for
each part of the graph available under the code snippet.

Example flow & script
=====================

.. code-block:: python
    :linenos:

    from chatsky import Pipeline, TRANSITIONS, RESPONSE, Transition as Tr
    import chatsky.conditions as cnd
    import chatsky.destinations as dst

    ping_pong_script = {
        "greeting_flow": {
            "start_node": {
                TRANSITIONS: [Tr(dst="greeting_node", cnd=cnd.ExactMatch("/start"))]
                # start node handles the initial handshake (command /start)
            },
            "greeting_node": {
                RESPONSE: "Hi!",
                TRANSITIONS: [
                    Tr(
                        dst=("ping_pong_flow", "game_start_node"),
                        cnd=cnd.ExactMatch("Hello!")
                    )
                ]
            },
            "fallback_node": {
                RESPONSE: "That was against the rules",
                TRANSITIONS: [Tr(dst="greeting_node")],
                                # this transition is unconditional
            },
        },
        "ping_pong_flow": {
            "game_start_node": {
                RESPONSE: "Let's play ping-pong!",
                TRANSITIONS: [Tr(dst="response_node", cnd=cnd.ExactMatch("Ping!"))],
            },
            "response_node": {
                RESPONSE: "Pong!",
                TRANSITIONS: [Tr(dst=dst.Current(), cnd=cnd.ExactMatch("Ping!"))],
            },
        },
    }

    pipeline = Pipeline(
        ping_pong_script,
        start_label=("greeting_flow", "start_node"),
        fallback_label=("greeting_flow", "fallback_node"),
    )

    if __name__ == "__main__":
        pipeline.run()

An example chat with this bot:

.. code-block::

    request: /start
    response: text='Hi!'
    request: Hello!
    response: text='Let's play ping-pong!'
    request: Ping!
    response: text='Pong!'
    request: Bye
    response: text='That was against the rules'

The order of request processing is, essentially:

1. Obtain user request
2. Travel to the next node (chosen based on transitions of the current node)
3. Send the response of the new node

Below is a breakdown of key features used in the example:

* ``ping_pong_script``: The dialog **script** mentioned above is a dictionary that has one or more
  dialog flows as its values.

* ``ping_pong_flow`` is the game emulation flow; it contains linked
  conversation nodes and possibly some extra data, transitions, etc.

* A node object is an atomic part of the script.
  The required fields of a node object are ``RESPONSE`` and ``TRANSITIONS``.

* The ``RESPONSE`` field specifies the response that the dialog agent gives to the user in the current turn.

* The ``TRANSITIONS`` field specifies the edges of the dialog graph that link the dialog states.
  This is a list of ``Transition`` instances. They specify the destination node of the potential transition
  and a condition for the transition to be valid.
  In the example script, we use build-in functions: ``ExactMatch`` requires the user request to
  fully match the provided text, while ``Current`` makes a transition to the current node.
  However, passing custom callbacks that implement arbitrary logic is also an option.

* ``start_node`` is the initial node, which contains an empty response and only transfers user to another node
  according to the first message user sends.
  It transfers user to ``greeting_node`` if user writes text message exactly equal to "Hello!".

* ``greeting_node`` is the node that will greet user and propose him a ping-pong game.
  It transfers user to ``response_node`` if user writes text message exactly equal to "Ping!".

* ``response_node`` is the node that will play ping-pong game with the user.
  It transfers user to ``response_node`` if user writes text message exactly equal to "Ping!".

* ``fallback_node`` is an "exception handling node"; user will be transferred here if
  none of the transition conditions (see ``TRANSITIONS``) is satisfied.
  It transfers user to ``greeting_node`` no matter what user writes.

* ``pipeline`` is a special object that traverses the script graph based on the values of user input.
  It is also capable of executing custom actions that you want to run on every turn of the conversation.
  The pipeline can be initialized with a script, and with labels of two nodes:
  the entrypoint of the graph, aka the 'start node', and the 'fallback node'
  (if not provided it defaults to 'start node').

.. note::

    See `tutorial on basic dialog structure <../tutorials/tutorials.script.core.1_basics.html>`_.

Processing Definition
=====================

.. note::

    The topic of this section is explained in greater detail in the following tutorials:

    * `Pre-response processing <../tutorials/tutorials.script.core.7_pre_response_processing.html>`_
    * `Pre-transition processing <../tutorials/tutorials.script.core.9_pre_transition_processing.html>`_
    * `Pipeline processors <../tutorials/tutorials.pipeline.2_pre_and_post_processors.html>`_

Processing user requests and extracting additional parameters is a crucial part of building a conversational bot. 
Chatsky allows you to define how user requests will be processed to extract additional parameters.
This is done by passing callbacks to a special ``PROCESSING`` fields in a Node dict.

* ``PRE_RESPONSE`` will happen **after** a transition has been made but **before** response generation. See `tutorial on pre-response processing`_.
* ``PRE_TRANSITION`` will happen **after** obtaining user request but **before** transition to the next node. See `tutorial on pre-transition processing`_.

Depending on the requirements of your bot and the dialog goal, you may need to interact with external databases or APIs to retrieve data. 
For instance, if a user wants to know a schedule, you may need to access a database and extract parameters such as date and location.

.. code-block:: python

    import requests
    from chatsky import BaseProcessing, PRE_TRANSITION
    ...
    class UseAPI(BaseProcessing):
        async def call(self, ctx):
            # save to the context field for custom info
            ctx.misc["api_call_results"] = requests.get("http://schedule.api/day1").json()
    ...
    node = {
        RESPONSE: ...
        TRANSITIONS: ...
        PRE_TRANSITION: {"use_api": UseAPI()}
    }

.. note::

    This function uses ``Context`` to store the result of a request for other functions to use.
    Context is a data structure that keeps all the information about a specific conversation.

    To learn more about ``Context`` see the `relevant guide <../user_guides/context_guide.html>`__.

If you retrieve data from the database or API, it's important to validate it to ensure it meets expectations.

Generating a bot Response
=========================

Response is defined in the ``RESPONSE`` section of each node and should be either a ``Message`` object,
that can contain text, images, audios, attachments, etc., or a callback that returns a ``Message``.
The latter allows you to customize the response based on the specific scenario and user input.

.. note::

    ``Message`` object can be instantiated from a string (filling its ``text`` field).
    We've used this feature for ``RESPONSE`` and will use it now.

.. code-block:: python

    class MyResponse(BaseResponse):
        async def call(self, ctx):
            if ctx.misc["user"] == 'vegan':
                return "Here is a list of vegan cafes."
            return "Here is a list of cafes."


For more information on responses, see the `tutorial on response functions`_.

Handling Fallbacks
==================

In Chatsky, you should provide handling for situations where the user makes requests
that do not trigger any of the transitions specified in the script graph. 
To cover that use case, Chatsky requires you to define a fallback node that the agent will move to
when no adequate transition has been found.

Like other nodes, the fallback node can either use a message or a callback to produce a response
which gives you a lot of freedom in creating situationally appropriate error messages.
Create friendly error messages and, if possible, suggest alternative options. 
This ensures a smoother user experience even when the bot encounters unexpected inputs.

.. code-block:: python

    class MyResponse(BaseResponse):
        """
        Generate a special fallback response depending on the situation.
        """
        async def call(self, ctx):
            if ctx.last_label == ctx.pipeline.start_label and ctx.last_request.text != "/start":
                # start_label can be obtained from the pipeline instance stored inside context
                return "You should've started the dialog with '/start'"
            else:
                return (
                    f"That was against the rules!\n"
                    f"You should've written 'Ping', not '{ctx.last_request.text}'!"
                )

Testing and Debugging
~~~~~~~~~~~~~~~~~~~~~

Periodically testing the conversational service is crucial to ensure it works correctly.
You should also be prepared to debug the code and dialogue logic if problems are discovered during testing. 
Thorough testing helps identify and resolve any potential problems in the conversation flow.

The basic testing procedure offered by Chatsky is end-to-end testing of the pipeline and the script
which ensures that the pipeline yields correct responses for any given input.
It requires a sequence of user request - bot response pairs that form the happy path of your
conversational service.

.. code-block:: python

    happy_path = (
        ("/start", "Hi!"),
        ("Hello!", "Let's play ping-pong!"),
        ("Ping!", "Pong!")
    )

A special function is then used to ascertain complete identity of the messages taken from
the happy path and the pipeline. The function will play out a dialog with the pipeline acting as a user while checking returned messages.

.. code-block:: python

    from chatsky.utils.testing.common import check_happy_path

    check_happy_path(pipeline, happy_path)

Monitoring and Analytics
~~~~~~~~~~~~~~~~~~~~~~~~

Setting up bot performance monitoring and usage analytics is essential to monitor its operation and identify potential issues. 
Monitoring helps you understand how users are interacting with the bot and whether any improvements are needed.
Analytics data can provide valuable insights for refining the bot's behavior and responses.

Chatsky provides a `statistics` module as an out-of-the-box solution for collecting arbitrary statistical metrics
from your service. Setting up the data collection is as easy as instantiating the relevant class in the same
context with the pipeline. 
What's more, the data you obtain can be visualized right away using Apache Superset as a charting engine.

.. note::

    More information is available in the respective `guide <../user_guides/superset_guide.html>`__.

Iterative Improvement
~~~~~~~~~~~~~~~~~~~~~

To continually enhance your chat-bot's performance, monitor user feedback and analyze data on bot usage.
For instance, the statistics or the charts may reveal that some flow is visited by users more frequently or
less frequently than planned. This would mean that adjustments to the transition structure
of the graph need to be made.

Gradually improve the transition logic and response content based on the data received. 
This iterative approach ensures that the bot becomes more effective over time.

Data Protection
~~~~~~~~~~~~~~~

Data protection is a critical consideration in bot development, especially when handling sensitive information.

.. note::

    The Chatsky framework helps ensure the safety of your application by storing the history and other user data present
    in the ``Context`` object under unique ids and abstracting the storage logic away from the user interface.
    As a result, it offers the basic level of data protection making it impossible to gain unlawful access to personal information.

Documentation
~~~~~~~~~~~~~

Creating documentation is essential for teamwork and future bot maintenance. 
Document how different parts of the script work and how the bot covers the expected interaction scenarios.
It is especially important to document the purpose and functionality of callback functions and pipeline services
that you may have in your project, using Python docstrings.

.. code-block:: python

    class FavCuisineResponse(BaseResponse):
        """
        This function returns a user-targeted response depending on the value
        of the 'cuisine preference' slot.
        """
        ...

This documentation serves as a reference for developers involved in the project.

Scaling
~~~~~~~

If your bot becomes popular and requires scaling, consider scalability during development.
Scalability ensures that the bot can handle a growing user base without performance issues.
While having only one application instance will suffice in most cases, there are many ways
how you can adapt the application to a high load environment.

* With the database connection support that Chatsky offers out of the box, Chatsky projects can be easily scaled through sharing the same database between multiple application instances. However, using an external database is required due to the fact that this is the only kind of storage that can be efficiently shared between processes.
* Likewise, using multiple database instances to ensure the availability of data is also an option.
* The structure of the `Context` object makes it easy to vertically partition the data storing different subsets of data across multiple database instances.

Further reading
~~~~~~~~~~~~~~~

* `Tutorial on basic dialog structure <../tutorials/tutorials.script.core.1_basics.html>`_
* `Tutorial on transitions <../tutorials/tutorials.script.core.4_transitions.html>`_
* `Tutorial on conditions <../tutorials/tutorials.script.core.2_conditions.html>`_
* `Tutorial on response functions <../tutorials/tutorials.script.core.3_responses.html>`_
* `Tutorial on pre-response processing <../tutorials/tutorials.script.core.7_pre_response_processing.html>`_
* `Tutorial on pre-transition processing <../tutorials/tutorials.script.core.9_pre_transition_processing.html>`_
* `Guide on Context <../user_guides/context_guide.html>`_
* `Tutorial on global and local nodes <../tutorials/tutorials.script.core.5_global_local.html>`_
* `Tutorial on context serialization <../tutorials/tutorials.script.core.6_context_serialization.html>`_
* `Tutorial on script MISC <../tutorials/tutorials.script.core.8_misc.html>`_
