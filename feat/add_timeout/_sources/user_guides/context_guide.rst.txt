Context guide
--------------

Introduction
~~~~~~~~~~~~

The ``Context`` class is a backbone component of the Chatsky API.
Like the name suggests, this data structure is used to store information
about the current state, or context, of a particular conversation.
Each individual user has their own ``Context`` instance and can be identified by it.

``Context`` is used to keep track of the user's requests, bot's replies,
user-related and request-related annotations, and any other information
that is relevant to the conversation with the user.

.. note::

    Since most callback functions used in Chatsky script and Chatsky pipeline (see the `basic guide <./basic_conceptions.rst>`__)
    need to either read or update the current dialog state,
    the framework-level convention is that all functions of this kind
    use ``Context`` as their first parameter. This dependency is being
    injected by the pipeline during its run. 
    Thus, understanding the ``Context`` class is essential for developing custom conversation logic
    which is mostly made up by the said functions.

As a callback parameter, ``Context`` provides a convenient interface for working with data,
allowing developers to easily add, retrieve,
and manipulate data as the conversation progresses.

Let's consider some of the built-in callback instances to see how the context can be leveraged:

.. code-block:: python
    :linenos:

    class Regexp(BaseCondition):
        pattern: str

        @cached_property
        def re_object(self) -> Pattern:
            return re.compile(self.pattern)

        async def call(self, ctx: Context) -> bool:
            request = ctx.last_request
            if request.text is None:
                return False
            return bool(self.re_object.search(request.text))

The code above is a condition function (see the `conditions tutorial <../tutorials/tutorials.script.core.2_conditions.py>`__)
that belongs to the ``TRANSITIONS`` section of the script and returns `True` or `False`
depending on whether the current user request matches the given pattern.

As can be seen from the code block, the current
request (``last_request``) can be retrieved as one of the attributes of the ``Context`` object.
Likewise, the ``last_response`` (bot's current reply) or the ``last_label``
(the name of the current node) attributes can be used in the same manner.

Another common use case is leveraging the ``misc`` field (see below for a detailed description):
pipeline functions or ``PROCESSING`` callbacks can write arbitrary values to the misc field,
making those available for other context-dependent functions.

.. code-block:: python
    :linenos:

    import urllib.request
    import urllib.error

    class PingExample(BaseProcessing):
        async def call(self, ctx):
            try:
                with urllib.request.urlopen("https://example.com/") as webpage:
                    web_content = webpage.read().decode(
                        webpage.headers.get_content_charset()
                    )
                    result = "Example Domain" in web_content
            except urllib.error.URLError:
                result = False
            ctx.misc["can_ping_example_com"] = result

..
    todo: link to the user defined functions tutorial

    .. note::
        For more information about user-defined functions see the `user functions guide <./user_functions.rst>`__.

API
~~~

This sections describes the API of the ``Context`` class.

For more information, such as method signatures, see
`API reference <../apiref/chatsky.core.context.html#chatsky.core.context.Context>`__.

Attributes
==========

* **id**: This attribute represents the unique context identifier. By default, it is randomly generated using uuid4.
  In most cases, this attribute will be used to identify a user.

* **labels**: The labels attribute stores the history of all passed labels within the conversation.
  It maps turn IDs to labels. The is a special dynamically-loaded collection of type
  `ContextDict <../apiref/chatsky.core.ctx_dict.html#chatsky.core.ctx_dict.ContextDict>`_,
  so some of its methods are asynchronous.
  In particular, it is required to await when getting items (e.g. ``await ctx.labels[1]``).

* **requests**: The requests attribute maintains the history of all received requests by the agent.
  It also maps turn IDs to requests. The is a special dynamically-loaded collection of type
  `ContextDict <../apiref/chatsky.core.ctx_dict.html#chatsky.core.ctx_dict.ContextDict>`_,
  so some of its methods are asynchronous.
  In particular, it is required to await when getting items (e.g. ``await ctx.requests[1]``).

* **responses**: This attribute keeps a record of all agent responses, mapping turn IDs to responses.
  The is a special dynamically-loaded collection of type
  `ContextDict <../apiref/chatsky.core.ctx_dict.html#chatsky.core.ctx_dict.ContextDict>`_,
  so some of its methods are asynchronous.
  In particular, it is required to await when getting items (e.g. ``await ctx.responses[1]``).

* **current_turn_id**: ID of the current turn. Can be used to access specific labels/requests/responses.
  E.g. ``ctx.requests[ctx.current_turn_id - 1]`` gives the request received on the previous turn.

  Turn ids are integers that go up by 1 every turn.
  Turn id 0 is reserved for start label and does not have any associated response or request.
  The first request from user has the turn id of 1.

* **misc**: The misc attribute is a dictionary object for storing custom data. This field is not used by any of the
  built-in Chatsky classes or functions, so the values that you write there are guaranteed to persist
  throughout the lifetime of the ``Context`` object.

* **framework_data**: This attribute is used for storing custom data required for pipeline execution.
  It is meant to be used by the framework only. Accessing it may result in pipeline breakage.
  But there are some methods that provide access to specific fields of framework data.
  These methods are described in the next section.

Methods
=======

The most useful methods of the ``Context`` class are the following:

* **last_request**: Return the last request of the context.

* **last_response**: Return the last response of the context, or `None` if the ``responses`` field is empty.

  Responses are added at the end of each turn, so an empty ``response`` field is something you should definitely consider.

* **last_label**: Return the last node label of the context (i.e. name of the current node).

* **current_node**: Return the current node of the context.
  Use this property to access properties of the current node.
  You can safely modify properties of this. The changes will be reflected in
  bot behaviour during this turn, bot are not permanent (the node stored inside the script is not changed).

  .. note::

    See the `preprocessing tutorial <../tutorials/tutorials.script.core.7_pre_response_processing.py>`__.

* **pipeline**: Return ``Pipeline`` object that is used to process this context.
  This can be used to get ``Script``, ``start_label`` or ``fallback_label``.

* **turns**: A list-like property that allows iterating over requests, labels and responses by turn id.

Context storages
~~~~~~~~~~~~~~~~

Since context instances contain all the information, relevant for a particular user, there needs to be a way
to persistently store that information and to make it accessible in different user sessions.
This functionality is implemented by the ``context storages`` module that provides 
the uniform ``DBContextStorage`` interface as well as child classes thereof that integrate
various database types (see the
`api reference <../apiref/chatsky.context_storages.database.html#chatsky.context_storages.database.DBContextStorage>`_).

The supported storage options are as follows:

* `In-memory storage`
* `JSON <https://www.json.org/json-en.html>`_
* `pickle <https://docs.python.org/3/library/pickle.html>`_
* `shelve <https://docs.python.org/3/library/shelve.html>`_
* `SQLite <https://www.sqlite.org/index.html>`_
* `PostgreSQL <https://www.postgresql.org/>`_
* `MySQL <https://www.mysql.com/>`_
* `MongoDB <https://www.mongodb.com/>`_
* `Redis <https://redis.io/>`_
* `Yandex DataBase <https://ydb.tech/>`_

``DBContextStorage`` instances can be uniformly constructed using the ``context_storage_factory`` function.
The function's only parameter is a connection string that specifies both the database type
and the connection parameters, for example, *mongodb://admin:pass@localhost:27016/admin*.
(`see the reference <../apiref/chatsky.context_storages.database.html#chatsky.context_storages.database.context_storage_factory>`_)

.. note::
    To learn how to use ``context_storage_factory`` in your pipeline, see our `Context Storage Tutorials <../tutorials/index_context_storages.html>`__.

The GitHub-based distribution of Chatsky includes Docker images for each of the supported database types.
Therefore, the easiest way to deploy your service together with a database is to clone the GitHub
distribution and to take advantage of the packaged :github_source_link:`docker compose file <compose.yml>`.

.. code-block:: shell
  :linenos:

  git clone https://github.com/deeppavlov/chatsky.git
  cd chatsky
  # assuming we need to deploy mongodb
  docker compose up mongo

The images can be configured using the docker compose file or the
:github_source_link:`environment file <.env_file>`,
also available in the distribution. Consult these files for more options.
