Pipeline YAML import guide
--------------------------

Introduction
~~~~~~~~~~~~

Instead of passing all the arguments to pipeline from a python environment,
you can initialize pipeline by getting the arguments from a file.

The details of this process are described in this guide.

Basics
~~~~~~

To initialize ``Pipeline`` from a file, call its `from_file <../apiref/chatsky.core.pipeline.html#chatsky.core.pipeline.Pipeline.from_file>`_
method. It accepts a path to a file, a path to a custom code directory and overrides.

File
====

The file should be a json or yaml file that contains a dictionary.
They keys in the dictionary are the names of pipeline init parameters and the values are the values of the parameters.

Below is a minimalistic example of such a file:

.. code-block:: yaml

    script:
        flow:
            node:
                RESPONSE: Hi
                TRANSITIONS:
                    - dst: node
                      cnd: true
                      priority: 2
    start_label:
        - flow
        - node

.. note::

    If you're using yaml files, you need to install pyyaml:

    .. code-block:: sh

        pip install chatsky[yaml]


Custom dir
==========

Custom directory allows using any objects inside the yaml file.

More on that in the :ref:`object-import` section.

Overrides
=========

Any pipeline init parameters can be passed to ``from_file``.
They will override parameters defined in the file (or add them if they are not defined in the file).

.. _object-import:

Object Import
~~~~~~~~~~~~~

JSON values are often not enough to build any serious script.

For this reason, the init parameters in the pipeline file are preprocessed in two ways:

String reference replacement
============================

Any string that begins with either ``chatsky.``, ``custom.`` or ``external:`` is replaced with a corresponding object.

The ``chatsky.`` prefix indicates that an object should be found inside the ``chatsky`` library.
For example, string ``chatsky.cnd.ExactMatch`` will be replaced with the ``chatsky.cnd.ExactMatch`` object (which is a class).

The ``custom.`` prefix allows importing object from the custom directory passed to ``Pipeline.from_file``.
For example, string ``custom.my_response`` will be replaced with the ``my_response`` object defined in ``custom/__init__.py``
(or will throw an exception if there's no such object).

The ``external:`` prefix can be used to import any objects (primarily, from external libraries).
For example, string ``external:os.getenv`` will be replaced with the function ``os.getenv``.

.. note::

    It is highly recommended to read about the import process for these strings
    `here <../apiref/chatsky.core.script_parsing.html#chatsky.core.script_parsing.JSONImporter.resolve_string_reference>`_.

.. note::

    If you want to use different prefixes, you can edit the corresponding class variables of the
    `JSONImporter <../apiref/chatsky.core.script_parsing.html#chatsky.core.script_parsing.JSONImporter>`_ class:

    .. code-block:: python

        from chatsky.core.script_parsing import JSONImporter
        from chatsky import Pipeline

        JSONImporter.CHATSKY_NAMESPACE_PREFIX = "_chatsky:"

        pipeline = Pipeline.from_file(...)

    After changing the prefix variable, ``from_file`` will no longer replace strings that start with ``chatsky.``.
    (and will replace strings that start with ``_chatsky:``)

Single-key dict replacement
===========================

Any dictionary containing a **single** key that **begins with any of the prefixes** described in the previous section
will be replaced with a call result of the object referenced by the key.

Call is made with the arguments passed as a value of the dictionary:

- If the value is a dictionary; it is passed as kwargs;
- If the value is a list; it is passed as args;
- If the value is ``None``; no arguments are passed;
- Otherwise, the value is passed as the only arg.

.. list-table:: Examples
    :widths: auto
    :header-rows: 1

    * - YAML string
      - Resulting object
      - Note
    * - .. code-block:: yaml

            external:os.getenv: TOKEN
      - .. code-block:: python

            os.getenv("TOKEN")
      - This falls into the 4th condition (value is not a dict, list or None) so it is passed as the only argument.
    * - .. code-block:: yaml

            chatsky.dst.Previous:
      - .. code-block:: python

            chatsky.dst.Previous()
      - The value is ``None``, so there are no arguments.
    * - .. code-block:: yaml

            chatsky.dst.Previous
      - .. code-block:: python

            chatsky.dst.Previous
      - This is not a dictionary, the resulting object is a class!
    * - .. code-block:: yaml

            chatsky.cnd.Regexp:
                pattern: "yes"
                flags: external:re.I
      - .. code-block:: python

            chatsky.cnd.Regexp(
                pattern="yes",
                flags=re.I
            )
      - The value is a dictionary; it is passed as kwargs.
        This also showcases that replacement is recursive ``external:re.I`` is replaced as well.
    * - .. code-block:: yaml

            chatsky.proc.Extract:
                - person.name
                - person.age
      - .. code-block:: python

            chatsky.proc.Extract(
                "person.name",
                "person.age"
            )
      - The value is a list; it is passed as args.

Further reading
~~~~~~~~~~~~~~~

* `API ref <../apiref/chatsky.core.script_parsing.html>`_
* `Comprehensive example <https://github.com/deeppavlov/chatsky/tree/master/utils/pipeline_yaml_import_example>`_
