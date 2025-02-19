Optimization Guide
------------------

Introduction
~~~~~~~~~~~~

When optimizing a dialog service to provide the best possible user experience,
it's essential to identify and address performance issues.
Similar to any complex system, a dialog service can have performance bottlenecks at various levels.
These bottlenecks can occur during I/O operations like receiving and sending messages,
as well as when synchronizing service states with a database.
As the number of callbacks in the script and pipeline increases,
the performance of Chatsky classes can degrade leading to longer response time.

As a result, it becomes necessary to locate the part of the pipeline that is causing issues, so that
further optimization steps can be taken. Chatsky provides several tools that address the need for
profiling individual system components. This guide will walk you through the process
of using these tools in practice and optimizing the profiled application.

Profiling with Locust testing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`Locust <https://locust.io/>`__ is a tool for load testing web applications that
simultaneously spawns several user agents that execute a pre-determined behavior
against the target application. Assuming that your pipeline is integrated into a web
server application, like Flask or FastAPI, that is not strongly impacted by the load,
the load testing reveals how well your pipeline would scale to a highly loaded environment.
Using this approach, you can also measure the scalability of each component in your pipeline,
if you take advantage of the Opentelemetry package bundled with the library (`stats` extra required)
as described below.

Since Locust testing can only target web apps,
this approach only applies if you integrate your dialog pipeline into a web application.
The `FastAPI integration tutorial <../tutorials/tutorials.messengers.web_api_interface.1_fastapi.py>`_
shows the most straightforward way to do this.
At this stage, you will also need to instrument the pipeline components that you want to additionally profile
using `extractor functions`. Put simply, you are decorating the components of the pipeline
with functions that can report their performance, e.g. their execution time or the CPU load.

.. note::

    You can get more info on how instrumentation is done and statistics are collected
    in the `stats tutorial <../tutorials/tutorials.stats.1_extractor_functions.py>`__.

When you are done setting up the instrumentation, you can launch the web server to accept connections from locust.

The final step is to run a Locust file which will result in artificial load traffic being generated and sent to your server.
A Locust file is a script that implements the behavior of artificial users,
i.e. the requests to the server that will be made during testing.

.. note::

    An example Locust script along with instructions on how to run it can be found in the
    `load testing tutorial <../tutorials/tutorials.messengers.web_api_interface.3_load_testing_with_locust.py>`_.
    The simplest way, however, is to pass a locust file to the Python interpreter.

Once Locust is running, you can access its GUI, where you can set the number of users to emulate.
After configuring this parameter, the active phase of testing will begin,
and the results will become accessible on an interactive dashboard.
These reported values include timing data, such as the average response time of your service,
allowing you to assess the performance's reasonableness and impact on user experience.

The data provided by extractor functions will be available in the Clickhouse database;
you can view it using the Superset dashboard (see `instructions <./superset_guide.html>`__)
or analyze it with your own queries using the Clickhouse client.

Profiling context storages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Benchmarking the performance of context storage is crucial to understanding
how different storage methods impact your dialog service's efficiency.
This process involves running tests to measure the speed and reliability of various context storage solutions.
Given the exact configuration of your system, one or the other database type may be performing more efficiently,
so you may prefer to change your database depending on the testing results.

.. note::
    The exact instructions of how the testing can be carried out are available in the
    `DB benchmarking tutorial <../tutorials/tutorials.context_storages.8_db_benchmarking.py>`__.

Optimization techniques
~~~~~~~~~~~~~~~~~~~~~~~

Aside from choosing an appropriate database type, there exists a number of other recommendations
that may help you improve the efficiency of your service.

* Firstly, follow the DRY principle not only with regard to your code, but also with regard to
  computational operations. In other words, you have to make sure that your callback functions work only once
  during a dialog turn and only when needed. E.g. you can take note of the `conditions` api available as a part
  of the `Pipeline` module: while normally a pipeline service runs every turn, you can restrict it
  to only run on turns when a particular condition is satisfied, greatly reducing
  the number of performed actions (see the
  `Groups and Conditions tutorial <../tutorials/tutorials.pipeline.4_groups_and_conditions_full.py>`__).

* Using caching for resource-consuming callbacks and actions may also prove to be a helpful strategy.
  In this manner, you can improve the computational efficiency of your pipeline,
  while making very few changes to the code itself.

* Finally, be mindful about the use of computationally expensive algorithms, like NLU classifiers
  or LLM-based generative networks, since those require a great deal of time and resources
  to produce an answer. In case you need to use one, take full advantage of caching along with
  other means to relieve the computational load imposed by neural networks such as message queueing.

..
    todo: add a link to a user guide about using message queueing.
