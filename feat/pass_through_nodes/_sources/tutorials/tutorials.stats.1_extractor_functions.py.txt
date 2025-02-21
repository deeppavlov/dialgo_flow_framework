# %% [markdown]
"""
# 1. Extractor Functions

The following example covers the basics of using the `stats` module.

Statistics are collected from pipeline services by extractor functions
that report the state of one or more pipeline components. The `stats` module
provides several default extractors, but users are free to define their own
extractor functions. You can find API reference for default extractors
[here](%doclink(api,stats.default_extractors)).

It is a preferred practice to define extractors as asynchronous functions.
Extractors need to have the following uniform signature:
the expected arguments are always `Context`, and `ExtraHandlerRuntimeInfo`,
while the expected return value is an arbitrary `dict` or a `None`.
The returned value gets persisted to Clickhouse as JSON
which is why it can contain arbitrarily nested dictionaries,
but it cannot contain any complex objects that cannot be trivially serialized.

The output of these functions will be captured by an OpenTelemetry
instrumentor and directed to the Opentelemetry collector server which
in its turn batches and persists data to Clickhouse or other OLAP storages.

<div class="alert alert-info">

Both the Opentelemetry collector and the Clickhouse instance must be running
during statistics collection.
If you cloned the Chatsky repo, launch them using `docker compose`:
```bash
docker compose --profile stats up
```

</div>

For more information on OpenTelemetry instrumentation,
refer to the body of this tutorial as well as [OpenTelemetry documentation](
https://opentelemetry.io/docs/instrumentation/python/manual/
).

"""

# %pip install chatsky[stats]=={chatsky}

# %%
import asyncio

from chatsky.core.service import (
    ExtraHandlerRuntimeInfo,
    ExtraHandlerType,
    to_service,
)
from chatsky import Context, Pipeline
from chatsky.stats import OtelInstrumentor, default_extractors
from chatsky.utils.testing import is_interactive_mode, check_happy_path
from chatsky.utils.testing.toy_script import TOY_SCRIPT_KWARGS, HAPPY_PATH

# %% [markdown]
"""
The cells below configure log export with the help of OTLP instrumentation.

* The initial step is to configure the export destination.
`from_url` method of the `OtelInstrumentor` class simplifies this task
allowing you to only pass the url of the OTLP Collector server.

* Alternatively, you can use the utility functions
provided by the `stats` module:
`set_logger_destination`, `set_tracer_destination`, or `set_meter_destination`.
These accept an appropriate Opentelemetry exporter instance
and bind it to provider classes.

* Nextly, the `OtelInstrumentor` class should be constructed to log the output
of extractor functions. Custom extractors need to be decorated
with the `OtelInstrumentor` instance. Default extractors are instrumented
by calling the `instrument` method.

* The entirety of the process is illustrated in the example below.

"""


# %%
chatsky_instrumentor = OtelInstrumentor.from_url("grpc://localhost:4317")
chatsky_instrumentor.instrument()

# %% [markdown]
"""
The following cell shows a custom extractor function. The data obtained from
the context and the runtime information gets shaped as a dict and returned
from the function body. The `chatsky_instrumentor` decorator then ensures
that the output is logged by OpenTelemetry.

"""


# %%
# decorated by an OTLP Instrumentor instance
@chatsky_instrumentor
async def get_service_state(ctx: Context, info: ExtraHandlerRuntimeInfo):
    # extract the execution state of a target service
    data = {
        "execution_state": info.component.get_state(ctx),
    }
    # return the state as an arbitrary dict for further logging
    return data


# %%
# configure `get_service_state` to run after the `heavy_service`
@to_service(after_handler=[get_service_state])
async def heavy_service(ctx: Context):
    _ = ctx  # get something from ctx if needed
    await asyncio.sleep(0.02)


# %%
pipeline = Pipeline(**TOY_SCRIPT_KWARGS, pre_services=[heavy_service])

pipeline.actor.add_extra_handler(
    ExtraHandlerType.BEFORE, default_extractors.get_current_label
)
if __name__ == "__main__":
    check_happy_path(pipeline, HAPPY_PATH, printout=True)
    if is_interactive_mode():
        pipeline.run()
