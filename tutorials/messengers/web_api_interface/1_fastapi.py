# %% [markdown]
"""
# Web API: 1. FastAPI

This tutorial shows how to create an API for DFF using FastAPI and
introduces messenger interfaces.

You can see the result at http://127.0.0.1:8000/docs.

Here, %mddoclink(api,messengers.common.interface,CallbackMessengerInterface)
is used to process requests.

%mddoclink(api,script.core.message,Message) is used in creating a JSON Schema for the endpoint.
"""

# %pip install dff uvicorn fastapi

# %%
from dff.messengers.common.interface import CallbackMessengerInterface
from dff.script import Message
from dff.pipeline import Pipeline
from dff.utils.testing import TOY_SCRIPT_ARGS, is_interactive_mode

import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI

# %% [markdown]
"""
Messenger interfaces establish communication between users and the pipeline.
They manage message channel initialization and termination
as well as pipeline execution on every user request.
There are two built-in messenger interface types that can be extended
through inheritance:

* `PollingMessengerInterface` - Starts polling for user requests
    in a loop upon initialization,
    it has following methods:

    * `_request()` - Method that is used to retrieve user requests from a messenger,
        should return list of tuples: (user request, unique dialog id).
    * `_respond(responses)` - Method that sends responses generated by pipeline
        to users through a messenger,
        accepts list of dialog `Contexts`.
    * `_on_exception(e)` - Method that is called when a critical exception occurs
        i.e. exception from context storage or messenger interface, not a service exception.

        Such exceptions lead to the termination of the loop.
    * `connect(pipeline_runner, loop, timeout)` -
        Method that starts the polling loop.

        This method is called inside `pipeline.run` method.

        It accepts 3 arguments:

        * a callback that runs pipeline,
        * a function that should return True to continue polling,
        * and time to wait between loop executions.

* `CallbackMessengerInterface` - Creates message channel
    and provides a callback for pipeline execution,
    it has following methods:

    * `on_request(request, ctx_id)` or `on_request_async(request, ctx_id)` -
        Method that should be called each time
        user provides new input to pipeline,
        returns dialog Context.
    * `connect(pipeline_runner)` - Method that sets `pipeline_runner` as
        a function to be called inside `on_request`.

        This method is called inside `pipeline.run` method.

You can find API reference for these classes [here](%doclink(api,messengers.common.interface)).

Here the default `CallbackMessengerInterface` is used to setup
communication between the pipeline on the server side and the messenger client.
"""

# %%
messenger_interface = CallbackMessengerInterface()
# CallbackMessengerInterface instantiating the dedicated messenger interface
pipeline = Pipeline.from_script(
    *TOY_SCRIPT_ARGS, messenger_interface=messenger_interface
)


# %%
app = FastAPI()


class Output(BaseModel):
    user_id: str
    response: Message


@app.post("/chat", response_model=Output)
async def respond(
    user_id: str,
    user_message: Message,
):
    context = await messenger_interface.on_request_async(user_message, user_id)
    return {"user_id": user_id, "response": context.last_response}


# %%
if __name__ == "__main__":
    if is_interactive_mode():  # do not run this during doc building
        pipeline.run()  # runs the messenger_interface.connect method
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8000,
        )
