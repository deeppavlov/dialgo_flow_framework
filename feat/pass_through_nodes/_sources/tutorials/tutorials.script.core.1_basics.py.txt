# %% [markdown]
"""
# Core: 1. Basics

This notebook shows a basic example of creating a simple dialog bot (agent).

Here, basic usage of %mddoclink(api,core.pipeline,Pipeline) is shown.

Additionally, function %mddoclink(api,utils.testing.common,check_happy_path)
that can be used for Pipeline testing is presented.

Let's do all the necessary imports from Chatsky:
"""

# %pip install chatsky=={chatsky}

# %%
from chatsky import (
    TRANSITIONS,
    RESPONSE,
    Pipeline,
    Transition as Tr,
    conditions as cnd,
    # all the aliases used in tutorials are available for direct import
    # e.g. you can do `from chatsky import Tr` instead
)

from chatsky.utils.testing.common import (
    check_happy_path,
    is_interactive_mode,
)


# %% [markdown]
"""
First of all, to create a dialog agent, we need to create a dialog script.
Below script means a dialog script.

A script is a dictionary, where the keys are the names of the flows.
A script can contain multiple scripts, which is needed in order to divide
a dialog into sub-dialogs and process them separately.

For example, the separation can be tied to the topic of the dialog.
In this tutorial there is one flow called `greeting_flow`.

Flow describes a sub-dialog using linked nodes.
Each node has the keywords `RESPONSE` and `TRANSITIONS`.

* `RESPONSE` contains the response
    that the agent will return from the current node.
* `TRANSITIONS` is a list of %mddoclink(api,core.transition,Transition)s
    that describes possible transitions from the current node as well as their
    conditions and priorities.
"""


# %%
toy_script = {
    "greeting_flow": {
        "start_node": {  # This is the initial node,
            # it doesn't contain a `RESPONSE`.
            TRANSITIONS: [Tr(dst="node1", cnd=cnd.ExactMatch("Hi"))],
            # This transition means that the next node would be "node1"
            # if user's message is "Hi"
        },
        "node1": {
            RESPONSE: "Hi, how are you?",
            # When the bot enters node1,
            # return "Hi, how are you?".
            TRANSITIONS: [
                Tr(dst="node2", cnd=cnd.ExactMatch("I'm fine, how are you?"))
            ],
        },
        "node2": {
            RESPONSE: "Good. What do you want to talk about?",
            TRANSITIONS: [
                Tr(dst="node3", cnd=cnd.ExactMatch("Let's talk about music."))
            ],
        },
        "node3": {
            RESPONSE: "Sorry, I can not talk about music now.",
            TRANSITIONS: [Tr(dst="node4", cnd=cnd.ExactMatch("Ok, goodbye."))],
        },
        "node4": {
            RESPONSE: "Bye",
            TRANSITIONS: [Tr(dst="node1", cnd=cnd.ExactMatch("Hi"))],
        },
        "fallback_node": {
            # We get to this node if the conditions
            # for switching to other nodes are not performed.
            RESPONSE: "Ooops",
            TRANSITIONS: [Tr(dst="node1", cnd=cnd.ExactMatch("Hi"))],
        },
    }
}


happy_path = (
    (
        "Hi",
        "Hi, how are you?",
    ),  # start_node -> node1
    (
        "I'm fine, how are you?",
        "Good. What do you want to talk about?",
    ),  # node1 -> node2
    (
        "Let's talk about music.",
        "Sorry, I can not talk about music now.",
    ),  # node2 -> node3
    ("Ok, goodbye.", "Bye"),  # node3 -> node4
    ("Hi", "Hi, how are you?"),  # node4 -> node1
    ("stop", "Ooops"),  # node1 -> fallback_node
    (
        "stop",
        "Ooops",
    ),  # fallback_node -> fallback_node
    (
        "Hi",
        "Hi, how are you?",
    ),  # fallback_node -> node1
    (
        "I'm fine, how are you?",
        "Good. What do you want to talk about?",
    ),  # node1 -> node2
    (
        "Let's talk about music.",
        "Sorry, I can not talk about music now.",
    ),  # node2 -> node3
    ("Ok, goodbye.", "Bye"),  # node3 -> node4
)


# %% [markdown]
"""
`Pipeline` is an object that processes user
inputs and produces responses.

To create the pipeline you need to pass the script (`script`),
initial node (`start_label`) and
the node to which the default transition will take place
if none of the current conditions are met (`fallback_label`).

If `fallback_label` is not set, it defaults to `start_label`.

Roughly, the process is as follows:

1. Pipeline receives a user request.
2. The next node is determined with the help of `TRANSITIONS`.
3. Response of the chosen node is sent to the user.

For a more detailed description, see [here](
%doclink(api,core.pipeline,Pipeline._run_pipeline)
).
"""


# %%
pipeline = Pipeline(
    script=toy_script,
    start_label=("greeting_flow", "start_node"),
    fallback_label=("greeting_flow", "fallback_node"),
)

if __name__ == "__main__":
    check_happy_path(
        pipeline,
        happy_path,
        printout=True,
    )  # This is a function for automatic tutorial
    # running (testing tutorial) with `happy_path`.

    if is_interactive_mode():
        pipeline.run()
        # this method runs the pipeline with the preconfigured interface
        # which is CLI by default: it allows chatting with the bot
        # via command line
