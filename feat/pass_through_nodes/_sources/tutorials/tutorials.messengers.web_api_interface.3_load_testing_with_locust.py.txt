# %% [markdown]
"""
# Web API: 3. Load testing with Locust

This tutorial shows how to use an API endpoint
created in the FastAPI tutorial in load testing.
"""

# %pip install chatsky=={chatsky} locust=={locust}

# %% [markdown]
"""
## Running Locust

1. Run this file directly:
   ```bash
   python {file_name}
   ```
2. Run locust targeting this file:
   ```bash
   locust -f {file_name}
   ```
3. Run from python:
   ```python
   import sys
   from locust import main

   sys.argv = ["locust", "-f", {file_name}]
   main.main()
   ```

You should see the result at http://127.0.0.1:8089.

Make sure that your POST endpoint is also running (run the FastAPI tutorial).

If using the FastAPI tutorial, set "Host" to `http://127.0.0.1:8000`,
when prompted by Locust.
"""


# %%
################################################################################
# this patch is only needed to run this file in IPython kernel
# and can be safely removed
import gevent.monkey

gevent.monkey.patch_all()
################################################################################


# %%
import uuid
import time
import sys

from locust import FastHttpUser, task, constant, main

from chatsky import Message
from chatsky.utils.testing import HAPPY_PATH, is_interactive_mode


# %%
class ChatskyUser(FastHttpUser):
    wait_time = constant(1)

    def check_happy_path(self, happy_path):
        """
        Check a happy path.
        For each `(request, response)` pair in `happy_path`:
        1. Send request to the API endpoint and catch its response.
        2. Compare API response with the `response`.
           If they do not match, fail the request.

        :param happy_path:
            An iterable of tuples of
            `(Message, Message | Callable(Message->str|None) | None)`.

            If the second element is `Message`,
            check that API response matches it.

            If the second element is `None`,
            do not check the API response.

            If the second element is a `Callable`,
            call it with the API response as its argument.
            If the function returns a string,
            that string is considered an error message.
            If the function returns `None`,
            the API response is considered correct.
        """
        user_id = str(uuid.uuid4())

        for request, response in happy_path:
            request = Message.model_validate(request)
            with self.client.post(
                f"/chat?user_id={user_id}",
                headers={
                    "accept": "application/json",
                    "Content-Type": "application/json",
                },
                # Name is the displayed name of the request.
                name=f"/chat?user_message={request.model_dump_json()}",
                data=request.model_dump_json(),
                catch_response=True,
            ) as candidate_response:
                candidate_response.raise_for_status()
                text_response = Message.model_validate(
                    candidate_response.json()
                )

                if response is not None:
                    if callable(response):
                        error_message = response(text_response)
                        if error_message is not None:
                            candidate_response.failure(error_message)
                    elif text_response != Message.model_validate(response):
                        candidate_response.failure(
                            f"Expected: {response.model_dump_json()}\n"
                            f"Got: {text_response.model_dump_json()}"
                        )

            time.sleep(self.wait_time())

    @task(3)  # <- this task is 3 times more likely than the other
    def dialog_1(self):
        self.check_happy_path(HAPPY_PATH)

    @task
    def dialog_2(self):
        def check_first_message(msg: Message) -> str | None:
            if msg.text is None:
                return f"Message does not contain text: {msg.model_dump_json()}"
            if "Hi" not in msg.text:
                return (
                    f'"Hi" is not in the response message: '
                    f"{msg.model_dump_json()}"
                )
            return None

        self.check_happy_path(
            [
                # a function can be used to check the return message
                ("Hi", check_first_message),
                # a None is used if return message should not be checked
                ("i'm fine, how are you?", None),
                # this should fail
                ("Hi", check_first_message),
            ]
        )


# %%
if __name__ == "__main__":
    if is_interactive_mode():
        sys.argv = ["locust", "-f", __file__]
        main.main()
