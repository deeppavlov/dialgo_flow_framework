# %% [markdown]
"""
# Web API: 2. WebSocket Chat

This tutorial shows how to create a Web chat on FastAPI using websockets.

You can see the result at http://127.0.0.1:8000/.

This tutorial is a modified version of the FastAPI tutorial on WebSockets:
https://fastapi.tiangolo.com/advanced/websockets/.

As mentioned in that tutorial,

> ... for this example, we'll use a very simple HTML document
> with some JavaScript, all inside a long string.
> This, of course, is not optimal and you wouldn't use it for production.

Here, %mddoclink(api,messengers.common.interface,CallbackMessengerInterface)
is used to process requests.

%mddoclink(api,core.message,Message) is used to represent text messages.
"""

# %pip install chatsky=={chatsky} uvicorn=={uvicorn} fastapi=={fastapi}

# %%
from chatsky.messengers.common.interface import CallbackMessengerInterface
from chatsky import Message, Pipeline
from chatsky.utils.testing import TOY_SCRIPT_KWARGS, is_interactive_mode

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse


# %%
messenger_interface = CallbackMessengerInterface()
pipeline = Pipeline(
    **TOY_SCRIPT_KWARGS, messenger_interface=messenger_interface
)


# %%
app = FastAPI()
PORT = 8000

html = f"""
<!DOCTYPE html>
<html>
    <head>
        <title>Chat</title>
    </head>
    <body>
        <h1>WebSocket Chat</h1>
        <form action="" onsubmit="sendMessage(event)">
            <input type="text" id="messageText" autocomplete="off"/>
            <button>Send</button>
        </form>
        <ul id='messages'>
        </ul>
        <script>
            var client_id = Date.now();
            var ws = new WebSocket(`ws://localhost:{PORT}/ws/${{client_id}}`);
            ws.onmessage = function(event) {{
                var messages = document.getElementById('messages')
                var message = document.createElement('li')
                var content = document.createTextNode(event.data)
                message.appendChild(content)
                messages.appendChild(message)
            }};
            function sendMessage(event) {{
                var input = document.getElementById("messageText")
                ws.send(input.value)
                input.value = ''
                event.preventDefault()
            }}
        </script>
    </body>
</html>
"""


@app.get("/")
async def get():
    return HTMLResponse(html)


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"User: {data}")
            request = Message(data)
            context = await messenger_interface.on_request_async(
                request, client_id
            )
            response = context.last_response.text
            if response is not None:
                await websocket.send_text(f"Bot: {response}")
            else:
                await websocket.send_text("Bot did not return text.")
    except WebSocketDisconnect:  # ignore disconnections
        pass


# %%
if __name__ == "__main__":
    if is_interactive_mode():  # do not run this during doc building
        pipeline.run()
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=PORT,
        )
