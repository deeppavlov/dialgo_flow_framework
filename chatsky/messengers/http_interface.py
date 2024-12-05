import os
import time
from typing import Optional

import dotenv
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from chatsky.core import Message
from chatsky.messengers.common import MessengerInterface

dotenv.load_dotenv()

HTTP_INTERFACE_PORT = int(os.getenv("HTTP_INTERFACE_PORT", 8020))


class HealthStatus(BaseModel):
    status: str
    uptime: Optional[float]


class HTTPMessengerInterface(MessengerInterface):
    async def connect(self, pipeline_runner):
        app = FastAPI()

        class Output(BaseModel):
            user_id: str
            response: Message

        @app.post("/chat", response_model=Output)
        async def respond(
            user_id: str,
            user_message: Message,
        ):
            message = Message(text=user_message)
            context = await pipeline_runner(message, user_id)
            return {"user_id": user_id, "response": context.last_response}

        @app.get("/health", response_model=HealthStatus)
        async def health_check():
            return {
                "status": "ok",
                "uptime": str(time.time() - self.start_time),
            }

        self.start_time = time.time()
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=HTTP_INTERFACE_PORT,
        )
