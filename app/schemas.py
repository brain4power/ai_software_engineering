from pydantic import BaseModel

__all__ = [
    "PingResponse",
]


class PingResponse(BaseModel):
    response: str = "pong"
