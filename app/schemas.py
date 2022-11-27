from pydantic import BaseModel


__all__ = [
    "PingResponse", "RecognizeResponse",
]


class PingResponse(BaseModel):
    response: str = "pong"


class RecognizeResponse(BaseModel):
    text: str


class EnhancementResponse(BaseModel):
    playload: bytes