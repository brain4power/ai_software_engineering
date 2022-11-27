from pydantic import BaseModel


__all__ = [
    "PingResponse", "RecognizeResponse", "EnhancementResponse"
]


class PingResponse(BaseModel):
    response: str = "pong"


class RecognizeResponse(BaseModel):
    text: str


class EnhancementResponse(BaseModel):
    payload: str
