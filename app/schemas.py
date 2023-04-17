from pydantic import BaseModel


__all__ = ["PingResponse", "RecognizeResponse", "EnhancementResponse", "SeparateResponse"]


class PingResponse(BaseModel):
    response: str = "pong"


class RecognizeResponse(BaseModel):
    text: str


class EnhancementResponse(BaseModel):
    payload: str


class SeparateResponse(BaseModel):
    source: str
    content: str
