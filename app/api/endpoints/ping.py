from fastapi import APIRouter

from app.schemas import PingResponse

router = APIRouter()


@router.get(
    "/",
    response_model=PingResponse,
)
async def ping():
    return {"response": "pong"}
