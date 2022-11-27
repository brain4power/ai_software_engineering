from fastapi import APIRouter, UploadFile

from app.api.utils import speech_enhancement
from app.schemas import EnhancementResponse

router = APIRouter()


@router.post(
    "/enhancement",
    response_model=EnhancementResponse,
)
async def get_enhancement(file: UploadFile):
    data = await speech_enhancement(file)
    return EnhancementResponse(payload=data)
