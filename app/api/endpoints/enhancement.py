from fastapi import UploadFile

from app.api.utils import speech_enhancement, APIRouter
from app.schemas import EnhancementResponse

router = APIRouter()


@router.post(
    "/",
    response_model=EnhancementResponse,
)
async def get_enhancement(file: UploadFile):
    data = await speech_enhancement(file)
    return EnhancementResponse(payload=data)
