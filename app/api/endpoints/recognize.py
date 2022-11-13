from fastapi import APIRouter, UploadFile

from app.api.utils import recognize_media_file
from app.schemas import RecognizeResponse

router = APIRouter()


@router.post(
    "/",
    response_model=RecognizeResponse,
)
async def recognize(file: UploadFile):
    result_text = await recognize_media_file(file=file)
    result = RecognizeResponse(text=result_text)
    return result
