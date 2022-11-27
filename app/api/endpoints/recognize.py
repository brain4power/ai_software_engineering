from fastapi import APIRouter
from fastapi import UploadFile

from app.api.utils import speech2text
from app.schemas import RecognizeResponse

router = APIRouter()


@router.post(
    "/",
    response_model=RecognizeResponse,
)
async def get_recognition(file: UploadFile):
    result_text = await speech2text(file)
    return RecognizeResponse(text=result_text)
