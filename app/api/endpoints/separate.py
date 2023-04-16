from typing import List
from fastapi import UploadFile

from app.api.utils import separate_audio_files, APIRouter
from app.schemas import SeparateResponse

router = APIRouter()


@router.post(
    "/",
    response_model=List[SeparateResponse],
)
async def get_recognition(file: UploadFile):
    sources = await separate_audio_files(file)
    return [SeparateResponse(source=source[0], content=source[1]) for source in sources]
