from fastapi import UploadFile

from app.api.utils import separate_audio_files, APIRouter
from app.schemas import SeparateResponse

router = APIRouter()


@router.post(
    "/",
    response_model=SeparateResponse,
)
async def get_separated_file(file: UploadFile):
    output_files = await separate_audio_files(file)
    return SeparateResponse(output_files=output_files)
