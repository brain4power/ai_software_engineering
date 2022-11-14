import os
import tempfile

from fastapi import UploadFile, HTTPException
from speechbrain.pretrained import EncoderDecoderASR

from app.core.config import settings

__all__ = ["recognize_media_file"]


async def recognize_media_file(file: UploadFile) -> str:
    if file.content_type != "audio/wav":
        raise HTTPException(status_code=400, detail="Only .wav file type supported")
    asr_model_save_dir = os.path.join(settings.SB_PRETRAINED_MODELS_FOLDER, "asr-wav2vec2-commonvoice-en")
    asr_model = EncoderDecoderASR.from_hparams(
        source="speechbrain/asr-wav2vec2-commonvoice-en",
        savedir=asr_model_save_dir,
        overrides={"wav2vec2": {"save_path": os.path.join(settings.SB_FOLDER, "model_checkpoints")}}
    )
    with tempfile.NamedTemporaryFile("a+b", delete=False) as media_file:
        file_data = await file.read()
        media_file.write(file_data)
        media_file.seek(0)
        result_text = asr_model.transcribe_file(media_file.name)
    return result_text
