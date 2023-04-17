import base64
import io
import os
from typing import Any, Callable

import numpy as np
import pydub
import torch
from fastapi import APIRouter as FastAPIRouter
from fastapi import File, UploadFile, HTTPException
from fastapi.types import DecoratedCallable
from speechbrain.pretrained import EncoderDecoderASR, SpectralMaskEnhancement
from speechbrain.pretrained import SepformerSeparation

from app.core.config import settings

__all__ = ["APIRouter", "speech2text", "speech_enhancement", "separate_audio_files"]


class APIRouter(FastAPIRouter):
    def api_route(
        self, path: str, *, include_in_schema: bool = True, **kwargs: Any
    ) -> Callable[[DecoratedCallable], DecoratedCallable]:
        if path.endswith("/"):
            path = path[:-1]

        add_path = super().api_route(path, include_in_schema=include_in_schema, **kwargs)

        alternate_path = path + "/"
        add_alternate_path = super().api_route(alternate_path, include_in_schema=False, **kwargs)

        def decorator(func: DecoratedCallable) -> DecoratedCallable:
            add_alternate_path(func)
            return add_path(func)

        return decorator


async def handle_uploaded_audio_file(file: UploadFile, sample_rate=settings.AUDIO_RATE):
    # validation
    VALID_AUDIO_TYPES = {"audio/wav", "audio/mpeg", "audio/x-wav"}
    if file.content_type not in VALID_AUDIO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Only {', '.join(VALID_AUDIO_TYPES)} file types are supported",
        )
    file_data = await file.read()
    if len(file_data) > settings.MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"Max file length: {settings.MAX_FILE_SIZE}")

    # transform data
    handler = pydub.AudioSegment.from_file(io.BytesIO(file_data))
    source = handler.set_frame_rate(sample_rate).set_channels(1).get_array_of_samples()
    fp_arr = np.array(source).astype(np.float32) / np.iinfo(source.typecode).max
    return torch.FloatTensor(fp_arr[np.newaxis, :]), torch.tensor([1.0])


async def speech2text(file: UploadFile):
    batch, rel_length = await handle_uploaded_audio_file(file)

    asr_model_save_dir = os.path.join(settings.SB_PRETRAINED_MODELS_FOLDER, "asr-wav2vec2-commonvoice-en")
    asr_model = EncoderDecoderASR.from_hparams(
        source="speechbrain/asr-wav2vec2-commonvoice-en",
        savedir=asr_model_save_dir,
        overrides={"wav2vec2": {"save_path": os.path.join(settings.SB_FOLDER, "model_checkpoints")}},
    )
    result = asr_model.transcribe_batch(batch, rel_length)
    return " ".join(token for token in result[0])


async def speech_enhancement(file: UploadFile):
    batch, rel_length = await handle_uploaded_audio_file(file)

    enhancer_model_save_dir = os.path.join(settings.SB_PRETRAINED_MODELS_FOLDER, "metricgan-plus-voicebank")
    enhancer = SpectralMaskEnhancement.from_hparams(
        source="speechbrain/metricgan-plus-voicebank", savedir=enhancer_model_save_dir
    )
    enhanced = enhancer.enhance_batch(batch, rel_length)
    enhanced = enhanced.cpu().detach().numpy()
    return base64.b64encode(enhanced.tobytes())


async def separate_audio_files(file: UploadFile = File(...)):
    model = SepformerSeparation.from_hparams(
        source="speechbrain/sepformer-whamr",
        savedir="pretrained_models/sepformer-whamr",
    )

    batch, rel_length = await handle_uploaded_audio_file(file, settings.SEPFORMER_WHAMR_RATE)

    sources = model.separate_batch(batch)
    sources = sources.cpu().detach().numpy()
    output_files = []
    for i in range(sources.shape[2]):
        source = sources[:, :, i]
        output_filename = f"source_{i + 1}"
        output_files.append((output_filename, base64.b64encode(source.tobytes())))
    return output_files
