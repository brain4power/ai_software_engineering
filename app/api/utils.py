import base64
import io
import os

import numpy as np
import pydub
import resampy
import torch
from fastapi import UploadFile, HTTPException
from speechbrain.pretrained import EncoderDecoderASR
from speechbrain.pretrained import SpectralMaskEnhancement

from app.core.config import settings

__all__ = [
    "speech2text", "speech_enhancement",
]


async def handle_uploaded_audio_file(file: UploadFile):
    if file.content_type == "audio/wav":
        handler = pydub.AudioSegment.from_wav
    elif file.content_type == "audio/mpeg":
        handler = pydub.AudioSegment.from_mp3
    else:
        raise HTTPException(status_code=400, detail="Only .wav or .mp3 file types are supported")
    load_bytes = io.BytesIO(await file.read())
    audio = handler(load_bytes)

    channel_sounds = audio.split_to_mono()
    samples = [s.get_array_of_samples() for s in channel_sounds]

    fp_arr = np.array(samples).T.astype(np.float32)
    fp_arr /= np.iinfo(samples[0].typecode).max
    source = fp_arr[:, 0]
    sample_rate = audio.frame_rate
    source = resampy.resample(source, sample_rate, settings.AUDIO_RATE, axis=0, filter="kaiser_best")
    source = torch.FloatTensor(source)

    batch = source.unsqueeze(0)
    rel_length = torch.tensor([1.0])
    return batch, rel_length


async def speech2text(file: UploadFile):
    batch, rel_length = await handle_uploaded_audio_file(file)

    asr_model_save_dir = os.path.join(settings.SB_PRETRAINED_MODELS_FOLDER, "asr-wav2vec2-commonvoice-en")
    asr_model = EncoderDecoderASR.from_hparams(
        source="speechbrain/asr-wav2vec2-commonvoice-en",
        savedir=asr_model_save_dir,
        overrides={"wav2vec2": {"save_path": os.path.join(settings.SB_FOLDER, "model_checkpoints")}}
    )
    result = asr_model.transcribe_batch(batch, rel_length)
    result_text = ' '.join(token for token in result[0])
    return result_text


async def speech_enhancement(file: UploadFile):
    batch, rel_length = await handle_uploaded_audio_file(file)

    asr_model_save_dir = os.path.join(settings.SB_PRETRAINED_MODELS_FOLDER, "metricgan-plus-voicebank")
    enhancer = SpectralMaskEnhancement.from_hparams(source="speechbrain/metricgan-plus-voicebank",
                                                    savedir=asr_model_save_dir)
    enhanced = enhancer.enhance_batch(batch, rel_length)
    enhanced = enhanced.cpu().detach().numpy()
    return base64.b64encode(enhanced.tobytes())
