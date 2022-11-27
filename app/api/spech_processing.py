import os
import io
import numpy as np
import torch

import base64

from speechbrain.pretrained import EncoderDecoderASR
from speechbrain.pretrained import SpectralMaskEnhancement

from app.core.config import settings


def prepare_tensor(file: bytes) -> (torch.Tensor, torch.Tensor):
    load_bytes = io.BytesIO(file)
    source = np.load(load_bytes)
    source = source.astype(np.float)
    source = torch.FloatTensor(source)
    return source.unsqueeze(0), torch.tensor([1.0])


def speech2text(file: bytes):
    batch, rel_length = prepare_tensor(file)
    asr_model_save_dir = os.path.join(settings.SB_PRETRAINED_MODELS_FOLDER, "asr-wav2vec2-commonvoice-en")
    asr_model = EncoderDecoderASR.from_hparams(
        source="speechbrain/asr-wav2vec2-commonvoice-en",
        savedir=asr_model_save_dir,
        overrides={"wav2vec2": {"save_path": os.path.join(settings.SB_FOLDER, "model_checkpoints")}}
    )
    result = asr_model.transcribe_batch(batch, rel_length)
    result_text = ' '.join(token for token in result[0])
    return result_text


def speech_enhancement(file: bytes):
    batch, rel_length = prepare_tensor(file)
    asr_model_save_dir = os.path.join(settings.SB_PRETRAINED_MODELS_FOLDER, "metricgan-plus-voicebank")
    enhancer = SpectralMaskEnhancement.from_hparams(source="speechbrain/metricgan-plus-voicebank",
                                                    savedir=asr_model_save_dir)
    enhanced = enhancer.enhance_batch(batch, rel_length)
    enhanced = enhanced.cpu().detach().numpy()
    return base64.b64encode(enhanced.tobytes())
