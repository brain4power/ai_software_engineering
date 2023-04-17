import operator
import os
import secrets
from functools import reduce
from pathlib import Path
from typing import List, Union

from pydantic import AnyHttpUrl, BaseSettings, validator


class Settings(BaseSettings):
    API_STR: str = "/api"
    SECRET_KEY: str = secrets.token_urlsafe(32)
    APP_PROJECT_NAME: str
    # SERVER_HOST: AnyHttpUrl
    # BACKEND_CORS_ORIGINS is a JSON-formatted list of origins
    # e.g: '["http://localhost", "http://localhost:4200", "http://localhost:3000", \
    # "http://localhost:8080", "http://local.dockertoolbox.tiangolo.com"]'
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []

    @classmethod
    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    MAX_FILE_SIZE: str

    @validator("MAX_FILE_SIZE")
    def max_file_size_transform_and_mul_int(cls, value: str) -> int:
        return reduce(operator.mul, map(int, value.split("*")))

    BASE_DIR = Path(__file__).resolve().parent.parent
    SYSTEM_STATIC_FOLDER = os.path.join(BASE_DIR, "system-static/")

    # speechbrain configs
    SB_FOLDER = os.path.join(SYSTEM_STATIC_FOLDER, "speechbrain/")
    SB_PRETRAINED_MODELS_FOLDER = os.path.join(SB_FOLDER, "pretrained_models/")
    AUDIO_RATE: int
    SEPFORMER_WHAMR_RATE: int = 8000

    class Config:
        case_sensitive = True


settings = Settings()
