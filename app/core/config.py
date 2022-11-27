import os
import secrets
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

    BASE_DIR = Path(__file__).resolve().parent.parent
    SYSTEM_STATIC_FOLDER = os.path.join(BASE_DIR, 'system-static/')

    # speechbrain configs
    SB_FOLDER = os.path.join(SYSTEM_STATIC_FOLDER, 'speechbrain/')
    SB_PRETRAINED_MODELS_FOLDER = os.path.join(SB_FOLDER, 'pretrained_models/')
    AUDIO_RATE = 16000

    class Config:
        case_sensitive = True


settings = Settings()
