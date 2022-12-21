import operator
import os
from functools import reduce

import pytest
from httpx import AsyncClient

from app.main import app as fa_app


def transform_and_mul_int(value: str) -> int:
    return reduce(operator.mul, map(int, value.split('*')))


@pytest.mark.anyio
async def test_ping():
    async with AsyncClient(app=fa_app, base_url="http://test") as ac:
        response = await ac.get("/api/ping")
    assert response.status_code == 200
    assert response.json() == {"response": "pong"}


@pytest.mark.anyio
async def test_incorrect_file_format():
    async with AsyncClient(app=fa_app, base_url="http://test") as ac:
        response = await ac.post("/api/recognize",
                                 files={"file": ("file.wav",
                                                 b"",
                                                 "text/plain")})
    assert response.status_code == 400


@pytest.mark.anyio
async def test_file_too_big():
    max_file_size = transform_and_mul_int(os.getenv("MAX_FILE_SIZE"))
    async with AsyncClient(app=fa_app, base_url="http://test") as ac:
        response = await ac.post("/api/recognize",
                                 files={"file": ("file.wav",
                                                 f"{'a' * (max_file_size + 1)}",
                                                 "audio/wav")})
    assert response.status_code == 400


async def mock_speech2text(*args, **kwargs):
    return "some text"


@pytest.mark.anyio
async def test_recognize(monkeypatch):
    from app.api.endpoints import recognize

    monkeypatch.setattr(recognize, "speech2text", mock_speech2text)
    async with AsyncClient(app=fa_app, base_url="http://test") as ac:
        response = await ac.post("/api/recognize",
                                 files={"file": ("file.wav",
                                                 b"",
                                                 "audio/wav")})
    assert response.status_code == 200
    assert response.json() == {"text": "some text"}


async def mock_enhancement(*args, **kwargs):
    return b"some audio data"


@pytest.mark.anyio
async def test_enhancement(monkeypatch):
    from app.api.endpoints import enhancement

    monkeypatch.setattr(enhancement, "speech_enhancement", mock_enhancement)
    async with AsyncClient(app=fa_app, base_url="http://test") as ac:
        response = await ac.post("/api/enhancement",
                                 files={"file": ("file.wav",
                                                 b"",
                                                 "audio/wav")})
    assert response.status_code == 200
    assert response.json() == {"payload": "some audio data"}
