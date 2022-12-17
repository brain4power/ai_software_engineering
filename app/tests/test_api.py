import operator
import os
from functools import reduce

from fastapi.testclient import TestClient

from app.main import app as fa_app

client = TestClient(fa_app)


def transform_and_mul_int(value: str) -> int:
    return reduce(operator.mul, map(int, value.split('*')))


def test_ping():
    response = client.get("/api/ping")
    assert response.status_code == 200
    assert response.json() == {"response": "pong"}


def test_incorrect_file_format():
    response = client.post("/api/recognize",
                           files={"file": ("file.wav",
                                           b"",
                                           "text/plain")})
    assert response.status_code == 400


def test_file_too_big():
    max_file_size = transform_and_mul_int(os.getenv("MAX_FILE_SIZE"))
    response = client.post("/api/recognize",
                           files={"file": ("file.wav",
                                           f"{'a' * (max_file_size + 1)}",
                                           "audio/wav")})
    assert response.status_code == 400
