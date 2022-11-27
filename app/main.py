import argparse
import logging

from sys import platform

import uvicorn
from fastapi import FastAPI, File
from starlette.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import settings
from app.api.spech_processing import speech2text, speech_enhancement

from schemas import RecognizeResponse, EnhancementResponse


logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.APP_PROJECT_NAME,
    version="0.3.0",
    openapi_url=f"{settings.API_STR}/openapi.json"
)


@app.post("/recognition", response_model=RecognizeResponse)
def get_segmentation_map(file: bytes = File(...)):
    result_text = speech2text(file)
    return {'text': result_text}


@app.post("/enhancement", response_model=EnhancementResponse)
def get_segmentation_map(file: bytes = File(...)):
    data = speech_enhancement(file)
    return {'playload': data}

@app.get("/")
async def root():
    return

# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(api_router, prefix=settings.API_STR)

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--log-level",
        action="store",
        dest="log_level",
        help="Logging level",
        default="INFO",
        type=str,
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        dest="reload",
        help="Enable auto-reload (code change watch dog)",
        default=False,
    )

    options = parser.parse_args()
    logger.warning(f"parsed args: options={options}")

    # run worker
    uvicorn.run(
        app,
        port=5049,
        host="127.0.0.1" if platform == "win32" else "0.0.0.0",
        log_level=options.log_level.lower(),
        workers=1,
        reload=options.reload,
    )




