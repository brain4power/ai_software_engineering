import argparse
import logging

from sys import platform

import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import settings

logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.APP_PROJECT_NAME,
    version="0.3.0",
    openapi_url=f"{settings.API_STR}/openapi.json"
)

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
        port=8000,
        host="127.0.0.1" if platform == "win32" else "0.0.0.0",
        log_level=options.log_level.lower(),
        workers=1,
        reload=options.reload,
    )
