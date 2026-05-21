"""Server entrypoint for local and container execution."""

from __future__ import annotations

import logging
import os

import uvicorn


class HealthCheckFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/health" not in record.getMessage()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
access_logger = logging.getLogger("uvicorn.access")
access_logger.addFilter(HealthCheckFilter())


def main() -> None:
    """Run the FastAPI application with uvicorn."""

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("src.api_server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()

