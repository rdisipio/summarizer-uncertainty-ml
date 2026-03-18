"""Server entrypoint for local and container execution."""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    """Run the FastAPI application with uvicorn."""

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("src.api_server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()

