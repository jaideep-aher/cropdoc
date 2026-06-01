"""
main.py
-------
Entry point: starts the CropDoc FastAPI inference server.

Usage:
    python main.py [--host HOST] [--port PORT]
"""

import argparse
import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Start CropDoc inference server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run(
        "app.backend.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
