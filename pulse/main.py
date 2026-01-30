import logging

import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    uvicorn.run(
        "pulse.app:app",
        host="0.0.0.0",
        port=8100,
        log_level="info",
    )


if __name__ == "__main__":
    main()
