import logging
import os
import sys

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
    # Force exit — background threads (fundus crawler, stats sampler) can
    # outlive uvicorn's shutdown and keep the process alive as a zombie.
    os._exit(0)


if __name__ == "__main__":
    main()
