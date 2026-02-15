"""Entry point: python run.py"""

import uvicorn

from app.config import get_config

if __name__ == "__main__":
    config = get_config()
    uvicorn.run(
        "app.main:app",
        host=config.server_host,
        port=config.server_port,
        reload=False,
    )
