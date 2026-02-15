"""API key authentication dependency for FastAPI."""

import logging
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import AppConfig, get_config

logger = logging.getLogger(__name__)

security_scheme = HTTPBearer()


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Security(security_scheme),
    config: AppConfig = Depends(get_config),
) -> str:
    """Validate Bearer token against the configured API key."""
    if credentials.credentials != config.api_key:
        logger.warning("Authentication failed: invalid API key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    return credentials.credentials
