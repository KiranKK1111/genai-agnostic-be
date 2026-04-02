"""JWT authentication — decode Bearer tokens, provide get_current_user dependency.

Usage in endpoints:
    from app.auth import get_current_user, User

    @router.get("/protected")
    async def protected(user: User = Depends(get_current_user)):
        return {"message": f"Hello {user.username}"}

When AUTH_ENABLED=false (local dev):
    All requests pass through with a local admin user.
    The Authorize button still appears in Swagger for testing.
"""
import jwt
import logging
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional
from app.config import get_settings

logger = logging.getLogger(__name__)

# This registers the "Authorize" button in Swagger UI
security = HTTPBearer(auto_error=False)


class User(BaseModel):
    """Decoded JWT user payload."""
    id: str
    username: str
    role: str = "user"


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> User:
    """FastAPI dependency that extracts and validates the JWT token.

    When AUTH_ENABLED=false: returns a local admin user (no token required).
    When AUTH_ENABLED=true: decodes the JWT and returns the user.
    """
    settings = get_settings()

    # Local dev mode — skip auth entirely
    if not settings.AUTH_ENABLED:
        return User(id="local", username="User", role="admin")

    # Auth enabled — token required
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
        user_id = payload.get("sub", "")
        username = payload.get("username", "") or payload.get("name", "") or "User"
        return User(
            id=user_id,
            username=username,
            role=payload.get("role", "user"),
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_admin(user: User = Depends(get_current_user)) -> User:
    """Dependency that requires admin role."""
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user
