"""Auth endpoints — login, register, refresh, profile, preferences."""
import logging
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from app.config import get_settings
from app.database import get_pool
from app.auth import get_current_user, User

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["auth"])


# ── Token helpers ─────────────────────────────────────────────
def _build_tokens(user_id: str, username: str, name: str, role: str) -> dict:
    """Mint a fresh (access_token, refresh_token) pair for the given user.

    The access token is short-lived (JWT_EXPIRE_MIN) and carries the user
    identity claims that protected routes consume. The refresh token is
    long-lived (JWT_REFRESH_EXPIRE_MIN) and carries only enough identity
    for /api/auth/refresh to mint a new access token — it is tagged with
    ``type=refresh`` so get_current_user rejects it on any other route.
    """
    import jwt
    settings = get_settings()
    now = datetime.now(timezone.utc)

    access_payload = {
        "sub": user_id,
        "username": username,
        "name": name,
        "role": role,
        "type": "access",
        "iat": now,
        "exp": now + timedelta(minutes=settings.JWT_EXPIRE_MIN),
    }
    refresh_payload = {
        "sub": user_id,
        "username": username,
        "type": "refresh",
        "iat": now,
        "exp": now + timedelta(minutes=settings.JWT_REFRESH_EXPIRE_MIN),
    }
    return {
        "access_token": jwt.encode(access_payload, settings.JWT_SECRET, algorithm="HS256"),
        "refresh_token": jwt.encode(refresh_payload, settings.JWT_SECRET, algorithm="HS256"),
        "expires_in": settings.JWT_EXPIRE_MIN * 60,
        "refresh_expires_in": settings.JWT_REFRESH_EXPIRE_MIN * 60,
    }


class RegisterRequest(BaseModel):
    username: str
    password: str
    name: Optional[str] = None
    email: Optional[str] = None


class LoginRequest(BaseModel):
    username: str
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class PreferencesRequest(BaseModel):
    theme: Optional[str] = None
    default_viz: Optional[str] = None
    language: Optional[str] = None


@router.post("/register")
async def register(req: RegisterRequest):
    """Create user (first user = admin)."""
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA

    async with pool.acquire() as conn:
        count = await conn.fetchval(f"SELECT COUNT(*) FROM {schema}.users")
        role = "admin" if count == 0 else "user"

        import bcrypt as _bcrypt
        hashed = _bcrypt.hashpw(req.password.encode(), _bcrypt.gensalt()).decode()

        display_name = (req.name or req.username).strip()
        try:
            row = await conn.fetchrow(
                f"""INSERT INTO {schema}.users (username, name, email, hashed_password, role)
                    VALUES ($1, $2, $3, $4, $5) RETURNING id, username, name, role""",
                req.username, display_name, req.email, hashed, role
            )
            return {
                "id": str(row["id"]),
                "username": row["username"],
                "name": row["name"],
                "role": row["role"],
            }
        except Exception as e:
            if "unique" in str(e).lower() or "duplicate" in str(e).lower():
                raise HTTPException(409, "Username already exists")
            raise HTTPException(500, str(e))


@router.post("/login")
async def login(req: LoginRequest):
    """Authenticate user and return JWT token with expiry."""
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA

    async with pool.acquire() as conn:
        user = await conn.fetchrow(
            f"SELECT id, username, name, email, hashed_password, role, is_active FROM {schema}.users WHERE username=$1",
            req.username
        )
        if not user:
            raise HTTPException(401, "Invalid credentials")

        if not user["is_active"]:
            raise HTTPException(403, "Account is disabled")

        import bcrypt as _bcrypt
        if not _bcrypt.checkpw(req.password.encode(), user["hashed_password"].encode()):
            raise HTTPException(401, "Invalid credentials")

        display_name = user["name"] or user["username"]

        tokens = _build_tokens(
            user_id=str(user["id"]),
            username=user["username"],
            name=display_name,
            role=user["role"],
        )

        return {
            "token": tokens["access_token"],
            "refresh_token": tokens["refresh_token"],
            "user": {
                "id": str(user["id"]),
                "username": user["username"],
                "name": display_name,
                "email": user["email"],
                "role": user["role"],
            },
            "expires_in": tokens["expires_in"],
            "refresh_expires_in": tokens["refresh_expires_in"],
        }


@router.post("/refresh")
async def refresh(req: RefreshRequest):
    """Exchange a valid refresh token for a new access token + refresh token.

    - Verifies the refresh JWT signature and expiry
    - Verifies ``type=refresh`` (access tokens can't be used here)
    - Re-reads the user row from the DB so revoked / deactivated accounts
      can't refresh into new access tokens
    - Returns a fully new token pair (refresh rotation) so a leaked refresh
      token has a bounded useful lifetime
    """
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA

    import jwt
    try:
        payload = jwt.decode(req.refresh_token, settings.JWT_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Refresh token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Invalid refresh token")

    if payload.get("type") != "refresh":
        raise HTTPException(401, "Not a refresh token")

    user_id = payload.get("sub", "")
    if not user_id:
        raise HTTPException(401, "Invalid refresh token payload")

    # Re-fetch the user so any change in role / active status is honoured.
    async with pool.acquire() as conn:
        user = await conn.fetchrow(
            f"SELECT id, username, name, email, role, is_active FROM {schema}.users WHERE id=$1",
            user_id,
        )
    if not user:
        raise HTTPException(401, "User no longer exists")
    if not user["is_active"]:
        raise HTTPException(403, "Account is disabled")

    display_name = user["name"] or user["username"]
    tokens = _build_tokens(
        user_id=str(user["id"]),
        username=user["username"],
        name=display_name,
        role=user["role"],
    )

    return {
        "token": tokens["access_token"],
        "refresh_token": tokens["refresh_token"],
        "user": {
            "id": str(user["id"]),
            "username": user["username"],
            "name": display_name,
            "email": user["email"],
            "role": user["role"],
        },
        "expires_in": tokens["expires_in"],
        "refresh_expires_in": tokens["refresh_expires_in"],
    }


@router.post("/logout")
async def logout(user: User = Depends(get_current_user)):
    """Logout (client should discard the token)."""
    return {"status": "logged_out"}


@router.get("/me")
async def get_me(user: User = Depends(get_current_user)):
    """Get current user profile from JWT."""
    settings = get_settings()
    pool = get_pool()
    schema = settings.APP_SCHEMA

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            f"SELECT id, username, name, email, role, preferences, created_at FROM {schema}.users WHERE id=$1",
            user.id
        )
        if not row:
            return {
                "id": user.id,
                "username": user.username,
                "name": user.name or user.username,
                "role": user.role,
                "preferences": {},
            }

        return {
            "id": str(row["id"]),
            "username": row["username"],
            "name": row["name"] or row["username"],
            "email": row["email"],
            "role": row["role"],
            "preferences": row["preferences"] or {},
            "created_at": str(row["created_at"]),
        }


@router.patch("/preferences")
async def update_preferences(req: PreferencesRequest, user: User = Depends(get_current_user)):
    """Update user preferences."""
    settings = get_settings()
    prefs = {}
    if req.theme is not None:
        prefs["theme"] = req.theme
    if req.default_viz is not None:
        prefs["default_viz"] = req.default_viz
    if req.language is not None:
        prefs["language"] = req.language

    pool = get_pool()
    schema = settings.APP_SCHEMA
    async with pool.acquire() as conn:
        import json
        await conn.execute(
            f"UPDATE {schema}.users SET preferences = preferences || $1::jsonb WHERE id=$2",
            json.dumps(prefs), user.id
        )
    return {"status": "updated", "preferences": prefs}
