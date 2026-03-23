"""Auth endpoints — login, register, profile, preferences."""
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


class RegisterRequest(BaseModel):
    username: str
    password: str
    email: Optional[str] = None


class LoginRequest(BaseModel):
    username: str
    password: str


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

        try:
            row = await conn.fetchrow(
                f"""INSERT INTO {schema}.users (username, email, hashed_password, role)
                    VALUES ($1, $2, $3, $4) RETURNING id, username, role""",
                req.username, req.email, hashed, role
            )
            return {"id": str(row["id"]), "username": row["username"], "role": row["role"]}
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
            f"SELECT id, username, hashed_password, role, is_active FROM {schema}.users WHERE username=$1",
            req.username
        )
        if not user:
            raise HTTPException(401, "Invalid credentials")

        if not user["is_active"]:
            raise HTTPException(403, "Account is disabled")

        import bcrypt as _bcrypt
        if not _bcrypt.checkpw(req.password.encode(), user["hashed_password"].encode()):
            raise HTTPException(401, "Invalid credentials")

        import jwt
        now = datetime.now(timezone.utc)
        payload = {
            "sub": str(user["id"]),
            "username": user["username"],
            "role": user["role"],
            "iat": now,
            "exp": now + timedelta(minutes=settings.JWT_EXPIRE_MIN),
        }
        token = jwt.encode(payload, settings.JWT_SECRET, algorithm="HS256")

        return {
            "token": token,
            "user": {
                "id": str(user["id"]),
                "username": user["username"],
                "role": user["role"],
            },
            "expires_in": settings.JWT_EXPIRE_MIN * 60,
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
            f"SELECT id, username, email, role, preferences, created_at FROM {schema}.users WHERE id=$1",
            user.id
        )
        if not row:
            return {"id": user.id, "username": user.username, "role": user.role, "preferences": {}}

        return {
            "id": str(row["id"]),
            "username": row["username"],
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
