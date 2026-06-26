"""FastAPI 공통 의존성 (현재 사용자, 권한)."""
from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.security import decode_token
from app.db.session import get_db
from app.models import Doctor, UserRole

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.api_v1_prefix}/auth/login"
)


async def get_current_doctor(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> Doctor:
    payload = decode_token(token)
    if payload is None or "sub" not in payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )
    doctor = await db.get(Doctor, payload["sub"])
    if doctor is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found"
        )
    return doctor


def require_roles(*roles: UserRole):
    """역할 기반 접근 제어 의존성 팩토리."""

    async def _checker(
        doctor: Doctor = Depends(get_current_doctor),
    ) -> Doctor:
        if roles and doctor.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )
        return doctor

    return _checker
