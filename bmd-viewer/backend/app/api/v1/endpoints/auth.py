"""인증 엔드포인트."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import create_access_token, hash_password, verify_password
from app.db.session import get_db
from app.models import Doctor
from app.schemas import DoctorCreate, DoctorOut, Token

router = APIRouter()


@router.post("/register", response_model=DoctorOut, status_code=201)
async def register(payload: DoctorCreate, db: AsyncSession = Depends(get_db)):
    existing = await db.scalar(
        select(Doctor).where(Doctor.email == payload.email)
    )
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    doctor = Doctor(
        email=payload.email,
        hashed_password=hash_password(payload.password),
        full_name=payload.full_name,
        role=payload.role,
    )
    db.add(doctor)
    await db.commit()
    await db.refresh(doctor)
    return doctor


@router.post("/login", response_model=Token)
async def login(
    form: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
):
    doctor = await db.scalar(
        select(Doctor).where(Doctor.email == form.username)
    )
    if not doctor or not verify_password(form.password, doctor.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )
    token = create_access_token(subject=doctor.id, role=doctor.role.value)
    return Token(access_token=token)
