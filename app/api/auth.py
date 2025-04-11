from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from app.models.user import UserCreate, UserResponse, UserInDB
from app.core.database import mysql
from app.core.security import get_password_hash, verify_password, create_access_token
from datetime import timedelta
from app.config import settings
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Any

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_PREFIX}/auth/login")

@router.post("/register", response_model=UserResponse)
async def register(
    user_data: UserCreate,
    session: AsyncSession = Depends(mysql.get_session)
) -> Any:
    """
    Register a new user
    """
    # Check if username already exists
    result = await session.execute(select(User).where(User.username == user_data.username))
    existing_user = result.scalars().first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Check if email already exists
    result = await session.execute(select(User).where(User.email == user_data.email))
    existing_email = result.scalars().first()
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    user_in_db = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=get_password_hash(user_data.password),
        full_name=user_data.full_name,
        phone_number=user_data.phone_number,
        preferences=user_data.preferences
    )
    
    # Insert user into database
    session.add(user_in_db)
    await session.commit()
    await session.refresh(user_in_db)
    
    return UserResponse(
        id=user_in_db.id,
        username=user_in_db.username,
        email=user_in_db.email,
        full_name=user_in_db.full_name,
        phone_number=user_in_db.phone_number,
        preferences=user_in_db.preferences,
        created_at=user_in_db.created_at
    )

@router.post("/login")
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    session: AsyncSession = Depends(mysql.get_session)
) -> Any:
    """
    OAuth2 compatible token login, get an access token for future requests
    """
    # Find user by username
    result = await session.execute(select(User).where(User.username == form_data.username))
    user = result.scalars().first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Verify password
    if not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user.id)},
        expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": str(user.id)
    }

@router.post("/logout")
async def logout(token: str = Depends(oauth2_scheme)) -> Any:
    """
    Logout user (revoke token)
    
    Note: In a stateless JWT system, we can't actually revoke tokens.
    For a more secure implementation, you might want to use a token blacklist.
    """
    return {"message": "Successfully logged out"} 