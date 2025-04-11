from fastapi import APIRouter, Depends, HTTPException, status
from app.models.user import UserResponse, UserUpdate, UserPreferences
from app.core.database import mysql
from app.core.security import oauth2_scheme
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Any

router = APIRouter()

@router.get("/me", response_model=UserResponse)
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    session: AsyncSession = Depends(mysql.get_session)
) -> Any:
    """
    Get current user
    """
    # Extract user_id from token
    # In a real implementation, you would decode the JWT token
    user_id = "user123"  # Replace with actual user_id from token
    
    # Query user from MySQL database
    result = await session.execute(select(User).where(User.id == user_id))
    user = result.scalars().first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        phone_number=user.phone_number,
        preferences=user.preferences,
        created_at=user.created_at
    )

@router.put("/me", response_model=UserResponse)
async def update_user(
    user_update: UserUpdate,
    token: str = Depends(oauth2_scheme),
    session: AsyncSession = Depends(mysql.get_session)
) -> Any:
    """
    Update current user
    """
    # Extract user_id from token
    # In a real implementation, you would decode the JWT token
    user_id = "user123"  # Replace with actual user_id from token
    
    # Get current user
    result = await session.execute(select(User).where(User.id == user_id))
    user = result.scalars().first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Update user
    update_data = user_update.dict(exclude_unset=True)
    
    if update_data:
        for key, value in update_data.items():
            setattr(user, key, value)
        
        await session.commit()
        await session.refresh(user)
    
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        phone_number=user.phone_number,
        preferences=user.preferences,
        created_at=user.created_at
    )

@router.put("/preferences", response_model=UserPreferences)
async def update_preferences(
    preferences: UserPreferences,
    token: str = Depends(oauth2_scheme),
    session: AsyncSession = Depends(mysql.get_session)
) -> Any:
    """
    Update user preferences
    """
    # Extract user_id from token
    # In a real implementation, you would decode the JWT token
    user_id = "user123"  # Replace with actual user_id from token
    
    # Get current user
    result = await session.execute(select(User).where(User.id == user_id))
    user = result.scalars().first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Update preferences
    user.preferences = preferences.dict()
    await session.commit()
    await session.refresh(user)
    
    if not user.preferences:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Preferences not found"
        )
    
    return UserPreferences(**user.preferences)