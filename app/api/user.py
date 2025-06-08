from fastapi import APIRouter, Depends, HTTPException, status
from app.models.user import UserResponse, UserUpdate
from app.models.user import User
from app.core.database import mysql
from app.core.security import get_current_user_id
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Any

router = APIRouter()

@router.get("/debug/all")
async def debug_get_all_users(
    session: AsyncSession = Depends(mysql.get_session)
) -> Any:
    """
    Debug endpoint to see all users in database
    """
    try:
        result = await session.execute(select(User))
        users = result.scalars().all()
        
        users_info = []
        for user in users:
            users_info.append({
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "full_name": user.full_name
            })
        
        return {
            "total_users": len(users_info),
            "users": users_info
        }
    except Exception as e:
        return {"error": str(e)}

@router.get("/me", response_model=UserResponse)
async def get_current_user(
    session: AsyncSession = Depends(mysql.get_session),
    current_user_id: str = Depends(get_current_user_id)
) -> Any:
    """
    Get current user
    
    **Requires Authentication**: Bearer token in Authorization header
    """
    print(f"=== USER ENDPOINT DEBUG ===")
    print(f"Current user ID from token: {current_user_id}")
    print(f"User ID type: {type(current_user_id)}")
    
    # Convert string ID to integer for database query
    try:
        user_id_int = int(current_user_id)
        print(f"Converted to int: {user_id_int}")
    except ValueError:
        print(f"Cannot convert user_id to int: {current_user_id}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user ID format"
        )
    
    # Query user from MySQL database using decoded user_id from JWT
    result = await session.execute(select(User).where(User.id == user_id_int))
    user = result.scalars().first()
    
    print(f"Database query result: {user}")
    print(f"User found: {user is not None}")
    if user:
        print(f"Found user: id={user.id}, username={user.username}")
    print("===========================")

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
        created_at=user.created_at
    )

@router.put("/me", response_model=UserResponse)
async def update_user(
    user_update: UserUpdate,
    session: AsyncSession = Depends(mysql.get_session),
    current_user_id: str = Depends(get_current_user_id)
) -> Any:
    """
    Update current user
    
    **Requires Authentication**: Bearer token in Authorization header
    """
    print(f"=== UPDATE USER ENDPOINT DEBUG ===")
    print(f"Current user ID from token: {current_user_id}")
    print(f"User ID type: {type(current_user_id)}")
    
    # Convert string ID to integer for database query (same as get_current_user)
    try:
        user_id_int = int(current_user_id)
        print(f"Converted to int: {user_id_int}")
    except ValueError:
        print(f"Cannot convert user_id to int: {current_user_id}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user ID format"
        )
    
    # Get current user using decoded user_id from JWT
    result = await session.execute(select(User).where(User.id == user_id_int))
    user = result.scalars().first()
    
    print(f"Database query result: {user}")
    print(f"User found: {user is not None}")
    print("===================================")

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
        created_at=user.created_at
    )