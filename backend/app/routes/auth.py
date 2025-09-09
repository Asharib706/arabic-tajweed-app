from fastapi import APIRouter, Depends, HTTPException, status, Body
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta
from app.schemas.user import UserCreate, UserLogin, Token, UserInDB, TokenData
from app.utils.security import (
    get_password_hash,
    verify_password,
    create_access_token,
    get_current_user
)
from app.database import get_db
from app.models.user import UserInDB as UserInDBModel
from app.config import settings
from datetime import datetime
from typing import Annotated

router = APIRouter(tags=["Authentication"])

@router.post("/register", response_model=Token)
async def register(
    user_data: UserCreate,  # Use Pydantic model for JSON input
    db = Depends(get_db)
):
    if db.users.find_one({"username": user_data.username}):
        raise HTTPException(status_code=400, detail="Username already registered")
    if db.users.find_one({"email": user_data.email}):
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = get_password_hash(user_data.password)
    user_dict = {
        "username": user_data.username,
        "email": user_data.email,
        "hashed_password": hashed_password,
        "disabled": False,
        "created_at": datetime.utcnow()
    }
    
    result = db.users.insert_one(user_dict)
    user_dict["_id"] = str(result.inserted_id)
    
    access_token = create_access_token(
        data={"sub": user_data.username},
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/login", response_model=TokenData)
async def login(
    login_data: UserLogin,  # Use Pydantic model for JSON input
    db = Depends(get_db)
):
    user = db.users.find_one({"username": login_data.username})
    if not user or not verify_password(login_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(
        data={"sub": user["username"]},
    )
    return {"username":login_data.username,"access_token": access_token, "token_type": "bearer"}