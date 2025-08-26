# services/auth.py
from app.database import get_db
from app.models.user import UserInDB
from app.utils.security import verify_password

def authenticate_user(db, username: str, password: str) -> UserInDB | None:
    user_data = db.users.find_one({"username": username})
    if not user_data:
        return None
    
    if not verify_password(password, user_data["hashed_password"]):
        return None
    
    user_data["_id"] = str(user_data["_id"])
    return UserInDB(**user_data)