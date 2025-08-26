from fastapi import Request, HTTPException, status
from fastapi import APIRouter, Depends, HTTPException, Response
from datetime import datetime
from app.database import get_db
from fastapi import Cookie

router=APIRouter()
@router.get("/protected")
async def protected_route(session_id: str = Cookie(None), db=Depends(get_db)):
    if not session_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # Check if the session exists in the database
    session = db.sessions.find_one({"session_id": session_id})
    if not session:
        raise HTTPException(status_code=401, detail="Invalid session")
    
    # Fetch the user associated with the session
    user = db.users.find_one({"username": session["username"]})
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    return {"message": f"Hello, {user['username']}! You are authenticated."}