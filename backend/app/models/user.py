from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
from datetime import datetime
from bson import ObjectId

class UserInDB(BaseModel):
    username: str
    email: str
    disabled: Optional[bool] = False
    hashed_password: str
    created_at: datetime
    
    # Handle both '_id' and 'id' fields
    id: str = Field(alias="_id")
    
    # Configuration to allow population by field name
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True
    )

    @classmethod
    def from_mongo(cls, data: dict):
        """Convert MongoDB document to UserInDB"""
        if not data:
            return None
        data['_id'] = str(data['_id'])  # Convert ObjectId to string
        return cls(**data)