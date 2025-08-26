from datetime import datetime
from typing import List, Optional
from app.database import get_db
from app.models.accent import AccentComparisonInDB, AccentComparisonCreate
from bson import ObjectId

def create_accent_comparison(user_id: str, comparison_data: dict) -> AccentComparisonInDB:
    """Save accent comparison to database"""
    db = get_db()
    
    comparison_dict = {
        "user_id": user_id,
        **comparison_data,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    result = db.accent_comparisons.insert_one(comparison_dict)
    comparison_dict["_id"] = result.inserted_id
    return AccentComparisonInDB(**comparison_dict)

def get_user_comparisons(user_id: str, limit: int = 10, skip: int = 0) -> List[AccentComparisonInDB]:
    """Get user's accent comparisons"""
    db = get_db()
    comparisons = db.accent_comparisons.find(
        {"user_id": user_id}
    ).sort("created_at", -1).skip(skip).limit(limit)
    
    return [AccentComparisonInDB(**comp) for comp in comparisons]

def get_comparison_by_id(comparison_id: str, user_id: str) -> Optional[AccentComparisonInDB]:
    """Get specific comparison by ID"""
    db = get_db()
    comparison = db.accent_comparisons.find_one({
        "_id": ObjectId(comparison_id),
        "user_id": user_id
    })
    
    return AccentComparisonInDB(**comparison) if comparison else None

def delete_comparison(comparison_id: str, user_id: str) -> bool:
    """Delete accent comparison"""
    db = get_db()
    result = db.accent_comparisons.delete_one({
        "_id": ObjectId(comparison_id),
        "user_id": user_id
    })
    
    return result.deleted_count > 0