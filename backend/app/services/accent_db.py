from datetime import datetime
from typing import List, Optional
from app.database import get_db
from app.models.accent import AccentComparisonInDB, AccentComparisonCreate
from bson import ObjectId

def create_accent_comparison(user_id: str, comparison_data: dict):
    """Save accent comparison to database"""
    db = get_db()
    
    # Prepare the document for MongoDB
    comparison_dict = {
        "user_id": user_id,
        "reference_audio": comparison_data["reference_audio"],
        "comparison_audio": comparison_data["comparison_audio"],
        "speaker_similarity": comparison_data["speaker_similarity"],
        "acoustic_comparison": comparison_data["acoustic_comparison"],
        "pronunciation_differences": comparison_data["pronunciation_differences"],
        "transcriptions": comparison_data["transcriptions"],
        "overall_score": comparison_data["overall_score"],
        "visualization_url": comparison_data.get("visualization_url"),
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    result = db.accent_comparisons.insert_one(comparison_dict)
    
    # Return the complete document with string ID
    inserted_doc = db.accent_comparisons.find_one({"_id": result.inserted_id})
    
    # Convert ObjectId to string for JSON serialization
    if inserted_doc and "_id" in inserted_doc:
        inserted_doc["_id"] = str(inserted_doc["_id"])
    
    return inserted_doc

def get_user_comparisons(user_id: str, limit: int = 10, skip: int = 0) -> List[AccentComparisonInDB]:
    """Get user's accent comparisons"""
    db = get_db()
    comparisons = list(db.accent_comparisons.find(
        {"user_id": user_id}
    ).sort("created_at", -1).skip(skip).limit(limit))
    
    # Convert ObjectId to string for each document
    results = []
    for comp in comparisons:
        comp_data = dict(comp)
        comp_data["_id"] = str(comp_data["_id"])  # Convert ObjectId to string
        results.append(AccentComparisonInDB(**comp_data))
    
    return results

def get_comparison_by_id(comparison_id: str, user_id: str) -> Optional[AccentComparisonInDB]:
    """Get specific comparison by ID"""
    db = get_db()
    
    try:
        # Convert string ID to ObjectId
        obj_id = ObjectId(comparison_id)
        comparison = db.accent_comparisons.find_one({
            "_id": obj_id,
            "user_id": user_id
        })
        
        if comparison:
            comp_data = dict(comparison)
            comp_data["_id"] = str(comp_data["_id"])  # Convert ObjectId to string
            return AccentComparisonInDB(**comp_data)
        return None
        
    except:
        return None

def delete_comparison(comparison_id: str, user_id: str) -> bool:
    """Delete accent comparison"""
    db = get_db()
    
    try:
        obj_id = ObjectId(comparison_id)
        result = db.accent_comparisons.delete_one({
            "_id": obj_id,
            "user_id": user_id
        })
        
        return result.deleted_count > 0
    except:
        return False