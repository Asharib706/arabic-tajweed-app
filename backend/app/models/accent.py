from datetime import datetime
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from bson import ObjectId

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid objectid")
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")

class AccentComparisonInDB(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    user_id: str
    reference_audio: str
    comparison_audio: str
    speaker_similarity: float
    acoustic_comparison: Dict[str, float]
    pronunciation_differences: Dict[str, Any]
    transcriptions: Dict[str, str]
    overall_score: float
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "user_id": "user123",
                "reference_audio": "ref_audio.wav",
                "comparison_audio": "comp_audio.wav",
                "speaker_similarity": 0.85,
                "acoustic_comparison": {"mfcc_dtw_distance": 123.45, "pitch_correlation": 0.78},
                "pronunciation_differences": {"levenshtein_distance": 5, "length_difference": 2},
                "transcriptions": {"speaker1": "مرحبا", "speaker2": "مرحبا"},
                "overall_score": 0.82
            }
        }

class AccentComparisonCreate(BaseModel):
    reference_audio: str
    comparison_audio: str

class AccentComparisonResponse(BaseModel):
    id: str
    user_id: str
    reference_audio: str
    comparison_audio: str
    speaker_similarity: float
    acoustic_comparison: Dict[str, float]
    pronunciation_differences: Dict[str, Any]
    transcriptions: Dict[str, str]
    overall_score: float
    created_at: datetime

    class Config:
        schema_extra = {
            "example": {
                "id": "507f1f77bcf86cd799439011",
                "user_id": "user123",
                "reference_audio": "ref_audio.wav",
                "comparison_audio": "comp_audio.wav",
                "speaker_similarity": 0.85,
                "acoustic_comparison": {"mfcc_dtw_distance": 123.45, "pitch_correlation": 0.78},
                "pronunciation_differences": {"levenshtein_distance": 5, "length_difference": 2},
                "transcriptions": {"speaker1": "مرحبا", "speaker2": "مرحبا"},
                "overall_score": 0.82,
                "created_at": "2023-12-07T10:30:00Z"
            }
        }