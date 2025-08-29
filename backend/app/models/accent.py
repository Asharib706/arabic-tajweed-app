from datetime import datetime
from typing import Dict, List, Any, Optional, Annotated
from pydantic import BaseModel, Field, ConfigDict
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema
from bson import ObjectId

class PyObjectId(ObjectId):
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler) -> core_schema.CoreSchema:
        return core_schema.general_after_validator_function(
            cls.validate,
            core_schema.str_schema(),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda x: str(x)
            ),
        )

    @classmethod
    def validate(cls, v, info):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid objectid")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: core_schema.CoreSchema, handler
    ) -> JsonSchemaValue:
        json_schema = handler(core_schema)
        json_schema.update(type="string", format="objectid")
        return json_schema

class AccentComparisonInDB(BaseModel):
    id: str = Field(alias="_id")
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

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str},
        json_schema_extra={
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
    )

class AccentComparisonCreate(BaseModel):
    reference_audio: str
    comparison_audio: str

class AccentComparisonResponse(BaseModel):
    id: str = Field(alias="_id")
    user_id: str
    reference_audio: str
    comparison_audio: str
    speaker_similarity: float
    acoustic_comparison: Dict[str, float]
    pronunciation_differences: Dict[str, Any]
    transcriptions: Dict[str, str]
    overall_score: float
    created_at: datetime

    model_config = ConfigDict(
        json_schema_extra={
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
    )