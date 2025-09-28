from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from typing import List
from app.services.accent import accent_comparator
from app.services.accent_db import (
    create_accent_comparison,
    get_user_comparisons,
    get_comparison_by_id,
    delete_comparison,
)
from app.models.accent import AccentComparisonResponse, AccentComparisonCreate
from app.utils.security import get_current_user
from app.models.user import UserInDB

router = APIRouter()
@router.post("/compare")
async def compare_accents(
    reference_audio: UploadFile = File(...),
    comparison_audio: UploadFile = File(...),
    current_user: UserInDB = Depends(get_current_user)
):
    """Compare two Arabic audio files for accent differences"""
    try:
        # Read audio files without validation
        ref_content = await reference_audio.read()
        comp_content = await comparison_audio.read()

        # Perform comparison
        comparison_result = accent_comparator.compare_accents(ref_content, comp_content)
        
        # Save to database
        comparison_data = {
            "reference_audio": reference_audio.filename,
            "comparison_audio": comparison_audio.filename,
            **comparison_result
        }
        
        saved_comparison = create_accent_comparison(str(current_user.id), comparison_data)
        
        # Convert ObjectId to string for JSON serialization
        if "_id" in saved_comparison:
            saved_comparison["_id"] = str(saved_comparison["_id"])
        
        return saved_comparison
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing audio files: {str(e)}"
        )
    
# ... (rest of the routes remain the same)
@router.get("/comparisons", response_model=List[AccentComparisonResponse])
async def get_user_accent_comparisons(
    limit: int = 10,
    skip: int = 0,
    current_user: UserInDB = Depends(get_current_user)
):
    """Get user's accent comparison history"""
    comparisons = get_user_comparisons(str(current_user.id), limit, skip)
    return [
        AccentComparisonResponse(
            id=str(comp.id),
            user_id=comp.user_id,
            reference_audio=comp.reference_audio,
            comparison_audio=comp.comparison_audio,
            speaker_similarity=comp.speaker_similarity,
            acoustic_comparison=comp.acoustic_comparison,
            pronunciation_differences=comp.pronunciation_differences,
            transcriptions=comp.transcriptions,
            overall_score=comp.overall_score,
            created_at=comp.created_at
        ) for comp in comparisons
    ]

@router.get("/comparisons/{comparison_id}", response_model=AccentComparisonResponse)
async def get_accent_comparison(
    comparison_id: str,
    current_user: UserInDB = Depends(get_current_user)
):
    """Get specific accent comparison"""
    comparison = get_comparison_by_id(comparison_id, str(current_user.id))
    if not comparison:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Comparison not found"
        )
    
    return AccentComparisonResponse(
        id=str(comparison.id),
        user_id=comparison.user_id,
        reference_audio=comparison.reference_audio,
        comparison_audio=comparison.comparison_audio,
        speaker_similarity=comparison.speaker_similarity,
        acoustic_comparison=comparison.acoustic_comparison,
        pronunciation_differences=comparison.pronunciation_differences,
        transcriptions=comparison.transcriptions,
        overall_score=comparison.overall_score,
        created_at=comparison.created_at
    )

@router.delete("/comparisons/{comparison_id}")
async def delete_accent_comparison(
    comparison_id: str,
    current_user: UserInDB = Depends(get_current_user)
):
    """Delete accent comparison"""
    success = delete_comparison(comparison_id, str(current_user.id))
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Comparison not found"
        )
    
    return {"message": "Comparison deleted successfully"}


import numpy as np

@router.post("/transcription")
async def transcribe_single_audio(
    audio: UploadFile = File(...)
):
    """Transcribe a single Arabic audio file and extract acoustic features"""
    try:
        # Read audio file
        audio_content = await audio.read()

        # Run transcription
        transcription = accent_comparator.transcribe(audio_content)

        # Extract features
        features = accent_comparator.extract_acoustic_features(audio_content)

        def safe_array(arr):
            return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).tolist()

        return {
            "filename": audio.filename,
            "transcription": transcription,
            "features": {
                # "pitch": safe_array(features["pitch"]) if features["pitch"] is not None else None,
                "amplitude": safe_array(features["ampitude"]),
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error transcribing audio: {str(e)}"
        )
