# from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
# from typing import List
# from app.services.receipt import (
#     create_receipt_manual,
#     get_receipts,
#     get_receipt_by_id,
#     update_receipt,
#     delete_receipt
# )
# from app.schemas.receipt import ReceiptCreate, ReceiptResponse
# from app.models.receipt import ReceiptInDB,ReceiptsAnalysis
# from app.utils.security import get_current_user
# from app.utils.aimodel import extract_info_from_file
# from app.models.user import UserInDB
# from app.services.analysis import get_receipts_analysis
# router = APIRouter()

# @router.post("/upload", response_model=dict)
# async def upload_receipts(
#     files: List[UploadFile] = File(...),
#     current_user: UserInDB = Depends(get_current_user)
# ):
#     successful_receipts = []
#     failed_files = []
    
#     for file in files:
#         try:
#             file_content = await file.read()
#             extracted_data = extract_info_from_file(file_content)
            
#             # Ensure we only store the user ID
#             if 'user_id' in extracted_data:
#                 extracted_data['user_id'] = str(current_user.id)
            
#             receipt_data = ReceiptCreate(**extracted_data)
#             created_receipt = create_receipt_manual(str(current_user.id), receipt_data)
            
#             successful_receipts.append({
#                 "receipt_id": created_receipt.id,
#                 "filename": file.filename
#             })
#         except Exception as e:
#             failed_files.append({
#                 "file_name": file.filename,
#                 "error": str(e)
#             })
    
#     return {
#         "message": "Receipt processing completed",
#         "successful_receipts": successful_receipts,
#         "failed_files": failed_files
#     }

# @router.post("/manual", response_model=ReceiptInDB)
# async def create_manual_receipt(
#     receipt: ReceiptCreate,
#     current_user: UserInDB = Depends(get_current_user)
# ):
#     return create_receipt_manual(str(current_user.id), receipt)

# @router.get("/", response_model=List[ReceiptInDB])
# async def list_receipts(
#     current_user: UserInDB = Depends(get_current_user)
# ):
#     return get_receipts(str(current_user.id))
# import sys
# @router.get("/analysis")
# async def analyze_user_receipts(
#     current_user: UserInDB = Depends(get_current_user)
# ):
#     """Get analysis of user's receipts data"""
#     print(f"[ANALYSIS ROUTE] Starting analysis for user {current_user.id}")
#     try:
#         analysis_data = get_receipts_analysis(str(current_user.id))
#         if not analysis_data:
#             print("[ANALYSIS ROUTE] No analysis data returned", file=sys.stderr)
#             raise HTTPException(status_code=404, detail="No receipts found to analyze")
        
#         print("[ANALYSIS ROUTE] Analysis completed, returning results")
#         return analysis_data
            
#     except HTTPException:
#         raise
#     except Exception as e:
#         print(f"[ANALYSIS ROUTE] Unexpected error during analysis: {e}", file=sys.stderr)
#         raise HTTPException(status_code=500, detail="Error processing receipt analysis")

# @router.get("/{receipt_id}", response_model=ReceiptInDB)
# async def get_single_receipt(
#     receipt_id: str,
#     current_user: UserInDB = Depends(get_current_user)
# ):
#     receipt = get_receipt_by_id(str(current_user.id), receipt_id)
#     if not receipt:
#         raise HTTPException(status_code=404, detail="Receipt not found")
#     return receipt

# @router.put("/{receipt_id}", response_model=ReceiptInDB)
# async def update_existing_receipt(
#     receipt_id: str,
#     receipt: ReceiptCreate,
#     current_user: UserInDB = Depends(get_current_user)
# ):
#     updated_receipt = update_receipt(str(current_user.id), receipt_id, receipt)
#     if not updated_receipt:
#         raise HTTPException(status_code=404, detail="Receipt not found")
#     return updated_receipt

# @router.delete("/{receipt_id}")
# async def remove_receipt(
#     receipt_id: str,
#     current_user: UserInDB = Depends(get_current_user)
# ):
#     success = delete_receipt(str(current_user.id), receipt_id)
#     if not success:
#         raise HTTPException(status_code=404, detail="Receipt not found")
#     return {"message": "Receipt deleted successfully"}
