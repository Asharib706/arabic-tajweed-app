# from bson import ObjectId
# from datetime import datetime
# from typing import List, Optional
# from app.database import get_db
# from app.schemas.receipt import ReceiptCreate, ReceiptResponse
# from app.models.receipt import ReceiptInDB

# def create_receipt_manual(user_id: str, receipt: ReceiptCreate) -> ReceiptInDB:
#     db = get_db()
#     receipt_dict = receipt.dict()
#     receipt_dict.update({
#         "user_id": user_id,
#         "created_at": datetime.utcnow(),
#         "updated_at": datetime.utcnow()
#     })
    
#     result = db.receipts.insert_one(receipt_dict)
#     receipt_dict["_id"] = str(result.inserted_id)
#     return ReceiptInDB(**receipt_dict)

# def get_receipts(user_id: str) -> List[ReceiptInDB]:
#     db = get_db()
#     receipts = []
#     for receipt in db.receipts.find({"user_id": user_id}):
#         receipt["_id"] = str(receipt["_id"])
#         receipts.append(ReceiptInDB(**receipt))
#     return receipts

# def get_receipt_by_id(user_id: str, receipt_id: str) -> Optional[ReceiptInDB]:
#     db = get_db()
#     try:
#         receipt = db.receipts.find_one({
#             "user_id": user_id,
#             "_id": ObjectId(receipt_id)
#         })
#         if receipt:
#             receipt["_id"] = str(receipt["_id"])
#             return ReceiptInDB(**receipt)
#         return None
#     except:
#         return None

# def update_receipt(user_id: str, receipt_id: str, receipt: ReceiptCreate) -> Optional[ReceiptInDB]:
#     db = get_db()
#     receipt_dict = receipt.dict(exclude_unset=True)
#     receipt_dict["updated_at"] = datetime.utcnow()
    
#     try:
#         db.receipts.update_one(
#             {"user_id": user_id, "_id": ObjectId(receipt_id)},
#             {"$set": receipt_dict}
#         )
#         return get_receipt_by_id(user_id, receipt_id)
#     except:
#         return None

# def delete_receipt(user_id: str, receipt_id: str) -> bool:
#     db = get_db()
#     try:
#         result = db.receipts.delete_one({
#             "user_id": user_id,
#             "_id": ObjectId(receipt_id)
#         })
#         return result.deleted_count > 0
#     except:
#         return False