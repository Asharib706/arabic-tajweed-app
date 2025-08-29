from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import secrets
import os

load_dotenv()  # Load environment variables from .env file
class Settings(BaseSettings):
    SECRET_KEY: str=secrets.token_urlsafe(32)
    MONGO_URI: str=os.getenv("MONGO_URI")
    MONGO_DB: str=os.getenv("MONGO_DB", "tajweed_app")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int =  1440
    CLOUDINARY_CLOUD_NAME: str = os.getenv("CLOUDINARY_CLOUD_NAME", "")
    CLOUDINARY_API_KEY: str = os.getenv("CLOUDINARY_API_KEY", "")
    CLOUDINARY_API_SECRET: str = os.getenv("CLOUDINARY_API_SECRET", "")

settings = Settings()