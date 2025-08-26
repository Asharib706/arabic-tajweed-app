from fastapi import FastAPI
from app.routes import auth, accent
from app.database import get_db
app = FastAPI()


app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(accent.router, prefix="/accent", tags=["accent"])
@app.get("/")
def read_root():
    return {"message": "Welcome to Tajweed App API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

