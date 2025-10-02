from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from app.api.resume_api import router as resume_router

app = FastAPI(title="BERT Resume Parser")

# Enable CORS - MUST be before other routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(resume_router, prefix="/resume", tags=["Resume Parser"])

# Get frontend path
frontend_path = Path(__file__).parent.parent.parent / "frontend"

# Serve static files (CSS, JS)
app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

# Serve index.html at root
@app.get("/")
async def serve_frontend():
    return FileResponse(str(frontend_path / "index.html"))