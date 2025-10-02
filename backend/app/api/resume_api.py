from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.services.resume_parser import ResumeParser
from app.model.resume_schema import ResumeParseResult
import logging

router = APIRouter()  # This line must exist!
logger = logging.getLogger(__name__)

@router.post("/parse-resume", response_model=ResumeParseResult)
async def parse_resume(file: UploadFile = File(...)):
    """
    API endpoint to parse an uploaded resume (PDF or DOCX).
    Returns structured JSON output following ResumeParseResult schema.
    """
    try:
        contents = await file.read()
        
        logger.info(f"Received file: {file.filename}, size: {len(contents)} bytes")
        
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file received")
        
        parser = ResumeParser()
        result = parser.parse(contents, file.filename)
        
        return result
        
    except Exception as e:
        logger.exception("Error parsing resume")
        raise HTTPException(status_code=500, detail=f"Error parsing resume: {str(e)}")
    finally:
        await file.close()


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "resume_parser"}