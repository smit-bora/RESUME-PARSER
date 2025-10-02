import pytest
from pathlib import Path
from app.services.resume_parser import ResumeParser
from app.services.text_extractor import TextExtractor

# Initialize parser and extractor
extractor = TextExtractor()
parser = ResumeParser()

# Path to sample resumes
SAMPLES_DIR = Path(__file__).parent / "sample_resumes"

def test_pdf_resume_parsing():
    sample_pdf = SAMPLES_DIR / "sample_resume.pdf"
    if not sample_pdf.exists():
        pytest.skip("Sample PDF not found")
    
    with open(sample_pdf, "rb") as f:
        content = f.read()
    
    text, method = extractor.extract_text(content, sample_pdf.name)
    result = parser.parse_resume(text, filename=sample_pdf.name)
    
    assert result.contact_info.name is not None
    assert isinstance(result.skills, list)
    assert isinstance(result.experience, list)
    assert isinstance(result.education, list)
    assert result.career_analysis.overall_candidate_score >= 0

def test_docx_resume_parsing():
    sample_docx = SAMPLES_DIR / "sample_resume.docx"
    if not sample_docx.exists():
        pytest.skip("Sample DOCX not found")
    
    with open(sample_docx, "rb") as f:
        content = f.read()
    
    text, method = extractor.extract_text(content, sample_docx.name)
    result = parser.parse_resume(text, filename=sample_docx.name)
    
    assert result.contact_info.email is not None
    assert isinstance(result.skills, list)
