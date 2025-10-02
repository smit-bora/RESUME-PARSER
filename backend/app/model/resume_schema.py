from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import date, datetime
from enum import Enum

class CompanyType(str, Enum):
    STARTUP = "startup"
    MNC = "mnc" 
    TIER1 = "tier1"
    SME = "sme"
    CONSULTING = "consulting"

class ProgressionTrend(str, Enum):
    PROGRESSIVE = "progressive"
    STAGNANT = "stagnant"
    RESET = "reset"
    MIXED = "mixed"

class ContactInfo(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    portfolio: Optional[str] = None

class Experience(BaseModel):
    company: str
    title: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    duration_months: Optional[int] = None
    is_current: bool = False
    description: str
    achievements: List[str] = []
    skills_used: List[str] = []
    team_size: Optional[int] = None
    budget_handled: Optional[str] = None
    seniority_score: float = 0.0
    confidence_score: float = 0.0

class Education(BaseModel):
    institution: str
    degree: str
    field_of_study: Optional[str] = None
    graduation_date: Optional[str] = None
    gpa: Optional[float] = None
    relevant: bool = True
    confidence_score: float = 0.0

class Skill(BaseModel):
    name: str
    category: str  # core_technical, supporting_technical, soft
    confidence: float
    years_experience: Optional[int] = None
    proficiency_level: Optional[str] = None  # beginner, intermediate, advanced, expert

class CareerGap(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    duration_months: int
    explanation: Optional[str] = None
    gap_type: str  # unexplained, education, personal, sabbatical, unemployment

class Achievement(BaseModel):
    description: str
    impact_type: str  # quantitative, qualitative
    metrics: Optional[str] = None
    confidence_score: float

class RiskIndicator(BaseModel):
    risk_type: str
    severity: float  # 0-1 scale
    description: str
    evidence: List[str] = []

class CareerAnalysis(BaseModel):
    # Career Stability & Employment History
    average_tenure_months: float
    role_consistency_score: float  # 0-1 scale
    consistency_explanation: str
    career_gaps: List[CareerGap] = []
    company_type_diversity: List[CompanyType] = []
    career_progression_trend: ProgressionTrend
    trend_explanation: str
    
    # Career Progression & Evolution
    seniority_progression: List[float] = []  # Track progression over time
    skill_evolution_score: float
    internal_promotion_evidence: int = 0
    responsibility_growth_score: float
    leadership_progression: float
    
    # Skills & Competency Mapping
    core_technical_skills: List[Skill] = []
    supporting_skills: List[Skill] = []
    soft_skills: List[Skill] = []
    skill_depth_score: float  # 0-1 (specialist vs generalist)
    skill_breadth_score: float
    transferable_skills: List[str] = []
    emerging_tech_familiarity: float
    
    # Resume Quality & Communication
    language_clarity_score: float
    grammar_spelling_score: float
    structure_formatting_score: float
    achievement_quantification_score: float
    career_storytelling_score: float
    data_usage_score: float
    
    # Attitude, Aptitude & Psychological Indicators
    stability_indicator: float
    learning_aptitude_score: float
    risk_appetite_score: float
    adaptability_score: float
    ambition_indicator: float
    ownership_orientation_score: float
    confidence_level: str  # under, appropriate, over
    
    # Achievements & Differentiators
    quantifiable_achievements: List[Achievement] = []
    awards_recognitions: List[str] = []
    thought_leadership_evidence: List[str] = []
    portfolio_strength: float
    
    # Cultural & Organizational Fit
    team_leadership_experience: float
    work_environment_fit: Dict[str, float] = {}  # startup_fit, corporate_fit
    cross_cultural_experience: float
    role_level_alignment: float
    
    # Risk Indicators
    job_hopping_risk: float
    overqualification_risk: float
    underqualification_risk: float
    resume_inconsistencies: List[str] = []
    online_profile_mismatch_risk: float
    unexplained_gaps_risk: float
    
    # Overall Scores
    overall_candidate_score: float  # 0-100
    hire_recommendation: str  # strong_yes, yes, maybe, no, strong_no
    fit_score_breakdown: Dict[str, float] = {}

class ParsingMetadata(BaseModel):
    filename: str
    file_size_mb: float
    parsing_time_seconds: float
    parsing_method: str  # pdf, docx, ocr
    bert_model_used: str
    confidence_threshold: float
    total_text_length: int
    sections_identified: List[str]
    parsing_errors: List[str] = []
    extraction_quality: float

class ResumeParseResult(BaseModel):
    contact_info: ContactInfo
    experience: List[Experience] = []
    education: List[Education] = []
    skills: List[Skill] = []
    career_analysis: CareerAnalysis
    confidence_scores: Dict[str, float] = {}
    parsing_metadata: ParsingMetadata
    raw_text: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat()
        }