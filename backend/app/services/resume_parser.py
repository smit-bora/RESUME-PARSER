# backend/app/services/resume_parser.py - IMPROVED VERSION

import json
import logging
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dateutil import parser as date_parser

from app.core.bert_analyzer import BERTResumeAnalyzer
from app.model.resume_schema import (
    Achievement,
    CareerAnalysis,
    CareerGap,
    CompanyType,    
    ContactInfo,
    Education,
    Experience,
    ParsingMetadata,
    ProgressionTrend,
    ResumeParseResult,
    RiskIndicator,
    Skill,
)

from app.services.text_extractor import TextExtractor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Lazy singleton for BERT analyzer
_BERT_ANALYZER: Optional[BERTResumeAnalyzer] = None


def get_bert_analyzer() -> BERTResumeAnalyzer:
    global _BERT_ANALYZER
    if _BERT_ANALYZER is None:
        _BERT_ANALYZER = BERTResumeAnalyzer()
    return _BERT_ANALYZER


def _load_skill_taxonomy() -> List[str]:
    """Load skills taxonomy from JSON file or use fallback"""
    try:
        project_root = Path(__file__).resolve().parents[3]
        tax_path = project_root / "data" / "skills_taxonomy.json"
        if tax_path.exists():
            with open(tax_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Flatten all skill categories
                all_skills = []
                if isinstance(data, dict):
                    for category, skills in data.items():
                        if isinstance(skills, list):
                            all_skills.extend(skills)
                        elif isinstance(skills, dict):
                            for subcategory, subskills in skills.items():
                                if isinstance(subskills, list):
                                    all_skills.extend(subskills)
                    return all_skills
                elif isinstance(data, list):
                    return data
    except Exception as e:
        logger.warning("Failed to load skills taxonomy: %s", e)

    # Fallback
    return [
        "Python", "Java", "C++", "JavaScript", "HTML", "CSS", "SQL", "R",
        "React", "Angular", "Vue", "Django", "Flask", "Node.js", "Spring Boot",
        "Machine Learning", "Deep Learning", "NLP", "TensorFlow", "PyTorch",
        "AWS", "Azure", "Google Cloud", "Docker", "Kubernetes",
        "MySQL", "PostgreSQL", "MongoDB", "Redis",
        "Git", "Jenkins", "CI/CD"
    ]


SKILL_TAXONOMY = _load_skill_taxonomy()


# Improved regex patterns
NAME_PATTERNS = [
    r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',  # Title case names at start
    r'([A-Z][a-z]+\s+[A-Z][a-z]+)',  # Any title case name
]

EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
PHONE_PATTERN = r'(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
LINKEDIN_PATTERN = r'(?:linkedin\.com/in/|linkedin:?\s*)([\w\-]+)'
GITHUB_PATTERN = r'(?:github\.com/|github:?\s*)([\w\-]+)'

# Date patterns - more flexible
DATE_PATTERN = r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}|\d{4}'
DATE_RANGE_PATTERN = rf'({DATE_PATTERN})\s*[-–—to]+\s*(present|current|{DATE_PATTERN})'


def _extract_name(text: str) -> Optional[str]:
    """Extract candidate name from resume text"""
    lines = text.split('\n')[:5]  # Check first 5 lines
    
    for line in lines:
        line = line.strip()
        if not line or len(line) > 50:  # Skip long lines
            continue
        
        # Try each name pattern
        for pattern in NAME_PATTERNS:
            match = re.search(pattern, line)
            if match:
                name = match.group(1).strip()
                # Validate: must be 2-4 words, each 2+ chars
                words = name.split()
                if 2 <= len(words) <= 4 and all(len(w) >= 2 for w in words):
                    return name
    
    return None


def _extract_contact_info(text: str) -> ContactInfo:
    """Enhanced contact information extraction"""
    
    # Email
    email_match = re.search(EMAIL_PATTERN, text, re.IGNORECASE)
    email = email_match.group(0) if email_match else None
    
    # Phone - improved cleaning
    phone_match = re.search(PHONE_PATTERN, text)
    phone = None
    if phone_match:
        phone = re.sub(r'[^\d+]', '', phone_match.group(0))
    
    # LinkedIn
    linkedin_match = re.search(LINKEDIN_PATTERN, text, re.IGNORECASE)
    linkedin = None
    if linkedin_match:
        linkedin = f"linkedin.com/in/{linkedin_match.group(1)}"
    
    # GitHub
    github_match = re.search(GITHUB_PATTERN, text, re.IGNORECASE)
    github = None
    if github_match:
        github = f"github.com/{github_match.group(1)}"
    
    # Name
    name = _extract_name(text)
    
    return ContactInfo(
        name=name,
        email=email,
        phone=phone,
        location=None,
        linkedin=linkedin,
        github=github,
        portfolio=None
    )


def _split_into_sections(text: str) -> Tuple[Dict[str, str], List[str]]:
    """Improved section splitting with better header detection"""
    
    # Define section headers with variations
    section_headers = {
        'education': ['education', 'academic background', 'academic qualifications'],
        'experience': ['experience', 'work experience', 'professional experience', 
                      'employment history', 'work history'],
        'projects': ['projects', 'personal projects', 'academic projects'],
        'skills': ['skills', 'technical skills', 'core competencies', 'technologies'],
        'certifications': ['certifications', 'certificates', 'licenses'],
        'achievements': ['achievements', 'accomplishments', 'awards', 'honors'],
        'publications': ['publications', 'research', 'papers'],
        'summary': ['summary', 'professional summary', 'objective', 'profile'],
    }
    
    lines = text.split('\n')
    sections = {}
    current_section = 'summary'
    section_content = []
    identified_sections = []
    
    for line in lines:
        line_lower = line.strip().lower()
        
        # Check if line is a section header
        matched_section = None
        for section_key, keywords in section_headers.items():
            for keyword in keywords:
                if line_lower == keyword or (len(line_lower) < 30 and keyword in line_lower):
                    matched_section = section_key
                    break
            if matched_section:
                break
        
        if matched_section:
            # Save previous section
            if section_content:
                sections[current_section] = '\n'.join(section_content).strip()
                section_content = []
            
            # Start new section
            current_section = matched_section
            if current_section not in identified_sections:
                identified_sections.append(current_section)
        else:
            # Add to current section
            section_content.append(line)
    
    # Save last section
    if section_content:
        sections[current_section] = '\n'.join(section_content).strip()
    
    return sections, identified_sections


def _parse_experience_section(exp_text: str) -> List[Dict]:
    """Improved experience parsing with better structure detection"""
    
    if not exp_text or not exp_text.strip():
        return []
    
    experiences = []
    
    # Split by blank lines or date patterns (likely new entries)
    blocks = []
    current_block = []
    lines = exp_text.split('\n')
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Check if this line starts a new experience entry
        # Heuristic: line with date range likely starts new entry
        if re.search(DATE_RANGE_PATTERN, line, re.IGNORECASE):
            if current_block:
                blocks.append('\n'.join(current_block))
                current_block = []
        
        if line:
            current_block.append(line)
        elif current_block:  # Blank line
            # Check if next non-blank line looks like a new entry
            next_line_idx = i + 1
            while next_line_idx < len(lines) and not lines[next_line_idx].strip():
                next_line_idx += 1
            
            if next_line_idx < len(lines):
                next_line = lines[next_line_idx].strip()
                # If next line looks like a title or has dates, save current block
                if re.search(DATE_RANGE_PATTERN, next_line, re.IGNORECASE) or \
                   any(keyword in next_line.lower() for keyword in ['engineer', 'developer', 'manager', 'analyst', 'founder', 'intern']):
                    blocks.append('\n'.join(current_block))
                    current_block = []
    
    if current_block:
        blocks.append('\n'.join(current_block))
    
    # Parse each block
    for block in blocks:
        exp = _parse_single_experience(block)
        if exp:
            experiences.append(exp)
    
    return experiences


def _parse_single_experience(text: str) -> Optional[Dict]:
    """Parse a single experience entry"""
    
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if not lines:
        return None
    
    title = ""
    company = ""
    location = ""
    start_date = None
    end_date = None
    description_lines = []
    
    # Find date range in the block
    date_match = None
    date_line_idx = -1
    for i, line in enumerate(lines):
        match = re.search(DATE_RANGE_PATTERN, line, re.IGNORECASE)
        if match:
            date_match = match
            date_line_idx = i
            start_date = match.group(1)
            end_date = match.group(2)
            break
    
    # Parse title and company
    # Common patterns:
    # Line 1: Title    Location
    # Line 2: Company  Dates
    # OR
    # Line 1: Title
    # Line 2: Company    Dates    Location
    
    if date_line_idx >= 0:
        # Lines before dates are likely title/company
        title_company_lines = lines[:date_line_idx+1]
        description_lines = lines[date_line_idx+1:]
        
        for line in title_company_lines:
            # Remove dates from line
            line_clean = re.sub(DATE_RANGE_PATTERN, '', line, flags=re.IGNORECASE).strip()
            line_clean = re.sub(r'\s+', ' ', line_clean)
            
            # Check for location patterns
            loc_match = re.search(r'([A-Z][a-z]+(?:,\s*[A-Z]{2,})?)\s*$', line_clean)
            if loc_match:
                location = loc_match.group(1)
                line_clean = line_clean[:loc_match.start()].strip()
            
            # Assign to title or company
            if not title:
                title = line_clean
            elif not company and line_clean:
                company = line_clean
    else:
        # No dates found - use heuristics
        if len(lines) >= 2:
            title = lines[0]
            company = lines[1]
            description_lines = lines[2:]
        else:
            title = lines[0]
            description_lines = lines[1:] if len(lines) > 1 else []
    
    # Clean up
    title = title.strip()
    company = company.strip()
    
    # If company looks like a title, swap
    if company and any(word in company.lower() for word in ['engineer', 'developer', 'manager', 'designer', 'analyst']):
        title, company = company, title
    
    description = '\n'.join(description_lines).strip()
    
    return {
        'title': title,
        'company': company,
        'location': location,
        'start_date': start_date,
        'end_date': end_date,
        'description': description,
        'raw': text
    }


def _parse_education_section(edu_text: str) -> List[Education]:
    """Improved education parsing"""
    
    items = []
    if not edu_text:
        return items
    
    # Split by double newlines or date patterns
    blocks = re.split(r'\n\s*\n', edu_text)
    
    for block in blocks:
        lines = [l.strip() for l in block.split('\n') if l.strip()]
        if not lines:
            continue
        
        institution = ""
        degree = ""
        graduation_date = None
        
        # Look for date
        for line in lines:
            date_match = re.search(DATE_RANGE_PATTERN, line, re.IGNORECASE)
            if date_match:
                graduation_date = date_match.group(2)
                break
        
        # Look for degree keywords
        degree_keywords = ['bachelor', 'master', 'phd', 'diploma', 'b.tech', 'm.tech', 
                          'b.e.', 'm.e.', 'bsc', 'msc', 'ba', 'ma', 'mba']
        
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in degree_keywords):
                degree = line
            elif not institution and len(line) > 10:
                # Likely institution name
                institution = line
        
        # If still not found, use first two lines
        if not degree and len(lines) >= 1:
            degree = lines[0]
        if not institution and len(lines) >= 2:
            institution = lines[1]
        
        items.append(Education(
            institution=institution or "",
            degree=degree or "",
            field_of_study=None,
            graduation_date=graduation_date,
            gpa=None,
            relevant=True,
            confidence_score=0.0
        ))
    
    return items


def _enhanced_skill_extraction(text: str, taxonomy: List[str]) -> List[str]:
    """Enhanced skill extraction with fuzzy matching"""
    
    found_skills = set()
    text_lower = text.lower()
    
    # Exact matching with word boundaries
    for skill in taxonomy:
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.add(skill)
    
    # Extract from comma-separated lists
    skill_section_pattern = r'(?:skills|technologies|tools)[\s:]*([^\n]+(?:\n[^\n]+)*?)(?:\n\n|\Z)'
    skill_matches = re.finditer(skill_section_pattern, text, re.IGNORECASE)
    
    for match in skill_matches:
        skill_text = match.group(1)
        # Split by commas, semicolons, pipes
        tokens = re.split(r'[,;|]', skill_text)
        for token in tokens:
            token = token.strip()
            if token and 2 <= len(token) <= 40:
                # Remove bullet points and clean
                token = re.sub(r'^[•◦▪▫-]\s*', '', token)
                if token and not token.isdigit():
                    found_skills.add(token)
    
    return list(found_skills)


def _months_between(start_str: Optional[str], end_str: Optional[str]) -> Optional[int]:
    """Calculate months between dates"""
    if not start_str:
        return None
    
    try:
        start = date_parser.parse(start_str, fuzzy=True)
    except:
        year_match = re.search(r'(19|20)\d{2}', start_str or "")
        if year_match:
            start = datetime(int(year_match.group()), 1, 1)
        else:
            return None
    
    if not end_str or re.search(r'present|current', end_str or "", re.IGNORECASE):
        end = datetime.now()
    else:
        try:
            end = date_parser.parse(end_str, fuzzy=True)
        except:
            year_match = re.search(r'(19|20)\d{2}', end_str or "")
            if year_match:
                end = datetime(int(year_match.group()), 12, 31)
            else:
                end = datetime.now()
    
    if end < start:
        return 0
    
    months = (end.year - start.year) * 12 + (end.month - start.month)
    return max(0, months)


def _determine_confidence_level(overall_score: float) -> str:
    """Map score to confidence level"""
    if overall_score < 0.35:
        return "under"
    if overall_score > 0.75:
        return "over"
    return "appropriate"


def parse_resume(file_content: bytes, filename: str, confidence_threshold: float = 0.25) -> ResumeParseResult:
    """Main parsing function with improved logic"""
    
    start_time = time.time()
    extractor = TextExtractor()
    parsing_errors: List[str] = []
    
    # Extract text
    try:
        text, method = extractor.extract_text(file_content, filename)
    except Exception as e:
        logger.exception("Extraction failed")
        parsing_errors.append(str(e))
        text = ""
        method = "failed"
    
    # Extract contact info
    contact = _extract_contact_info(text)
    
    # Split into sections
    sections, identified_sections = _split_into_sections(text or "")
    
    # Extract skills
    skills_text = sections.get('skills', '') + '\n' + sections.get('summary', '')
    raw_skills = _enhanced_skill_extraction(skills_text + '\n' + text, SKILL_TAXONOMY)
    
    # Categorize skills with BERT
    bert = get_bert_analyzer()
    categorized_skills: List[Skill] = []
    core_skills: List[Skill] = []
    supporting_skills: List[Skill] = []
    soft_skills: List[Skill] = []
    
    for skill in raw_skills:
        try:
            cat, conf = bert.categorize_skill(skill)
        except Exception as e:
            logger.warning(f"Skill categorization failed for {skill}: {e}")
            cat, conf = "other", 0.0
        
        if cat in ("programming_languages", "frameworks_libraries", "databases", "cloud_devops"):
            mapped_category = "core_technical"
        elif cat == "soft_skills":
            mapped_category = "soft"
        else:
            mapped_category = "supporting_technical"
        
        skill_obj = Skill(
            name=skill,
            category=mapped_category,
            confidence=float(conf or 0.0),
            years_experience=None,
            proficiency_level=None
        )
        categorized_skills.append(skill_obj)
        
        if mapped_category == "core_technical":
            core_skills.append(skill_obj)
        elif mapped_category == "soft":
            soft_skills.append(skill_obj)
        else:
            supporting_skills.append(skill_obj)
    
    # Parse experiences
    exp_blocks = _parse_experience_section(sections.get('experience', ''))
    experiences_result: List[Experience] = []
    durations = []
    short_stint_count = 0
    
    for blk in exp_blocks:
        title = blk.get('title', '')
        company = blk.get('company', '')
        start_date = blk.get('start_date')
        end_date = blk.get('end_date')
        description = blk.get('description', '')
        
        duration_months = _months_between(start_date, end_date) if start_date else None
        if duration_months is not None:
            durations.append(duration_months)
            if duration_months < 18:
                short_stint_count += 1
        
        # BERT analysis
        seniority_score = 0.0
        confidence_score = 0.0
        achievements_found = []
        
        try:
            seniority_dist = bert.calculate_seniority_score(f"{title} {description}")
            seniority_score = bert.get_overall_seniority_score(seniority_dist)
            confidence_score = float(max(seniority_dist.values())) if seniority_dist else 0.0
        except Exception as e:
            logger.debug(f"Seniority scoring failed: {e}")
        
        try:
            ach = bert.extract_achievements_with_metrics(description)
            for a in ach:
                achievements_found.append(a.get('description', ''))
        except:
            pass
        
        skills_used = _enhanced_skill_extraction(description, SKILL_TAXONOMY)
        
        experiences_result.append(Experience(
            company=company or "",
            title=title or "",
            start_date=start_date,
            end_date=end_date,
            duration_months=int(duration_months) if duration_months is not None else None,
            is_current=bool(end_date and re.search(r'present|current', end_date, re.IGNORECASE)),
            description=description or "",
            achievements=achievements_found,
            skills_used=skills_used,
            team_size=None,
            budget_handled=None,
            seniority_score=float(seniority_score),
            confidence_score=float(confidence_score)
        ))
    
    # Parse education
    education_result = _parse_education_section(sections.get('education', ''))
    
    # Extract achievements
    quant_achievements = []
    try:
        achs = bert.extract_achievements_with_metrics(text)
        for a in achs:
            quant_achievements.append(Achievement(
                description=a.get('description', ''),
                impact_type="quantitative" if a.get('has_metrics') else "qualitative",
                metrics=";".join(a.get('metrics_found', [])) if a.get('metrics_found') else None,
                confidence_score=float(a.get('confidence', 0.0))
            ))
    except Exception as e:
        logger.debug(f"Achievement extraction failed: {e}")
    
    # Career progression analysis
    try:
        progression_input = [
            {'title': e.title, 'description': e.description} 
            for e in experiences_result
        ]
        progression_res = bert.analyze_career_progression(progression_input)
    except Exception as e:
        logger.debug(f"Progression analysis failed: {e}")
        progression_res = bert._empty_progression_analysis()
    
    # Skill depth vs breadth
    try:
        depth_breadth = bert.calculate_skill_depth_vs_breadth([s.name for s in categorized_skills])
    except Exception as e:
        logger.debug(f"Skill depth calculation failed: {e}")
        depth_breadth = {
            'depth_score': 0.0,
            'breadth_score': 0.0,
            'specialist_score': 0.5,
            'categories': {}
        }
    
    # Language quality
    try:
        lang_quality = bert.analyze_language_quality(
            sections.get('summary', '') or text[:3000]
        )
    except Exception as e:
        logger.debug(f"Language analysis failed: {e}")
        lang_quality = {
            'clarity': 0.0,
            'sentiment': 0.0,
            'complexity': 0.0,
            'professionalism': 0.0
        }
    
    # Career gaps
    career_gaps: List[CareerGap] = []
    company_type_diversity: List[CompanyType] = []
    
    # Risk indicators
    job_hopping_risk = min(1.0, short_stint_count / max(1, len(experiences_result))) if experiences_result else 0.0
    
    # Average tenure
    avg_tenure = float(sum(durations) / len(durations)) if durations else 0.0
    role_consistency_score = 1.0
    if len(durations) >= 2:
        std_dev = (sum((d - avg_tenure) ** 2 for d in durations) / len(durations)) ** 0.5
        role_consistency_score = max(0.0, min(1.0, 1.0 - (std_dev / (avg_tenure + 1e-6))))
    
    # Trend mapping
    trend = progression_res.get('trend', 'unknown')
    if trend == 'progressive':
        progression_trend = ProgressionTrend.PROGRESSIVE
    elif trend in ('stable', 'unknown'):
        progression_trend = ProgressionTrend.MIXED
    elif trend in ('regressive', 'reset'):
        progression_trend = ProgressionTrend.RESET
    else:
        progression_trend = ProgressionTrend.MIXED
    
    # Build career analysis
    career_analysis = CareerAnalysis(
        average_tenure_months=float(avg_tenure),
        role_consistency_score=float(role_consistency_score),
        consistency_explanation=f"Detected {short_stint_count} short stints out of {len(experiences_result)} roles" if experiences_result else "No experience data",
        career_gaps=career_gaps,
        company_type_diversity=company_type_diversity,
        career_progression_trend=progression_trend,
        trend_explanation=f"avg_progression={progression_res.get('average_progression', 0.0):.3f}",
        seniority_progression=[float(x) for x in progression_res.get('progression_scores', [])],
        skill_evolution_score=float(depth_breadth.get('specialist_score', 0.5)),
        internal_promotion_evidence=0,
        responsibility_growth_score=float(min(1.0, abs(progression_res.get('average_progression', 0.0)))),
        leadership_progression=float(progression_res.get('final_seniority', 0.0)),
        core_technical_skills=core_skills,
        supporting_skills=supporting_skills,
        soft_skills=soft_skills,
        skill_depth_score=float(depth_breadth.get('depth_score', 0.0)),
        skill_breadth_score=float(depth_breadth.get('breadth_score', 0.0)),
        transferable_skills=[],
        emerging_tech_familiarity=float(1.0 if any('machine' in s.lower() or 'learning' in s.lower() for s in raw_skills) else 0.0),
        language_clarity_score=float(lang_quality.get('clarity', 0.0)),
        grammar_spelling_score=float(lang_quality.get('clarity', 0.0)),
        structure_formatting_score=float(min(1.0, len(identified_sections) / 6)),
        achievement_quantification_score=float(min(1.0, len(quant_achievements) / 5)),
        career_storytelling_score=float(min(1.0, len(sections.get('summary', '').split()) / 150.0)),
        data_usage_score=float(min(1.0, sum(1 for a in quant_achievements if a.metrics) / max(1, len(quant_achievements)))) if quant_achievements else 0.0,
        stability_indicator=float(1.0 - job_hopping_risk),
        learning_aptitude_score=float(min(1.0, 0.5 + (len([s for s in raw_skills if 'ml' in s.lower() or 'machine' in s.lower()]) * 0.1))),
        risk_appetite_score=float(min(1.0, 0.5 + (1.0 if any('startup' in (e.company or '').lower() for e in experiences_result) else 0.0))),
        adaptability_score=float(min(1.0, len(set([s.category for s in categorized_skills])) / 4 if categorized_skills else 0.0)),
        ambition_indicator=0.0,
        ownership_orientation_score=float(lang_quality.get('professionalism', 0.0)),
        confidence_level=_determine_confidence_level(0.5),
        quantifiable_achievements=quant_achievements,
        awards_recognitions=[m.group(0) for m in re.finditer(r'\b(?:award|awarded|scholarship|fellowship|patent|publication)\b', text, re.IGNORECASE)],
        thought_leadership_evidence=[m.group(0) for m in re.finditer(r'\b(?:blog|conference|talk|speaker|webinar)\b', text, re.IGNORECASE)],
        portfolio_strength=float(1.0 if (contact.portfolio or contact.github) else 0.0),
        team_leadership_experience=float(progression_res.get('final_seniority', 0.0)),
        work_environment_fit={
            "startup_fit": float(min(1.0, sum(1 for e in experiences_result if 'startup' in (e.company or '').lower()) / max(1, len(experiences_result)))),
            "corporate_fit": float(min(1.0, sum(1 for e in experiences_result if any(x in (e.company or '').lower() for x in ['inc','ltd','corp','technologies','systems','solutions'])) / max(1, len(experiences_result))))
        },
        cross_cultural_experience=0.0,
        role_level_alignment=float(min(1.0, progression_res.get('final_seniority', 0.0))),
        job_hopping_risk=float(job_hopping_risk),
        overqualification_risk=0.0,
        underqualification_risk=0.0,
        resume_inconsistencies=[],
        online_profile_mismatch_risk=0.0,
        unexplained_gaps_risk=0.0,
        overall_candidate_score=float(min(100, max(0, 50))),
        hire_recommendation="maybe",
        fit_score_breakdown={
            "skills": float(depth_breadth.get('depth_score', 0.0)),
            "experience": float(progression_res.get('final_seniority', 0.0)),
            "communication": float(lang_quality.get('clarity', 0.0))
        }
    )
    
    # Parsing metadata
    parsing_time = time.time() - start_time
    parsing_metadata = ParsingMetadata(
        filename=filename,
        file_size_mb=len(file_content) / (1024 * 1024),
        parsing_time_seconds=float(parsing_time),
        parsing_method=method,
        bert_model_used=get_bert_analyzer().model_name if get_bert_analyzer() else "unknown",
        confidence_threshold=float(confidence_threshold),
        total_text_length=len(text or ""),
        sections_identified=identified_sections,
        parsing_errors=parsing_errors,
        extraction_quality=float(min(1.0, (len(text or "") / 400.0)))
    )
    
    # Build result
    result = ResumeParseResult(
        contact_info=contact,
        experience=experiences_result,
        education=education_result,
        skills=categorized_skills,
        career_analysis=career_analysis,
        confidence_scores={"overall": 0.5},
        parsing_metadata=parsing_metadata,
        raw_text=text
    )
    
    return result


class ResumeParser: 
    """Class wrapper for API compatibility"""
    
    def __init__(self, confidence_threshold: float = 0.25):
        self.confidence_threshold = confidence_threshold
        logger.info(f"ResumeParser initialized with confidence_threshold={confidence_threshold}")
    
    def parse(self, file_content: bytes, filename: str) -> ResumeParseResult:
        """Parse resume file and return structured results"""
        return parse_resume(file_content, filename, self.confidence_threshold)