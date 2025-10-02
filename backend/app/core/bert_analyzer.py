import torch
from transformers import AutoTokenizer, AutoModel, pipeline
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import List, Dict, Tuple, Any
import nltk
from collections import Counter
import logging

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class BERTResumeAnalyzer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {self.device}")
        
        # Load BERT model
        self.model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        
        # Load sentiment analyzer for tone analysis
        self.sentiment_analyzer = pipeline("sentiment-analysis", 
                                         model="cardiffnlp/twitter-roberta-base-sentiment-latest")
        
        # Initialize reference embeddings
        self.reference_embeddings = {}
        self.seniority_embeddings = {}
        self.skill_category_embeddings = {}
        
        self.initialize_reference_embeddings()
    
    def initialize_reference_embeddings(self):
        """Create comprehensive reference embeddings for classification"""
        
        # Section classification references
        section_references = {
            'contact_info': [
                'John Smith software engineer contact information',
                'email phone address personal details candidate',
                'linkedin profile github portfolio contact'
            ],
            'experience': [
                'worked as software developer at microsoft google',
                'professional experience employment history job responsibilities',
                'led team delivered project managed developed implemented',
                'achieved results improved performance increased efficiency'
            ],
            'education': [
                'bachelor degree computer science university college',
                'educational background academic qualification masters degree',
                'graduated from stanford mit harvard school education',
                'coursework GPA academic achievements university'
            ],
            'skills': [
                'programming languages python javascript java react',
                'technical skills software development frameworks',
                'technologies tools expertise proficient experienced',
                'machine learning data science cloud computing'
            ],
            'achievements': [
                'improved performance by 30 percent increased revenue',
                'led team of 10 people managed project successfully',
                'achieved targets delivered on time under budget',
                'won award recognition patent publication conference'
            ],
            'projects': [
                'built application using react node.js database',
                'developed system architecture microservices cloud',
                'project portfolio github open source contribution',
                'personal projects side projects technical blog'
            ]
        }
        
        # Seniority level references
        seniority_references = {
            'entry': [
                'junior developer entry level associate trainee intern',
                'fresh graduate new grad starting career beginner',
                'learning growing developing skills first job'
            ],
            'mid': [
                'software developer engineer analyst specialist consultant',
                'experienced professional 2-5 years independent contributor',
                'intermediate level competent skilled proficient'
            ],
            'senior': [
                'senior developer lead engineer principal architect',
                'expert specialist advanced professional 5+ years',
                'senior level experienced mature technical leader'
            ],
            'leadership': [
                'manager director head VP chief executive officer',
                'team lead engineering manager technical manager',
                'leadership management strategic oversight executive'
            ]
        }
        
        # Skill category references
        skill_category_references = {
            'programming_languages': [
                'python java javascript typescript c++ c# php ruby go rust',
                'programming languages coding development scripting',
                'software development programming technical skills'
            ],
            'frameworks_libraries': [
                'react angular vue django flask spring boot express',
                'frameworks libraries tools development environment',
                'web development mobile development backend frontend'
            ],
            'databases': [
                'mysql postgresql mongodb redis elasticsearch oracle',
                'database management data storage sql nosql queries',
                'data modeling database design database administration'
            ],
            'cloud_devops': [
                'aws azure gcp docker kubernetes jenkins terraform',
                'cloud computing devops deployment infrastructure',
                'continuous integration continuous deployment automation'
            ],
            'soft_skills': [
                'leadership teamwork communication problem solving',
                'project management stakeholder management mentoring',
                'analytical thinking creative problem solving'
            ]
        }
        
        # Generate embeddings for all categories
        self.reference_embeddings = self._generate_category_embeddings(section_references)
        self.seniority_embeddings = self._generate_category_embeddings(seniority_references)
        self.skill_category_embeddings = self._generate_category_embeddings(skill_category_references)
        
        logging.info("Reference embeddings initialized successfully")
    
    def _generate_category_embeddings(self, references: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        """Generate averaged embeddings for each category"""
        category_embeddings = {}
        
        for category, texts in references.items():
            embeddings = self.get_embeddings(texts)
            category_embeddings[category] = np.mean(embeddings, axis=0)
        
        return category_embeddings
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate BERT embeddings for list of texts"""
        embeddings = []
        
        for text in texts:
            # Tokenize and encode
            inputs = self.tokenizer(text, return_tensors='pt', 
                                 truncation=True, padding=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Mean pooling
                embedding = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                embeddings.append(embedding[0])
        
        return np.array(embeddings)
    
    def classify_text_section(self, text: str) -> Tuple[str, float]:
        """Classify text section using BERT embeddings"""
        if not text.strip():
            return 'unknown', 0.0
        
        text_embedding = self.get_embeddings([text])[0]
        
        best_category = 'other'
        best_similarity = 0
        
        for category, ref_embedding in self.reference_embeddings.items():
            similarity = cosine_similarity([text_embedding], [ref_embedding])[0][0]
            if similarity > best_similarity:
                best_similarity = similarity
                best_category = category
        
        return best_category, float(best_similarity)
    
    def calculate_seniority_score(self, text: str) -> Dict[str, float]:
        """Calculate seniority distribution using BERT embeddings"""
        if not text.strip():
            return {'entry': 0, 'mid': 0, 'senior': 0, 'leadership': 0}
        
        text_embedding = self.get_embeddings([text])[0]
        scores = {}
        
        for level, ref_embedding in self.seniority_embeddings.items():
            similarity = cosine_similarity([text_embedding], [ref_embedding])[0][0]
            scores[level] = float(similarity)
        
        # Normalize scores
        total_score = sum(scores.values())
        if total_score > 0:
            scores = {k: v / total_score for k, v in scores.items()}
        
        return scores
    
    def get_overall_seniority_score(self, seniority_distribution: Dict[str, float]) -> float:
        """Convert seniority distribution to single score (0-1)"""
        level_values = {'entry': 0.2, 'mid': 0.5, 'senior': 0.8, 'leadership': 1.0}
        return sum(seniority_distribution[level] * level_values[level] 
                  for level in seniority_distribution)
    
    def analyze_career_progression(self, experiences: List[Dict]) -> Dict[str, Any]:
        """Analyze career progression using BERT"""
        if not experiences:
            return self._empty_progression_analysis()
        
        progression_scores = []
        role_changes = []
        
        for i, exp in enumerate(experiences):
            title = exp.get('title', '')
            description = exp.get('description', '')
            
            # Calculate seniority for this role
            combined_text = f"{title} {description}"
            seniority_dist = self.calculate_seniority_score(combined_text)
            seniority_score = self.get_overall_seniority_score(seniority_dist)
            
            progression_scores.append(seniority_score)
            
            # Analyze role changes
            if i > 0:
                prev_score = progression_scores[i-1]
                change = seniority_score - prev_score
                role_changes.append(change)
        
        # Calculate progression metrics
        avg_progression = np.mean(role_changes) if role_changes else 0
        progression_consistency = 1.0 - (np.std(role_changes) if role_changes else 0)
        
        # Determine trend
        if avg_progression > 0.1:
            trend = 'progressive'
        elif avg_progression < -0.1:
            trend = 'regressive'
        else:
            trend = 'stable'
        
        return {
            'progression_scores': progression_scores,
            'average_progression': float(avg_progression),
            'progression_consistency': float(max(0, progression_consistency)),
            'trend': trend,
            'role_changes': role_changes,
            'final_seniority': progression_scores[-1] if progression_scores else 0
        }
    
    def _empty_progression_analysis(self) -> Dict[str, Any]:
        """Return empty progression analysis"""
        return {
            'progression_scores': [],
            'average_progression': 0.0,
            'progression_consistency': 0.0,
            'trend': 'unknown',
            'role_changes': [],
            'final_seniority': 0.0
        }
    
    def categorize_skill(self, skill: str) -> Tuple[str, float]:
        """Categorize skill using BERT embeddings"""
        skill_embedding = self.get_embeddings([skill])[0]
        
        best_category = 'other'
        best_similarity = 0
        
        for category, ref_embedding in self.skill_category_embeddings.items():
            similarity = cosine_similarity([skill_embedding], [ref_embedding])[0][0]
            if similarity > best_similarity:
                best_similarity = similarity
                best_category = category
        
        return best_category, float(best_similarity)
    
    def analyze_language_quality(self, text: str) -> Dict[str, float]:
        """Analyze language quality using BERT and NLP techniques"""
        if not text.strip():
            return {'clarity': 0, 'sentiment': 0, 'complexity': 0, 'professionalism': 0}
        
        # Sentiment analysis
        sentiment_result = self.sentiment_analyzer(text[:512])  # Limit text length
        sentiment_score = sentiment_result[0]['score'] if sentiment_result[0]['label'] == 'POSITIVE' else 1 - sentiment_result[0]['score']
        
        # Language complexity (simple metrics)
        sentences = nltk.sent_tokenize(text)
        words = text.split()
        
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        complexity_score = min(1.0, avg_sentence_length / 20)  # Normalize to 0-1
        
        # Professional language indicators
        professional_keywords = [
            'achieved', 'delivered', 'implemented', 'managed', 'led', 'developed',
            'improved', 'increased', 'optimized', 'designed', 'created', 'established'
        ]
        
        text_lower = text.lower()
        professional_count = sum(1 for keyword in professional_keywords if keyword in text_lower)
        professionalism_score = min(1.0, professional_count / 10)
        
        return {
            'clarity': float(sentiment_score * 0.8 + (1 - complexity_score * 0.5) * 0.2),
            'sentiment': float(sentiment_score),
            'complexity': float(complexity_score),
            'professionalism': float(professionalism_score)
        }
    
    def extract_achievements_with_metrics(self, text: str) -> List[Dict[str, Any]]:
        """Extract quantified achievements using BERT and regex"""
        achievements = []
        
        # Patterns for quantified achievements
        metric_patterns = [
            r'(\d+(?:\.\d+)?)\s*%',  # Percentages
            r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)',  # Money
            r'(\d+(?:,\d{3})*)\s+(?:users|customers|clients|people|team|members)',  # Numbers
            r'(\d+(?:\.\d+)?)\s*(?:x|times)',  # Multipliers
            r'reduced.*?by\s+(\d+(?:\.\d+)?)',  # Reductions
            r'increased.*?by\s+(\d+(?:\.\d+)?)',  # Increases
        ]
        
        sentences = nltk.sent_tokenize(text)
        
        for sentence in sentences:
            # Check if sentence contains metrics
            has_metrics = any(re.search(pattern, sentence, re.IGNORECASE) for pattern in metric_patterns)
            
            if has_metrics:
                # Use BERT to classify if this is likely an achievement
                achievement_embedding = self.get_embeddings([sentence])[0]
                achievement_ref = self.reference_embeddings.get('achievements')
                
                if achievement_ref is not None:
                    similarity = cosine_similarity([achievement_embedding], [achievement_ref])[0][0]
                    
                    if similarity > 0.3:  # Threshold for achievement classification
                        achievements.append({
                            'description': sentence,
                            'has_metrics': True,
                            'confidence': float(similarity),
                            'metrics_found': [match.group() for pattern in metric_patterns 
                                            for match in re.finditer(pattern, sentence, re.IGNORECASE)]
                        })
        
        return achievements
    
    def calculate_skill_depth_vs_breadth(self, skills: List[str]) -> Dict[str, float]:
        """Calculate skill depth vs breadth using BERT clustering"""
        if not skills:
            return {'depth_score': 0, 'breadth_score': 0, 'specialist_score': 0.5}
        
        # Categorize all skills
        skill_categories = {}
        for skill in skills:
            category, confidence = self.categorize_skill(skill)
            if category not in skill_categories:
                skill_categories[category] = []
            skill_categories[category].append((skill, confidence))
        
        # Calculate breadth (number of categories)
        breadth_score = min(1.0, len(skill_categories) / 6)  # Normalize to max 6 categories
        
        # Calculate depth (skills per category)
        max_skills_in_category = max(len(skills) for skills in skill_categories.values()) if skill_categories else 1
        depth_score = min(1.0, max_skills_in_category / 10)  # Normalize to max 10 skills
        
        # Specialist score (depth vs breadth)
        specialist_score = depth_score / (depth_score + breadth_score) if (depth_score + breadth_score) > 0 else 0.5
        
        return {
            'depth_score': float(depth_score),
            'breadth_score': float(breadth_score),
            'specialist_score': float(specialist_score),
            'categories': skill_categories
        }