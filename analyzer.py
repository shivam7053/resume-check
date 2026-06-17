import re
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy model for NLP tasks
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # Fallback if model isn't linked correctly in some environments
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

STRONG_VERBS = {
    "achieved", "acquired", "addressed", "orchestrated", "spearheaded", "developed",
    "managed", "led", "increased", "decreased", "negotiated", "implemented",
    "designed", "engineered", "optimized", "streamlined", "transformed", "executed"
}

WEAK_VERBS = {
    "assisted", "helped", "worked", "responsible", "handled", "supported",
    "participated", "contributed", "tasks", "duties"
}

def compute_similarity(resume_text: str, jd_text: str) -> float:
    """Compute TF-IDF Cosine Similarity between Resume and JD."""
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return float(similarity[0][0])
    except Exception:
        return 0.0

def keyword_match_analysis(resume_text: str, jd_text: str) -> dict:
    """Extract keywords from JD and check presence in Resume."""
    def extract_keywords(text):
        doc = nlp(text.lower())
        # Filter for Nouns, Proper Nouns, and Adjectives that aren't stop words
        keywords = {
            token.text for token in doc 
            if token.pos_ in ["NOUN", "PROPN", "ADJ"] and not token.is_stop and len(token.text) > 2
        }
        return keywords

    jd_keywords = extract_keywords(jd_text)
    resume_text_lower = resume_text.lower()
    
    matched = [kw for kw in jd_keywords if kw in resume_text_lower]
    missing = [kw for kw in jd_keywords if kw not in resume_text_lower]
    
    total = len(jd_keywords)
    rate = len(matched) / total if total > 0 else 0.0

    return {
        "matched_keywords": sorted(matched[:15]),  # Limit for readability
        "missing_keywords": sorted(missing[:15]),
        "keyword_match_rate": rate,
        "total_jd_keywords": total
    }

def detect_quantification(text: str) -> dict:
    """Detect metrics and numbers used in bullet points."""
    lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 10]
    total_bullets = len(lines)
    
    # Look for percentages, dollar amounts, or general numbers
    quant_pattern = re.compile(r'\d+%|\$\d+|(\d+ (users|clients|revenue|growth|reduction|team))')
    
    quantified_count = sum(1 for line in lines if quant_pattern.search(line) or re.search(r'\b\d{2,}\b', line))
    
    return {
        "total_bullet_points": total_bullets,
        "quantified_bullets": quantified_count,
        "quantification_rate": quantified_count / total_bullets if total_bullets > 0 else 0.0
    }

def detect_action_verbs(text: str) -> dict:
    """Find strong and weak action verbs in the text."""
    doc = nlp(text.lower())
    verbs = [token.text for token in doc if token.pos_ == "VERB"]
    
    strong_found = list(set([v for v in verbs if v in STRONG_VERBS]))
    weak_found = list(set([v for v in verbs if v in WEAK_VERBS]))
    
    return {
        "strong_verbs_found": strong_found,
        "weak_verbs_found": weak_found
    }

def detect_ats_formatting_issues(text: str, sections: dict) -> list[str]:
    """Check for common ATS readability issues."""
    issues = []
    
    # Check for missing critical sections
    if not sections.get("skills"):
        issues.append("Missing explicit 'Skills' section.")
    if not sections.get("experience"):
        issues.append("Missing 'Experience' section.")
    
    # Check for non-standard characters (potential parsing artifacts)
    if len(re.findall(r'[^\x00-\x7F]+', text)) > 50:
        issues.append("Detected high number of special characters; check for layout/column issues.")
    
    # Very long lines might indicate table artifacts
    lines = text.split('\n')
    if any(len(line) > 200 for line in lines):
        issues.append("Some lines are unusually long; ensure you aren't using complex tables or columns.")
        
    return issues

def generate_hints(sections, quant, verbs, keyword_data, contact, similarity) -> list[str]:
    """Generate prioritized suggestions for resume improvement."""
    hints = []
    
    if keyword_data["keyword_match_rate"] < 0.4:
        hints.append("High Priority: Add more industry-specific keywords found in the job description.")
    
    if quant["quantification_rate"] < 0.3:
        hints.append("Impact: Use more numbers/metrics to quantify your achievements (e.g., 'Increased sales by 20%').")
        
    if len(verbs["strong_verbs_found"]) < 5:
        hints.append("Style: Use more powerful action verbs like 'Spearheaded' or 'Optimized' instead of 'Assisted'.")
        
    if not contact.get("linkedin"):
        hints.append("Formatting: Add your LinkedIn profile URL to increase recruiter engagement.")
        
    if not sections.get("summary"):
        hints.append("Structure: Consider adding a professional summary to highlight your top value propositions.")

    return hints

def summarize_resume(resume_text: str, jd_text: str) -> str:
    """Generate a brief comparison summary."""
    doc = nlp(resume_text)
    # Basic summary: take first few sentences or identify top skills
    sents = list(doc.sents)
    intro = " ".join([s.text.strip() for s in sents[:2]])
    
    if len(resume_text) < 100:
        return "Resume content is too short for a detailed summary."
        
    return f"Professional profile: {intro}... This candidate shows alignment with the JD, but should focus on bridging gaps in specific technical keywords identified in the analysis."