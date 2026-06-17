"""
Pydantic request/response models for the Resume Analyzer API.
"""

from typing import Optional
from pydantic import BaseModel, Field


# ── Response sub-models ────────────────────────────────────────────────────────

class ContactInfo(BaseModel):
    email:    Optional[str] = None
    phone:    Optional[str] = None
    linkedin: Optional[str] = None
    github:   Optional[str] = None


class ScoreBreakdown(BaseModel):
    keyword_score:        float = Field(description="Out of 30")
    similarity_score:     float = Field(description="Out of 25")
    quantification_score: float = Field(description="Out of 15")
    section_score:        float = Field(description="Out of 15")
    verb_score:           float = Field(description="Out of 10")
    contact_score:        float = Field(description="Out of 5")
    total:                float = Field(description="Out of 100")
    label:                str   = Field(description="Human-readable rating")


class KeywordAnalysis(BaseModel):
    matched_keywords:    list[str]
    missing_keywords:    list[str]
    keyword_match_rate:  float = Field(description="0.0–1.0")
    total_jd_keywords:   int


class QuantificationAnalysis(BaseModel):
    total_bullet_points:  int
    quantified_bullets:   int
    quantification_rate:  float = Field(description="0.0–1.0")


class VerbAnalysis(BaseModel):
    strong_verbs_found: list[str]
    weak_verbs_found:   list[str]


class SectionPresence(BaseModel):
    contact:          bool
    summary:          bool
    experience:       bool
    education:        bool
    skills:           bool
    projects:         bool
    certifications:   bool
    achievements:     bool


# ── Main response ──────────────────────────────────────────────────────────────

class AnalysisResponse(BaseModel):
    ats_score:          ScoreBreakdown
    summary:            str = Field(description="Plain-English overview of the resume vs JD")
    contact_info:       ContactInfo
    sections_detected:  SectionPresence
    keyword_analysis:   KeywordAnalysis
    quantification:     QuantificationAnalysis
    verb_analysis:      VerbAnalysis
    formatting_issues:  list[str] = Field(description="ATS formatting warnings")
    hints:              list[str]  = Field(description="Prioritized improvement suggestions")
    resume_word_count:  int
    similarity_raw:     float      = Field(description="Raw TF-IDF cosine similarity (0–1)")


class HealthResponse(BaseModel):
    status: str
    model:  str
    version: str