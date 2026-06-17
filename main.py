"""
Resume Analyzer API — FastAPI entry point.

POST /analyze   Upload resume (PDF/DOCX/TXT) + job description text
GET  /health    Liveness check
GET  /docs      Swagger UI (auto-generated)
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import AnalysisResponse, HealthResponse, ScoreBreakdown as ScoreModel
from parser  import extract_text, detect_sections, extract_contact_info
from analyzer import (
    compute_similarity,
    keyword_match_analysis,
    detect_quantification,
    detect_action_verbs,
    detect_ats_formatting_issues,
    generate_hints,
    summarize_resume,
)
from scorer import compute_ats_score, score_label

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VERSION = "1.0.0"
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 5))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load spaCy model on startup to avoid cold-start lag on first request."""
    logger.info("Loading NLP model...")
    import analyzer  # triggers spacy.load at module level
    logger.info("NLP model ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Resume Analyzer API",
    description=(
        "Analyzes a resume against a job description and returns:\n"
        "- ATS score (0–100) with breakdown\n"
        "- Resume summary\n"
        "- Keyword match / missing keywords\n"
        "- Formatting issues\n"
        "- Prioritized improvement hints\n\n"
        "No external AI API — runs fully offline using spaCy + scikit-learn."
    ),
    version=VERSION,
    lifespan=lifespan,
)

# Allow all origins for dev; tighten for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse(
        status="ok",
        model="spaCy en_core_web_sm + scikit-learn TF-IDF",
        version=VERSION,
    )


# ── Main analysis endpoint ─────────────────────────────────────────────────────

@app.post(
    "/analyze",
    response_model=AnalysisResponse,
    tags=["Analysis"],
    summary="Analyze a resume against a job description",
    response_description="Full ATS analysis with score, keywords, hints, and summary",
)
async def analyze_resume(
    resume: UploadFile = File(..., description="Resume file — PDF, DOCX, or TXT"),
    job_description: str = Form(
        ...,
        min_length=50,
        description="Paste the full job description text here",
    ),
):
    # ── 1. File size check ────────────────────────────────────────────────────
    file_bytes = await resume.read()
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Maximum allowed: {MAX_FILE_SIZE_MB} MB.",
        )

    # ── 2. Extract text ───────────────────────────────────────────────────────
    try:
        resume_text = extract_text(file_bytes, resume.filename or "resume.pdf")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error during text extraction")
        raise HTTPException(status_code=500, detail=f"Failed to extract text: {e}")

    jd_text = job_description.strip()
    if len(jd_text) < 50:
        raise HTTPException(
            status_code=422,
            detail="Job description too short. Please provide at least 50 characters.",
        )

    logger.info(
        "Analyzing resume '%s' (%d chars) against JD (%d chars)",
        resume.filename, len(resume_text), len(jd_text),
    )

    # ── 3. Parse & analyse ────────────────────────────────────────────────────
    sections = detect_sections(resume_text)
    contact  = extract_contact_info(resume_text)

    similarity    = compute_similarity(resume_text, jd_text)
    keyword_data  = keyword_match_analysis(resume_text, jd_text)
    quant         = detect_quantification(resume_text)
    verbs         = detect_action_verbs(resume_text)
    fmt_issues    = detect_ats_formatting_issues(resume_text, sections)
    hints         = generate_hints(sections, quant, verbs, keyword_data, contact, similarity)
    summary       = summarize_resume(resume_text, jd_text)

    # ── 4. Score ──────────────────────────────────────────────────────────────
    breakdown = compute_ats_score(
        keyword_match_rate=keyword_data["keyword_match_rate"],
        similarity=similarity,
        quant=quant,
        sections=sections,
        verbs=verbs,
        contact=contact,
    )

    word_count = len(resume_text.split())

    # ── 5. Build response ─────────────────────────────────────────────────────
    return AnalysisResponse(
        ats_score=ScoreModel(
            keyword_score=breakdown.keyword_score,
            similarity_score=breakdown.similarity_score,
            quantification_score=breakdown.quantification_score,
            section_score=breakdown.section_score,
            verb_score=breakdown.verb_score,
            contact_score=breakdown.contact_score,
            total=breakdown.total,
            label=score_label(breakdown.total),
        ),
        summary=summary,
        contact_info=contact,
        sections_detected=sections,
        keyword_analysis=keyword_data,
        quantification=quant,
        verb_analysis=verbs,
        formatting_issues=fmt_issues,
        hints=hints,
        resume_word_count=word_count,
        similarity_raw=round(similarity, 4),
    )