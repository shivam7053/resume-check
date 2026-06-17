"""
ATS Score calculator.

Scoring breakdown (total = 100):
  - Keyword match rate       30 pts  (how much JD vocab appears in resume)
  - TF-IDF cosine similarity 25 pts  (overall semantic overlap)
  - Quantification           15 pts  (metrics and numbers in bullets)
  - Section completeness     15 pts  (required resume sections present)
  - Action verb strength     10 pts  (strong vs weak verbs)
  - Contact info             5  pts  (email, phone present)
"""

from dataclasses import dataclass


@dataclass
class ScoreBreakdown:
    keyword_score:       float   # 0–30
    similarity_score:    float   # 0–25
    quantification_score: float  # 0–15
    section_score:       float   # 0–15
    verb_score:          float   # 0–10
    contact_score:       float   # 0–5
    total:               float   # 0–100


def compute_ats_score(
    keyword_match_rate: float,
    similarity: float,
    quant: dict,
    sections: dict,
    verbs: dict,
    contact: dict,
) -> ScoreBreakdown:
    """
    Compute weighted ATS score from analysis sub-results.

    All inputs are dicts/floats produced by analyzer.py functions.
    Returns a ScoreBreakdown with each component and the total.
    """

    # 1. Keyword match (0–30)
    keyword_score = round(min(keyword_match_rate, 1.0) * 30, 1)

    # 2. Similarity (0–25)
    # Raw cosine similarity tends to be low (<0.4) even for good matches,
    # so we scale it up with a sqrt to give more credit for moderate overlap.
    import math
    similarity_scaled = math.sqrt(min(similarity, 1.0))
    similarity_score  = round(similarity_scaled * 25, 1)

    # 3. Quantification (0–15)
    qrate              = quant.get("quantification_rate", 0.0)
    quantification_score = round(min(qrate / 0.6, 1.0) * 15, 1)   # 60% rate = full marks

    # 4. Section completeness (0–15)
    required_sections = ["contact", "experience", "education", "skills"]
    bonus_sections    = ["summary", "projects", "certifications", "achievements"]
    req_present   = sum(1 for s in required_sections if sections.get(s))
    bonus_present = sum(1 for s in bonus_sections    if sections.get(s))
    section_score = round(
        (req_present / len(required_sections)) * 11
        + min(bonus_present / len(bonus_sections), 1.0) * 4,
        1,
    )

    # 5. Action verb strength (0–10)
    strong = len(verbs.get("strong_verbs_found", []))
    weak   = len(verbs.get("weak_verbs_found",   []))
    total_verbs = strong + weak
    if total_verbs == 0:
        verb_ratio = 0.0
    else:
        verb_ratio = strong / total_verbs
    # Bonus for having many strong verbs at all
    coverage_bonus = min(strong / 6, 1.0)   # 6+ strong verbs = full coverage
    verb_score = round((verb_ratio * 0.6 + coverage_bonus * 0.4) * 10, 1)

    # 6. Contact info (0–5)
    contact_fields = ["email", "phone", "linkedin", "github"]
    present        = sum(1 for f in contact_fields if contact.get(f))
    contact_score  = round(min(present / 2, 1.0) * 5, 1)   # 2 fields = full marks

    total = round(
        keyword_score + similarity_score + quantification_score
        + section_score + verb_score + contact_score,
        1,
    )
    total = min(total, 100.0)   # cap at 100

    return ScoreBreakdown(
        keyword_score=keyword_score,
        similarity_score=similarity_score,
        quantification_score=quantification_score,
        section_score=section_score,
        verb_score=verb_score,
        contact_score=contact_score,
        total=total,
    )


def score_label(total: float) -> str:
    if total >= 80:
        return "Excellent — strong ATS match"
    elif total >= 65:
        return "Good — likely to pass ATS screening"
    elif total >= 50:
        return "Fair — needs improvement to reliably pass ATS"
    elif total >= 35:
        return "Poor — significant gaps vs. job description"
    else:
        return "Very poor — resume needs a major rewrite for this role"