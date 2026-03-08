"""
Response format schemas for structured LLM outputs.
"""
from pydantic import BaseModel
from typing import Optional, List


class LRAM_paper_rough(BaseModel):
    """Standard LRAM paper extraction format."""
    title: str
    authors: str
    abstract: str
    method: str
    results: str
    conclusion: str

class LRAM_paper(BaseModel):
    type_of_metamaterial_design: Optional[str] = None
    active_control_present: Optional[bool] = None
    attenuation_band_min_hz: Optional[float] = None
    attenuation_band_max_hz: Optional[float] = None
    attenuation_band_min_normalized: Optional[float] = None
    attenuation_band_max_normalized: Optional[float] = None
    peak_attenuation_db: Optional[float] = None
    material_of_metamaterial: Optional[List[str]] = None
    evidence_text: Optional[List[str]] = None

class PaperComparison(BaseModel):
    """Comparison between query paper and related work."""
    query_paper_summary: str
    methodological_differences: str
    key_findings_comparison: str
    novel_contributions: str
    similarities: str
    citations: List[int]


class MethodologyAnalysis(BaseModel):
    """Detailed methodology extraction."""
    research_methods: str
    experimental_design: str
    data_collection: str
    data_analysis: str
    limitations: Optional[str] = None
    citations: List[int]


class PaperSummary(BaseModel):
    """Concise paper summary."""
    title: str
    main_contribution: str
    key_findings: List[str]
    methodology_brief: str
    implications: str


class DetailedExtraction(BaseModel):
    """Comprehensive paper information extraction."""
    title: str
    authors: str
    year: Optional[str] = None
    abstract: str
    introduction: str
    method: str
    results: str
    discussion: str
    conclusion: str
    future_work: Optional[str] = None
    limitations: Optional[str] = None
