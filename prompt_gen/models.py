"""Page, prompt, validation, and analytics records shared by the pipeline."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class PromptStatus(str, Enum):
    """Status of a prompt through the validation pipeline."""

    EXTRACTED = "extracted"
    VALIDATED = "validated"
    FACTCHECKED = "factchecked"
    CORRELATED = "correlated"
    REJECTED = "rejected"


class QualityRating(str, Enum):
    """Quality rating for a prompt."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    REJECTED = "rejected"


@dataclass
class CrawledPage:
    """Represents a single crawled web page."""

    url: str
    title: str = ""
    meta_description: str = ""
    h1_tags: list[str] = field(default_factory=list)
    h2_tags: list[str] = field(default_factory=list)
    h3_tags: list[str] = field(default_factory=list)
    body_text: str = ""
    internal_links: list[str] = field(default_factory=list)
    external_links: list[str] = field(default_factory=list)
    word_count: int = 0
    crawled_at: datetime = field(default_factory=datetime.utcnow)
    status_code: int = 200
    content_type: str = "text/html"
    depth: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "url": self.url,
            "title": self.title,
            "meta_description": self.meta_description,
            "h1_tags": self.h1_tags,
            "h2_tags": self.h2_tags,
            "h3_tags": self.h3_tags,
            "body_text": self.body_text[:500],  # Truncate for storage
            "word_count": self.word_count,
            "crawled_at": self.crawled_at.isoformat(),
            "status_code": self.status_code,
            "depth": self.depth,
        }


@dataclass
class ExtractedPrompt:
    """A prompt/query extracted from website content analysis."""

    prompt_text: str
    source_urls: list[str] = field(default_factory=list)
    relevance_score: float = 0.0
    search_intent: str = ""  # informational, navigational, transactional, commercial
    topic_cluster: str = ""
    estimated_volume: str = ""  # high, medium, low
    difficulty: str = ""  # easy, medium, hard
    content_match_score: float = 0.0
    status: PromptStatus = PromptStatus.EXTRACTED
    extraction_reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prompt_text": self.prompt_text,
            "source_urls": self.source_urls,
            "relevance_score": self.relevance_score,
            "search_intent": self.search_intent,
            "topic_cluster": self.topic_cluster,
            "estimated_volume": self.estimated_volume,
            "difficulty": self.difficulty,
            "content_match_score": self.content_match_score,
            "status": self.status.value,
            "extraction_reasoning": self.extraction_reasoning,
        }


@dataclass
class ValidationResult:
    """Prompt validation scores and a summary of each debate turn."""

    prompt_text: str
    is_accurate: bool = False
    accuracy_score: float = 0.0
    accuracy_reasoning: str = ""
    is_faithful: bool = False
    faithfulness_score: float = 0.0
    faithfulness_reasoning: str = ""
    quality_rating: QualityRating = QualityRating.LOW
    quality_score: float = 0.0
    quality_reasoning: str = ""
    overall_score: float = 0.0
    agent_chain_of_thought: list[str] = field(default_factory=list)
    validation_timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prompt_text": self.prompt_text,
            "is_accurate": self.is_accurate,
            "accuracy_score": self.accuracy_score,
            "accuracy_reasoning": self.accuracy_reasoning,
            "is_faithful": self.is_faithful,
            "faithfulness_score": self.faithfulness_score,
            "faithfulness_reasoning": self.faithfulness_reasoning,
            "quality_rating": self.quality_rating.value,
            "quality_score": self.quality_score,
            "quality_reasoning": self.quality_reasoning,
            "overall_score": self.overall_score,
            "agent_chain_of_thought": self.agent_chain_of_thought,
            "validation_timestamp": self.validation_timestamp.isoformat(),
        }


@dataclass
class FactcheckClaim:
    """A single claim extracted for factchecking."""

    claim_text: str
    source_context: str = ""
    is_verifiable: bool = True
    verification_status: str = "pending"  # verified, refuted, unverifiable, pending
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class FactcheckResult:
    """Result of factcheck validation."""

    prompt_text: str
    claims: list[FactcheckClaim] = field(default_factory=list)
    overall_factual_score: float = 0.0
    decomposition_reasoning: str = ""
    verification_summary: str = ""
    is_factually_sound: bool = False
    confidence_level: float = 0.0
    factcheck_timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prompt_text": self.prompt_text,
            "claims": [
                {
                    "claim_text": c.claim_text,
                    "verification_status": c.verification_status,
                    "confidence": c.confidence,
                    "reasoning": c.reasoning,
                }
                for c in self.claims
            ],
            "overall_factual_score": self.overall_factual_score,
            "verification_summary": self.verification_summary,
            "is_factually_sound": self.is_factually_sound,
            "confidence_level": self.confidence_level,
        }


@dataclass
class GSCData:
    """Google Search Console data for a query."""

    query: str
    clicks: int = 0
    impressions: int = 0
    ctr: float = 0.0
    position: float = 0.0
    pages: list[str] = field(default_factory=list)
    date_range: str = ""


@dataclass
class GAData:
    """Google Analytics data for a page/query."""

    page_path: str = ""
    sessions: int = 0
    users: int = 0
    pageviews: int = 0
    avg_session_duration: float = 0.0
    bounce_rate: float = 0.0
    conversions: int = 0
    organic_sessions: int = 0


@dataclass
class CorrelatedPrompt:
    """A prompt with its validation results, matched analytics, and ranking scores."""

    prompt: ExtractedPrompt
    validation: Optional[ValidationResult] = None
    factcheck: Optional[FactcheckResult] = None
    gsc_data: Optional[GSCData] = None
    ga_data: Optional[GAData] = None

    data_correlation_score: float = 0.0
    impact_score: float = 0.0
    priority_rank: int = 0
    recommendation: str = ""
    correlation_reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the prompt, available metrics, and scores."""
        result = {
            "prompt": self.prompt.to_dict(),
            "validation": self.validation.to_dict() if self.validation else None,
            "factcheck": self.factcheck.to_dict() if self.factcheck else None,
            "gsc_data": {
                "query": self.gsc_data.query,
                "clicks": self.gsc_data.clicks,
                "impressions": self.gsc_data.impressions,
                "ctr": self.gsc_data.ctr,
                "position": self.gsc_data.position,
            }
            if self.gsc_data
            else None,
            "ga_data": {
                "page_path": self.ga_data.page_path,
                "sessions": self.ga_data.sessions,
                "organic_sessions": self.ga_data.organic_sessions,
                "pageviews": self.ga_data.pageviews,
                "bounce_rate": self.ga_data.bounce_rate,
            }
            if self.ga_data
            else None,
            "data_correlation_score": self.data_correlation_score,
            "impact_score": self.impact_score,
            "priority_rank": self.priority_rank,
            "recommendation": self.recommendation,
            "correlation_reasoning": self.correlation_reasoning,
        }
        return result
