"""Load application settings from environment variables and .env."""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load .env file early so all os.environ lookups work
load_dotenv(dotenv_path=Path(".env"), override=False)


def _env(key: str, default: str = "") -> str:
    """Get an environment variable with a default."""
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    """Get an integer environment variable."""
    val = os.environ.get(key)
    return int(val) if val is not None else default


def _env_float(key: str, default: float) -> float:
    """Get a float environment variable."""
    val = os.environ.get(key)
    return float(val) if val is not None else default


def _env_bool(key: str, default: bool) -> bool:
    """Get a boolean environment variable."""
    val = os.environ.get(key)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes")


class OpenRouterConfig(BaseModel):
    """OpenRouter API configuration."""

    api_key: str = ""
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "anthropic/claude-sonnet-4"
    max_tokens: int = 4096
    temperature: float = 0.3
    request_timeout: int = 120
    enable_prompt_cache: bool = True


class CrawlerConfig(BaseModel):
    """Web crawler configuration."""

    max_pages: int = 500
    crawl_delay: float = 1.0
    max_concurrent_requests: int = 5
    user_agent: str = "PromptGen Bot/1.0 (+https://github.com/promptgen)"
    request_timeout: int = 30
    respect_robots_txt: bool = True
    max_depth: int = 5


class PromptExtractionConfig(BaseModel):
    """Prompt extraction configuration."""

    top_prompts_count: int = 500
    min_quality_score: float = 0.7
    batch_size: int = 10
    dedup_similarity_threshold: float = 0.85


class FactcheckConfig(BaseModel):
    """Factcheck validation configuration."""

    confidence_threshold: float = 0.8
    max_claims_per_prompt: int = 10
    verification_rounds: int = 3


class GoogleConfig(BaseModel):
    """Google API configuration."""

    gsc_credentials_file: str = "credentials/gsc_credentials.json"
    gsc_token_file: str = "credentials/gsc_token.json"
    ga_credentials_file: str = "credentials/ga_credentials.json"
    ga_property_id: str = ""
    gsc_site_url: Optional[str] = None
    data_lookback_days: int = 90


class CacheConfig(BaseModel):
    """Cache configuration."""

    cache_dir: str = ".cache"
    cache_ttl: int = 86400
    enable_cache: bool = True


class LoggingConfig(BaseModel):
    """Logging configuration."""

    log_level: str = "INFO"
    log_file: str = "logs/prompt_gen.log"
    log_format: str = "json"


class Settings(BaseModel):
    """Application settings grouped by service and pipeline stage."""

    openrouter: OpenRouterConfig = Field(default_factory=OpenRouterConfig)
    crawler: CrawlerConfig = Field(default_factory=CrawlerConfig)
    prompt_extraction: PromptExtractionConfig = Field(
        default_factory=PromptExtractionConfig
    )
    factcheck: FactcheckConfig = Field(default_factory=FactcheckConfig)
    google: GoogleConfig = Field(default_factory=GoogleConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    output_dir: str = "output"
    report_format: str = "json"

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        dirs = [
            self.cache.cache_dir,
            self.output_dir,
            Path(self.logging.log_file).parent,
            "credentials",
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)


def load_settings() -> Settings:
    """Load and return application settings from environment variables."""
    settings = Settings(
        openrouter=OpenRouterConfig(
            api_key=_env("OPENROUTER_API_KEY"),
            base_url=_env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            model=_env("OPENROUTER_MODEL", "anthropic/claude-sonnet-4"),
            max_tokens=_env_int("OPENROUTER_MAX_TOKENS", 4096),
            temperature=_env_float("OPENROUTER_TEMPERATURE", 0.3),
            request_timeout=_env_int("OPENROUTER_TIMEOUT", 120),
            enable_prompt_cache=_env_bool("ENABLE_PROMPT_CACHE", True),
        ),
        crawler=CrawlerConfig(
            max_pages=_env_int("MAX_PAGES", 500),
            crawl_delay=_env_float("CRAWL_DELAY", 1.0),
            max_concurrent_requests=_env_int("MAX_CONCURRENT_REQUESTS", 5),
            user_agent=_env("USER_AGENT", "PromptGen Bot/1.0 (+https://github.com/promptgen)"),
            request_timeout=_env_int("CRAWL_REQUEST_TIMEOUT", 30),
            respect_robots_txt=_env_bool("RESPECT_ROBOTS_TXT", True),
            max_depth=_env_int("MAX_CRAWL_DEPTH", 5),
        ),
        prompt_extraction=PromptExtractionConfig(
            top_prompts_count=_env_int("TOP_PROMPTS_COUNT", 500),
            min_quality_score=_env_float("MIN_PROMPT_QUALITY_SCORE", 0.7),
            batch_size=_env_int("PROMPT_BATCH_SIZE", 10),
            dedup_similarity_threshold=_env_float("DEDUP_SIMILARITY_THRESHOLD", 0.85),
        ),
        factcheck=FactcheckConfig(
            confidence_threshold=_env_float("FACTCHECK_CONFIDENCE_THRESHOLD", 0.8),
            max_claims_per_prompt=_env_int("MAX_CLAIMS_PER_PROMPT", 10),
            verification_rounds=_env_int("VERIFICATION_ROUNDS", 3),
        ),
        google=GoogleConfig(
            gsc_credentials_file=_env("GSC_CREDENTIALS_FILE", "credentials/gsc_credentials.json"),
            gsc_token_file=_env("GSC_TOKEN_FILE", "credentials/gsc_token.json"),
            ga_credentials_file=_env("GA_CREDENTIALS_FILE", "credentials/ga_credentials.json"),
            ga_property_id=_env("GA_PROPERTY_ID"),
            gsc_site_url=_env("GSC_SITE_URL") or None,
            data_lookback_days=_env_int("DATA_LOOKBACK_DAYS", 90),
        ),
        cache=CacheConfig(
            cache_dir=_env("CACHE_DIR", ".cache"),
            cache_ttl=_env_int("CACHE_TTL", 86400),
            enable_cache=_env_bool("ENABLE_CACHE", True),
        ),
        logging=LoggingConfig(
            log_level=_env("LOG_LEVEL", "INFO"),
            log_file=_env("LOG_FILE", "logs/prompt_gen.log"),
            log_format=_env("LOG_FORMAT", "json"),
        ),
        output_dir=_env("OUTPUT_DIR", "output"),
        report_format=_env("REPORT_FORMAT", "json"),
    )
    settings.ensure_directories()
    return settings
