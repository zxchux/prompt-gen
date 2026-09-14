# PromptGen

PromptGen crawls a website and suggests search queries based on its content. It uses
models through OpenRouter to evaluate those queries, check their claims against
source pages, and rank them alongside Google Search Console and Analytics data.
The default limit is 500 queries; the results are suggestions, not measured rankings.

The pipeline has six stages:

1. **Crawl:** Collect page text, headings, metadata, links, and sitemap URLs.
2. **Extract:** Suggest queries from page batches and site summaries, then merge duplicates.
3. **Validate:** Run proposer, challenger, defender, and judge turns in one conversation.
4. **Factcheck:** Extract claims, make them self-contained, verify them, and combine verdicts.
5. **Correlate:** Match queries and source URLs to analytics data and score opportunities.
6. **Report:** Write JSON, CSV, and text reports.

Validation and factchecking are optional. Analytics data can come from CSV exports
or Google APIs; analysis can also run without it.

## Installation

### Prerequisites
- Python 3.11+
- OpenRouter API key ([get one here](https://openrouter.ai/keys))

### Setup

Run these commands from the project directory:

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package and its dependencies
pip install -e .

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Configuration

Edit `.env` with your settings:

```env
# Required
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Optional - LLM model (default: anthropic/claude-sonnet-4)
OPENROUTER_MODEL=anthropic/claude-sonnet-4

# Optional - Google APIs (for correlation stage)
GSC_CREDENTIALS_FILE=credentials/gsc_credentials.json
GA_CREDENTIALS_FILE=credentials/ga_credentials.json
GA_PROPERTY_ID=your_ga4_property_id

# Optional - Crawler settings
MAX_PAGES=500
CRAWL_DELAY=1.0
TOP_PROMPTS_COUNT=500
```

### Google API Setup (Optional)

CSV imports do not need Google credentials. To fetch data through the APIs:

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project and enable:
   - Search Console API
   - Google Analytics Data API
3. For GSC: Create OAuth2 credentials → download as `credentials/gsc_credentials.json`
4. For GA: Create a service account → download as `credentials/ga_credentials.json`
5. Add the service account email to your GA4 property with Viewer access

## Usage

### Full Pipeline

```bash
# Analyze a website (full pipeline)
promptgen analyze https://example.com

# With options
promptgen analyze https://example.com \
  --max-pages 200 \
  --top-prompts 500 \
  --model anthropic/claude-sonnet-4

# Skip optional stages
promptgen analyze https://example.com \
  --skip-google \
  --skip-factcheck

# Quick analysis (minimal LLM calls)
promptgen analyze https://example.com \
  --max-pages 50 \
  --top-prompts 100 \
  --skip-validation \
  --skip-factcheck \
  --skip-google
```

### Use CSV exports

Export the Queries tab from Search Console and the Pages and screens report from
GA4. Preview the columns before running an analysis:

```bash
promptgen preview-csv --gsc-csv queries.csv --ga-csv pages.csv
promptgen analyze https://example.com \
  --gsc-csv queries.csv \
  --ga-csv pages.csv \
  --skip-google
```

CSV input takes precedence for each service. `--skip-google` prevents API calls
for any service without a CSV input.

### Individual Commands

```bash
# Just crawl a website
promptgen crawl https://example.com --max-pages 50

# Check local configuration
promptgen check-config

# Authenticate Search Console through browser OAuth
promptgen auth --service gsc

# Check Analytics service account credentials
promptgen auth --service ga
```

### Python API

```python
from prompt_gen.config import load_settings
from prompt_gen.pipeline import Pipeline

# Load settings from .env
settings = load_settings()

# Create and run pipeline
pipeline = Pipeline(settings)
reports = pipeline.run(
    target_url="https://example.com",
    skip_google_apis=True,  # Skip if no Google credentials
)

# Access results programmatically
for result in pipeline.correlated[:10]:
    print(f"#{result.priority_rank}: {result.prompt.prompt_text}")
    print(f"  Impact: {result.impact_score:.2f}")
    print(f"  Recommendation: {result.recommendation}")
```

### Running as a Module

```bash
python -m prompt_gen analyze https://example.com
```

## Output

Reports are generated in the `output/` directory:

| File | Format | Description |
|------|--------|-------------|
| `prompt_report_TIMESTAMP.json` | JSON | Prompt results, scores, matched metrics, and recommendations |
| `prompt_summary_TIMESTAMP.txt` | Text | Human-readable summary report |
| `prompts_TIMESTAMP.csv` | CSV | Spreadsheet-ready prompt list |

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | (required) | OpenRouter API key |
| `OPENROUTER_MODEL` | `anthropic/claude-sonnet-4` | LLM model to use |
| `MAX_PAGES` | `500` | Maximum pages to crawl |
| `CRAWL_DELAY` | `1.0` | Seconds between requests |
| `MAX_CRAWL_DEPTH` | `5` | Maximum crawl depth |
| `TOP_PROMPTS_COUNT` | `500` | Number of top prompts |
| `MIN_PROMPT_QUALITY_SCORE` | `0.7` | Minimum validation score |
| `FACTCHECK_CONFIDENCE_THRESHOLD` | `0.8` | Minimum factcheck score |
| `DATA_LOOKBACK_DAYS` | `90` | Days of GSC/GA data |
| `ENABLE_CACHE` | `true` | Cache intermediate results |
| `OUTPUT_DIR` | `output` | Report output directory |

## Project Structure

```
prompt-gen/
├── prompt_gen/
│   ├── __init__.py              # Package initialization
│   ├── __main__.py              # Module entry point
│   ├── cli.py                   # CLI interface (Click + Rich)
│   ├── config.py                # Environment settings (Pydantic)
│   ├── models.py                # Data models
│   ├── pipeline.py              # Pipeline orchestrator
│   ├── llm_client.py            # OpenRouter LLM client
│   ├── crawler.py               # Web crawler
│   ├── prompt_extractor.py      # Query suggestions and deduplication
│   ├── csv_importer.py          # Search Console and GA4 CSV imports
│   ├── agentic_validator.py     # Four-turn prompt evaluation
│   ├── factcheck.py             # Claim extraction and verification
│   ├── google_search_console.py # GSC integration
│   ├── google_analytics.py      # GA4 integration
│   ├── correlation_engine.py    # Data correlation engine
│   └── reporter.py              # Report generator
├── .env.example                 # Environment template
├── .gitignore                   # Git ignore rules
├── pyproject.toml               # Project metadata & build config
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## License

MIT License
