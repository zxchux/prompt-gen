# PromptGen - AI-Powered Website Prompt Discovery & Validation Engine

PromptGen crawls any website, auto-discovers competitors, identifies the top 500 search prompts the site and its competitors rank for, validates them using adversarial agentic LLM reasoning and the Profound factcheck methodology, then correlates with Google Search Console and Analytics data to surface high-impact monitoring opportunities.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          PromptGen Pipeline                              │
├──────────┬────────────┬──────────┬───────────┬───────────┬──────┬───────┤
│ Stage 1  │ Stage 1.5  │ Stage 2  │  Stage 3  │  Stage 4  │  5   │  6   │
│  Crawl   │ Competitor │ Extract  │ Validate  │ Factcheck │Corr. │Report│
│  Target  │ Discovery  │ Prompts  │ (Agentic  │(Profound) │GSC+GA│      │
│          │ & Analysis │ + Merge  │  Debate)  │           │      │      │
└────┬─────┴─────┬──────┴────┬─────┴─────┬─────┴─────┬─────┴──┬───┴──┬───┘
     │           │           │           │           │        │      │
     ▼           ▼           ▼           ▼           ▼        ▼      ▼
  Website    Auto-find    Top 500    Proposer    Decompose  Fuzzy  JSON
  Pages      Competitors  Prompts    Challenger  Decontext  Match  CSV
  Content    Crawl them   (target +  Defender    Verify     Impact Text
             Extract      competitor Judge       Aggregate  Rank
             prompts      merged)
                                               
     └──────────────────── All LLM calls via OpenRouter ─────────────────┘
```

## Features

### 1. Website Crawler
- BFS-based crawling with configurable depth and page limits
- Respects `robots.txt` and rate limits
- Extracts structured content: titles, headings, meta descriptions, body text
- Sitemap discovery for comprehensive coverage
- SPA (Single Page Application) support via trafilatura and metadata extraction
- Handles SSL certificate issues gracefully

### 2. Competitor Analysis (NEW)
- **Auto-discovers competitors** using three methods:
  - LLM analysis of the target site's content (identifies industry, audience, competitors)
  - External link analysis (finds competitor domains linked from the target site)
  - Manual competitor list via `--competitors` CLI flag
- Crawls each competitor site (limited scope: 15 pages, depth 3)
- Extracts prompts competitors rank for that are relevant to the target
- Merges competitor prompts into the main list with deduplication
- Configurable: `--max-competitors`, `--skip-competitors`

### 3. Prompt Extraction Engine
- LLM-powered analysis of page content to identify ranking queries
- Batch processing for efficiency
- Site-level analysis for content gap discovery
- Competitor prompt merging and deduplication
- Covers all search intents: informational, navigational, transactional, commercial

### 4. Adversarial Agentic Reasoning Validation
- Multi-turn debate pipeline where the LLM takes on different roles:
  - **Proposer**: Argues FOR the prompt's validity with evidence
  - **Challenger**: Devil's advocate — attacks every weakness
  - **Defender**: Counters challenges, must concede valid points
  - **Judge**: Weighs both sides, delivers final scored verdict
- Single multi-turn conversation so each agent sees all prior reasoning
- Weighted scoring: accuracy (30%) + faithfulness (30%) + quality (40%)
- Only prompts that survive the debate pass (0.7 threshold)

### 5. Profound Factcheck Methodology
- Implements the 4-step Profound factcheck pipeline:
  - **Decompose**: Break prompt into atomic verifiable claims
  - **Decontextualize**: Make claims self-contained
  - **Verify**: Check claims against source content (multiple rounds)
  - **Aggregate**: Combine verdicts into overall factual score
- Ensures prompts are grounded in factual content

### 6. Google Search Console Integration
- OAuth2 authentication with token caching (CLI-based flow)
- Fetches query-level performance data (clicks, impressions, CTR, position)
- **CSV import**: Export from GSC web interface and import directly
- Configurable lookback period

### 7. Google Analytics 4 Integration
- Service account authentication
- Page-level traffic metrics (sessions, users, pageviews, bounce rate)
- Organic traffic segmentation
- **CSV import**: Export from GA4 web interface and import directly

### 8. Correlation Engine
- Fuzzy matching between prompts and GSC queries
- URL-based matching for GA page data
- Multi-signal impact scoring
- LLM-enhanced correlation analysis for top candidates
- Priority ranking with actionable recommendations

## Installation

### Prerequisites
- Python 3.11+
- OpenRouter API key ([get one here](https://openrouter.ai/keys))

### Setup

```bash
# Clone the repository
git clone https://github.com/your-org/prompt-gen.git
cd prompt-gen

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install as a package
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

You have two options for bringing in GSC/GA data:

#### Option A: CSV Import (Easiest)
1. Export data from GSC/GA web interfaces as CSV
2. Pass CSV files directly via CLI flags

#### Option B: OAuth Authentication
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project and enable Search Console API / Analytics Data API
3. Download credentials to `credentials/` directory
4. Run `python3 -m prompt_gen auth` to authenticate via browser

## Usage

### Full Pipeline

```bash
# Analyze a website (full pipeline with auto competitor discovery)
python3 -m prompt_gen analyze https://example.com

# With specific competitors
python3 -m prompt_gen analyze https://example.com \
  --competitors "competitor1.com,competitor2.com,competitor3.com"

# Control competitor analysis
python3 -m prompt_gen analyze https://example.com \
  --max-competitors 3 \
  --max-pages 200

# With CSV data imports
python3 -m prompt_gen analyze https://example.com \
  --gsc-csv gsc_queries.csv \
  --ga-csv ga_pages.csv

# Skip optional stages for faster analysis
python3 -m prompt_gen analyze https://example.com \
  --skip-google \
  --skip-factcheck \
  --skip-competitors

# Quick analysis (minimal LLM calls)
python3 -m prompt_gen analyze https://example.com \
  --max-pages 50 \
  --top-prompts 100 \
  --skip-validation \
  --skip-factcheck \
  --skip-google \
  --skip-competitors
```

### Individual Commands

```bash
# Just crawl a website
python3 -m prompt_gen crawl https://example.com --max-pages 50

# Authenticate with Google APIs (opens browser)
python3 -m prompt_gen auth
python3 -m prompt_gen auth --service gsc    # Just GSC
python3 -m prompt_gen auth --service ga     # Just GA

# Preview CSV data before importing
python3 -m prompt_gen preview-csv --gsc-csv queries.csv --ga-csv pages.csv

# Check configuration
python3 -m prompt_gen check-config
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
    skip_google_apis=True,
    manual_competitors=["competitor1.com", "competitor2.com"],
    max_competitors=3,
)

# Access results programmatically
for cp in pipeline.correlated[:10]:
    print(f"#{cp.priority_rank}: {cp.prompt.prompt_text}")
    print(f"  Impact: {cp.impact_score:.2f}")
    print(f"  Recommendation: {cp.recommendation}")

# Access competitor data
for comp in pipeline.competitors:
    print(f"Competitor: {comp['domain']} ({comp['strength']})")
    print(f"  Prompts found: {comp.get('prompts_found', 0)}")
```

### Running as a Module

```bash
python3 -m prompt_gen analyze https://example.com
```

## Output

Reports are generated in the `output/` directory:

| File | Format | Description |
|------|--------|-------------|
| `prompt_report_TIMESTAMP.json` | JSON | Full detailed report with all data |
| `prompt_summary_TIMESTAMP.txt` | Text | Human-readable summary report |
| `prompts_TIMESTAMP.csv` | CSV | Spreadsheet-ready prompt list |

### JSON Report Structure

```json
{
  "metadata": { "target_url": "...", "total_pages_crawled": 150 },
  "summary": { "total_prompts": 500, "high_impact_prompts": 45 },
  "top_prompts": [
    {
      "prompt": { "prompt_text": "...", "search_intent": "informational" },
      "validation": { "overall_score": 0.85, "is_accurate": true },
      "factcheck": { "overall_factual_score": 0.92 },
      "gsc_data": { "clicks": 150, "impressions": 5000, "ctr": 0.03 },
      "impact_score": 0.88,
      "recommendation": "Optimize meta description for higher CTR"
    }
  ],
  "topic_clusters": { ... },
  "recommendations": [ ... ]
}
```

## Pipeline Stages Detail

### Stage 1: Crawl
- BFS traversal from the start URL
- Extracts: title, meta description, H1-H3 tags, body text, links
- Sitemap discovery seeds the crawl queue
- SPA support via trafilatura + metadata extraction
- Configurable: max pages, crawl delay, max depth

### Stage 1.5: Competitor Analysis
- Auto-discovers competitors via LLM analysis of target site content
- Identifies industry, target audience, and direct/indirect competitors
- Crawls each competitor (limited: 15 pages, depth 3)
- Extracts prompts competitors rank for that are relevant to the target
- Supports manual competitor list via `--competitors` flag
- Configurable: `--max-competitors`, `--skip-competitors`

### Stage 2: Extract Prompts
- Processes pages in batches through LLM
- Each page analyzed for 10-15 potential search queries
- Site-level analysis adds content gap prompts
- **Merges competitor prompts** into the list with deduplication
- Returns top N prompts ranked by relevance

### Stage 3: Agentic Validation (Adversarial Debate)
- 4-turn multi-turn conversation per prompt:
  1. **Proposer**: Makes the case FOR the prompt with evidence
  2. **Challenger**: Devil's advocate, finds every weakness
  3. **Defender**: Counters challenges, concedes valid points
  4. **Judge**: Weighs both sides, delivers scored verdict
- Each agent sees the full conversation history
- Prompts below 0.7 threshold are rejected

### Stage 4: Profound Factcheck
- 4-step factcheck pipeline per prompt:
  1. Decompose into atomic claims
  2. Decontextualize for independent verification
  3. Verify each claim (multiple rounds)
  4. Aggregate into overall factual score
- Only processes validated prompts

### Stage 5: Correlate
- Fetches GSC query data and GA page data (API or CSV import)
- Fuzzy matches prompts to existing search queries
- Calculates data correlation and impact scores
- LLM-enhanced analysis for top candidates
- Generates actionable recommendations

### Stage 6: Report
- JSON report with full data
- Human-readable summary
- CSV for spreadsheet analysis

## CLI Reference

### `analyze` - Full Pipeline

```
Options:
  --max-pages INTEGER          Maximum pages to crawl (default: 500)
  --top-prompts INTEGER        Number of top prompts (default: 500)
  --competitors TEXT            Comma-separated competitor domains
  --max-competitors INTEGER    Max competitors to analyze (default: 5)
  --skip-competitors           Skip competitor analysis
  --skip-validation            Skip agentic validation
  --skip-factcheck             Skip Profound factcheck
  --skip-google                Skip Google API calls
  --gsc-csv PATH               GSC query data CSV file
  --ga-csv PATH                GA page data CSV file
  --model TEXT                 Override LLM model
  --output-dir TEXT            Output directory
  --log-level [debug|info|warning|error]
```

### `auth` - Google OAuth

```
Options:
  --service [gsc|ga|both]      Which service to authenticate (default: both)
```

### `preview-csv` - Preview CSV Data

```
Options:
  --gsc-csv PATH               GSC query CSV to preview
  --ga-csv PATH                GA page CSV to preview
```

### `crawl` - Crawl Only

```
Options:
  --max-pages INTEGER          Max pages to crawl (default: 50)
```

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
│   ├── config.py                # Configuration management
│   ├── models.py                # Data models
│   ├── pipeline.py              # Pipeline orchestrator
│   ├── llm_client.py            # OpenRouter LLM client
│   ├── crawler.py               # Web crawler (BFS + sitemap + SPA)
│   ├── competitor_analyzer.py   # Competitor discovery & analysis
│   ├── prompt_extractor.py      # Prompt extraction engine
│   ├── agentic_validator.py     # Adversarial debate validator
│   ├── factcheck.py             # Profound factcheck module
│   ├── csv_importer.py          # GSC/GA CSV data importer
│   ├── google_search_console.py # GSC API integration
│   ├── google_analytics.py      # GA4 API integration
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
