"""Commands for website analysis, crawling, CSV previews, and Google authentication."""

import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from prompt_gen.config import load_settings
from prompt_gen.pipeline import Pipeline

console = Console()


def setup_logging(log_level: str, log_file: str) -> None:
    """Configure logging with rich handler."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    handlers = [
        RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            markup=True,
        ),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
    )


@click.group()
@click.version_option(version="1.0.0", prog_name="PromptGen")
def cli():
    """Discover and evaluate search queries for a website.

    Crawl pages, suggest queries through OpenRouter, and compare results with
    Google Search Console and Analytics data.
    """
    pass


@cli.command()
@click.argument("url")
@click.option(
    "--max-pages",
    default=None,
    type=int,
    help="Maximum pages to crawl (overrides config).",
)
@click.option(
    "--top-prompts",
    default=None,
    type=int,
    help="Number of top prompts to extract (default: 500).",
)
@click.option(
    "--skip-validation",
    is_flag=True,
    default=False,
    help="Skip the agentic validation stage.",
)
@click.option(
    "--skip-factcheck",
    is_flag=True,
    default=False,
    help="Skip the factcheck stage.",
)
@click.option(
    "--skip-google",
    is_flag=True,
    default=False,
    help="Skip Google Search Console and Analytics integration.",
)
@click.option(
    "--gsc-csv",
    default=None,
    type=click.Path(exists=True),
    help="Path to GSC query data CSV (exported from Search Console).",
)
@click.option(
    "--ga-csv",
    default=None,
    type=click.Path(exists=True),
    help="Path to GA page data CSV (exported from Google Analytics).",
)
@click.option(
    "--competitors",
    default=None,
    type=str,
    help="Comma-separated list of competitor domains (e.g., 'comp1.com,comp2.com').",
)
@click.option(
    "--max-competitors",
    default=5,
    type=int,
    help="Maximum number of competitors to analyze (default: 5).",
)
@click.option(
    "--skip-competitors",
    is_flag=True,
    default=False,
    help="Skip competitor analysis stage.",
)
@click.option(
    "--consensus-passes",
    default=1,
    type=int,
    help="Number of extraction passes for consistency (1=single, 3=recommended for stability).",
)
@click.option(
    "--model",
    default=None,
    type=str,
    help="Override the LLM model (e.g., anthropic/claude-sonnet-4).",
)
@click.option(
    "--output-dir",
    default=None,
    type=str,
    help="Output directory for reports.",
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    help="Logging level.",
)
def analyze(
    url: str,
    max_pages: int,
    top_prompts: int,
    skip_validation: bool,
    skip_factcheck: bool,
    skip_google: bool,
    gsc_csv: str,
    ga_csv: str,
    competitors: str,
    max_competitors: int,
    skip_competitors: bool,
    consensus_passes: int,
    model: str,
    output_dir: str,
    log_level: str,
):
    """Run the full analysis pipeline on a website URL.

    Example:
        promptgen analyze https://example.com
        promptgen analyze https://example.com --max-pages 100 --skip-google
        promptgen analyze https://example.com --gsc-csv gsc_queries.csv --ga-csv ga_pages.csv
        promptgen analyze https://example.com --competitors "labster.com,chemcollective.org"
        promptgen analyze https://example.com --consensus-passes 3
    """
    # Parse manual competitors
    manual_competitors = None
    if competitors:
        manual_competitors = [c.strip() for c in competitors.split(",") if c.strip()]

    settings = load_settings()

    if max_pages:
        settings.crawler.max_pages = max_pages
    if top_prompts:
        settings.prompt_extraction.top_prompts_count = top_prompts
    if model:
        settings.openrouter.model = model
    if output_dir:
        settings.output_dir = output_dir

    setup_logging(log_level, settings.logging.log_file)

    if not settings.openrouter.api_key:
        console.print(
            Panel(
                "[red]OpenRouter API key not configured![/red]\n\n"
                "Set OPENROUTER_API_KEY in your .env file or environment:\n"
                "  export OPENROUTER_API_KEY=your_key_here\n\n"
                "Get your API key at: https://openrouter.ai/keys",
                title="Configuration Error",
                border_style="red",
            )
        )
        sys.exit(1)

    data_source_parts = []
    if gsc_csv:
        data_source_parts.append(f"GSC CSV: {gsc_csv}")
    if ga_csv:
        data_source_parts.append(f"GA CSV: {ga_csv}")
    if not gsc_csv and not ga_csv and not skip_google:
        data_source_parts.append("Google APIs (OAuth)")
    if skip_google and not gsc_csv and not ga_csv:
        data_source_parts.append("None (skipped)")
    data_source = "\n".join(f"[bold cyan]  {p}[/bold cyan]" for p in data_source_parts)

    console.print(
        Panel(
            f"[bold cyan]Target URL:[/bold cyan] {url}\n"
            f"[bold cyan]Max Pages:[/bold cyan] {settings.crawler.max_pages}\n"
            f"[bold cyan]Top Prompts:[/bold cyan] {settings.prompt_extraction.top_prompts_count}\n"
            f"[bold cyan]LLM Model:[/bold cyan] {settings.openrouter.model}\n"
            f"[bold cyan]Validation:[/bold cyan] "
            f"{'Enabled' if not skip_validation else 'Skipped'}\n"
            f"[bold cyan]Factcheck:[/bold cyan] "
            f"{'Enabled' if not skip_factcheck else 'Skipped'}\n"
            f"[bold cyan]Data Sources:[/bold cyan]\n{data_source}",
            title="[bold green]PromptGen Analysis[/bold green]",
            border_style="green",
        )
    )

    pipeline = Pipeline(settings)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            crawl_task = progress.add_task(
                "[cyan]Crawling website...", total=settings.crawler.max_pages
            )

            def crawl_progress(current, total):
                progress.update(crawl_task, completed=current, total=total)

            pipeline.stage_crawl(url, progress_callback=crawl_progress)
            progress.update(crawl_task, completed=len(pipeline.pages))

            if not pipeline.pages:
                console.print("[red]No pages crawled. Check the URL and try again.[/red]")
                sys.exit(1)

            console.print(f"  ✓ Crawled [green]{len(pipeline.pages)}[/green] pages")

            # Stage 1.5: Competitor Analysis
            if not skip_competitors:
                comp_task = progress.add_task(
                    "[cyan]Analyzing competitors...", total=max_competitors
                )

                def comp_progress(current, total, name):
                    progress.update(
                        comp_task,
                        completed=current,
                        total=total,
                        description=f"[cyan]Analyzing competitor: {name}...",
                    )

                pipeline.stage_competitors(
                    target_url=url,
                    manual_competitors=manual_competitors,
                    max_competitors=max_competitors,
                    progress_callback=comp_progress,
                )
                progress.update(comp_task, completed=max_competitors)
                console.print(
                    f"  ✓ Analyzed [green]{len(pipeline.competitors)}[/green] competitors, "
                    f"found [green]{len(pipeline.competitor_prompts)}[/green] competitor prompts"
                )
            else:
                console.print("  ⊘ Competitor analysis skipped")

            # Stage 2: Extract (merges competitor prompts)
            extract_task = progress.add_task(
                "[cyan]Extracting prompts...", total=100
            )

            def extract_progress(batch, total):
                progress.update(
                    extract_task,
                    completed=int(batch / total * 100),
                    total=100,
                )

            pipeline.stage_extract(progress_callback=extract_progress)
            progress.update(extract_task, completed=100)
            comp_note = (
                f" (includes {len(pipeline.competitor_prompts)} from competitors)"
                if pipeline.competitor_prompts
                else ""
            )
            console.print(
                f"  ✓ Extracted [green]{len(pipeline.prompts)}[/green] prompts{comp_note}"
            )

            # Stage 2.5: Consistency reconciliation
            consistency_task = progress.add_task(
                "[cyan]Reconciling consistency...",
                total=max(consensus_passes, 1),
            )
            pipeline.stage_consistency(
                target_url=url,
                consensus_passes=consensus_passes,
            )
            progress.update(consistency_task, completed=max(consensus_passes, 1))

            drift = pipeline.drift_report
            if drift.get("has_history"):
                drift_score = drift["drift_score"]
                drift_color = (
                    "green" if drift_score < 0.2
                    else "yellow" if drift_score < 0.5
                    else "red"
                )
                console.print(
                    f"  ✓ Consistency: [{drift_color}]drift {drift_score:.0%}[/{drift_color}] "
                    f"({drift['new_count']} new, {drift['dropped_count']} dropped, "
                    f"{drift['stable_count']} stable across {drift['total_historical_runs']} runs)"
                )
            else:
                console.print(
                    f"  ✓ Consistency: first run — {len(pipeline.prompts)} prompts baselined"
                )

            if not skip_validation:
                validate_task = progress.add_task(
                    "[cyan]Validating prompts...",
                    total=len(pipeline.prompts),
                )

                def validate_progress(current, total):
                    progress.update(validate_task, completed=current, total=total)

                pipeline.stage_validate(progress_callback=validate_progress)
                passed = sum(
                    1
                    for v in pipeline.validations
                    if v.overall_score
                    >= settings.prompt_extraction.min_quality_score
                )
                console.print(
                    f"  ✓ Validated [green]{passed}[/green]/"
                    f"{len(pipeline.validations)} prompts"
                )
            else:
                for p in pipeline.prompts:
                    from prompt_gen.models import PromptStatus
                    p.status = PromptStatus.VALIDATED
                console.print("  ⊘ Validation skipped")

            if not skip_factcheck:
                validated_count = sum(
                    1
                    for p in pipeline.prompts
                    if p.status.value == "validated"
                )
                factcheck_task = progress.add_task(
                    "[cyan]Factchecking prompts...",
                    total=max(validated_count, 1),
                )

                def factcheck_progress(current, total):
                    progress.update(factcheck_task, completed=current, total=total)

                pipeline.stage_factcheck(progress_callback=factcheck_progress)
                passed = sum(1 for f in pipeline.factchecks if f.is_factually_sound)
                console.print(
                    f"  ✓ Factchecked [green]{passed}[/green]/"
                    f"{len(pipeline.factchecks)} prompts"
                )
            else:
                for p in pipeline.prompts:
                    from prompt_gen.models import PromptStatus
                    if p.status == PromptStatus.VALIDATED:
                        p.status = PromptStatus.FACTCHECKED
                console.print("  ⊘ Factcheck skipped")

            correlate_task = progress.add_task(
                "[cyan]Correlating with data...", total=100
            )

            def correlate_progress(current, total):
                progress.update(
                    correlate_task,
                    completed=int(current / total * 100) if total > 0 else 100,
                    total=100,
                )

            pipeline.stage_correlate(
                url,
                skip_google_apis=skip_google,
                gsc_csv_path=gsc_csv,
                ga_csv_path=ga_csv,
                progress_callback=correlate_progress,
            )
            progress.update(correlate_task, completed=100)
            console.print(
                f"  ✓ Correlated [green]{len(pipeline.correlated)}[/green] prompts"
            )

            report_task = progress.add_task(
                "[cyan]Generating reports...", total=3
            )
            reports = pipeline.stage_report(url)
            progress.update(report_task, completed=3)

        console.print()
        _display_results(pipeline, reports)

    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline interrupted by user.[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]Pipeline error: {e}[/red]")
        logging.exception("Pipeline failed")
        sys.exit(1)


@cli.command()
@click.argument("url")
@click.option("--max-pages", default=50, type=int, help="Max pages to crawl.")
def crawl(url: str, max_pages: int):
    """Crawl a website and display page information.

    Example:
        promptgen crawl https://example.com --max-pages 20
    """
    settings = load_settings()
    settings.crawler.max_pages = max_pages
    setup_logging("INFO", settings.logging.log_file)

    crawler = __import__("prompt_gen.crawler", fromlist=["WebCrawler"]).WebCrawler(
        settings.crawler
    )

    console.print(f"[cyan]Crawling {url} (max {max_pages} pages)...[/cyan]")
    pages = crawler.crawl(url)

    table = Table(title=f"Crawled Pages ({len(pages)})")
    table.add_column("URL", style="cyan", max_width=60)
    table.add_column("Title", max_width=40)
    table.add_column("Words", justify="right")
    table.add_column("Links", justify="right")

    for page in pages:
        table.add_row(
            page.url[:60],
            (page.title or "")[:40],
            str(page.word_count),
            str(len(page.internal_links)),
        )

    console.print(table)


@cli.command()
def check_config():
    """Check local settings and credential file paths."""
    settings = load_settings()

    console.print(Panel("[bold]Configuration Check[/bold]", border_style="blue"))

    if settings.openrouter.api_key:
        console.print("  ✓ OpenRouter API key: [green]configured[/green]")
        console.print(f"    Model: {settings.openrouter.model}")
    else:
        console.print("  ✗ OpenRouter API key: [red]not configured[/red]")

    gsc_creds = Path(settings.google.gsc_credentials_file)
    if gsc_creds.exists():
        console.print("  ✓ GSC credentials: [green]found[/green]")
    else:
        console.print(f"  ⊘ GSC credentials: [yellow]not found[/yellow] ({gsc_creds})")

    ga_creds = Path(settings.google.ga_credentials_file)
    if ga_creds.exists():
        console.print("  ✓ GA credentials: [green]found[/green]")
    else:
        console.print(f"  ⊘ GA credentials: [yellow]not found[/yellow] ({ga_creds})")

    if settings.google.ga_property_id:
        console.print(f"  ✓ GA Property ID: [green]{settings.google.ga_property_id}[/green]")
    else:
        console.print("  ⊘ GA Property ID: [yellow]not configured[/yellow]")

    console.print(f"\n  Output dir: {settings.output_dir}")
    console.print(f"  Cache dir: {settings.cache.cache_dir}")
    console.print(f"  Log file: {settings.logging.log_file}")


@cli.command()
@click.option(
    "--service",
    type=click.Choice(["gsc", "ga", "both"], case_sensitive=False),
    default="both",
    help="Which Google service to authenticate.",
)
def auth(service: str):
    """Authenticate with Google Search Console or Analytics.

    Search Console uses browser-based OAuth and saves a token for later runs.
    Analytics uses the configured service account credentials.

    Examples:
        promptgen auth
        promptgen auth --service gsc
    """
    settings = load_settings()
    setup_logging("INFO", settings.logging.log_file)

    console.print(
        Panel(
            "[bold]Google API Authentication[/bold]\n\n"
            "This will open your browser for OAuth consent.\n"
            "Make sure you have credentials files configured in .env",
            border_style="blue",
        )
    )

    if service in ("gsc", "both"):
        console.print("\n[cyan]Authenticating with Google Search Console...[/cyan]")
        from prompt_gen.google_search_console import SearchConsoleClient

        gsc = SearchConsoleClient(settings.google)
        if gsc.authenticate():
            console.print("  ✓ GSC authentication [green]successful[/green]")
            try:
                site_list = gsc.service.sites().list().execute()
                sites = site_list.get("siteEntry", [])
                if sites:
                    console.print("  Available sites:")
                    for site in sites:
                        console.print(f"    • {site.get('siteUrl', '')}")
                else:
                    console.print("  [yellow]No sites found in your GSC account[/yellow]")
            except Exception as e:
                console.print(f"  [yellow]Could not list sites: {e}[/yellow]")
        else:
            console.print("  ✗ GSC authentication [red]failed[/red]")

    if service in ("ga", "both"):
        console.print("\n[cyan]Authenticating with Google Analytics...[/cyan]")
        from prompt_gen.google_analytics import AnalyticsClient

        ga = AnalyticsClient(settings.google)
        if ga.authenticate():
            console.print("  ✓ GA authentication [green]successful[/green]")
            if settings.google.ga_property_id:
                console.print(f"  Property ID: {settings.google.ga_property_id}")
            else:
                console.print(
                    "  [yellow]Set GA_PROPERTY_ID in .env to specify your GA4 property[/yellow]"
                )
        else:
            console.print("  ✗ GA authentication [red]failed[/red]")

    console.print()


@cli.command()
@click.option(
    "--gsc-csv",
    default=None,
    type=click.Path(exists=True),
    help="Path to GSC query data CSV.",
)
@click.option(
    "--ga-csv",
    default=None,
    type=click.Path(exists=True),
    help="Path to GA page data CSV.",
)
def preview_csv(gsc_csv: str, ga_csv: str):
    """Preview imported CSV data to verify format before running analysis.

    Example:
        promptgen preview-csv --gsc-csv queries.csv
        promptgen preview-csv --ga-csv pages.csv
        promptgen preview-csv --gsc-csv queries.csv --ga-csv pages.csv
    """
    from prompt_gen.csv_importer import CSVImporter

    importer = CSVImporter()

    if not gsc_csv and not ga_csv:
        console.print("[red]Provide at least one CSV file: --gsc-csv or --ga-csv[/red]")
        return

    if gsc_csv:
        console.print(f"\n[bold cyan]GSC Data Preview[/bold cyan] ({gsc_csv})")
        gsc_data = importer.import_gsc_queries(gsc_csv)

        if gsc_data:
            table = Table(title=f"GSC Queries ({len(gsc_data)} total)")
            table.add_column("Query", style="cyan", max_width=50)
            table.add_column("Clicks", justify="right")
            table.add_column("Impressions", justify="right")
            table.add_column("CTR", justify="right")
            table.add_column("Position", justify="right")

            for row in gsc_data[:20]:
                table.add_row(
                    row.query[:50],
                    str(row.clicks),
                    str(row.impressions),
                    f"{row.ctr:.3f}",
                    f"{row.position:.1f}",
                )

            console.print(table)
            if len(gsc_data) > 20:
                console.print(f"  ... and {len(gsc_data) - 20} more rows")
        else:
            console.print("  [red]No data imported. Check CSV format.[/red]")

    if ga_csv:
        console.print(f"\n[bold cyan]GA Data Preview[/bold cyan] ({ga_csv})")
        ga_data = importer.import_ga_pages(ga_csv)

        if ga_data:
            table = Table(title=f"GA Pages ({len(ga_data)} total)")
            table.add_column("Page Path", style="cyan", max_width=50)
            table.add_column("Sessions", justify="right")
            table.add_column("Users", justify="right")
            table.add_column("Pageviews", justify="right")
            table.add_column("Bounce Rate", justify="right")

            for path, data in list(ga_data.items())[:20]:
                table.add_row(
                    path[:50],
                    str(data.sessions),
                    str(data.users),
                    str(data.pageviews),
                    f"{data.bounce_rate:.2f}",
                )

            console.print(table)
            if len(ga_data) > 20:
                console.print(f"  ... and {len(ga_data) - 20} more rows")
        else:
            console.print("  [red]No data imported. Check CSV format.[/red]")


def _display_results(pipeline: Pipeline, reports: dict) -> None:
    """Display final results in a rich table."""
    table = Table(title="Top 20 High-Impact Prompts")
    table.add_column("#", style="bold", width=4)
    table.add_column("Prompt", style="cyan", max_width=50)
    table.add_column("Impact", justify="right", style="green")
    table.add_column("Corr.", justify="right", style="yellow")
    table.add_column("Intent", width=14)
    table.add_column("Topic", max_width=20)

    for cp in pipeline.correlated[:20]:
        impact_color = (
            "green"
            if cp.impact_score >= 0.7
            else "yellow"
            if cp.impact_score >= 0.4
            else "red"
        )
        table.add_row(
            str(cp.priority_rank),
            cp.prompt.prompt_text[:50],
            f"[{impact_color}]{cp.impact_score:.2f}[/{impact_color}]",
            f"{cp.data_correlation_score:.2f}",
            cp.prompt.search_intent,
            (cp.prompt.topic_cluster or "")[:20],
        )

    console.print(table)

    high = sum(1 for cp in pipeline.correlated if cp.impact_score >= 0.7)
    medium = sum(
        1 for cp in pipeline.correlated if 0.4 <= cp.impact_score < 0.7
    )
    low = sum(1 for cp in pipeline.correlated if cp.impact_score < 0.4)

    console.print(
        Panel(
            f"[bold]Total Prompts:[/bold] {len(pipeline.correlated)}\n"
            f"[green]High Impact:[/green] {high}  "
            f"[yellow]Medium Impact:[/yellow] {medium}  "
            f"[red]Low Impact:[/red] {low}\n\n"
            f"[bold]Reports Generated:[/bold]\n"
            + "\n".join(f"  • {k}: {v}" for k, v in reports.items())
            + f"\n\n[bold]LLM Usage:[/bold] {pipeline.llm.get_usage_stats()}",
            title="[bold green]Analysis Complete[/bold green]",
            border_style="green",
        )
    )


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
