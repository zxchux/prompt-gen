"""
Consistency Reconciliation Module for PromptGen.

Addresses the inherent non-determinism of LLM outputs by:

1. MULTI-PASS CONSENSUS: Run extraction N times, keep only prompts that
   appear in a majority of passes (configurable threshold).

2. CROSS-RUN COMPARISON: Compare current results against previous runs,
   detect drift, and flag new/dropped prompts.

3. HISTORICAL ANCHORING: Use previous run results as a baseline to
   stabilize outputs across runs.

4. DETERMINISTIC SETTINGS: Use low temperatures and seeds where supported.

5. SIMILARITY CLUSTERING: Group near-duplicate prompts across runs and
   pick the canonical version.

This ensures that the final prompt list is stable and reproducible,
not subject to random LLM variation between runs.
"""

import json
import logging
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional

from prompt_gen.models import ExtractedPrompt

logger = logging.getLogger(__name__)


class ConsistencyReconciler:
    """
    Ensures consistency of prompt extraction across multiple runs.

    Strategies:
    - Multi-pass consensus: Extract prompts N times, keep majority
    - Cross-run drift detection: Compare against historical baselines
    - Similarity clustering: Merge near-duplicate prompts
    """

    def __init__(
        self,
        history_dir: str = ".cache/history",
        similarity_threshold: float = 0.80,
        consensus_threshold: float = 0.5,
    ):
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.similarity_threshold = similarity_threshold
        self.consensus_threshold = consensus_threshold

    # ── Similarity helpers ──────────────────────────────────────────

    def _normalize(self, text: str) -> str:
        """Normalize prompt text for comparison."""
        return " ".join(text.lower().strip().split())

    def _similarity(self, a: str, b: str) -> float:
        """Calculate similarity between two prompt texts."""
        na, nb = self._normalize(a), self._normalize(b)
        if na == nb:
            return 1.0

        seq_score = SequenceMatcher(None, na, nb).ratio()

        words_a, words_b = set(na.split()), set(nb.split())
        if words_a and words_b:
            overlap = len(words_a & words_b)
            word_score = overlap / max(len(words_a), len(words_b))
        else:
            word_score = 0.0

        return max(seq_score, word_score)

    def _find_match(
        self, prompt_text: str, candidates: list[str]
    ) -> Optional[tuple[str, float]]:
        """Find the best matching candidate for a prompt text."""
        best_match = None
        best_score = 0.0

        for candidate in candidates:
            score = self._similarity(prompt_text, candidate)
            if score > best_score:
                best_score = score
                best_match = candidate

        if best_match and best_score >= self.similarity_threshold:
            return best_match, best_score
        return None

    # ── Multi-pass consensus ────────────────────────────────────────

    def build_consensus(
        self, passes: list[list[ExtractedPrompt]]
    ) -> list[ExtractedPrompt]:
        """
        Build consensus from multiple extraction passes.

        A prompt is kept only if it appears in at least `consensus_threshold`
        fraction of passes (default: 50%). Scores are averaged across passes.

        Args:
            passes: List of prompt lists from multiple extraction runs.

        Returns:
            Consensus list of prompts that appeared consistently.
        """
        if not passes:
            return []

        if len(passes) == 1:
            logger.info("Only 1 pass provided, skipping consensus (need 2+)")
            return passes[0]

        num_passes = len(passes)
        min_appearances = max(1, int(num_passes * self.consensus_threshold))

        logger.info(
            f"Building consensus from {num_passes} passes "
            f"(min appearances: {min_appearances}/{num_passes})"
        )

        # Track prompt appearances across passes
        # Key: normalized text, Value: {appearances, scores, best_prompt}
        prompt_tracker: dict[str, dict[str, Any]] = {}

        for pass_idx, pass_prompts in enumerate(passes):
            for prompt in pass_prompts:
                normalized = self._normalize(prompt.prompt_text)

                # Check for exact or fuzzy match in tracker
                matched_key = None
                for existing_key in prompt_tracker:
                    if self._similarity(normalized, existing_key) >= self.similarity_threshold:
                        matched_key = existing_key
                        break

                if matched_key:
                    tracker = prompt_tracker[matched_key]
                    tracker["appearances"] += 1
                    tracker["scores"].append(prompt.relevance_score)
                    tracker["pass_indices"].add(pass_idx)
                    # Keep the version with the highest score
                    if prompt.relevance_score > tracker["best_score"]:
                        tracker["best_prompt"] = prompt
                        tracker["best_score"] = prompt.relevance_score
                else:
                    prompt_tracker[normalized] = {
                        "appearances": 1,
                        "scores": [prompt.relevance_score],
                        "pass_indices": {pass_idx},
                        "best_prompt": prompt,
                        "best_score": prompt.relevance_score,
                    }

        # Filter to consensus prompts
        consensus_prompts = []
        rejected_count = 0

        for key, tracker in prompt_tracker.items():
            if tracker["appearances"] >= min_appearances:
                prompt = tracker["best_prompt"]
                # Average the score across appearances
                avg_score = sum(tracker["scores"]) / len(tracker["scores"])
                prompt.relevance_score = avg_score
                # Add consistency metadata to reasoning
                prompt.extraction_reasoning = (
                    f"[Consensus: {tracker['appearances']}/{num_passes} passes, "
                    f"avg score: {avg_score:.2f}] "
                    + (prompt.extraction_reasoning or "")
                )
                consensus_prompts.append(prompt)
            else:
                rejected_count += 1

        # Sort by score
        consensus_prompts.sort(key=lambda p: p.relevance_score, reverse=True)

        logger.info(
            f"Consensus result: {len(consensus_prompts)} prompts kept, "
            f"{rejected_count} rejected (appeared in <{min_appearances} passes)"
        )

        return consensus_prompts

    # ── History management ──────────────────────────────────────────

    def _get_history_path(self, domain: str) -> Path:
        """Get the history file path for a domain."""
        safe_domain = domain.replace(".", "_").replace("/", "_")
        return self.history_dir / f"{safe_domain}_history.json"

    def save_run(
        self, domain: str, prompts: list[ExtractedPrompt], run_metadata: dict = None
    ) -> None:
        """
        Save current run results to history for future comparison.

        Args:
            domain: The target domain.
            prompts: The final prompt list from this run.
            run_metadata: Optional metadata about the run.
        """
        history_path = self._get_history_path(domain)

        # Load existing history
        history = self._load_history(domain)

        # Add current run
        run_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "prompt_count": len(prompts),
            "prompts": [
                {
                    "text": p.prompt_text,
                    "score": p.relevance_score,
                    "intent": p.search_intent,
                    "topic": p.topic_cluster,
                }
                for p in prompts
            ],
            "metadata": run_metadata or {},
        }

        history["runs"].append(run_entry)

        # Keep only last 10 runs
        if len(history["runs"]) > 10:
            history["runs"] = history["runs"][-10:]

        # Update stable prompts (appeared in 3+ of last 5 runs)
        history["stable_prompts"] = self._compute_stable_prompts(history["runs"])

        try:
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2, default=str)
            logger.info(f"Saved run history to {history_path}")
        except Exception as e:
            logger.error(f"Failed to save history: {e}")

    def _load_history(self, domain: str) -> dict:
        """Load run history for a domain."""
        history_path = self._get_history_path(domain)
        if history_path.exists():
            try:
                with open(history_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"domain": domain, "runs": [], "stable_prompts": []}

    def _compute_stable_prompts(self, runs: list[dict]) -> list[dict]:
        """Compute prompts that are stable across recent runs."""
        recent_runs = runs[-5:]  # Last 5 runs
        if len(recent_runs) < 2:
            return []

        # Count appearances
        prompt_counts: dict[str, dict] = {}
        for run in recent_runs:
            for prompt in run.get("prompts", []):
                normalized = self._normalize(prompt["text"])

                matched_key = None
                for existing_key in prompt_counts:
                    if self._similarity(normalized, existing_key) >= self.similarity_threshold:
                        matched_key = existing_key
                        break

                if matched_key:
                    prompt_counts[matched_key]["count"] += 1
                    prompt_counts[matched_key]["scores"].append(prompt["score"])
                else:
                    prompt_counts[normalized] = {
                        "text": prompt["text"],
                        "count": 1,
                        "scores": [prompt["score"]],
                        "intent": prompt.get("intent", ""),
                        "topic": prompt.get("topic", ""),
                    }

        # Stable = appeared in 60%+ of recent runs
        min_count = max(2, int(len(recent_runs) * 0.6))
        stable = []
        for key, data in prompt_counts.items():
            if data["count"] >= min_count:
                stable.append({
                    "text": data["text"],
                    "avg_score": sum(data["scores"]) / len(data["scores"]),
                    "appearances": data["count"],
                    "total_runs": len(recent_runs),
                    "stability": data["count"] / len(recent_runs),
                })

        stable.sort(key=lambda x: x["avg_score"], reverse=True)
        return stable

    # ── Cross-run drift detection ───────────────────────────────────

    def detect_drift(
        self, domain: str, current_prompts: list[ExtractedPrompt]
    ) -> dict[str, Any]:
        """
        Compare current results against historical baseline to detect drift.

        Returns a drift report with:
        - new_prompts: Prompts in current run but not in history
        - dropped_prompts: Prompts in history but not in current run
        - stable_prompts: Prompts consistent across runs
        - drift_score: 0.0 (identical) to 1.0 (completely different)

        Args:
            domain: The target domain.
            current_prompts: Current run's prompt list.

        Returns:
            Drift analysis report.
        """
        history = self._load_history(domain)

        if not history["runs"]:
            logger.info("No historical runs found - skipping drift detection")
            return {
                "has_history": False,
                "drift_score": 0.0,
                "new_prompts": [],
                "dropped_prompts": [],
                "stable_prompts": [],
                "message": "First run - no history to compare against",
            }

        # Get the most recent previous run
        prev_run = history["runs"][-1]
        prev_prompts = {
            self._normalize(p["text"]): p for p in prev_run.get("prompts", [])
        }
        curr_prompts = {
            self._normalize(p.prompt_text): p for p in current_prompts
        }

        # Find new, dropped, and stable prompts
        new_prompts = []
        matched_prev = set()

        for curr_key, curr_prompt in curr_prompts.items():
            match = self._find_match(curr_key, list(prev_prompts.keys()))
            if match:
                matched_prev.add(match[0])
            else:
                new_prompts.append(curr_prompt.prompt_text)

        dropped_prompts = [
            prev_prompts[key]["text"]
            for key in prev_prompts
            if key not in matched_prev
        ]

        # Calculate drift score
        total_unique = len(set(list(curr_prompts.keys()) + list(prev_prompts.keys())))
        if total_unique == 0:
            drift_score = 0.0
        else:
            changes = len(new_prompts) + len(dropped_prompts)
            drift_score = min(changes / total_unique, 1.0)

        # Get stable prompts from history
        stable = history.get("stable_prompts", [])

        report = {
            "has_history": True,
            "previous_run": prev_run.get("timestamp", "unknown"),
            "previous_count": len(prev_prompts),
            "current_count": len(curr_prompts),
            "drift_score": drift_score,
            "new_prompts": new_prompts[:20],  # Limit for readability
            "new_count": len(new_prompts),
            "dropped_prompts": dropped_prompts[:20],
            "dropped_count": len(dropped_prompts),
            "stable_prompts": [s["text"] for s in stable[:20]],
            "stable_count": len(stable),
            "total_historical_runs": len(history["runs"]),
        }

        # Log drift summary
        if drift_score < 0.2:
            drift_level = "LOW"
        elif drift_score < 0.5:
            drift_level = "MODERATE"
        else:
            drift_level = "HIGH"

        logger.info(
            f"Drift detection: {drift_level} drift ({drift_score:.2f}) - "
            f"{len(new_prompts)} new, {len(dropped_prompts)} dropped, "
            f"{len(stable)} stable across {len(history['runs'])} historical runs"
        )

        return report

    # ── Anchor to history ───────────────────────────────────────────

    def anchor_to_history(
        self,
        domain: str,
        current_prompts: list[ExtractedPrompt],
        anchor_weight: float = 0.3,
    ) -> list[ExtractedPrompt]:
        """
        Anchor current results to historical stable prompts.

        Boosts scores of prompts that have been consistently identified
        across runs, and adds back historically stable prompts that
        were missed in the current run.

        Args:
            domain: The target domain.
            current_prompts: Current run's prompt list.
            anchor_weight: How much to boost historically stable prompts (0-1).

        Returns:
            Adjusted prompt list with historical anchoring applied.
        """
        history = self._load_history(domain)
        stable_prompts = history.get("stable_prompts", [])

        if not stable_prompts:
            logger.info("No stable historical prompts - skipping anchoring")
            return current_prompts

        logger.info(
            f"Anchoring to {len(stable_prompts)} historically stable prompts "
            f"(weight: {anchor_weight})"
        )

        # Build lookup of current prompts
        current_map: dict[str, ExtractedPrompt] = {}
        for p in current_prompts:
            current_map[self._normalize(p.prompt_text)] = p

        # Boost scores of stable prompts found in current run
        boosted = 0
        for stable in stable_prompts:
            stable_norm = self._normalize(stable["text"])
            match = self._find_match(stable_norm, list(current_map.keys()))
            if match:
                prompt = current_map[match[0]]
                stability = stable.get("stability", 0.5)
                boost = anchor_weight * stability
                original_score = prompt.relevance_score
                prompt.relevance_score = min(1.0, original_score + boost)
                prompt.extraction_reasoning = (
                    f"[Historically stable: {stable.get('appearances', '?')}/"
                    f"{stable.get('total_runs', '?')} runs, "
                    f"score boosted {original_score:.2f}→{prompt.relevance_score:.2f}] "
                    + (prompt.extraction_reasoning or "")
                )
                boosted += 1

        # Add back historically stable prompts that were missed
        added_back = 0
        for stable in stable_prompts:
            stable_norm = self._normalize(stable["text"])
            match = self._find_match(stable_norm, list(current_map.keys()))
            if not match and stable.get("stability", 0) >= 0.6:
                # This prompt was stable but missing from current run - add it back
                recovered = ExtractedPrompt(
                    prompt_text=stable["text"],
                    relevance_score=stable["avg_score"] * 0.9,  # Slight penalty
                    search_intent=stable.get("intent", "informational"),
                    topic_cluster=stable.get("topic", ""),
                    extraction_reasoning=(
                        f"[Recovered from history: stable in "
                        f"{stable.get('appearances', '?')}/{stable.get('total_runs', '?')} "
                        f"runs but missing from current extraction]"
                    ),
                )
                current_prompts.append(recovered)
                current_map[stable_norm] = recovered
                added_back += 1

        logger.info(
            f"Historical anchoring: {boosted} prompts boosted, "
            f"{added_back} stable prompts recovered from history"
        )

        return current_prompts
