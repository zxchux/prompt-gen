"""OpenRouter chat requests with retries, rate limiting, and JSON response parsing.

Supports Anthropic-style prompt caching via ``cache_control`` breakpoints.
When enabled, system messages and the first user turn in multi-turn
conversations are annotated so that the provider can reuse cached
prefixes, significantly reducing cost on repeated calls with shared
system prompts.
"""

import copy
import json
import logging
import time
from typing import Any, Optional

import openai

from prompt_gen.config import OpenRouterConfig

logger = logging.getLogger(__name__)

# Minimum content length (chars) worth caching.  Anthropic requires at
# least 1024 tokens for a cache hit; we use a rough character proxy.
_MIN_CACHEABLE_CHARS = 2048


class OpenRouterClient:
    """Client for interacting with LLMs via OpenRouter API."""

    def __init__(self, config: OpenRouterConfig):
        self.config = config
        self.client = openai.OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.request_timeout,
        )
        self.total_tokens_used = 0
        self.total_requests = 0
        self.cache_read_tokens = 0
        self.cache_write_tokens = 0
        self._last_request_time = 0.0
        self._min_request_interval = 0.5  # seconds between requests

    # ── Prompt caching helpers ───────────────────────────────────────

    @staticmethod
    def _add_cache_control(message: dict) -> dict:
        """Add ``cache_control`` to a message's content.

        Converts plain-string ``content`` to the multi-part list format
        required by the Anthropic cache_control extension, then appends
        ``{"type": "ephemeral"}`` to the last content block.

        Already-annotated messages are returned unchanged.
        """
        content = message.get("content")
        if content is None:
            return message

        msg = copy.copy(message)

        # If content is a plain string, convert to list-of-blocks format.
        if isinstance(content, str):
            msg["content"] = [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        elif isinstance(content, list):
            # Annotate the *last* block if not already annotated.
            blocks = [copy.copy(b) for b in content]
            if blocks and "cache_control" not in blocks[-1]:
                blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral"}}
            msg["content"] = blocks

        return msg

    def _apply_prompt_caching(self, messages: list[dict]) -> list[dict]:
        """Return a copy of *messages* with cache_control breakpoints.

        Strategy (follows Anthropic best practices):
        1. Mark every **system** message for caching – these are large,
           static instruction blocks reused across hundreds of calls.
        2. In multi-turn conversations (≥4 messages), also mark the
           **first user turn** so the shared prefix is cached for
           subsequent turns in the same conversation.
        3. Only annotate content blocks that are long enough to benefit
           from caching (≥ ``_MIN_CACHEABLE_CHARS``).
        """
        if not messages:
            return messages

        result: list[dict] = []
        first_user_marked = False
        is_multi_turn = sum(1 for m in messages if m.get("role") == "user") > 1

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            content_len = len(content) if isinstance(content, str) else sum(
                len(b.get("text", "")) for b in content if isinstance(b, dict)
            )

            if role == "system" and content_len >= _MIN_CACHEABLE_CHARS:
                result.append(self._add_cache_control(msg))
            elif (
                role == "user"
                and is_multi_turn
                and not first_user_marked
                and content_len >= _MIN_CACHEABLE_CHARS
            ):
                result.append(self._add_cache_control(msg))
                first_user_marked = True
            else:
                result.append(msg)

        return result

    # ── Rate limiting ────────────────────────────────────────────────

    def _rate_limit(self) -> None:
        """Simple rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def chat(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
        retry_count: int = 3,
        seed: Optional[int] = None,
    ) -> str:
        """Send a chat request and return the response text.

        Model, temperature, and token limits default to the client configuration.
        retry_count is the total number of attempts. Raise RuntimeError if all fail.
        When seed is provided, it encourages deterministic output (model-dependent).

        When ``config.enable_prompt_cache`` is True, system messages and
        the first user turn in multi-turn conversations are annotated with
        ``cache_control`` breakpoints for Anthropic prompt caching via
        OpenRouter, reducing cost by up to 90% on cache hits.
        """
        self._rate_limit()

        # Apply prompt caching annotations when enabled.
        if self.config.enable_prompt_cache:
            messages = self._apply_prompt_caching(messages)

        kwargs: dict[str, Any] = {
            "model": model or self.config.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            "extra_headers": {
                "HTTP-Referer": "https://github.com/promptgen",
                "X-Title": "PromptGen",
            },
        }

        if response_format:
            kwargs["response_format"] = response_format

        if seed is not None:
            kwargs["seed"] = seed

        last_error = None
        for attempt in range(retry_count):
            try:
                response = self.client.chat.completions.create(**kwargs)
                self.total_requests += 1

                cache_read = 0
                cache_write = 0

                if response.usage:
                    self.total_tokens_used += response.usage.total_tokens

                    # Track prompt cache statistics from Anthropic via
                    # OpenRouter.  The fields live under usage as extra
                    # attributes returned by the provider.
                    usage_dict = (
                        response.usage.model_extra
                        if hasattr(response.usage, "model_extra")
                        else {}
                    ) or {}
                    cache_read = int(usage_dict.get("cache_read_input_tokens", 0))
                    cache_write = int(usage_dict.get("cache_creation_input_tokens", 0))
                    self.cache_read_tokens += cache_read
                    self.cache_write_tokens += cache_write

                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("Empty response from LLM")

                logger.debug(
                    "LLM response received",
                    extra={
                        "model": kwargs["model"],
                        "tokens": response.usage.total_tokens if response.usage else 0,
                        "cache_read": cache_read,
                        "cache_write": cache_write,
                        "attempt": attempt + 1,
                    },
                )
                return content

            except openai.RateLimitError as e:
                last_error = e
                wait_time = min(2 ** (attempt + 1), 60)
                logger.warning(
                    f"Rate limited, waiting {wait_time}s (attempt {attempt + 1}/{retry_count})"
                )
                time.sleep(wait_time)

            except openai.APIError as e:
                last_error = e
                if attempt < retry_count - 1:
                    wait_time = 2 ** attempt
                    logger.warning(
                        f"API error: {e}, retrying in {wait_time}s "
                        f"(attempt {attempt + 1}/{retry_count})"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"API error after {retry_count} attempts: {e}")

            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error in LLM call: {e}")
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)

        raise RuntimeError(
            f"Failed to get LLM response after {retry_count} attempts: {last_error}"
        )

    def chat_json(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        retry_count: int = 3,
    ) -> dict[str, Any]:
        """Request JSON, then try embedded objects or arrays if direct parsing fails."""
        response_text = self.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            retry_count=retry_count,
        )

        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(response_text[start:end])
                except json.JSONDecodeError:
                    pass

            # Try array format
            start = response_text.find("[")
            end = response_text.rfind("]") + 1
            if start >= 0 and end > start:
                try:
                    return {"items": json.loads(response_text[start:end])}
                except json.JSONDecodeError:
                    pass

            logger.error(f"Failed to parse JSON response: {response_text[:200]}")
            raise ValueError(f"Could not parse LLM response as JSON: {response_text[:200]}")

    def analyze(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send a system instruction and one user message; return the response text."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def analyze_json(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        """Send a system instruction and one user message; parse the response as JSON."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.chat_json(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def multi_turn(
        self,
        system_prompt: str,
        turns: list[str],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> list[str]:
        """Send user messages in sequence, preserving history, and return the replies."""
        messages = [{"role": "system", "content": system_prompt}]
        responses = []

        for turn in turns:
            messages.append({"role": "user", "content": turn})
            response = self.chat(
                messages=messages,
                model=model,
                temperature=temperature,
            )
            messages.append({"role": "assistant", "content": response})
            responses.append(response)

        return responses

    def get_usage_stats(self) -> dict[str, Any]:
        """Return usage statistics including prompt cache metrics."""
        stats: dict[str, Any] = {
            "total_requests": self.total_requests,
            "total_tokens_used": self.total_tokens_used,
            "prompt_cache_enabled": self.config.enable_prompt_cache,
        }
        if self.config.enable_prompt_cache:
            stats["cache_read_tokens"] = self.cache_read_tokens
            stats["cache_write_tokens"] = self.cache_write_tokens
            # Anthropic charges 10% for cache reads vs full price
            if self.total_tokens_used > 0:
                stats["estimated_cache_savings_pct"] = round(
                    (self.cache_read_tokens * 0.9)
                    / max(self.total_tokens_used, 1)
                    * 100,
                    1,
                )
        return stats
