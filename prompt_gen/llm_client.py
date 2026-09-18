"""OpenRouter chat requests with retries, rate limiting, and JSON response parsing."""

import json
import logging
import time
from typing import Any, Optional

import openai

from prompt_gen.config import OpenRouterConfig

logger = logging.getLogger(__name__)


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
        self._last_request_time = 0.0
        self._min_request_interval = 0.5  # seconds between requests

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
        """
        self._rate_limit()

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

                if response.usage:
                    self.total_tokens_used += response.usage.total_tokens

                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("Empty response from LLM")

                logger.debug(
                    "LLM response received",
                    extra={
                        "model": kwargs["model"],
                        "tokens": response.usage.total_tokens if response.usage else 0,
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
        """Return usage statistics."""
        return {
            "total_requests": self.total_requests,
            "total_tokens_used": self.total_tokens_used,
        }
