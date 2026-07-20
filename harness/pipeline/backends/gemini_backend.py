"""Gemini (Google) coding backend — API-direct, stdlib HTTP, no SDK.

Auth: ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``), sent via the
``x-goog-api-key`` header — never in the URL, where proxies and any future
URL logging would capture the credential. Model: ``PipelineConfig.gemini_model``
(default ``gemini-2.5-flash``; pin with ``HARNESS_GEMINI_MODEL``). Same ralph
loop + grounding/test gate as every other backend — see
:mod:`harness.pipeline.backends.api_base`.
"""

from __future__ import annotations

import os

from harness.log import get_logger

from .api_base import ApiBackend, http_post_json
from .base import ImplementContext

logger = get_logger(__name__)


class GeminiBackend(ApiBackend):
    name = "gemini"

    def available(self) -> tuple[bool, str]:
        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            logger.debug("gemini backend unavailable: neither GEMINI_API_KEY "
                         "nor GOOGLE_API_KEY is set")
            return False, "set GEMINI_API_KEY (or GOOGLE_API_KEY) to use the gemini backend"
        return True, ""

    def _resolve_model(self, config) -> str:
        return (getattr(config, "gemini_model", None)
                or os.environ.get("HARNESS_GEMINI_MODEL") or "gemini-2.5-flash")

    def _complete(self, ctx: ImplementContext, prompt: str) -> str:
        key = os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"]
        model = self._resolve_model(ctx.config)
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{model}:generateContent")
        body = {"contents": [{"parts": [{"text": prompt}]}]}
        # host/path, payload size and latency are logged by http_post_json;
        # the key travels in a header and is never logged.
        logger.debug("gemini generateContent call: model=%s prompt=%d chars",
                     model, len(prompt))
        resp = http_post_json(url, {"x-goog-api-key": key}, body, timeout=300)
        usage = resp.get("usageMetadata") or {}
        self.last_tokens = int(usage.get("totalTokenCount") or 0)
        if not usage:
            logger.debug("gemini response carried no usageMetadata — this call "
                         "counts 0 tokens toward the loop budget")
        candidates = resp.get("candidates") or []
        if not candidates:
            # Prompt-safety blocks return candidates: [] on HTTP 200 — surface
            # as an empty response (retried with feedback) instead of an
            # IndexError that would kill the whole task.
            logger.warning("gemini returned no candidates on HTTP 200 "
                           "(blockReason=%s) — treating as an empty response "
                           "so the loop re-prompts instead of dying",
                           (resp.get("promptFeedback") or {}).get("blockReason")
                           or "not reported")
            return ""
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts)
        logger.debug("gemini response: %d candidate(s), %d part(s), %d chars, "
                     "finishReason=%s, totalTokenCount=%d",
                     len(candidates), len(parts), len(text),
                     candidates[0].get("finishReason") or "?", self.last_tokens)
        return text
