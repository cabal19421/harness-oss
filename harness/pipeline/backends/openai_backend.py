"""OpenAI (ChatGPT) coding backend — API-direct, stdlib HTTP, no SDK.

Auth: ``OPENAI_API_KEY`` (and optionally ``OPENAI_BASE_URL`` for Azure / a
compatible gateway). Model: ``PipelineConfig.openai_model`` (default
``gpt-4.1``; pin with ``HARNESS_OPENAI_MODEL`` or ``--openai-model``). Same
ralph loop + grounding/test gate as every other backend — see
:mod:`harness.pipeline.backends.api_base`.
"""

from __future__ import annotations

import os

from harness.log import get_logger

from .api_base import ApiBackend, http_post_json
from .base import ImplementContext

logger = get_logger(__name__)


class OpenAIBackend(ApiBackend):
    name = "openai"

    def available(self) -> tuple[bool, str]:
        if not os.environ.get("OPENAI_API_KEY"):
            logger.debug("openai backend unavailable: OPENAI_API_KEY is not set")
            return False, "set OPENAI_API_KEY to use the openai backend"
        return True, ""

    def _resolve_model(self, config) -> str:
        return (getattr(config, "openai_model", None)
                or os.environ.get("HARNESS_OPENAI_MODEL") or "gpt-4.1")

    def _complete(self, ctx: ImplementContext, prompt: str) -> str:
        key = os.environ["OPENAI_API_KEY"]
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        model = self._resolve_model(ctx.config)
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        # host/path, payload size and latency are logged by http_post_json;
        # never the Authorization header or the prompt/response bodies.
        logger.debug("openai chat.completions call: model=%s prompt=%d chars",
                     model, len(prompt))
        resp = http_post_json(f"{base}/chat/completions",
                              {"Authorization": f"Bearer {key}"}, body, timeout=300)
        usage = resp.get("usage") or {}
        self.last_tokens = int(usage.get("total_tokens") or 0)
        if not usage:
            logger.debug("openai response carried no usage object — this call "
                         "counts 0 tokens toward the loop budget")
        content = resp["choices"][0]["message"]["content"]
        if content is None:
            # Refusal / content-filter results ship content: null on HTTP 200 —
            # surface it as an empty response (retried with format feedback)
            # instead of a TypeError that would kill the whole task.
            logger.warning("openai returned content: null on HTTP 200 "
                           "(refusal / content filter, finish_reason=%s) — "
                           "treating as an empty response so the loop "
                           "re-prompts instead of dying",
                           resp["choices"][0].get("finish_reason") or "?")
            return ""
        logger.debug("openai response: %d chars, finish_reason=%s, "
                     "total_tokens=%d",
                     len(content) if isinstance(content, str) else -1,
                     resp["choices"][0].get("finish_reason") or "?",
                     self.last_tokens)
        return content
