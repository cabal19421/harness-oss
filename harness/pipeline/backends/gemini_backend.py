"""Gemini (Google) coding backend — API-direct, stdlib HTTP, no SDK.

Auth: ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``), sent via the
``x-goog-api-key`` header — never in the URL, where proxies and any future
URL logging would capture the credential. Model: ``PipelineConfig.gemini_model``
(default ``gemini-3.6-flash``; pin with ``HARNESS_GEMINI_MODEL``). Same ralph
loop + grounding/test gate as every other backend — see
:mod:`harness.pipeline.backends.api_base`.

The default moved off ``gemini-2.5-flash`` ahead of its documented 2026-10-16
shutdown — a dated deadline, not a preference. Deliberately a PINNED id and not
a floating alias like ``gemini-flash-latest``: a silently hot-swapping model
changes oracle outcomes with no diff, which is the opposite of what a
gate-verified run is for. ``HARNESS_GEMINI_MODEL`` is there for anyone who wants
to float.

API-KEY NOTE: Google now rejects *Standard* keys on this endpoint (unrestricted
keys since 2026-06-19, the rest from ~Sept 2026). A run that suddenly 401s
usually needs the key recreated as an **auth key** in AI Studio — the transport
here (``x-goog-api-key``) is correct for both kinds and needs no change.
"""

from __future__ import annotations

import os

from harness.log import get_logger, redact, trunc

from .api_base import ApiBackend, ResponseIssue, http_post_json
from .base import ImplementContext

logger = get_logger(__name__)

#: Best-effort, not a verified contract: Google's FinishReason enum could not be
#: confirmed from primary docs, so these are treated defensively — an unknown
#: value is ignored and the text is used, exactly as before.
_UNUSABLE_FINISH = {
    "MAX_TOKENS": ("truncated", "the response hit the output-token limit"),
    "SAFETY": ("refusal", "the response was blocked by a safety filter"),
    "RECITATION": ("refusal", "the response was blocked as recitation"),
    "BLOCKLIST": ("refusal", "the response was blocked by a term blocklist"),
    "PROHIBITED_CONTENT": ("refusal", "the response was blocked as prohibited content"),
    "SPII": ("refusal", "the response was blocked as sensitive personal information"),
}
_TRUNCATED_FEEDBACK = (
    "Your previous response was TRUNCATED — it hit the output-token limit and "
    "ended mid-file, so it was discarded rather than written. Return the complete "
    "file and nothing else: no explanation, no commentary, no repeated code. If "
    "the file genuinely cannot fit in one response, reply with a single line "
    "`ABSTAIN: <reason>` instead of a partial file."
)


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
                or os.environ.get("HARNESS_GEMINI_MODEL") or "gemini-3.6-flash")

    def _complete(self, ctx: ImplementContext, prompt: str) -> str:
        key = os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"]
        model = self._resolve_model(ctx.config)
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{model}:generateContent")
        body = {"contents": [{"parts": [{"text": prompt}]}]}
        timeout = int(getattr(ctx.config, "api_timeout_seconds", 600) or 600)
        # host/path, payload size and latency are logged by http_post_json;
        # the key travels in a header and is never logged.
        logger.debug("gemini generateContent call: model=%s prompt=%d chars timeout=%ds",
                     model, len(prompt), timeout)
        resp = http_post_json(url, {"x-goog-api-key": key}, body, timeout=timeout)
        usage = resp.get("usageMetadata") or {}
        if not usage:
            logger.debug("gemini response carried no usageMetadata — this call "
                         "counts 0 tokens / $0 toward the loop budget")
        # totalTokenCount is documented to ALREADY include thought tokens, so it
        # is the right number for the token cap — do not "fix" it by adding
        # thoughtsTokenCount, which would double-count them. The itemised counts
        # below are only for pricing, where cached input bills differently.
        cached = int(usage.get("cachedContentTokenCount") or 0)
        self._account_usage(
            model,
            input_tokens=int(usage.get("promptTokenCount") or 0),
            output_tokens=int(usage.get("candidatesTokenCount") or 0)
            + int(usage.get("thoughtsTokenCount") or 0),
            cached_input_tokens=cached,
            total_tokens=int(usage.get("totalTokenCount") or 0))
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
        finish = str(candidates[0].get("finishReason") or "")
        logger.debug("gemini response: %d candidate(s), %d part(s), %d chars, "
                     "finishReason=%s, totalTokenCount=%d, ~$%.4f",
                     len(candidates), len(parts), len(text), finish or "?",
                     self.last_tokens, self.last_cost_usd)
        unusable = _UNUSABLE_FINISH.get(finish.upper())
        if unusable:
            kind, why = unusable
            logger.warning("gemini finishReason=%s — %s; DISCARDING the %d-char "
                           "response rather than writing a partial/blocked file",
                           finish, why, len(text))
            self.last_issue = ResponseIssue(
                kind=kind, detail=f"finishReason={finish}: {why}",
                feedback=(_TRUNCATED_FEEDBACK if kind == "truncated" else
                          ("Your previous response was blocked by the provider "
                           f"({why}) and was discarded. Produce the file without "
                           "whatever tripped it, or reply with a single line "
                           "`ABSTAIN: <reason>`.")))
            return ""
        if finish and finish.upper() not in ("STOP", "FINISH_REASON_UNSPECIFIED"):
            # Defensive: an unrecognised terminal reason is logged but still
            # trusted, so a future enum value cannot silently stall the loop.
            logger.info("gemini reported an unrecognised finishReason=%s — using "
                        "the response as-is: %s", finish, trunc(redact(text), 120))
        return text
