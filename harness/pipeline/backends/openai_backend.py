"""OpenAI (ChatGPT) coding backend — API-direct, stdlib HTTP, no SDK.

Auth: ``OPENAI_API_KEY`` (and optionally ``OPENAI_BASE_URL`` for Azure / a
compatible gateway). Model: ``PipelineConfig.openai_model`` (default
``gpt-5.6-terra``; pin with ``HARNESS_OPENAI_MODEL`` or ``openai_model=``). Same
ralph loop + grounding/test gate as every other backend — see
:mod:`harness.pipeline.backends.api_base`.

The default moved off ``gpt-4.1`` because that id is already unusable on the
Azure path this module advertises: its lifecycle stage there is *Deprecated*
(new deployments refused, retirement 2027-04-14). The request body stays
``{model, messages}`` plus a handful of additive fields, so a model swap needs no
rework — see ``_complete``.
"""

from __future__ import annotations

import os
from typing import Any

from harness.log import get_logger, redact, trunc

from .api_base import ApiBackend, ResponseIssue, http_post_json
from .base import ImplementContext

logger = get_logger(__name__)

#: Truncation is the dangerous one: the reply ends mid-fence, so the extractor
#: falls back to "the whole text" and prose-plus-half-a-function lands on disk as
#: the task's code. Feedback is explicit about what to do differently.
_TRUNCATED_FEEDBACK = (
    "Your previous response was TRUNCATED — it hit the output-token limit and "
    "ended mid-file, so it was discarded rather than written. Return the complete "
    "file and nothing else: no explanation, no commentary, no repeated code. If "
    "the file genuinely cannot fit in one response, reply with a single line "
    "`ABSTAIN: <reason>` instead of a partial file."
)


class OpenAIBackend(ApiBackend):
    name = "openai"

    def available(self) -> tuple[bool, str]:
        if not os.environ.get("OPENAI_API_KEY"):
            logger.debug("openai backend unavailable: OPENAI_API_KEY is not set")
            return False, "set OPENAI_API_KEY to use the openai backend"
        return True, ""

    def _resolve_model(self, config) -> str:
        return (getattr(config, "openai_model", None)
                or os.environ.get("HARNESS_OPENAI_MODEL") or "gpt-5.6-terra")

    def _complete(self, ctx: ImplementContext, prompt: str) -> str:
        key = os.environ["OPENAI_API_KEY"]
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        model = self._resolve_model(ctx.config)
        cfg = ctx.config
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            # Everything before the feedback section is byte-identical across
            # every iteration of a task (target file, title, description, wrapped
            # design doc, definition of done, frozen acceptance tests), which is
            # an ideal cache prefix — but with parallel workers hitting different
            # tasks the buckets interleave and hits become incidental. The task id
            # is the natural bucket. `prompt_cache_key`, NOT the deprecated `user`.
            "prompt_cache_key": f"harness:{ctx.task.id}",
        }
        cap = getattr(cfg, "openai_max_output_tokens", None)
        if cap:
            # `max_completion_tokens`, NEVER `max_tokens`: the latter is
            # deprecated and not accepted by the reasoning models, so setting it
            # would silently fail on exactly the models worth using. Paired with
            # the finish_reason=="length" branch below, so hitting the cap is a
            # diagnosable event rather than a corrupt file.
            body["max_completion_tokens"] = int(cap)
        tier = getattr(cfg, "openai_service_tier", None)
        if tier:
            # `flex` prices at Batch rates and is the textbook fit for an
            # unattended, budget-capped overnight loop where latency is
            # irrelevant. Its "resource unavailable" 429 is uncharged and already
            # handled by the transient path — but it needs a long timeout, hence
            # api_timeout_seconds defaulting to 600.
            body["service_tier"] = str(tier)
        timeout = int(getattr(cfg, "api_timeout_seconds", 600) or 600)
        # host/path, payload size and latency are logged by http_post_json;
        # never the Authorization header or the prompt/response bodies.
        logger.debug("openai chat.completions call: model=%s prompt=%d chars "
                     "cache_key=%s tier=%s max_completion_tokens=%s timeout=%ds",
                     model, len(prompt), body["prompt_cache_key"], tier or "default",
                     body.get("max_completion_tokens", "unset"), timeout)
        resp = http_post_json(f"{base}/chat/completions",
                              {"Authorization": f"Bearer {key}"}, body,
                              timeout=timeout)
        usage = resp.get("usage") or {}
        if not usage:
            logger.debug("openai response carried no usage object — this call "
                         "counts 0 tokens / $0 toward the loop budget")
        prompt_details = usage.get("prompt_tokens_details") or {}
        cached = int(prompt_details.get("cached_tokens") or 0)
        self._account_usage(
            model,
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            cached_input_tokens=cached,
            total_tokens=int(usage.get("total_tokens") or 0))

        choice = resp["choices"][0]
        finish = choice.get("finish_reason") or ""
        message = choice.get("message") or {}
        refusal = message.get("refusal")
        content = message.get("content")

        if refusal:
            # A first-class nullable sibling of `content` holding the actual
            # reason. Discarding it meant re-prompting a safety refusal with
            # "Your response contained no code" — useless feedback — and then
            # aborting having thrown away the one diagnostic the provider gave.
            text = str(refusal)
            logger.warning("openai REFUSED the request (finish_reason=%s): %s",
                           finish or "?", trunc(redact(text), 300))
            self.last_issue = ResponseIssue(
                kind="refusal", detail=text,
                feedback=("The model refused your previous request:\n"
                          f"{text[:600]}\n\n"
                          "Rewrite the file so it does not trip that refusal. If the "
                          "task genuinely cannot be done within those limits, reply "
                          "with a single line `ABSTAIN: <reason>`."))
            return ""
        if finish == "length":
            logger.warning("openai response hit the output-token limit "
                           "(finish_reason=length, %d chars) — DISCARDING it: a "
                           "reply cut off mid-file must never be written as code",
                           len(content) if isinstance(content, str) else 0)
            self.last_issue = ResponseIssue(
                kind="truncated",
                detail=("response truncated at the output-token limit "
                        f"(completion_tokens={usage.get('completion_tokens') or '?'})"),
                feedback=_TRUNCATED_FEEDBACK)
            return ""
        if finish == "content_filter":
            logger.warning("openai stopped on a content filter "
                           "(finish_reason=content_filter) — treating the partial "
                           "response as unusable rather than writing it")
            self.last_issue = ResponseIssue(
                kind="content_filter",
                detail="response stopped by the provider's content filter",
                feedback=("Your previous response was stopped by the provider's "
                          "content filter and was discarded. Produce the file "
                          "without whatever tripped it, or reply with a single "
                          "line `ABSTAIN: <reason>`."))
            return ""
        if content is None:
            # Refusal / content-filter results ship content: null on HTTP 200 —
            # surface it as an empty response (retried with format feedback)
            # instead of a TypeError that would kill the whole task. Reached only
            # when the provider gave no `refusal` and no explanatory
            # finish_reason, i.e. there is genuinely nothing to report.
            logger.warning("openai returned content: null on HTTP 200 with no "
                           "refusal text (finish_reason=%s) — treating as an "
                           "empty response so the loop re-prompts instead of dying",
                           finish or "?")
            return ""
        logger.debug("openai response: %d chars, finish_reason=%s, "
                     "total_tokens=%d (cached input %d), ~$%.4f",
                     len(content) if isinstance(content, str) else -1,
                     finish or "?", self.last_tokens, cached, self.last_cost_usd)
        return content
