"""Provider-agnostic LLM client.

Any OpenAI-compatible /chat/completions endpoint works (a free-tier provider or the local stub), so the
provider is changed by configuration only. Every call has a timeout, bounded retries with exponential
backoff + jitter, and goes through a circuit breaker. Token usage and cost are recorded per call.
"""
import json
import random
import re
import threading
import time
from dataclasses import dataclass

import httpx

from . import metrics
from .config import settings


class LLMUnavailable(Exception):
    """Provider unusable after retries, or breaker open. Callers must degrade, not guess."""


class BreakerOpen(LLMUnavailable):
    pass


class CircuitBreaker:
    """Closed -> (N consecutive failures) -> open -> (reset_s) -> half-open -> (success) closed."""

    def __init__(self):
        self.failures = 0
        self.opened_at: float | None = None
        self.lock = threading.Lock()

    def state(self) -> str:
        with self.lock:
            if self.opened_at is None:
                return "closed"
            if time.time() - self.opened_at >= settings.breaker_reset_s:
                return "half-open"
            return "open"

    def before_call(self):
        if self.state() == "open":
            raise BreakerOpen("circuit breaker open: provider recently failing")

    def success(self):
        with self.lock:
            self.failures, self.opened_at = 0, None
        metrics.BREAKER_STATE.set(0)

    def failure(self):
        with self.lock:
            self.failures += 1
            if self.failures >= settings.breaker_fail_threshold or self.opened_at is not None:
                self.opened_at = time.time()
                metrics.BREAKER_STATE.set(1)


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_s: float
    retries: int


BREAKER = CircuitBreaker()
_RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504}


def _retry_after(r) -> float:
    """Seconds the provider asked us to wait. Understands the Retry-After header and the message formats of
    Groq ("try again in 9.12s"), Gemini ("Please retry in 21.6s", "retryDelay": "21s") and OpenAI-style APIs."""
    try:
        return float(r.headers.get("retry-after")) + 0.5
    except (TypeError, ValueError):
        pass
    text = r.text or ""
    m = re.search(r"try again in (?:(\d+)m)?\s*([\d.]+)s", text)
    if m:
        return int(m.group(1) or 0) * 60 + float(m.group(2)) + 0.5
    m = re.search(r"retry in ([\d.]+)s", text, re.I) or re.search(r'"retryDelay"\s*:\s*"([\d.]+)s"', text)
    return float(m.group(1)) + 0.5 if m else 5.0


_pace_lock, _last_call = threading.Lock(), [0.0]


def _pace():
    """Keep calls at least LLM_MIN_INTERVAL_S apart so a free-tier requests-per-minute cap is never hit."""
    gap = settings.llm_min_interval_s
    if gap <= 0:
        return
    with _pace_lock:
        wait = _last_call[0] + gap - time.time()
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()


def chat(messages: list[dict], agent: str, step: str) -> LLMResult:
    BREAKER.before_call()
    url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {settings.llm_api_key}"}
    body = {"model": settings.llm_model, "messages": messages, "temperature": 0}
    if settings.llm_json_mode:
        body["response_format"] = {"type": "json_object"}  # providers that support it then always return a JSON object
    if settings.llm_reasoning_effort:
        body["reasoning_effort"] = settings.llm_reasoning_effort  # "low" keeps reasoning models from thinking at length
    retries, last_err, rate_waits = 0, None, 0
    t_start = time.time()
    for attempt in range(settings.llm_max_retries + 1):
        try:
            _pace()
            r = httpx.post(url, json=body, headers=headers, timeout=settings.llm_timeout_s)
            while r.status_code == 429 and settings.llm_rate_limit_wait_s > 0 and rate_waits < 40:
                # the provider says "slow down", not "broken": wait as told and ask again. This neither uses up the
                # retry attempts nor trips the breaker.
                rate_waits += 1
                metrics.LLM_RETRIES.labels(agent).inc()
                time.sleep(min(_retry_after(r), settings.llm_rate_limit_wait_s))
                _pace()
                r = httpx.post(url, json=body, headers=headers, timeout=settings.llm_timeout_s)
            if r.status_code in _RETRYABLE:
                raise httpx.HTTPStatusError(f"status {r.status_code} from {settings.llm_base_url}: {r.text[:300]}", request=r.request, response=r)
            if r.status_code >= 400:
                # not in the retry list (bad request, bad key, unknown model): retrying cannot help, so fail at once with the reason
                raise LLMUnavailable(f"provider rejected the request: status {r.status_code} from {settings.llm_base_url}: {r.text[:400]}")
            data = r.json()
            msg = data["choices"][0]["message"]
            text = msg.get("content") or ""
            if not text.strip():
                # some reasoning models leave content empty when the answer ended up elsewhere; say so instead of guessing
                text = msg.get("reasoning_content") or ""
            u = data.get("usage", {}) or {}
            pt, ct = int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0))
            cost = pt / 1000 * settings.price_prompt_per_1k + ct / 1000 * settings.price_completion_per_1k
            BREAKER.success()
            metrics.LLM_CALLS.labels(agent, "ok").inc()
            metrics.LLM_TOKENS.labels(agent, "prompt").inc(pt)
            metrics.LLM_TOKENS.labels(agent, "completion").inc(ct)
            metrics.LLM_COST.labels(agent, step).inc(cost)
            return LLMResult(text, pt, ct, cost, time.time() - t_start, retries)
        except LLMUnavailable:
            metrics.LLM_CALLS.labels(agent, "failed").inc()
            raise
        except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError, KeyError, ValueError) as e:
            last_err = e
            BREAKER.failure()
            if BREAKER.state() == "open":
                break  # stop hammering a provider the breaker has just declared down
            if attempt < settings.llm_max_retries:
                retries += 1
                metrics.LLM_RETRIES.labels(agent).inc()
                time.sleep(settings.llm_backoff_base_s * (2 ** attempt) * (0.5 + random.random() / 2))
    metrics.LLM_CALLS.labels(agent, "failed").inc()
    raise LLMUnavailable(f"LLM call failed after {retries} retries: {type(last_err).__name__}: {last_err}")


def parse_json(text: str) -> dict:
    """Models wrap JSON in prose or fences; take the first balanced object."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in model output")
    depth = 0
    for i in range(start, len(text)):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            return json.loads(text[start:i + 1])
    raise ValueError("unbalanced JSON in model output")
