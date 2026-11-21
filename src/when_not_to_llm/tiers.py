"""Execution tiers: Deterministic Python, Ollama Local, OpenAI-Compat (MiniMax), Gemini REST, Hybrid, and Mock."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Callable

from .datasets import BenchmarkCase
from .finops import compute_call_cost_usd
from .verifiers import extract_json_object


@dataclass
class TierResult:
    raw_output: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    thinking_tokens: int = 0
    load_duration_ms: float = 0.0
    cost_usd: float = 0.0
    api_usd: float = 0.0
    power_usd: float = 0.0
    routed_to: str = ""
    error: str | None = None


# ---------------------------------------------------------------------------
# Tier -1: Deterministic Python Solvers (honest, non-strawman implementations)
# ---------------------------------------------------------------------------

_LOG_ERR_RE = re.compile(
    r"\b(?:ERROR|CRITICAL)\s+\[([^\]]+)\]\s+code=(ERR_[A-Z0-9_]+)\b"
)
_INCIDENT_RE = re.compile(
    r"INCIDENT_ALERT\s+id=(INC-\d+)\s+owner=([\w-]+)\s+impact_usd=(\d+)"
)


def reconcile_tables_exact(
    platform_rows: list[dict[str, Any]],
    gl_rows: list[dict[str, Any]],
    vendor_map: dict[str, str],
) -> dict[str, Any]:
    """Cent-exact deterministic reconciliation given a platform_code -> gl_vendor mapping."""
    inv_map = {v.strip().lower(): k for k, v in vendor_map.items() if v}

    plat_by_key: dict[tuple[str, str], float] = defaultdict(float)
    total_plat = 0.0
    for r in platform_rows:
        amt = round(float(r["spend_usd"]), 2)
        total_plat += amt
        plat_by_key[(r["platform_code"], r["period"])] += amt

    gl_entries_by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    total_gl = 0.0
    for r in gl_rows:
        amt = round(float(r["booked_usd"]), 2)
        total_gl += amt
        pcode = inv_map.get(r["gl_vendor"].strip().lower())
        if pcode:
            gl_entries_by_key[(pcode, r["period"])].append(amt)

    anomalies: list[dict[str, Any]] = []
    for key, p_amt in sorted(plat_by_key.items()):
        pcode, period = key
        g_list = gl_entries_by_key.get(key, [])
        g_sum = round(sum(g_list), 2)
        p_amt = round(p_amt, 2)
        if len(g_list) == 0:
            anomalies.append(
                {
                    "platform_code": pcode,
                    "period": period,
                    "type": "missing_gl",
                    "variance_usd": round(p_amt, 2),
                }
            )
        elif len(g_list) > 1 and abs(g_list[0] - p_amt) <= 0.02:
            anomalies.append(
                {
                    "platform_code": pcode,
                    "period": period,
                    "type": "duplicate_gl",
                    "variance_usd": round(p_amt - g_sum, 2),
                }
            )
        elif abs(p_amt - g_sum) > 0.02:
            anomalies.append(
                {
                    "platform_code": pcode,
                    "period": period,
                    "type": "variance",
                    "variance_usd": round(p_amt - g_sum, 2),
                }
            )

    return {
        "net_variance_usd": round(total_plat - total_gl, 2),
        "anomalies": anomalies,
    }


def _naive_token_vendor_map(
    platform_codes: list[str], gl_vendors: list[str]
) -> dict[str, str]:
    """Best-effort deterministic token-overlap matcher without a curated synonym table."""
    def tokens(s: str) -> set[str]:
        return {t for t in re.split(r"[^a-z0-9]+", s.lower()) if len(t) >= 3}

    mapping: dict[str, str] = {}
    used_gl: set[str] = set()
    for p in platform_codes:
        pt = tokens(p)
        best_gl = ""
        best_score = 0.0
        for g in gl_vendors:
            if g in used_gl:
                continue
            gt = tokens(g)
            # Check exact token overlap or 4-char prefix match
            hits = sum(
                1 for a in pt if any(a == b or (len(a) >= 4 and b.startswith(a)) for b in gt)
            )
            score = hits / max(1, len(pt))
            if score > best_score:
                best_score = score
                best_gl = g
        if best_score >= 0.34 and best_gl:
            mapping[p] = best_gl
            used_gl.add(best_gl)
    return mapping


def _deterministic_note_extract(note: str) -> dict[str, Any]:
    # Honest regex heuristic: captures explicit "$X", simple name patterns, and keywords.
    amt = 0.0
    m_k = re.search(r"\$\s*([\d,]+(?:\.\d+)?)\s*k\b", note, re.IGNORECASE)
    m_d = re.search(r"\$\s*([\d,]+(?:\.\d+)?)", note)
    if m_k:
        amt = float(m_k.group(1).replace(",", "")) * 1000.0
    elif m_d:
        amt = float(m_d.group(1).replace(",", ""))

    low = note.lower()
    if any(w in low for w in ("renew", "extend", "roll over")):
        cat = "renewal"
    elif any(w in low for w in ("invoice", "discrepancy", "overcharge", "double-charged", "late fee")):
        cat = "invoice_dispute"
    elif any(w in low for w in ("cancel", "shutting down", "dropping", "churn")):
        cat = "cancellation"
    elif any(w in low for w in ("onboard", "kickoff", "implementation", "provisioning")):
        cat = "onboarding"
    elif any(w in low for w in ("upgrade", "add-on", "extra seats", "expanding")):
        cat = "upsell"
    else:
        cat = "unknown"

    followup = True
    if any(w in low for w in ("nothing left", "no outreach", "all set", "just logging", "zero open", "closed", "no further action")):
        followup = False

    # Naive Proper-Noun bigram extraction
    names = re.findall(r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b", note)
    contact = names[0] if len(names) >= 1 else ""
    company = names[1] if len(names) >= 2 else ""
    return {
        "contact": contact,
        "company": company,
        "amount_usd": round(amt, 2),
        "category": cat,
        "requires_followup": followup,
    }


def _deterministic_doc_contradictions(memos_text: str) -> dict[str, Any]:
    """Sentence-level keyword + numeric-divergence heuristic across memos."""
    sections = re.split(r"=== MEMO (M\d+)[^=]*===", memos_text)
    memos: dict[str, str] = {}
    for i in range(1, len(sections) - 1, 2):
        memos[sections[i].strip()] = sections[i + 1]

    topic_keywords = {
        "budget_scope": ("budget", "agency retainer", "media allocation"),
        "launch_date": ("launch", "public release", "go-live"),
        "compliance_status": ("soc 2", "attestation", "auditor"),
        "sla_target": ("uptime", "service credits", "sla"),
        "headcount_cap": ("headcount", "ftes", "requisitions"),
        "pricing_tier": ("self-serve", "pricing", "subscribers"),
    }
    found = []
    m_ids = sorted(memos.keys())
    for topic, kws in topic_keywords.items():
        hits: list[tuple[str, str]] = []
        for m in m_ids:
            for para in memos[m].split("\n\n"):
                low = para.lower()
                if any(k in low for k in kws):
                    hits.append((m, para))
                    break
        if len(hits) >= 2:
            (m_a, p_a), (m_b, p_b) = hits[0], hits[1]
            nums_a = set(re.findall(r"\$?\d+(?:[.,]\d+)?%?", p_a))
            nums_b = set(re.findall(r"\$?\d+(?:[.,]\d+)?%?", p_b))
            if nums_a != nums_b:
                found.append({"topic": topic, "memos": sorted([m_a, m_b]), "summary": "numeric diff"})
    return {"contradictions": found}


def run_deterministic(case: BenchmarkCase) -> TierResult:
    t0 = time.perf_counter()
    if case.task == "log_triage":
        counts = Counter(_LOG_ERR_RE.findall(case.payload["log_text"]))
        obj = {
            "errors": [
                {"service": svc, "code": code, "count": cnt}
                for (svc, code), cnt in sorted(counts.items())
            ]
        }
    elif case.task == "finance_reconcile":
        p_rows = case.payload["platform_rows"]
        g_rows = case.payload["gl_rows"]
        p_codes = sorted({r["platform_code"] for r in p_rows})
        g_vendors = sorted({r["gl_vendor"] for r in g_rows})
        vmap = _naive_token_vendor_map(p_codes, g_vendors)
        obj = reconcile_tables_exact(p_rows, g_rows, vmap)
    elif case.task == "note_to_json":
        obj = _deterministic_note_extract(case.payload["note"])
    elif case.task == "doc_contradictions":
        obj = _deterministic_doc_contradictions(case.payload["memos_text"])
    elif case.task == "context_cliff":
        matches = _INCIDENT_RE.findall(case.payload["digest"])
        obj = {
            "incidents": [
                {"id": inc_id, "owner": owner, "impact_usd": int(imp)}
                for inc_id, owner, imp in matches
            ]
        }
    else:
        raise ValueError(f"Unsupported task: {case.task}")

    latency_ms = (time.perf_counter() - t0) * 1000.0
    return TierResult(
        raw_output=json.dumps(obj),
        latency_ms=round(latency_ms, 3),
        input_tokens=0,
        output_tokens=0,
        cost_usd=0.0,
        api_usd=0.0,
        power_usd=0.0,
        routed_to="deterministic",
    )


# ---------------------------------------------------------------------------
# Direct HTTP Clients (Ollama, OpenAI-Compat/MiniMax, Gemini REST, Mock)
# ---------------------------------------------------------------------------

class RateLimiter:
    def __init__(self, rpm: float | None = None) -> None:
        self.min_interval = (60.0 / rpm) if rpm and rpm > 0 else 0.0
        self._last = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        now = time.monotonic()
        elapsed = now - self._last
        if self._last > 0 and elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last = time.monotonic()


def _http_post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: int = 180,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class OllamaClient:
    """Direct Ollama /api/chat client. Always sets keep_alive='15m' and think=False."""

    def __init__(
        self,
        tier_id: str,
        base_url: str,
        model: str,
        think: bool = False,
        keep_alive: str = "15m",
        timeout: int = 180,
    ) -> None:
        self.tier_id = tier_id
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.think = think
        self.keep_alive = keep_alive
        self.timeout = timeout

    def call(self, system_prompt: str, user_prompt: str, json_mode: bool = True) -> TierResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "think": self.think,
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0.0},
        }
        if json_mode:
            payload["format"] = "json"

        t0 = time.perf_counter()
        try:
            body = _http_post_json(
                f"{self.base_url}/api/chat",
                payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            content = body.get("message", {}).get("content", "")
            in_tok = int(body.get("prompt_eval_count", 0))
            out_tok = int(body.get("eval_count", 0))
            load_ms = float(body.get("load_duration", 0)) / 1_000_000.0
            cost = compute_call_cost_usd(self.tier_id, in_tok, out_tok, latency_ms)
            return TierResult(
                raw_output=content,
                latency_ms=round(latency_ms, 2),
                input_tokens=in_tok,
                output_tokens=out_tok,
                load_duration_ms=round(load_ms, 2),
                cost_usd=cost["total_usd"],
                api_usd=cost["api_usd"],
                power_usd=cost["power_usd"],
                routed_to=self.tier_id,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return TierResult(
                raw_output="",
                latency_ms=round(latency_ms, 2),
                input_tokens=0,
                output_tokens=0,
                routed_to=self.tier_id,
                error=f"{type(exc).__name__}: {exc}",
            )


class OpenAICompatClient:
    """Direct OpenAI-compatible client (supports MiniMax-M2.7 reasoning_split=True)."""

    def __init__(
        self,
        tier_id: str,
        url: str,
        model: str,
        api_key: str,
        reasoning_split: bool = False,
        max_tokens: int = 8000,
        timeout: int = 180,
        rpm: float | None = None,
    ) -> None:
        self.tier_id = tier_id
        self.url = url
        self.model = model
        self.api_key = api_key
        self.reasoning_split = reasoning_split
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.limiter = RateLimiter(rpm)

    def call(self, system_prompt: str, user_prompt: str, json_mode: bool = True) -> TierResult:
        self.limiter.wait()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
        }
        if self.reasoning_split:
            payload["reasoning_split"] = True

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        t0 = time.perf_counter()
        try:
            body = _http_post_json(self.url, payload, headers=headers, timeout=self.timeout)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            msg = body.get("choices", [{}])[0].get("message", {})
            content = msg.get("content", "") or ""
            usage = body.get("usage", {}) or {}
            in_tok = int(usage.get("prompt_tokens", 0))
            out_tok = int(usage.get("completion_tokens", 0))
            cost = compute_call_cost_usd(self.tier_id, in_tok, out_tok, latency_ms)
            return TierResult(
                raw_output=content,
                latency_ms=round(latency_ms, 2),
                input_tokens=in_tok,
                output_tokens=out_tok,
                cost_usd=cost["total_usd"],
                api_usd=cost["api_usd"],
                power_usd=cost["power_usd"],
                routed_to=self.tier_id,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return TierResult(
                raw_output="",
                latency_ms=round(latency_ms, 2),
                input_tokens=0,
                output_tokens=0,
                routed_to=self.tier_id,
                error=f"{type(exc).__name__}: {exc}",
            )


class GeminiClient:
    """Direct Google Generative Language REST client with free-tier RPM pacing & 429 retry."""

    def __init__(
        self,
        tier_id: str,
        model: str,
        api_key: str,
        thinking_budget: int | None = 0,
        rpm: float = 8.0,
        max_retries: int = 5,
        timeout: int = 180,
    ) -> None:
        self.tier_id = tier_id
        self.model = model
        self.api_key = api_key
        self.thinking_budget = thinking_budget
        self.limiter = RateLimiter(rpm)
        self.max_retries = max_retries
        self.timeout = timeout

    def call(self, system_prompt: str, user_prompt: str, json_mode: bool = True) -> TierResult:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        gen_cfg: dict[str, Any] = {"temperature": 0.0}
        if json_mode:
            gen_cfg["responseMimeType"] = "application/json"
        if self.thinking_budget is not None:
            if "gemini-3" in self.model and self.thinking_budget == 0:
                gen_cfg["thinkingConfig"] = {"thinkingLevel": "minimal"}
            else:
                gen_cfg["thinkingConfig"] = {"thinkingBudget": self.thinking_budget}

        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": gen_cfg,
        }
        headers = {"Content-Type": "application/json"}

        last_err = ""
        for attempt in range(self.max_retries):
            self.limiter.wait()
            t0 = time.perf_counter()
            try:
                body = _http_post_json(url, payload, headers=headers, timeout=self.timeout)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                parts = (
                    body.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [])
                )
                content = "".join(p.get("text", "") for p in parts if not p.get("thought"))
                usage = body.get("usageMetadata", {}) or {}
                in_tok = int(usage.get("promptTokenCount", 0))
                cand_tok = int(usage.get("candidatesTokenCount", 0))
                think_tok = int(usage.get("thoughtsTokenCount", 0))
                billable_out = cand_tok + think_tok
                cost = compute_call_cost_usd(self.tier_id, in_tok, billable_out, latency_ms)
                return TierResult(
                    raw_output=content,
                    latency_ms=round(latency_ms, 2),
                    input_tokens=in_tok,
                    output_tokens=billable_out,
                    thinking_tokens=think_tok,
                    cost_usd=cost["total_usd"],
                    api_usd=cost["api_usd"],
                    power_usd=cost["power_usd"],
                    routed_to=self.tier_id,
                )
            except urllib.error.HTTPError as exc:
                err_body = ""
                try:
                    err_body = exc.read().decode("utf-8", errors="ignore")[:300]
                except Exception:
                    pass
                last_err = f"HTTPError {exc.code}: {err_body}"
                if exc.code in (429, 503) and attempt + 1 < self.max_retries:
                    backoff = min(65.0, 8.0 * (2**attempt))
                    time.sleep(backoff)
                    continue
                return TierResult(
                    raw_output="",
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                    input_tokens=0,
                    output_tokens=0,
                    routed_to=self.tier_id,
                    error=last_err,
                )
            except Exception as exc:
                last_err = f"{type(exc).__name__}: {exc}"
                return TierResult(
                    raw_output="",
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                    input_tokens=0,
                    output_tokens=0,
                    routed_to=self.tier_id,
                    error=last_err,
                )

        return TierResult(
            raw_output="",
            latency_ms=0.0,
            input_tokens=0,
            output_tokens=0,
            routed_to=self.tier_id,
            error=last_err or "max_retries_exceeded",
        )


# ---------------------------------------------------------------------------
# Hybrid Pipeline: Code First -> Local SLM for narrow fuzzy steps -> Cloud only >1.5k prose
# ---------------------------------------------------------------------------

def run_hybrid(
    case: BenchmarkCase,
    slm_call: Callable[[str, str], TierResult],
    cloud_call: Callable[[str, str], TierResult],
) -> TierResult:
    """Routes each task to the cheapest reliable tier (and splits Task 2 into SLM map + Python math)."""
    if case.task in ("log_triage", "context_cliff"):
        res = run_deterministic(case)
        res.routed_to = "hybrid->deterministic"
        return res

    if case.task == "finance_reconcile":
        p_rows = case.payload["platform_rows"]
        g_rows = case.payload["gl_rows"]
        p_codes = sorted({r["platform_code"] for r in p_rows})
        g_vendors = sorted({r["gl_vendor"] for r in g_rows})
        sys_p = (
            "Map each abbreviated ad platform code to its exact matching General Ledger vendor name. "
            "Return ONLY JSON: {\"mapping\": {\"<PLATFORM_CODE>\": \"<EXACT_GL_VENDOR>\"}}"
        )
        usr_p = (
            f"PLATFORM_CODES: {json.dumps(p_codes)}\n"
            f"GL_VENDORS: {json.dumps(g_vendors)}"
        )
        t0 = time.perf_counter()
        slm_res = slm_call(sys_p, usr_p)
        parsed, ok = extract_json_object(slm_res.raw_output)
        vmap = parsed.get("mapping", {}) if (ok and isinstance(parsed, dict)) else {}
        if not isinstance(vmap, dict) or not vmap:
            vmap = _naive_token_vendor_map(p_codes, g_vendors)
        reconciled = reconcile_tables_exact(p_rows, g_rows, vmap)
        total_ms = (time.perf_counter() - t0) * 1000.0
        return TierResult(
            raw_output=json.dumps(reconciled),
            latency_ms=round(total_ms, 2),
            input_tokens=slm_res.input_tokens,
            output_tokens=slm_res.output_tokens,
            load_duration_ms=slm_res.load_duration_ms,
            cost_usd=slm_res.cost_usd,
            api_usd=slm_res.api_usd,
            power_usd=slm_res.power_usd,
            routed_to=f"hybrid->{slm_res.routed_to}+python",
            error=slm_res.error,
        )

    if case.task == "note_to_json":
        res = slm_call(case.system_prompt, case.prompt)
        res.routed_to = f"hybrid->{res.routed_to}"
        return res

    # Multi-doc unstructured prose (>1,500 tokens): skip small SLM, route directly to Cloud/74B
    res = cloud_call(case.system_prompt, case.prompt)
    res.routed_to = f"hybrid->{res.routed_to}"
    return res
