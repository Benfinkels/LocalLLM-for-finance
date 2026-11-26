"""Execution tiers: Deterministic Python, Ollama Local, OpenAI-Compat (MiniMax), Gemini REST, Hybrid, and Mock."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
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


_WORD_NUMS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}

_ALLOWED_NOTE_CATEGORIES = frozenset(
    {"renewal", "invoice_dispute", "onboarding", "cancellation", "upsell"}
)

_NO_FOLLOWUP_CUES = (
    "nothing left", "no outreach", "all set", "just logging", "log for record",
    "zero open", "closed", "no further action", "no follow-up", "archive only",
    "do not reach out", "not in our court", "completely wrapped up",
    "wait for them to re-engage", "already reversed it",
)


def parse_english_amount(text: str) -> float | None:
    """General deterministic parser for numeric and spelled-out English USD amounts."""
    low = text.lower()
    if any(p in low for p in ("zero-dollar", "zero-cost", "$0", "free zero")):
        return 0.0
    m_k = re.search(r"\$\s*([\d,]+(?:\.\d+)?)\s*k\b", low)
    if m_k:
        return round(float(m_k.group(1).replace(",", "")) * 1000.0, 2)
    m_d = re.search(r"\$\s*([\d,]+(?:\.\d+)?)", low)
    if m_d:
        return round(float(m_d.group(1).replace(",", "")), 2)
    if "half-a-grand" in low or "half a grand" in low:
        return 500.0

    # Handle "<num> point <digit> (grand|thousand)" and "<num> and a half (grand|thousand)"
    tokens = re.findall(r"[a-z]+|\d+(?:\.\d+)?", low.replace("-", " "))
    for i, tok in enumerate(tokens):
        if tok in ("grand", "thousand") and i >= 1:
            if i >= 4 and tokens[i - 3 : i] == ["and", "a", "half"] and tokens[i - 4] in _WORD_NUMS:
                return round((_WORD_NUMS[tokens[i - 4]] + 0.5) * 1000.0, 2)
            if i >= 3 and tokens[i - 2] == "point" and tokens[i - 3] in _WORD_NUMS and tokens[i - 1] in _WORD_NUMS:
                return round((_WORD_NUMS[tokens[i - 3]] + _WORD_NUMS[tokens[i - 1]] / 10.0) * 1000.0, 2)

    # Scan contiguous English number-word spans
    best_val: float | None = None
    best_len = 0
    n = len(tokens)
    i = 0
    while i < n:
        if tokens[i] not in _WORD_NUMS:
            i += 1
            continue
        j = i
        total = 0.0
        current = 0.0
        saw_scale = False
        while j < n:
            w = tokens[j]
            if w in _WORD_NUMS:
                val = _WORD_NUMS[w]
                # Spoken shorthand like "eighteen fifty" -> 1850
                if 11 <= current <= 19 and val >= 20 and not saw_scale:
                    current = current * 100 + val
                    saw_scale = True
                else:
                    current += val
            elif w == "hundred":
                current = max(1.0, current) * 100.0
                saw_scale = True
            elif w in ("thousand", "grand"):
                total += max(1.0, current) * 1000.0
                current = 0.0
                saw_scale = True
            elif w == "and" and j + 1 < n and tokens[j + 1] in _WORD_NUMS:
                pass
            else:
                break
            j += 1
        span_val = total + current
        span_len = j - i
        # Only treat as currency if scaled (>=100) or followed by currency words
        next_words = set(tokens[j : min(n, j + 4)])
        if (saw_scale or next_words & {"dollar", "dollars", "bucks", "usd", "acv", "month", "yr"}) and span_len > best_len:
            best_val = round(span_val, 2)
            best_len = span_len
        i = max(i + 1, j)
    return best_val


def _deterministic_note_extract(note: str) -> dict[str, Any]:
    """Improved general-purpose regex + English-number parser (no NER model)."""
    # Strip optional leading "Header Title: " prefix before scanning for names
    body = note.split(":", 1)[1].strip() if ":" in note[:45] else note
    amt = parse_english_amount(body)
    if amt is None:
        amt = 0.0

    low = note.lower()
    if any(w in low for w in ("renew", "extend", "extension", "roll over", "rollover", "keep their")):
        cat = "renewal"
    elif any(w in low for w in ("invoice", "discrepancy", "overcharge", "double-billed", "double-charged", "late fee", "penalty", "mismatch", "variance")):
        cat = "invoice_dispute"
    elif any(w in low for w in ("cancel", "shutting down", "winding down", "dropping", "churn", "terminate", "walk away", "deactivated", "end their")):
        cat = "cancellation"
    elif any(w in low for w in ("onboard", "kickoff", "implementation", "provisioning", "trial", "pipeline setup", "go-live")):
        cat = "onboarding"
    elif any(w in low for w in ("upgrade", "add-on", "extra seats", "expanding", "expansion", "move up", "step up")):
        cat = "upsell"
    else:
        cat = "unknown"

    followup = not any(cue in low for cue in _NO_FOLLOWUP_CUES)

    # Best-effort proper-noun extraction (parenthesized contact or capitalized spans)
    paren = re.search(r"\((?:contact:\s*)?([A-Z][a-z]+ [A-Z][a-z]+)\)", body)
    names = re.findall(r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b", body)
    contact = paren.group(1) if paren else (names[0] if names else "")
    company = next((nm for nm in names if nm != contact), "")
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


_REFUND_PREFIX_RE = re.compile(r"^(?:refund|cr|rev\*?)\s+", re.IGNORECASE)

_BANK_CATEGORY_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("cloud_infra", ("aws", "cloud", "hosting", "digitalocean")),
    ("saas_tools", ("slack", "github", "software", "saas")),
    ("travel_lodging", ("air lines", "airline", "hotel", "marriott", "delta", "lodging")),
    ("meals_dining", ("coffee", "starbucks", "bread", "catering", "cafe", "restaurant")),
    ("office_supplies", ("staples", "office", "supply", "depot")),
    ("shipping_logistics", ("fedex", "ups", "shipping", "freight")),
)


def _normalize_bank_merchant(description: str) -> str:
    cleaned = _REFUND_PREFIX_RE.sub("", description.strip())
    return re.sub(r"\s+", " ", cleaned).strip().lower()


def _naive_bank_category_map(descriptions: list[str]) -> dict[str, str]:
    """Honest deterministic keyword categorizer (no curated processor-code table)."""
    mapping: dict[str, str] = {}
    for desc in descriptions:
        low = _normalize_bank_merchant(desc)
        matched = "uncategorized"
        for category, keywords in _BANK_CATEGORY_KEYWORDS:
            if any(kw in low for kw in keywords):
                matched = category
                break
        mapping[desc] = matched
    return mapping


def analyze_bank_statement_exact(
    rows: list[dict[str, Any]],
    category_map: dict[str, str],
) -> dict[str, Any]:
    """Cent-exact deterministic anomaly detection & net spend given a description -> category map."""
    norm_cat_map = {k.strip().lower(): v.strip().lower() for k, v in category_map.items() if k and v}
    parsed = [
        {
            "txn_id": r["txn_id"],
            "date": date.fromisoformat(r["date"]),
            "description": r["description"],
            "merchant_key": _normalize_bank_merchant(r["description"]),
            "amount": round(float(r["amount_usd"]), 2),
        }
        for r in rows
    ]

    categories = [
        {
            "txn_id": p["txn_id"],
            "category": norm_cat_map.get(p["description"].strip().lower())
            or norm_cat_map.get(p["merchant_key"])
            or "uncategorized",
        }
        for p in parsed
    ]

    anomalies: list[dict[str, Any]] = []
    used_dup: set[str] = set()
    used_ref: set[str] = set()

    for i, a in enumerate(parsed):
        if a["amount"] >= 0:
            continue
        for j in range(i + 1, len(parsed)):
            b = parsed[j]
            day_diff = abs((b["date"] - a["date"]).days)
            if (
                b["amount"] < 0
                and a["txn_id"] not in used_dup
                and b["txn_id"] not in used_dup
                and a["merchant_key"] == b["merchant_key"]
                and abs(a["amount"] - b["amount"]) <= 0.02
                and day_diff <= 2
            ):
                anomalies.append(
                    {
                        "type": "duplicate_charge",
                        "txn_ids": sorted([a["txn_id"], b["txn_id"]]),
                        "amount_usd": round(abs(a["amount"]), 2),
                    }
                )
                used_dup.update((a["txn_id"], b["txn_id"]))
            elif (
                b["amount"] > 0
                and a["txn_id"] not in used_ref
                and b["txn_id"] not in used_ref
                and a["merchant_key"] == b["merchant_key"]
                and abs(a["amount"] + b["amount"]) <= 0.02
                and day_diff <= 5
            ):
                anomalies.append(
                    {
                        "type": "refund_reversal",
                        "txn_ids": sorted([a["txn_id"], b["txn_id"]]),
                        "amount_usd": round(abs(a["amount"]), 2),
                    }
                )
                used_ref.update((a["txn_id"], b["txn_id"]))

    net_spend = round(-sum(p["amount"] for p in parsed), 2)
    return {
        "categories": categories,
        "anomalies": anomalies,
        "net_spend_usd": net_spend,
    }


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
    elif case.task == "bank_statement":
        rows = case.payload["rows"]
        descriptions = sorted({r["description"] for r in rows})
        cmap = _naive_bank_category_map(descriptions)
        obj = analyze_bank_statement_exact(rows, cmap)
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

    if case.task == "bank_statement":
        rows = case.payload["rows"]
        descriptions = sorted({r["description"] for r in rows})
        sys_p = (
            "Map each messy bank-statement merchant description to one canonical category: "
            "cloud_infra, saas_tools, travel_lodging, meals_dining, office_supplies, shipping_logistics. "
            'Return ONLY JSON: {"mapping": {"<DESCRIPTION>": "<CATEGORY>"}}'
        )
        usr_p = f"MERCHANT_DESCRIPTIONS: {json.dumps(descriptions)}"
        t0 = time.perf_counter()
        slm_res = slm_call(sys_p, usr_p)
        parsed, ok = extract_json_object(slm_res.raw_output)
        cmap = parsed.get("mapping", {}) if (ok and isinstance(parsed, dict)) else {}
        if not isinstance(cmap, dict) or not cmap:
            cmap = _naive_bank_category_map(descriptions)
        analyzed = analyze_bank_statement_exact(rows, cmap)
        total_ms = (time.perf_counter() - t0) * 1000.0
        return TierResult(
            raw_output=json.dumps(analyzed),
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
        # Code + Local SLM agreement check: run both ($0), escalate ONLY disagreements to Cloud.
        det = _deterministic_note_extract(case.payload["note"])
        det_amt = parse_english_amount(case.payload["note"])
        slm_res = slm_call(case.system_prompt, case.prompt)
        parsed, ok = extract_json_object(slm_res.raw_output)
        needs_escalation = False
        if not ok or not isinstance(parsed, dict):
            needs_escalation = True
        else:
            cat = str(parsed.get("category", "")).strip().lower()
            try:
                slm_amt = float(parsed.get("amount_usd"))
            except Exception:
                slm_amt = None
            if (
                cat not in _ALLOWED_NOTE_CATEGORIES
                or not str(parsed.get("contact", "")).strip()
                or not str(parsed.get("company", "")).strip()
                or not isinstance(parsed.get("requires_followup"), bool)
                or (det_amt is not None and (slm_amt is None or abs(det_amt - slm_amt) > 0.02))
                or (det["category"] != "unknown" and cat != det["category"])
                or (parsed.get("requires_followup") != det["requires_followup"])
            ):
                needs_escalation = True

        if not needs_escalation:
            slm_res.routed_to = f"hybrid->{slm_res.routed_to}"
            return slm_res

        # Escalate only the small fraction of notes where Code and Local SLM disagree
        esc = cloud_call(case.system_prompt, case.prompt)
        return TierResult(
            raw_output=esc.raw_output,
            latency_ms=round(slm_res.latency_ms + esc.latency_ms, 2),
            input_tokens=slm_res.input_tokens + esc.input_tokens,
            output_tokens=slm_res.output_tokens + esc.output_tokens,
            thinking_tokens=slm_res.thinking_tokens + esc.thinking_tokens,
            load_duration_ms=slm_res.load_duration_ms,
            cost_usd=round(slm_res.cost_usd + esc.cost_usd, 8),
            api_usd=round(slm_res.api_usd + esc.api_usd, 8),
            power_usd=round(slm_res.power_usd + esc.power_usd, 8),
            routed_to=f"hybrid->{slm_res.routed_to}->escalate->{esc.routed_to}",
            error=esc.error,
        )

    # Multi-doc unstructured prose (>1,500 tokens): skip small SLM, route directly to Cloud/74B
    res = cloud_call(case.system_prompt, case.prompt)
    res.routed_to = f"hybrid->{res.routed_to}"
    return res
