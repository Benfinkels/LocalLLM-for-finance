"""FinOps unit-economics calculator and telemetry sanitizer.

Computes exact dollar cost per call and projected cost per 10,000 tasks across:
- Deterministic Python (CPU)
- Local GPU inference (marginal electricity cost at $0.15/kWh)
- Cloud API endpoints (published per-1M token input/output rates)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TierPricing:
    tier_id: str
    display_name: str
    category: str  # deterministic | local_slm | local_moe | cloud_api | hybrid
    input_per_1m_usd: float = 0.0
    output_per_1m_usd: float = 0.0
    gpu_tdp_watts: float = 0.0  # Marginal active power draw for local GPU tiers
    electricity_per_kwh_usd: float = 0.15


# Published API rates (late 2026) and measured local hardware TDP profiles
PRICING_CATALOG: dict[str, TierPricing] = {
    "deterministic": TierPricing(
        tier_id="deterministic",
        display_name="Deterministic Python",
        category="deterministic",
    ),
    "phi4_mini": TierPricing(
        tier_id="phi4_mini",
        display_name="Phi-4 Mini 3.8B (GTX 1080 8GB)",
        category="local_slm",
        gpu_tdp_watts=130.0,
    ),
    "gemma4_e2b": TierPricing(
        tier_id="gemma4_e2b",
        display_name="Gemma 4 E2B (RTX 2080 8GB)",
        category="local_slm",
        gpu_tdp_watts=175.0,
    ),
    "nemotron_4090": TierPricing(
        tier_id="nemotron_4090",
        display_name="Nemotron 74B-A9B (RTX 4090 24GB)",
        category="local_moe",
        gpu_tdp_watts=320.0,
    ),
    "minimax_m27": TierPricing(
        tier_id="minimax_m27",
        display_name="MiniMax-M2.7 (Cloud API)",
        category="cloud_api",
        input_per_1m_usd=0.30,
        output_per_1m_usd=1.20,
    ),
    "gemini_flash": TierPricing(
        tier_id="gemini_flash",
        display_name="Gemini 3 Flash (Cloud API)",
        category="cloud_api",
        input_per_1m_usd=0.50,
        output_per_1m_usd=3.00,
    ),
    "gemini_4_hydrogen": TierPricing(
        tier_id="gemini_4_hydrogen",
        display_name="Gemini 4 Hydrogen (Cloud API)",
        category="cloud_api",
        input_per_1m_usd=1.50,
        output_per_1m_usd=7.50,
    ),
    "gemini_4_neon": TierPricing(
        tier_id="gemini_4_neon",
        display_name="Gemini 4 Neon (Frontier API)",
        category="cloud_api",
        input_per_1m_usd=3.00,
        output_per_1m_usd=15.00,
    ),
    "gemini_4_carbon": TierPricing(
        tier_id="gemini_4_carbon",
        display_name="Gemini 4 Carbon (Frontier API)",
        category="cloud_api",
        input_per_1m_usd=4.00,
        output_per_1m_usd=20.00,
    ),
    "hybrid": TierPricing(
        tier_id="hybrid",
        display_name="Hybrid (Code + Gemma4 + Cloud)",
        category="hybrid",
        gpu_tdp_watts=175.0,
    ),
}


def compute_call_cost_usd(
    tier_id: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
    pricing: TierPricing | None = None,
) -> dict[str, float]:
    p = pricing or PRICING_CATALOG.get(tier_id, TierPricing(tier_id, tier_id, "cloud_api"))
    api_usd = (
        (max(0, input_tokens) / 1_000_000.0) * p.input_per_1m_usd
        + (max(0, output_tokens) / 1_000_000.0) * p.output_per_1m_usd
    )
    hours = max(0.0, latency_ms) / 3_600_000.0
    power_usd = (p.gpu_tdp_watts / 1000.0) * hours * p.electricity_per_kwh_usd
    total_usd = api_usd + power_usd
    return {
        "api_usd": round(api_usd, 8),
        "power_usd": round(power_usd, 8),
        "total_usd": round(total_usd, 8),
    }


def project_cost_per_10k(mean_cost_usd: float) -> float:
    return round(mean_cost_usd * 10_000.0, 2)


_PRIVATE_IP_RE = re.compile(
    r"\b(?:192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
    r"100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3})\b"
)
_TS_HOST_RE = re.compile(r"\b[a-zA-Z0-9_-]+\.ts\.net\b")


def sanitize_telemetry(obj: Any) -> Any:
    """Recursively scrub LAN/Tailscale IPs and internal hostnames before saving JSON."""
    if isinstance(obj, str):
        s = _PRIVATE_IP_RE.sub("192.0.2.10", obj)
        s = _TS_HOST_RE.sub("node.example.invalid", s)
        return s
    if isinstance(obj, list):
        return [sanitize_telemetry(x) for x in obj]
    if isinstance(obj, dict):
        return {k: sanitize_telemetry(v) for k, v in obj.items()}
    return obj
