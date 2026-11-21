"""Track 5B: Reproducible Ollama infrastructure cliff measurements.

Quantifies two common local-SLM latency traps:
1. OLLAMA_KEEP_ALIVE=0 cold-reload penalty (evicting & reloading weights every call)
   vs. warm resident inference (keep_alive='15m').
2. Varying num_ctx across callers (forcing silent KV-cache unload/reallocation).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any

from .finops import sanitize_telemetry


def _call_ollama(
    base_url: str,
    model: str,
    keep_alive: Any,
    num_ctx: int | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": 'Return {"status": "ok"}.'},
        ],
        "stream": False,
        "think": False,
        "format": "json",
        "keep_alive": keep_alive,
        "options": {"temperature": 0.0},
    }
    if num_ctx is not None:
        payload["options"]["num_ctx"] = num_ctx

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    wall_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "wall_ms": round(wall_ms, 2),
        "load_ms": round(float(body.get("load_duration", 0)) / 1_000_000.0, 2),
        "eval_ms": round(float(body.get("eval_duration", 0)) / 1_000_000.0, 2),
        "prompt_eval_ms": round(float(body.get("prompt_eval_duration", 0)) / 1_000_000.0, 2),
        "eval_count": int(body.get("eval_count", 0)),
    }


def run_infra_cliff(
    base_url: str,
    model: str,
    trials: int = 4,
    test_num_ctx_thrash: bool = True,
) -> dict[str, Any]:
    # 1. Warm resident calls (1 priming call + N timed warm calls)
    _call_ollama(base_url, model, keep_alive="15m")
    warm_runs = [_call_ollama(base_url, model, keep_alive="15m") for _ in range(trials)]

    # 2. keep_alive=0 cold-evict loop (evict first, then measure reload on every call)
    _call_ollama(base_url, model, keep_alive=0)
    time.sleep(0.5)
    cold_runs = []
    for _ in range(trials):
        cold_runs.append(_call_ollama(base_url, model, keep_alive=0))
        time.sleep(0.5)

    # 3. Optional num_ctx thrashing (alternating 4096 <-> 6144 with keep_alive="15m")
    thrash_runs = []
    if test_num_ctx_thrash:
        _call_ollama(base_url, model, keep_alive="15m", num_ctx=4096)
        for ctx in (6144, 4096, 6144, 4096)[:trials]:
            r = _call_ollama(base_url, model, keep_alive="15m", num_ctx=ctx)
            r["num_ctx"] = ctx
            thrash_runs.append(r)
        # Leave model warm at default context
        _call_ollama(base_url, model, keep_alive="15m")

    warm_med = round(statistics.median(r["wall_ms"] for r in warm_runs), 2)
    cold_med = round(statistics.median(r["wall_ms"] for r in cold_runs), 2)
    thrash_med = (
        round(statistics.median(r["wall_ms"] for r in thrash_runs), 2)
        if thrash_runs
        else None
    )

    return sanitize_telemetry(
        {
            "model": model,
            "trials": trials,
            "summary": {
                "warm_median_ms": warm_med,
                "cold_keep_alive_0_median_ms": cold_med,
                "cold_reload_multiplier": round(cold_med / max(1.0, warm_med), 2),
                "num_ctx_thrash_median_ms": thrash_med,
                "num_ctx_thrash_multiplier": (
                    round(thrash_med / max(1.0, warm_med), 2) if thrash_med else None
                ),
            },
            "warm_runs": warm_runs,
            "cold_keep_alive_0_runs": cold_runs,
            "num_ctx_thrash_runs": thrash_runs,
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure Ollama keep_alive=0 and num_ctx latency cliffs.")
    ap.add_argument("--base-url", default="http://127.0.0.1:11434")
    ap.add_argument("--model", required=True)
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--no-ctx-thrash", action="store_true")
    ap.add_argument("--output", default="results/infra_cliff.json")
    args = ap.parse_args()

    res = run_infra_cliff(
        base_url=args.base_url,
        model=args.model,
        trials=args.trials,
        test_num_ctx_thrash=not args.no_ctx_thrash,
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res["summary"], indent=2))


if __name__ == "__main__":
    main()
