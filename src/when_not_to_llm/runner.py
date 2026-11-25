"""CLI runner for when-not-to-llm with dry-run cost estimation, checkpoint resume, and FinOps summary."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .datasets import BenchmarkCase, load_all_cases
from .finops import (
    PRICING_CATALOG,
    compute_call_cost_usd,
    project_cost_per_10k,
    sanitize_telemetry,
)
from .tiers import (
    GeminiClient,
    OllamaClient,
    OpenAICompatClient,
    TierResult,
    run_deterministic,
    run_hybrid,
)
from .verifiers import score_case


def _percentile(vals: list[float], pct: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, int(round((pct / 100.0) * (len(s) - 1)))))
    return round(s[idx], 2)


def estimate_dry_run(cases: list[BenchmarkCase], tiers: list[str]) -> dict[str, Any]:
    est: dict[str, Any] = {}
    for tier in tiers:
        total_in = sum(c.approx_tokens for c in cases) if tier != "deterministic" else 0
        total_out = (len(cases) * 350) if tier != "deterministic" else 0
        if tier == "hybrid":
            # Only small SLM calls on finance/note + cloud on doc_contradictions
            total_in = sum(
                180 if c.task == "finance_reconcile"
                else (c.approx_tokens if c.task in ("note_to_json", "doc_contradictions") else 0)
                for c in cases
            )
            total_out = sum(
                200 if c.task in ("finance_reconcile", "note_to_json")
                else (600 if c.task == "doc_contradictions" else 0)
                for c in cases
            )
        c_est = compute_call_cost_usd(tier, total_in, total_out, latency_ms=len(cases) * 1500.0)
        est[tier] = {
            "calls": len(cases),
            "est_input_tokens": total_in,
            "est_output_tokens": total_out,
            "est_api_usd": round(c_est["api_usd"], 4),
            "est_total_usd": round(c_est["total_usd"], 4),
        }
    return est


def summarize_results(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_tier: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_tier_task: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    context_curve: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for r in records:
        tier = r["tier"]
        task = r["task"]
        by_tier[tier].append(r)
        by_tier_task[tier][task].append(r)
        if task == "context_cliff" and r.get("target_tokens"):
            context_curve[tier][str(r["target_tokens"])].append(float(r["score"]))

    tier_summary: dict[str, Any] = {}
    for tier, rows in sorted(by_tier.items()):
        core_rows = [r for r in rows if r["task"] != "context_cliff"] or rows
        lats = [float(r["latency_ms"]) for r in rows]
        mean_cost = statistics.mean(float(r["cost_usd"]) for r in rows)
        task_scores = {
            t: round(statistics.mean(float(x["score"]) for x in trows), 4)
            for t, trows in sorted(by_tier_task[tier].items())
        }
        task_lats = {
            t: round(statistics.median(float(x["latency_ms"]) for x in trows), 2)
            for t, trows in sorted(by_tier_task[tier].items())
        }
        # Macro average across core tasks (equal weight per task archetype)
        core_task_means = [v for k, v in task_scores.items() if k != "context_cliff"]
        macro_core = round(statistics.mean(core_task_means), 4) if core_task_means else 0.0
        tier_summary[tier] = {
            "display_name": PRICING_CATALOG.get(tier).display_name if tier in PRICING_CATALOG else tier,
            "cases": len(rows),
            "errors": sum(1 for r in rows if r.get("error")),
            "macro_core_accuracy": macro_core,
            "task_accuracy": task_scores,
            "task_median_latency_ms": task_lats,
            "p50_latency_ms": _percentile(lats, 50),
            "p95_latency_ms": _percentile(lats, 95),
            "total_input_tokens": sum(int(r["input_tokens"]) for r in rows),
            "total_output_tokens": sum(int(r["output_tokens"]) for r in rows),
            "total_cost_usd": round(sum(float(r["cost_usd"]) for r in rows), 6),
            "cost_per_10k_tasks_usd": project_cost_per_10k(mean_cost),
            "context_cliff_accuracy": {
                k: round(statistics.mean(vs), 4)
                for k, vs in sorted(context_curve[tier].items(), key=lambda kv: int(kv[0]))
            },
        }
    return tier_summary


def format_summary_markdown(summary: dict[str, Any]) -> str:
    """Render the benchmark's tier summary without exposing per-case payloads."""
    lines = [
        "| Tier | Cases | Errors | Core accuracy | p50 (ms) | p95 (ms) | Cost / 10k (USD) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for tier, row in sorted(summary.items()):
        name = str(row.get("display_name", tier)).replace("\\", "\\\\").replace("|", "\\|")
        name = name.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")
        lines.append(
            f"| {name} | {row['cases']} | {row['errors']} | {row['macro_core_accuracy']:.2%} | "
            f"{row['p50_latency_ms']:.2f} | {row['p95_latency_ms']:.2f} | "
            f"{row['cost_per_10k_tasks_usd']:.2f} |"
        )
    return "\n".join(lines)


def build_clients(args: argparse.Namespace) -> dict[str, Any]:
    clients: dict[str, Any] = {}
    if "phi4_mini" in args.tiers:
        clients["phi4_mini"] = OllamaClient(
            "phi4_mini",
            base_url=args.sheldon_gpu1_url,
            model=args.phi4_model,
            think=False,
        )
    if "gemma4_e2b" in args.tiers or "hybrid" in args.tiers:
        clients["gemma4_e2b"] = OllamaClient(
            "gemma4_e2b",
            base_url=args.sheldon_gpu0_url,
            model=args.gemma4_model,
            think=False,
        )
    if "nemotron_4090" in args.tiers:
        clients["nemotron_4090"] = OllamaClient(
            "nemotron_4090",
            base_url=args.hyperion_ollama_url,
            model=args.nemotron_model,
            think=False,
            timeout=300,
        )
    if "minimax_m27" in args.tiers or ("hybrid" in args.tiers and args.hybrid_cloud == "minimax_m27"):
        api_key = os.environ.get("MINIMAX_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("MINIMAX_API_KEY env var is required for minimax_m27 tier.")
        clients["minimax_m27"] = OpenAICompatClient(
            "minimax_m27",
            url="https://api.minimax.io/v1/chat/completions",
            model=args.minimax_model,
            api_key=api_key,
            reasoning_split=True,
            max_tokens=8000,
        )
    if "gemini_flash" in args.tiers or ("hybrid" in args.tiers and args.hybrid_cloud == "gemini_flash"):
        api_key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY env var is required for gemini_flash tier.")
        clients["gemini_flash"] = GeminiClient(
            "gemini_flash",
            model=args.gemini_model,
            api_key=api_key,
            thinking_budget=args.gemini_thinking_budget,
            rpm=args.gemini_rpm,
        )
    for g_tier, g_model in (
        ("gemini_4_hydrogen", args.hydrogen_model),
        ("gemini_4_neon", args.neon_model),
        ("gemini_4_carbon", args.frontier_model),
    ):
        if g_tier in args.tiers or ("hybrid" in args.tiers and args.hybrid_cloud == g_tier):
            api_key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
            if not api_key:
                raise RuntimeError(f"GEMINI_API_KEY env var is required for {g_tier} tier.")
            clients[g_tier] = GeminiClient(
                g_tier,
                model=g_model,
                api_key=api_key,
                thinking_budget=None,  # Full default Gemini 4 reasoning (no handicapping)
                rpm=args.gemini_rpm,
            )
    return clients


def _execute_tier(
    tier: str,
    case: BenchmarkCase,
    clients: dict[str, Any],
    hybrid_cloud: str,
) -> TierResult:
    if tier == "deterministic":
        return run_deterministic(case)
    if tier == "hybrid":
        slm = clients["gemma4_e2b"]
        cloud = clients[hybrid_cloud]
        return run_hybrid(
            case,
            slm_call=lambda s, u: slm.call(s, u, json_mode=True),
            cloud_call=lambda s, u: cloud.call(s, u, json_mode=True),
        )
    client = clients[tier]
    return client.call(case.system_prompt, case.prompt, json_mode=True)


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    cases = load_all_cases(args.tasks)
    if args.dry_run:
        est = estimate_dry_run(cases, args.tiers)
        print(json.dumps({"dry_run": True, "cases": len(cases), "estimates": est}, indent=2))
        return est

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    done_keys: set[tuple[str, str]] = set()
    if args.resume and out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        active_tiers = set(args.tiers)
        active_tasks = set(args.tasks) if args.tasks else None
        for r in existing.get("records", []):
            is_retry_target = r["tier"] in active_tiers and (active_tasks is None or r["task"] in active_tasks)
            if not r.get("error") or not is_retry_target:
                records.append(r)
                done_keys.add((r["tier"], r["case_id"]))
        print(f"[resume] Loaded {len(records)} completed records from {out_path}", file=sys.stderr)

    clients = build_clients(args)

    # Warm-up 1 priming ping per Ollama client so cold model loads don't distort Task 1 latency
    if not args.skip_warmup:
        for t in ("phi4_mini", "gemma4_e2b", "nemotron_4090"):
            if t in clients and (t in args.tiers or (t == "gemma4_e2b" and "hybrid" in args.tiers)):
                print(f"[warmup] Priming {t}...", file=sys.stderr)
                clients[t].call("Return JSON.", '{"ok": true}', json_mode=True)

    spent_usd = sum(float(r.get("api_usd", 0.0)) for r in records)

    for tier in args.tiers:
        for case in cases:
            if (tier, case.case_id) in done_keys:
                continue
            if spent_usd >= args.max_cost_usd:
                print(
                    f"[stop] Reached --max-cost-usd=${args.max_cost_usd:.2f} (spent=${spent_usd:.4f})",
                    file=sys.stderr,
                )
                break

            res = _execute_tier(tier, case, clients, args.hybrid_cloud)
            metrics = score_case(case.task, res.raw_output, case.ground_truth)
            spent_usd += res.api_usd

            rec = {
                "tier": tier,
                "task": case.task,
                "case_id": case.case_id,
                "approx_tokens": case.approx_tokens,
                "target_tokens": case.payload.get("target_tokens"),
                "score": metrics["score"],
                "json_valid": metrics["json_valid"],
                "metrics": metrics,
                "latency_ms": res.latency_ms,
                "load_duration_ms": res.load_duration_ms,
                "input_tokens": res.input_tokens,
                "output_tokens": res.output_tokens,
                "thinking_tokens": res.thinking_tokens,
                "cost_usd": res.cost_usd,
                "api_usd": res.api_usd,
                "power_usd": res.power_usd,
                "routed_to": res.routed_to,
                "error": res.error,
            }
            records.append(rec)
            print(
                f"  [{tier:<13}] {case.case_id:<26} score={metrics['score']:.2f} "
                f"lat={res.latency_ms:7.1f}ms tok={res.input_tokens}/{res.output_tokens} "
                f"err={'YES' if res.error else 'no'}",
                file=sys.stderr,
            )

            # Checkpoint after every call so free-tier interruptions never lose progress
            payload = sanitize_telemetry(
                {
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "models": {
                        "phi4_mini": args.phi4_model,
                        "gemma4_e2b": args.gemma4_model,
                        "nemotron_4090": args.nemotron_model,
                        "minimax_m27": args.minimax_model,
                        "gemini_flash": args.gemini_model,
                        "gemini_4_carbon": args.frontier_model,
                    },
                    "summary": summarize_results(records),
                    "records": records,
                }
            )
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    final_payload = sanitize_telemetry(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "models": {
                "phi4_mini": args.phi4_model,
                "gemma4_e2b": args.gemma4_model,
                "nemotron_4090": args.nemotron_model,
                "minimax_m27": args.minimax_model,
                "gemini_flash": args.gemini_model,
            },
            "summary": summarize_results(records),
            "records": records,
        }
    )
    out_path.write_text(json.dumps(final_payload, indent=2), encoding="utf-8")
    return final_payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the when-not-to-llm FinOps & accuracy benchmark.")
    ap.add_argument(
        "--tiers",
        nargs="+",
        default=["deterministic"],
        choices=[
            "deterministic",
            "phi4_mini",
            "gemma4_e2b",
            "nemotron_4090",
            "minimax_m27",
            "gemini_flash",
            "gemini_4_hydrogen",
            "gemini_4_neon",
            "gemini_4_carbon",
            "hybrid",
        ],
    )
    ap.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        choices=["log_triage", "finance_reconcile", "note_to_json", "doc_contradictions", "context_cliff"],
    )
    ap.add_argument("--format", choices=["json", "markdown"], default="json",
                    help="Summary output format (default: json).")
    ap.add_argument("--summary-from", type=Path,
                    help="Render a saved benchmark JSON summary without running models.")
    ap.add_argument("--output", default="results/benchmark_results.json")
    ap.add_argument("--resume", action="store_true", help="Resume from existing output JSON checkpoint.")
    ap.add_argument("--dry-run", action="store_true", help="Print token/cost estimates without making API calls.")
    ap.add_argument("--max-cost-usd", type=float, default=2.0, help="Hard cap on API spend before stopping.")
    ap.add_argument("--skip-warmup", action="store_true")
    ap.add_argument(
        "--hybrid-cloud",
        default="minimax_m27",
        choices=["minimax_m27", "gemini_flash", "gemini_4_hydrogen", "gemini_4_neon", "gemini_4_carbon"],
    )

    ap.add_argument("--sheldon-gpu0-url", default=os.environ.get("OLLAMA_GPU0_URL", "http://127.0.0.1:11434"))
    ap.add_argument("--sheldon-gpu1-url", default=os.environ.get("OLLAMA_GPU1_URL", "http://127.0.0.1:11435"))
    ap.add_argument("--hyperion-ollama-url", default=os.environ.get("HYPERION_OLLAMA_URL", "http://127.0.0.1:11434"))

    ap.add_argument("--phi4-model", default="phi4-mini:latest")
    ap.add_argument("--gemma4-model", default="gemma4:e2b-it-q4_K_M")
    ap.add_argument("--nemotron-model", default="nemotron-3.5-bolt:latest")
    ap.add_argument("--minimax-model", default="MiniMax-M2.7")
    ap.add_argument("--gemini-model", default=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"))
    ap.add_argument("--hydrogen-model", default=os.environ.get("HYDROGEN_MODEL", "gemini-4-hydrogen"))
    ap.add_argument("--neon-model", default=os.environ.get("NEON_MODEL", "gemini-4-neon"))
    ap.add_argument("--frontier-model", default=os.environ.get("FRONTIER_MODEL", "gemini-4-carbon"))
    ap.add_argument("--gemini-thinking-budget", type=int, default=0)
    ap.add_argument("--gemini-rpm", type=float, default=8.0, help="Free-tier friendly requests/minute cap.")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.summary_from:
        try:
            res = json.loads(args.summary_from.read_text(encoding="utf-8"))
            if not isinstance(res, dict) or not isinstance(res.get("summary"), dict):
                raise ValueError("benchmark JSON must contain a summary object")
        except (OSError, ValueError) as exc:
            raise SystemExit(f"Cannot read benchmark summary: {exc}") from exc
    else:
        res = run_benchmark(args)
    if args.summary_from or not args.dry_run:
        if args.format == "markdown":
            print(format_summary_markdown(res["summary"]))
        else:
            print(json.dumps(res["summary"], indent=2))


if __name__ == "__main__":
    main()
