"""Generate publication-ready PNG charts from committed benchmark JSON results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TIER_ORDER = [
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
]

TIER_LABELS = {
    "deterministic": "Plain Python\n(CPU)",
    "phi4_mini": "Phi-4 Mini\n(GTX 1080 8GB)",
    "gemma4_e2b": "Gemma 4 E2B\n(RTX 2080 8GB)",
    "nemotron_4090": "Nemotron 74B\n(RTX 4090 24GB)",
    "minimax_m27": "MiniMax-M2.7\n(Cloud API)",
    "gemini_flash": "Gemini 3 Flash\n(Cloud API)",
    "gemini_4_hydrogen": "Gemini 4 Hydrogen\n(Cloud API)",
    "gemini_4_neon": "Gemini 4 Neon\n(Frontier API)",
    "gemini_4_carbon": "Gemini 4 Carbon\n(Frontier API)",
    "hybrid": "Hybrid Blend\n(Python+Local+Cloud)",
}

TIER_COLORS = {
    "deterministic": "#10b981",
    "phi4_mini": "#94a3b8",
    "gemma4_e2b": "#3b82f6",
    "nemotron_4090": "#8b5cf6",
    "minimax_m27": "#f59e0b",
    "gemini_flash": "#ef4444",
    "gemini_4_hydrogen": "#14b8a6",
    "gemini_4_neon": "#ec4899",
    "gemini_4_carbon": "#9d174d",
    "hybrid": "#06b6d4",
}

TASK_ORDER = ["log_triage", "finance_reconcile", "note_to_json", "doc_contradictions"]
TASK_LABELS = {
    "log_triage": "1. Noisy Logs\n(~10k tok)",
    "finance_reconcile": "2. Table Matching\n& Dollar Math",
    "note_to_json": "3. Messy Note\n-> JSON",
    "doc_contradictions": "4. Multi-Doc\nContradictions",
}


def _style_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.25)


def _fmt_ms(v: float) -> str:
    if v < 1.0:
        return f"{v:.2f}"
    if v < 10.0:
        return f"{v:.1f}"
    return f"{v:,.0f}"


def plot_task_heatmap(summary: dict[str, Any], out_path: Path) -> None:
    tiers = [t for t in TIER_ORDER if t in summary]
    tasks = [t for t in TASK_ORDER if any(t in summary[tr].get("task_accuracy", {}) for tr in tiers)]
    if not tiers or not tasks:
        return

    mat = np.zeros((len(tiers), len(tasks)))
    for i, tr in enumerate(tiers):
        for j, tk in enumerate(tasks):
            mat[i, j] = summary[tr].get("task_accuracy", {}).get(tk, 0.0) * 100.0

    fig, ax = plt.subplots(figsize=(10.5, max(4.6, 0.85 * len(tiers) + 1.8)), dpi=150)
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([TASK_LABELS[t] for t in tasks], fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(tiers)))
    ax.set_yticklabels([TIER_LABELS.get(t, t).replace("\n", " ") for t in tiers], fontsize=10)

    for i in range(len(tiers)):
        for j in range(len(tasks)):
            val = mat[i, j]
            txt_color = "#111827" if 30 <= val <= 85 else "#ffffff"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color=txt_color, fontsize=11, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Exact Accuracy Score (%)", fontsize=9)
    ax.set_title("When Not to Use an LLM: Accuracy Across Code, Local GPUs, Cloud & Hybrid", fontsize=12.2, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_finops_pareto(summary: dict[str, Any], out_path: Path) -> None:
    tiers = [t for t in TIER_ORDER if t in summary]
    if not tiers:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.4), dpi=150)

    # Custom offsets to guarantee zero label overlap on the log-scale scatter
    offsets = {
        "deterministic": (10, -4),
        "phi4_mini": (10, -4),
        "gemma4_e2b": (10, -4),
        "nemotron_4090": (10, -4),
        "hybrid": (-138, -2),
        "minimax_m27": (-118, -32),
        "gemini_flash": (10, -28),
        "gemini_4_hydrogen": (10, 8),
        "gemini_4_neon": (-135, -36),
        "gemini_4_carbon": (-135, 14),
    }

    for t in tiers:
        s = summary[t]
        cost_10k = max(0.005, float(s["cost_per_10k_tasks_usd"]))
        acc = float(s["macro_core_accuracy"]) * 100.0
        ax1.scatter(
            cost_10k,
            acc,
            s=170,
            color=TIER_COLORS.get(t, "#64748b"),
            edgecolors="#111827",
            linewidth=1.1,
            zorder=4,
        )
        label = TIER_LABELS.get(t, t).split("\n")[0]
        dx, dy = offsets.get(t, (8, 4))
        ax1.annotate(
            f"{label}\n({acc:.1f}%, ${s['cost_per_10k_tasks_usd']:.2f}/10k)",
            (cost_10k, acc),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=8.4,
            fontweight="semibold",
            arrowprops={"arrowstyle": "-", "color": "#64748b", "lw": 0.8} if abs(dx) > 50 else None,
        )

    ax1.set_xscale("log")
    ax1.set_xlim(0.003, max(350.0, max(float(summary[t]["cost_per_10k_tasks_usd"]) for t in tiers) * 14.0))
    ax1.set_ylim(0, 110)
    ax1.set_xlabel("Estimated Total Cost per 10,000 Tasks (USD, log scale; local = GPU electricity)", fontsize=9.2)
    ax1.set_ylabel("Macro Core Accuracy Across 4 Tasks (%)", fontsize=9.5)
    ax1.set_title("FinOps Pareto Frontier: Accuracy vs. Unit Cost", fontsize=11.5, fontweight="bold")
    _style_ax(ax1)

    # Panel B: Median Latency (p50) vs p95 Latency (log scale)
    y_pos = np.arange(len(tiers))
    raw_p50 = [float(summary[t]["p50_latency_ms"]) for t in tiers]
    raw_p95 = [float(summary[t]["p95_latency_ms"]) for t in tiers]
    plot_p50 = [max(0.25, v) for v in raw_p50]
    plot_p95 = [max(0.35, v) for v in raw_p95]
    colors = [TIER_COLORS.get(t, "#64748b") for t in tiers]

    ax2.barh(y_pos - 0.18, plot_p50, height=0.34, color=colors, alpha=0.92, label="p50 Median (ms)")
    ax2.barh(y_pos + 0.18, plot_p95, height=0.34, color=colors, alpha=0.45, hatch="//", edgecolor="#334155", label="p95 Tail (ms)")
    ax2.set_xscale("log")
    ax2.set_xlim(0.15, max(plot_p95) * 12.0)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([TIER_LABELS.get(t, t).split("\n")[0] for t in tiers], fontsize=9.5)
    ax2.invert_yaxis()
    ax2.set_xlabel("Wall-Clock Latency (ms, log scale)", fontsize=9.5)
    ax2.set_title("End-to-End Latency Distribution (p50 vs. p95)", fontsize=11.5, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
    _style_ax(ax2)

    for i, (m50, m95, p95_bar) in enumerate(zip(raw_p50, raw_p95, plot_p95)):
        ax2.text(p95_bar * 1.25, i, f"{_fmt_ms(m50)} / {_fmt_ms(m95)} ms", va="center", fontsize=8.3, color="#1f2937")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_cliffs(summary: dict[str, Any], infra: dict[str, Any] | None, out_path: Path) -> None:
    ncols = 2 if infra else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6.8 * ncols, 4.9), dpi=150)
    ax1 = axes[0] if ncols == 2 else axes

    # Small visual offsets so coincident 100% / 0% lines remain clearly distinguishable
    line_styles = {
        "deterministic": ("-", 2.4, 2.0),
        "nemotron_4090": ("--", 2.0, 1.0),
        "minimax_m27": ("-.", 2.0, 0.0),
        "gemini_flash": (":", 2.2, -1.0),
        "gemini_4_carbon": ("--", 2.2, -2.0),
        "hybrid": ("-", 2.4, 3.0),
        "phi4_mini": ("--", 2.4, -1.2),
        "gemma4_e2b": ("-", 2.4, 1.2),
    }

    for t in TIER_ORDER:
        if t not in summary:
            continue
        curve = summary[t].get("context_cliff_accuracy", {})
        if not curve:
            continue
        ls, lw, dy = line_styles.get(t, ("-", 2.0, 0.0))
        xs = [int(k) for k in curve.keys()]
        ys = [float(v) * 100.0 + dy for v in curve.values()]
        ax1.plot(
            xs,
            ys,
            linestyle=ls,
            marker="o",
            linewidth=lw,
            markersize=5.5,
            color=TIER_COLORS.get(t, "#64748b"),
            label=TIER_LABELS.get(t, t).split("\n")[0],
        )

    ax1.axvline(1500, color="#ef4444", linestyle=":", linewidth=1.5, alpha=0.85)
    ax1.annotate(
        "1,500-tok SLM\nsweet-spot ceiling",
        xy=(1500, 72),
        xytext=(1950, 72),
        fontsize=8.6,
        fontweight="bold",
        color="#b91c1c",
        arrowprops={"arrowstyle": "->", "color": "#b91c1c", "lw": 1.0},
    )
    ax1.annotate(
        "8GB Local SLMs (Phi-4 & Gemma 4):\n100% -> 0% silent truncation cliff",
        xy=(4000, 1.2),
        xytext=(4350, 22),
        fontsize=8.4,
        fontweight="semibold",
        color="#1e40af",
        arrowprops={"arrowstyle": "->", "color": "#1e40af", "lw": 1.0},
    )
    ax1.set_xlabel("Target Prompt Length (Approx. Tokens)", fontsize=9.5)
    ax1.set_ylabel("Exact Incident Recall / F1 (%)", fontsize=9.5)
    ax1.set_xlim(200, 8400)
    ax1.set_ylim(-8, 110)
    ax1.set_title("Track 5A: Recall vs. Input Context Length", fontsize=11.5, fontweight="bold")
    ax1.legend(loc="center right", bbox_to_anchor=(0.98, 0.56), fontsize=8.0, framealpha=0.94, ncol=2)
    _style_ax(ax1)

    if infra and ncols == 2:
        ax2 = axes[1]
        s = infra["summary"]
        labels = ["Warm Resident\n(keep_alive='15m')", "Cold Reload\n(keep_alive=0)"]
        vals = [float(s["warm_median_ms"]), float(s["cold_keep_alive_0_median_ms"])]
        bar_colors = ["#10b981", "#ef4444"]
        if s.get("num_ctx_thrash_median_ms") is not None:
            labels.append("num_ctx Thrash\n(4096 <-> 6144)")
            vals.append(float(s["num_ctx_thrash_median_ms"]))
            bar_colors.append("#f59e0b")

        bars = ax2.bar(labels, vals, color=bar_colors, width=0.52, edgecolor="#111827", linewidth=0.9)
        for b, v in zip(bars, vals):
            mult = v / max(1.0, vals[0])
            tag = f"{v:,.0f} ms\n({mult:.1f}x)" if v != vals[0] else f"{v:,.0f} ms\n(1.0x)"
            ax2.text(b.get_x() + b.get_width() / 2.0, v + max(vals) * 0.025, tag, ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax2.set_ylim(0, max(vals) * 1.24)
        ax2.set_ylabel("Median Wall-Clock Latency (ms)", fontsize=9.5)
        ax2.set_title(f"Track 5B: Local Ollama Reload Cliffs ({infra.get('model', 'SLM')})", fontsize=11.5, fontweight="bold")
        _style_ax(ax2)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render PNG charts from when-not-to-llm benchmark JSON.")
    ap.add_argument("--results", default="results/benchmark_results.json")
    ap.add_argument("--infra", default="results/infra_cliff.json")
    ap.add_argument("--out-dir", default="docs/charts")
    args = ap.parse_args()

    res_path = Path(args.results)
    data = json.loads(res_path.read_text(encoding="utf-8"))
    summary = data["summary"]

    infra_data = None
    infra_path = Path(args.infra)
    if infra_path.exists():
        infra_data = json.loads(infra_path.read_text(encoding="utf-8"))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_task_heatmap(summary, out_dir / "task_accuracy_heatmap.png")
    plot_finops_pareto(summary, out_dir / "finops_pareto.png")
    plot_cliffs(summary, infra_data, out_dir / "context_and_infra_cliffs.png")
    print(f"Rendered charts in {out_dir}")


if __name__ == "__main__":
    main()
