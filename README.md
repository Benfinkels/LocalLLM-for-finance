# LocalLLM for Finance & Ops (`LocalLLM-for-finance`)

[![CI](https://github.com/Benfinkels/LocalLLM-for-finance/actions/workflows/ci.yml/badge.svg)](https://github.com/Benfinkels/LocalLLM-for-finance/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Why I Built This

I've been trying to move more of my AI workflows onto local models running on my own GPUs to cut API costs and keep things fast. When I actually tried doing it, I kept running into frustrating errors—models silently dropping details on longer inputs, stalling for 5+ seconds between calls, or botching simple math. At the same time, even strong cloud models were getting basic arithmetic wrong on tables that plain Python handles effortlessly.

Most AI benchmarks only compare one model against another. They don't answer the practical question: **how should regular code, small local models, and cloud APIs actually work together?**

This repo tests **4 everyday tasks** across plain Python, local GPUs (`GTX 1080 8GB`, `RTX 2080 8GB`, `RTX 4090 24GB`), five cloud APIs (`MiniMax-M2.7`, `Gemini 3 Flash`, and the full Gemini 4 lineup: `Hydrogen`, `Neon`, and `Carbon`), and a **hybrid blend** where Python and AI split the job—measuring exact accuracy, speed, and estimated cost per 10,000 runs.

## When You'd Use This Repo

- **Moving work to local GPUs:** Run these tests against your own Ollama setup to see exactly where small local models hold up and where they fall off a cliff.
- **Designing how code and AI split a job:** See a concrete, working pattern for keeping math and structured parsing in Python while using small models only for the messy text parts.
- **Catching local Ollama slowdowns:** Run Track 5B (`infra_cliff.py`) to spot hidden 10x reload penalties (`keep_alive=0` or changing `num_ctx` between calls) before relying on a local model.

---

## What Happened (`470` Live Test Runs Across 10 Setups, Exact Ground-Truth Scoring)

- **All three Gemini 4 models (`Hydrogen`, `Carbon`, `Neon`) hit a perfect `100.0%` (`47/47`)**—at **`3.4–7.5 s` median (`10–35 s` p95)** and **`$120.81–$330.03` per 10,000 runs** (Neon actually ran pricier than Carbon under default reasoning because it spent `70.9k` thinking+output tokens vs. Carbon's `24.9k`).
- **The Hybrid Blend (`Python + 8GB Gemma 4 + Cloud tie-breaker`) matched that exact `100.0%` (`47/47`) at `1.1 s` median (`3.2 s` p95) and `$2.86` per 10,000 runs**—**`42x` cheaper than Hydrogen, `85x` cheaper than Carbon, `115x` cheaper than Neon**, and **`3x–7x` faster** than all three.
- **How Code and Local AI hit `100.0%` together:**
  - *Tables + math:* An 8GB local model (`Gemma 4 E2B`, `~160` tokens, `$0`) maps only the 6 messy vendor names, then Python runs exact math → **`100.0%` in `3.2 s` at `$0` API cost** (vs. `Gemini 3 Flash` missing the net total by `$8k–$16k` on 5/5 runs, and Gemini 4 taking `9.5–32.9 s` median).
  - *Short messy notes:* Plain Python (`76.0%` in `0.03 ms`) and `Gemma 4 E2B` (`97.0%` in `1.1 s`) both run locally for `$0` API cost. When they agree on the amount, category, and follow-up flag (`17/20` notes = `85%`), we keep the local answer; *only* the `3/20` disagreements escalate to Cloud as a tie-breaker → **`100.0%`**.
- **Small 8GB local models are genuinely useful—just keep inputs short.** Once inputs grew past ~1,500–4,000 tokens, recall dropped straight from **`100%` to `0%`**.

![Accuracy Matrix by Task and Compute Tier](docs/charts/task_accuracy_heatmap.png)

---

## Full Results (November 2026 Live Hardware + Cloud Run)

Every test uses seeded synthetic data (`47` test cases per tier) with exact programmatic checks:

| Setup | 1. Noisy Logs (~10k tok) | 2. Table Matching & Math | 3. Messy Note → JSON | 4. Multi-Doc Contradictions | **Overall Average** | Median / p95 Speed | **Est. Cost / 10k Runs** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Plain Python (CPU)** | **100.0%** *(0.3 ms)* | `75.9%` *(0.2 ms)* | `76.0%` *(0.03 ms)* | `81.9%` *(0.3 ms)* | `83.5%` | **0.02 / 0.35 ms** | **`$0.00`** |
| **Phi-4 Mini 3.8B** *(GTX 1080 8GB)* | `0.0%` *(15.7 s)* | `3.2%` *(6.9 s)* | `77.0%` *(2.7 s)* | `6.7%` *(11.5 s)* | `21.7%` | 6.9 / 180.1 s | `$0.36`* |
| **Gemma 4 E2B** *(RTX 2080 8GB)* | `0.0%` *(23.3 s)* | `4.8%` *(22.2 s)* | **`97.0%`** *(1.1 s)* | `56.4%` *(14.1 s)* | `39.6%` | 12.1 / 34.3 s | `$0.92`* |
| **Nemotron 74B** *(RTX 4090 24GB)* | `55.2%` *(109 s, 2 TO)* | `41.3%` *(2.5 s)* | **`99.0%`** *(0.79 s)* | **`97.8%`** *(5.5 s)* | `73.3%` | 2.5 / 109.4 s | `$0.68`* |
| **MiniMax-M2.7** *(Cloud API)* | **100.0%** *(2.8 s)* | `79.8%` *(8.7 s)* | **`99.0%`** *(0.90 s)* | **`97.8%`** *(4.3 s)* | `94.2%` | 1.4 / 8.8 s | `$12.36` |
| **Gemini 3 Flash** *(Cloud API)* | **100.0%** *(1.7 s)* | `75.0%` *(1.3 s)* | **100.0%** *(0.78 s)* | **100.0%** *(1.8 s)* | `93.8%` | **1.1 / 1.8 s** | `$21.51` |
| **Gemini 4 Hydrogen** *(Cloud API)* | **100.0%** *(8.3 s)* | **100.0%** *(9.5 s)* | **100.0%** *(2.5 s)* | **100.0%** *(4.9 s)* | **100.0%** | 3.4 / 10.0 s | `$120.81` |
| **Gemini 4 Carbon** *(Frontier API)* | **100.0%** *(19.9 s)* | **100.0%** *(15.5 s)* | **100.0%** *(3.7 s)* | **100.0%** *(12.5 s)* | **100.0%** | 6.4 / 32.6 s | `$244.43` |
| **Gemini 4 Neon** *(Frontier API)* | **100.0%** *(25.2 s)* | **100.0%** *(32.9 s)* | **100.0%** *(6.0 s)* | **100.0%** *(17.3 s)* | **100.0%** | 7.5 / 35.3 s | `$330.03` |
| **Hybrid Blend** *(Python + Gemma 4 + Cloud)* | **100.0%** *(0.4 ms)* | **100.0%** *(3.2 s)* | **100.0%** *(1.1 s)* | **100.0%** *(1.8 s)* | **100.0%** | **1.1 / 3.2 s** | **`$2.86`** |

*\*Local GPU costs estimate electricity (`$0.15/kWh × GPU power × active time`), with `$0.00` in API fees. Cloud costs apply published standard per-million token rates (`MiniMax-M2.7`: `$0.30/$1.20`; `Gemini 3 Flash`: `$0.50/$3.00`; `Gemini 4 Hydrogen`: `$1.50/$7.50`; `Gemini 4 Neon`: `$3.00/$15.00`; `Gemini 4 Carbon`: `$4.00/$20.00` standard / `$2.00/$10.00` Nov promo) to exact measured token counts.*

![Accuracy vs. Cost and Latency Distribution](docs/charts/finops_pareto.png)

---

## Five Practical Takeaways

### 1. Split Messy Text and Exact Math Between AI and Code
Task 2 (`finance_reconcile`) compares two 18-row spend tables where vendor names don't match cleanly (`"GOOG_YT_BRND"` vs `"Alphabet / YouTube Brand Awareness"`, `"TTK_CRTR_US"` vs `"ByteDance TikTok Creator Marketplace"`) and 3 cent-exact issues are hidden (a dollar mismatch, a duplicate booking, and a missing entry).
- **Plain Python (`75.9%`):** Gets every calculation right, but simple string matching misses unfamiliar abbreviations.
- **Standalone LLMs (`3.2%–100.0%`):** `Nemotron 74B` scored **`41.3%`**. `Gemini 3 Flash` caught all 3 individual issues on every run, yet still missed the overall net dollar difference by **`$8,376` to `$16,247` on 5/5 runs**. Flagship `Gemini 4 Carbon` *did* hit `100.0%`, but needed **`15.5 s` median** and heavy reasoning tokens to do it.
- **Hybrid (`100.0%`):** Sends **only the 6 unique vendor names (`~160` tokens)** to `Gemma 4 E2B` to line them up, then runs Python to add, deduplicate, and compare the numbers. Accuracy jumps from **`4.8%` to `100.0%`** with `$0` in API fees.

### 2. Don't Throw AI at Structured Logs
Task 1 (`log_triage`) hides 5 error types inside `~15 KB` (`~10k` tokens) of noisy service logs, with 2 error types only appearing near the top of the file.
- **Plain Python:** **`100.0%`** in **`0.3 ms`** (`$0.00`).
- **8GB Local Models (`Phi-4 Mini`, `Gemma 4 E2B`):** **`0.0%`** in `15–23 s` because Ollama's default 4k context quietly cut off the top of the file.
- **24GB Local (`Nemotron 74B` on RTX 4090):** **`55.2%`** (`2/5` timed out at `300 s`, `109 s` median).

### 3. Small 8GB Local Models Have a Clear `~1,500`-Token Sweet Spot
- On **Task 3 (`note_to_json`, `<200` tokens)**—pulling names, companies, amounts like *"four point two grand"* or *"half-a-grand"*, and follow-up flags out of casual notes—regex dropped to **`34.0%`**, while **`Gemma 4 E2B` hit `97.0%`** in `1.1 s` for free.
- On **Track 5A (`context_cliff`)**, both `Phi-4 Mini` and `Gemma 4 E2B` hit **`100.0%`** at `500` and `1,500` tokens, then fell straight to **`0.0%`** at `4,000` and `8,000` tokens.

### 4. Two Ollama Defaults Can Quietly Make Local Models `10–11x` Slower
Track 5B (`results/infra_cliff.json`) times two common setup mistakes on `gemma4:e2b-it-q4_K_M` (RTX 2080 8GB):
- **Warm in VRAM (`keep_alive="15m"`):** **`464 ms`** (`1.0x`)
- **Unloading after every call (`OLLAMA_KEEP_ALIVE=0`):** **`5,225 ms`** (**`11.3x` slower**)
- **Changing `num_ctx` between callers (`4096 <-> 6144`):** **`4,807 ms`** (**`10.4x` slower** — forces Ollama to rebuild the KV cache even when `keep_alive="15m"` is set)

![Context-Length Drop-Off and Local Ollama Reload Delays](docs/charts/context_and_infra_cliffs.png)

### 5. Save Cloud APIs (or a 24GB Local GPU) for Longer Reading Tasks
On **Task 4 (`doc_contradictions`, `~3.1k` tokens across 5 memos)**—catching subtle conflicts like *"all-inclusive budget cap"* vs. *"agency fee billed on top of the cap"*—the 8GB local models dropped to `6.7%–56.4%`, while `Nemotron 74B` (`97.8%`), `MiniMax-M2.7` (`97.8%`), and `Gemini 3 Flash` (`100.0%`) handled it reliably.

---

## Simple Rule of Thumb

| What You're Doing | Best Fit | Why |
| :--- | :--- | :--- |
| Searching logs, counting events, checking formats | **Plain Python** | `100%` accurate, `<1 ms`, free, no length limits. |
| Comparing tables with messy names + exact numbers | **Hybrid (Local AI maps names → Python does math)** | Fixes brittle string matching without letting AI touch the arithmetic. |
| Pulling structured fields from short messy notes (`<1,500` tok) | **Warm 8GB Local Model** (`keep_alive="15m"`) | `97%` accurate at `$0` API cost; keep `num_ctx` consistent. |
| Reading across multiple longer docs (`>1,500` tok) | **Cloud API or 24GB+ Local GPU** | Small local models silently lose context here; pay for tokens only when needed. |

---

## Quick Start

### 1. Run the Offline Tests (No GPU or API Keys Needed)
```bash
pip install pytest
PYTHONPATH=src pytest -v
```

### 2. Preview Token & Cost Estimates (`--dry-run`)
```bash
PYTHONPATH=src python -m when_not_to_llm.runner \
  --tiers deterministic gemma4_e2b minimax_m27 gemini_flash hybrid \
  --dry-run
```

### 3. Run Against Your Own Local GPUs or Cloud Keys
```bash
# Plain Python baseline (instant, $0)
PYTHONPATH=src python -m when_not_to_llm.runner --tiers deterministic

# Local Ollama + Cloud APIs (set GEMINI_API_KEY / MINIMAX_API_KEY in env)
PYTHONPATH=src python -m when_not_to_llm.runner \
  --tiers deterministic gemma4_e2b gemini_flash hybrid \
  --gemini-model gemini-3-flash-preview --gemini-rpm 8 \
  --max-cost-usd 1.00 --resume

# Track 5B: Check your Ollama box for keep_alive=0 / num_ctx reload slowdowns
PYTHONPATH=src python -m when_not_to_llm.infra_cliff \
  --base-url http://127.0.0.1:11434 --model gemma4:e2b-it-q4_K_M --trials 4
```

### 4. Regenerate Charts from Saved Results
```bash
pip install matplotlib numpy
python scripts/generate_charts.py
```

All saved results in [`results/benchmark_results.json`](results/benchmark_results.json) and [`results/infra_cliff.json`](results/infra_cliff.json) come from synthetic test data (`src/when_not_to_llm/datasets.py`) and are checked in CI (`tests/test_suite.py`) so no private IPs or keys are committed.
