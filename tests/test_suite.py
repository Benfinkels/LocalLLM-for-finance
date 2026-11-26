"""Offline CI unit test suite for when-not-to-llm (no GPU or API keys required)."""

from __future__ import annotations

import json
from pathlib import Path

from when_not_to_llm.datasets import (
    generate_context_cliff_cases,
    generate_doc_contradictions_cases,
    generate_finance_reconcile_cases,
    generate_log_triage_cases,
    generate_note_to_json_cases,
    load_all_cases,
)
from when_not_to_llm.finops import compute_call_cost_usd, sanitize_telemetry
from when_not_to_llm.runner import (
    estimate_dry_run,
    format_markdown_summary,
    load_summary_from_file,
    main as runner_main,
    summarize_results,
)
from when_not_to_llm.tiers import (
    TierResult,
    parse_english_amount,
    reconcile_tables_exact,
    run_deterministic,
    run_hybrid,
)
from when_not_to_llm.verifiers import extract_json_object, score_case


def test_dataset_counts_and_determinism() -> None:
    a = load_all_cases()
    b = load_all_cases()
    assert len(a) == 47  # 5 + 5 + 20 + 5 + 12
    assert [c.prompt for c in a] == [c.prompt for c in b]
    assert [c.ground_truth for c in a] == [c.ground_truth for c in b]


def test_extract_json_strips_think_and_fences() -> None:
    raw = '<think>long reasoning {"fake": 1}</think>\n```json\n{"errors": []}\n```'
    obj, ok = extract_json_object(raw)
    assert ok is True
    assert obj == {"errors": []}


def test_verifiers_score_perfect_ground_truth_as_one() -> None:
    for case in load_all_cases():
        res = score_case(case.task, json.dumps(case.ground_truth), case.ground_truth)
        assert res["json_valid"] is True
        assert res["score"] == 1.0, f"Expected 1.0 on {case.case_id}, got {res}"


def test_deterministic_crushes_structured_logs_and_context_cliff() -> None:
    for case in generate_log_triage_cases() + generate_context_cliff_cases():
        out = run_deterministic(case)
        scored = score_case(case.task, out.raw_output, case.ground_truth)
        assert scored["score"] == 1.0, f"Deterministic failed {case.case_id}: {scored}"
        assert out.cost_usd == 0.0
        assert out.latency_ms < 100.0


def test_finance_reconcile_exact_with_perfect_vendor_map_and_fails_with_naive_map() -> None:
    cases = generate_finance_reconcile_cases()
    naive_scores = []
    for case in cases:
        # With exact vendor map -> deterministic math is 100% cent-accurate
        exact = reconcile_tables_exact(
            case.payload["platform_rows"],
            case.payload["gl_rows"],
            case.ground_truth["vendor_map"],
        )
        assert score_case("finance_reconcile", json.dumps(exact), case.ground_truth)["score"] == 1.0

        # Naive deterministic token-overlap struggles on opaque abbreviations
        det_out = run_deterministic(case)
        naive_scores.append(score_case("finance_reconcile", det_out.raw_output, case.ground_truth)["score"])
    assert min(naive_scores) < 1.0


def test_finance_verifier_penalizes_arithmetic_drift() -> None:
    case = generate_finance_reconcile_cases(n=1)[0]
    bad = json.loads(json.dumps(case.ground_truth))
    bad["net_variance_usd"] = round(bad["net_variance_usd"] + 12.50, 2)
    bad["anomalies"][0]["variance_usd"] = round(bad["anomalies"][0]["variance_usd"] + 5.00, 2)
    res = score_case("finance_reconcile", json.dumps(bad), case.ground_truth)
    assert res["net_variance_exact"] is False
    assert res["score"] < 0.70


def test_hybrid_pipeline_routes_each_task_correctly() -> None:
    def mock_slm(sys_p: str, usr_p: str) -> TierResult:
        if "PLATFORM_CODES" in usr_p:
            # Find matching case vendor_map from our test fixture
            return TierResult(
                raw_output=json.dumps({"mapping": active_vmap}),
                latency_ms=250.0,
                input_tokens=150,
                output_tokens=90,
                cost_usd=0.000001,
                routed_to="gemma4_e2b",
            )
        return TierResult(
            raw_output=json.dumps(active_truth),
            latency_ms=300.0,
            input_tokens=200,
            output_tokens=80,
            cost_usd=0.000002,
            routed_to="gemma4_e2b",
        )

    def mock_cloud(sys_p: str, usr_p: str) -> TierResult:
        return TierResult(
            raw_output=json.dumps(active_truth),
            latency_ms=1400.0,
            input_tokens=5000,
            output_tokens=400,
            cost_usd=0.005,
            routed_to="gemini_flash",
        )

    fin_case = generate_finance_reconcile_cases(n=1)[0]
    active_vmap = fin_case.ground_truth["vendor_map"]
    active_truth = fin_case.ground_truth
    r_fin = run_hybrid(fin_case, mock_slm, mock_cloud)
    assert r_fin.routed_to == "hybrid->gemma4_e2b+python"
    assert score_case("finance_reconcile", r_fin.raw_output, fin_case.ground_truth)["score"] == 1.0

    log_case = generate_log_triage_cases(n=1)[0]
    r_log = run_hybrid(log_case, mock_slm, mock_cloud)
    assert r_log.routed_to == "hybrid->deterministic"
    assert r_log.input_tokens == 0

    note_case = generate_note_to_json_cases(n=1)[0]
    active_truth = note_case.ground_truth
    r_note = run_hybrid(note_case, mock_slm, mock_cloud)
    assert r_note.routed_to == "hybrid->gemma4_e2b"

    doc_case = generate_doc_contradictions_cases(n=1)[0]
    active_truth = doc_case.ground_truth
    r_doc = run_hybrid(doc_case, mock_slm, mock_cloud)
    assert r_doc.routed_to == "hybrid->gemini_flash"


def test_finops_cost_and_sanitizer() -> None:
    c = compute_call_cost_usd("gemini_flash", input_tokens=100_000, output_tokens=20_000, latency_ms=1000.0)
    assert abs(c["api_usd"] - (0.05 + 0.06)) < 1e-6

    dirty = {"host": "192.168.86.42", "ts": "100.112.182.12", "fqdn": "sheldon.tail123.ts.net"}
    clean = sanitize_telemetry(dirty)
    blob = json.dumps(clean)
    assert "192.168." not in blob
    assert "100.112." not in blob
    assert ".ts.net" not in blob


def test_dry_run_and_summarize() -> None:
    cases = load_all_cases()
    est = estimate_dry_run(cases, ["deterministic", "gemini_flash", "hybrid"])
    assert est["deterministic"]["est_api_usd"] == 0.0
    assert est["gemini_flash"]["calls"] == 47
    assert est["hybrid"]["est_input_tokens"] < est["gemini_flash"]["est_input_tokens"]

    summary = summarize_results(
        [
            {
                "tier": "deterministic",
                "task": "log_triage",
                "score": 1.0,
                "latency_ms": 1.2,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
            }
        ]
    )
    assert summary["deterministic"]["macro_core_accuracy"] == 1.0


def test_parse_english_amount_and_hybrid_note_escalation() -> None:
    assert parse_english_amount("renew at four point two grand") == 4200.0
    assert parse_english_amount("paid the half-a-grand setup fee") == 500.0
    assert parse_english_amount("roughly eighteen fifty a month") == 1850.0
    assert parse_english_amount("five and a half grand contract") == 5500.0
    assert parse_english_amount("twelve hundred bucks") == 1200.0
    assert parse_english_amount("free zero-dollar trial") == 0.0

    # Verify Hybrid escalates when local SLM disagrees with deterministic currency parser
    case = generate_note_to_json_cases()[12]  # note_to_json_13 ("half-a-grand" = 500.0)
    bad_slm_obj = dict(case.ground_truth, amount_usd=5000.0)
    slm = lambda s, u: TierResult(json.dumps(bad_slm_obj), 250.0, 150, 50, routed_to="gemma4_e2b")
    cloud = lambda s, u: TierResult(json.dumps(case.ground_truth), 800.0, 150, 50, cost_usd=0.0002, routed_to="gemini_flash")
    res = run_hybrid(case, slm, cloud)
    assert "escalate" in res.routed_to
    assert score_case("note_to_json", res.raw_output, case.ground_truth)["score"] == 1.0


def test_committed_results_have_no_private_ips() -> None:
    root = Path(__file__).resolve().parents[1]
    res_dir = root / "results"
    if res_dir.exists():
        for jf in res_dir.glob("*.json"):
            txt = jf.read_text(encoding="utf-8")
            assert "192.168." not in txt, f"Private LAN IP leaked in {jf}"
            assert "100.112." not in txt, f"Tailscale IP leaked in {jf}"
            assert ".ts.net" not in txt, f"Tailscale hostname leaked in {jf}"

    # Repo-wide secret hygiene check before any public push
    for p in list((root / "src").rglob("*.py")) + list(res_dir.glob("*.json")) + [root / "README.md"]:
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        assert "AIzaSy" not in txt, f"Google API key pattern leaked in {p}"
        assert "sk-ant-" not in txt, f"Anthropic API key pattern leaked in {p}"
        assert "sk-proj-" not in txt, f"OpenAI API key pattern leaked in {p}"


def test_format_markdown_summary_synthetic_and_committed_results() -> None:
    synthetic_summary = summarize_results(
        [
            {
                "tier": "deterministic",
                "task": "log_triage",
                "score": 1.0,
                "latency_ms": 0.25,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
            },
            {
                "tier": "deterministic",
                "task": "finance_reconcile",
                "score": 0.75,
                "latency_ms": 0.40,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
            },
            {
                "tier": "hybrid",
                "task": "log_triage",
                "score": 1.0,
                "latency_ms": 0.40,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
            },
            {
                "tier": "hybrid",
                "task": "finance_reconcile",
                "score": 1.0,
                "latency_ms": 3200.0,
                "input_tokens": 180,
                "output_tokens": 120,
                "cost_usd": 0.000286,
            },
        ]
    )
    md = format_markdown_summary(synthetic_summary)
    lines = md.splitlines()
    assert len(lines) == 4  # header + align + 2 tiers
    assert lines[0] == "| Setup | Log Triage | Finance Reconcile | Overall | Median / p95 | Est. Cost / 10k |"
    assert "Deterministic Python" in lines[2]
    assert "87.5%" in lines[2]
    assert "$0.00" in lines[2]
    assert "Hybrid (Code + Gemma4 + Cloud)" in lines[3]
    assert "100.0%" in lines[3]

    # Smoke test against the committed benchmark_results.json
    root = Path(__file__).resolve().parents[1]
    saved_path = root / "results" / "benchmark_results.json"
    saved_summary = load_summary_from_file(saved_path)
    saved_md = format_markdown_summary(saved_summary)
    saved_lines = saved_md.splitlines()
    assert len(saved_lines) == 2 + len(saved_summary)
    col_count = saved_lines[0].count("|")
    assert all(line.count("|") == col_count for line in saved_lines)
    assert "Deterministic Python" in saved_md
    assert "Hybrid (Code + Gemma4 + Cloud)" in saved_md
    assert "$2.86" in saved_md


def test_cli_summarize_and_markdown_format(capsys, tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    saved_path = root / "results" / "benchmark_results.json"

    runner_main(["--summarize", str(saved_path), "--format", "markdown"])
    out = capsys.readouterr().out
    assert "| Setup |" in out
    assert "Deterministic Python" in out
    assert "$2.86" in out

    # Also works on a records-only JSON file without a precomputed summary
    records_only = tmp_path / "custom_run.json"
    records_only.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "tier": "deterministic",
                        "task": "log_triage",
                        "score": 1.0,
                        "latency_ms": 0.30,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cost_usd": 0.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    runner_main(["--summarize", str(records_only), "--format", "markdown"])
    custom_out = capsys.readouterr().out
    assert "| Setup | Log Triage | Overall | Median / p95 | Est. Cost / 10k |" in custom_out
    assert "Deterministic Python" in custom_out

    # Running a real offline deterministic pass with --format markdown
    run_out = tmp_path / "det_run.json"
    runner_main(
        [
            "--tiers",
            "deterministic",
            "--tasks",
            "log_triage",
            "--output",
            str(run_out),
            "--format",
            "markdown",
        ]
    )
    md_out = capsys.readouterr().out
    assert "| Setup | Log Triage | Overall | Median / p95 | Est. Cost / 10k |" in md_out
    assert "100.0%" in md_out
