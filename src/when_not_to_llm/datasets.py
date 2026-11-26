"""Seeded synthetic dataset generators for all 5 when-not-to-llm benchmark tracks.

Zero PII, zero proprietary data, 100% deterministic ground truth from fixed seeds.
"""

from __future__ import annotations

import csv
import io
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any


@dataclass(frozen=True)
class BenchmarkCase:
    task: str
    case_id: str
    prompt: str
    system_prompt: str
    ground_truth: dict[str, Any]
    approx_tokens: int
    payload: dict[str, Any] = field(default_factory=dict)


def estimate_tokens(text: str) -> int:
    return max(1, round(len(text) / 4))


# ---------------------------------------------------------------------------
# Task 1: Noisy Service Log Triage (~14-16 KB, 5 injected root causes)
# ---------------------------------------------------------------------------

_LOG_SERVICES = [
    "auth-gateway",
    "billing-sync",
    "edge-proxy",
    "ingest-worker",
    "cache-node",
    "scheduler",
]

_NOISE_TEMPLATES = [
    "INFO  [{svc}] healthcheck status=200 latency_ms={lat} worker={w}",
    "DEBUG [{svc}] pool_stats active={a} idle={i} queue_depth={q}",
    "INFO  [{svc}] batch_flushed records={r} duration_ms={lat}",
    "WARN  [{svc}] slow_query table=events latency_ms={lat} threshold=200",
    "INFO  [{svc}] heartbeat uptime_sec={u} rss_mb={m}",
    "WARN  [{svc}] retry_succeeded attempt=2 upstream=internal-{w}",
    "DEBUG [{svc}] cache_hit ratio=0.9{d} keys={r}",
]

_ERROR_CATALOG = [
    ("auth-gateway", "ERR_JWT_CLOCK_SKEW", "token rejected clock_skew_sec=142"),
    ("billing-sync", "ERR_LEDGER_DEADLOCK", "tx aborted serialization failure"),
    ("edge-proxy", "ERR_TLS_CERT_EXPIRED", "upstream cert expired days_ago=1"),
    ("ingest-worker", "ERR_SCHEMA_DRIFT", "missing required column event_ts"),
    ("cache-node", "ERR_OOM_EVICT_STORM", "memory limit 4096MB exceeded"),
    ("scheduler", "ERR_CRON_LOCK_STALE", "zombie lock held pid=19421"),
    ("billing-sync", "ERR_FX_RATE_TIMEOUT", "ecb feed timeout after 5000ms"),
    ("ingest-worker", "ERR_PARQUET_CORRUPT", "magic bytes mismatch shard=09"),
]


def generate_log_triage_cases(n: int = 5, seed: int = 101) -> list[BenchmarkCase]:
    system_prompt = (
        "You are a log triage assistant. Extract EVERY distinct ERROR or CRITICAL "
        "root-cause signature from the log stream and its exact occurrence count. "
        "Ignore INFO, DEBUG, and WARN lines. Return ONLY a JSON object with schema: "
        '{"errors": [{"service": "<service>", "code": "<ERR_CODE>", "count": <int>}]}'
    )
    cases: list[BenchmarkCase] = []
    for idx in range(n):
        rng = random.Random(seed + idx)
        chosen = rng.sample(_ERROR_CATALOG, 5)
        counts = [rng.randint(2, 6) for _ in chosen]
        total_noise = 215
        lines: list[str] = []
        for t in range(total_noise):
            day = 10 + (t // 35)
            hour = (t * 7) % 24
            minute = (t * 13) % 60
            sec = (t * 19) % 60
            svc = rng.choice(_LOG_SERVICES)
            tmpl = rng.choice(_NOISE_TEMPLATES)
            msg = tmpl.format(
                svc=svc,
                lat=rng.randint(12, 340),
                w=rng.randint(1, 8),
                a=rng.randint(2, 16),
                i=rng.randint(1, 10),
                q=rng.randint(0, 12),
                r=rng.randint(50, 900),
                u=10000 + t * 60,
                m=rng.randint(210, 680),
                d=rng.randint(1, 9),
            )
            lines.append(f"2026-11-{day:02d}T{hour:02d}:{minute:02d}:{sec:02d}Z {msg}")

        # Deliberately place 2 error families exclusively in the first 25% of the log
        # so small-context models that truncate the top of the prompt miss them.
        bands = [
            (4, 38),
            (18, 52),
            (85, 130),
            (145, 188),
            (175, 212),
        ]
        truth_errors = []
        for (svc, code, detail), cnt, (lo, hi) in zip(chosen, counts, bands):
            level = "CRITICAL" if "OOM" in code or "DEADLOCK" in code else "ERROR"
            for _ in range(cnt):
                pos = rng.randint(lo, min(hi, len(lines)))
                ts = f"2026-11-{10 + (pos // 35):02d}T{(pos * 5) % 24:02d}:{(pos * 11) % 60:02d}:{(pos * 17) % 60:02d}Z"
                lines.insert(pos, f"{ts} {level:<8} [{svc}] code={code} {detail}")
            truth_errors.append({"service": svc, "code": code, "count": cnt})

        log_text = "\n".join(lines)
        prompt = (
            "Analyze the following production service log and return all distinct "
            "ERROR/CRITICAL signatures with exact counts as JSON.\n\n"
            f"--- BEGIN LOG ---\n{log_text}\n--- END LOG ---"
        )
        cases.append(
            BenchmarkCase(
                task="log_triage",
                case_id=f"log_triage_{idx + 1:02d}",
                prompt=prompt,
                system_prompt=system_prompt,
                ground_truth={"errors": truth_errors},
                approx_tokens=estimate_tokens(system_prompt + prompt),
                payload={"log_text": log_text},
            )
        )
    return cases


# ---------------------------------------------------------------------------
# Task 2: Marketing / Finance Spend-to-Ledger Reconciliation (Fuzzy + Exact Math)
# ---------------------------------------------------------------------------

_VENDOR_PAIRS = [
    ("GOOG_YT_BRND", "Alphabet / YouTube Brand Awareness"),
    ("META_IG_PRSP", "Meta Platforms - Instagram Prospecting"),
    ("TTK_CRTR_US", "ByteDance TikTok Creator Marketplace"),
    ("LI_ABM_ENT", "LinkedIn Corp ABM Enterprise"),
    ("AMZN_DSP_RET", "Amazon Advertising DSP Retargeting"),
    ("PINT_SHOP_Q4", "Pinterest Shopping Catalog Campaign"),
    ("SNAP_DR_MOB", "Snap Inc Direct Response Mobile"),
    ("REDDIT_CONV", "Reddit Ads Conversion Retargeting"),
]

_PERIODS = ["2026-W42", "2026-W43", "2026-W44"]


def _rows_to_csv(rows: list[dict[str, Any]], fieldnames: list[str]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue().strip()


def generate_finance_reconcile_cases(n: int = 5, seed: int = 201) -> list[BenchmarkCase]:
    system_prompt = (
        "You are a marketing finance reconciliation analyst. Match each row in "
        "MEDIA_PLATFORM_SPEND to GL_LEDGER by canonical vendor + billing period "
        "(note: vendor names differ across systems). Detect all anomalies:\n"
        "- 'variance': same vendor+period exists in both tables but amounts differ\n"
        "- 'duplicate_gl': same vendor+period is booked twice in GL_LEDGER\n"
        "- 'missing_gl': vendor+period exists in MEDIA_PLATFORM_SPEND but is absent from GL_LEDGER\n"
        "Compute variance_usd = round(platform_usd - gl_total_usd_for_that_vendor_period, 2) "
        "and overall net_variance_usd = round(sum(platform_usd) - sum(gl_usd), 2).\n"
        "Return ONLY JSON matching:\n"
        '{"net_variance_usd": <float>, "anomalies": [{"platform_code": "<CODE>", '
        '"period": "<YYYY-W##>", "type": "variance|duplicate_gl|missing_gl", '
        '"variance_usd": <float>}]}'
    )
    cases: list[BenchmarkCase] = []
    for idx in range(n):
        rng = random.Random(seed + idx)
        vendors = rng.sample(_VENDOR_PAIRS, 6)
        platform_rows: list[dict[str, Any]] = []
        gl_rows: list[dict[str, Any]] = []
        gl_seq = 8001 + idx * 100

        # Pick 3 distinct (vendor_idx, period_idx) slots for the 3 anomaly types
        all_slots = [(vi, pi) for vi in range(len(vendors)) for pi in range(len(_PERIODS))]
        rng.shuffle(all_slots)
        var_slot, dup_slot, miss_slot = all_slots[0], all_slots[1], all_slots[2]

        anomalies: list[dict[str, Any]] = []
        total_plat = 0.0
        total_gl = 0.0

        for vi, (pcode, gl_name) in enumerate(vendors):
            for pi, period in enumerate(_PERIODS):
                amt = round(rng.uniform(2400.0, 18500.0), 2)
                platform_rows.append(
                    {"platform_code": pcode, "period": period, "spend_usd": f"{amt:.2f}"}
                )
                total_plat += amt

                slot = (vi, pi)
                if slot == var_slot:
                    delta = round(rng.choice([-1, 1]) * rng.uniform(115.13, 845.89), 2)
                    gl_amt = round(amt - delta, 2)
                    gl_rows.append(
                        {
                            "voucher_id": f"GL-{gl_seq}",
                            "gl_vendor": gl_name,
                            "period": period,
                            "booked_usd": f"{gl_amt:.2f}",
                        }
                    )
                    gl_seq += 1
                    total_gl += gl_amt
                    anomalies.append(
                        {
                            "platform_code": pcode,
                            "period": period,
                            "type": "variance",
                            "variance_usd": round(amt - gl_amt, 2),
                        }
                    )
                elif slot == dup_slot:
                    for _ in range(2):
                        gl_rows.append(
                            {
                                "voucher_id": f"GL-{gl_seq}",
                                "gl_vendor": gl_name,
                                "period": period,
                                "booked_usd": f"{amt:.2f}",
                            }
                        )
                        gl_seq += 1
                        total_gl += amt
                    anomalies.append(
                        {
                            "platform_code": pcode,
                            "period": period,
                            "type": "duplicate_gl",
                            "variance_usd": round(-amt, 2),
                        }
                    )
                elif slot == miss_slot:
                    anomalies.append(
                        {
                            "platform_code": pcode,
                            "period": period,
                            "type": "missing_gl",
                            "variance_usd": round(amt, 2),
                        }
                    )
                else:
                    gl_rows.append(
                        {
                            "voucher_id": f"GL-{gl_seq}",
                            "gl_vendor": gl_name,
                            "period": period,
                            "booked_usd": f"{amt:.2f}",
                        }
                    )
                    gl_seq += 1
                    total_gl += amt

        rng.shuffle(platform_rows)
        rng.shuffle(gl_rows)
        plat_csv = _rows_to_csv(platform_rows, ["platform_code", "period", "spend_usd"])
        gl_csv = _rows_to_csv(gl_rows, ["voucher_id", "gl_vendor", "period", "booked_usd"])
        net_var = round(total_plat - total_gl, 2)

        prompt = (
            "Reconcile MEDIA_PLATFORM_SPEND against GL_LEDGER and return the exact "
            "net variance and all anomalies as JSON.\n\n"
            f"MEDIA_PLATFORM_SPEND:\n{plat_csv}\n\n"
            f"GL_LEDGER:\n{gl_csv}"
        )
        cases.append(
            BenchmarkCase(
                task="finance_reconcile",
                case_id=f"finance_reconcile_{idx + 1:02d}",
                prompt=prompt,
                system_prompt=system_prompt,
                ground_truth={
                    "net_variance_usd": net_var,
                    "total_platform_usd": round(total_plat, 2),
                    "total_gl_usd": round(total_gl, 2),
                    "anomalies": anomalies,
                    "vendor_map": {p: g for p, g in vendors},
                },
                approx_tokens=estimate_tokens(system_prompt + prompt),
                payload={
                    "platform_rows": platform_rows,
                    "gl_rows": gl_rows,
                },
            )
        )
    return cases


# ---------------------------------------------------------------------------
# Task 3: Short Unstructured Note -> Strict JSON (20 cases, <350 tokens)
# ---------------------------------------------------------------------------

_NOTE_FIXTURES: list[tuple[str, dict[str, Any]]] = [
    (
        "Follow-up from Q4 Sync: Northstar's buyer Maya Lin wants to renew their annual plan at four point two grand, ball's in our court to send the order form before Friday.",
        {"contact": "Maya Lin", "company": "Northstar", "amount_usd": 4200.0, "category": "renewal", "requires_followup": True},
    ),
    (
        "Billing Audit Log: Over at Cobalt Health Group, Devon Brooks confirmed the twelve hundred bucks overcharge was credited and wrapped up. Ball's not in our court.",
        {"contact": "Devon Brooks", "company": "Cobalt Health Group", "amount_usd": 1200.0, "category": "invoice_dispute", "requires_followup": False},
    ),
    (
        "Enterprise Analytics Inquiry — Harborline Retail (contact: Priya Nair) asked to move up to the pro tier, roughly eighteen fifty a month. Send revised quote Monday.",
        {"contact": "Priya Nair", "company": "Harborline Retail", "amount_usd": 1850.0, "category": "upsell", "requires_followup": True},
    ),
    (
        "End-of-Month Churn Notice: Redrock (Tom Alvarez) is winding down their nine hundred fifty dollar pilot. Billing already stopped the recurring charge; nothing left on our side.",
        {"contact": "Tom Alvarez", "company": "Redrock", "amount_usd": 950.0, "category": "cancellation", "requires_followup": False},
    ),
    (
        "Sandbox Kickoff Call: Hannah Okafor with Pinecrest Supply Chain Solutions is starting a free zero-dollar trial today — I owe her the API credentials doc tomorrow morning.",
        {"contact": "Hannah Okafor", "company": "Pinecrest Supply Chain Solutions", "amount_usd": 0.0, "category": "onboarding", "requires_followup": True},
    ),
    (
        "October Billing Pushback: Carlos Mendez from Veridian says we double-billed $3,475.50 for media fees. Finance owes him a ledger reconciliation reply.",
        {"contact": "Carlos Mendez", "company": "Veridian", "amount_usd": 3475.5, "category": "invoice_dispute", "requires_followup": True},
    ),
    (
        "Signed Paperwork Filed: Apex Global Robotics (Elena Rostova) countersigned their multi-year extension at twenty-five grand flat. Archive only — do not reach out.",
        {"contact": "Elena Rostova", "company": "Apex Global Robotics", "amount_usd": 25000.0, "category": "renewal", "requires_followup": False},
    ),
    (
        "Seat Expansion Ask: Summit Gear's Jordan Patel wants 15 extra seats (~two thousand four hundred/yr) pending Okta SSO confirmation from our engineering team.",
        {"contact": "Jordan Patel", "company": "Summit Gear", "amount_usd": 2400.0, "category": "upsell", "requires_followup": True},
    ),
    (
        "Prod Go-Live Complete: Brightleaf (Rachel Kim) finished their complimentary zero-cost pipeline setup today. Green across the board; log for record only.",
        {"contact": "Rachel Kim", "company": "Brightleaf", "amount_usd": 0.0, "category": "onboarding", "requires_followup": False},
    ),
    (
        "At-Risk Alert: Ironclad Cyber Defense (Marcus Vance) emailed that they'll terminate their six thousand eight hundred dollar quarterly package unless we send a concession by Wednesday.",
        {"contact": "Marcus Vance", "company": "Ironclad Cyber Defense", "amount_usd": 6800.0, "category": "cancellation", "requires_followup": True},
    ),
    (
        "Executive Call Recap: Fjord Energy CFO Sofia Lindqvist agreed verbally to extend for fourteen and a half thousand; waiting on our legal team's redlines.",
        {"contact": "Sofia Lindqvist", "company": "Fjord Energy", "amount_usd": 14500.0, "category": "renewal", "requires_followup": True},
    ),
    (
        "Q3 Co-Op Variance: Cascade Craft Brewing's Derek Cho flagged a seven hundred fifty dollar mismatch on last month's statement — accounting still needs to respond.",
        {"contact": "Derek Cho", "company": "Cascade Craft Brewing", "amount_usd": 750.0, "category": "invoice_dispute", "requires_followup": True},
    ),
    (
        "Workspace Ready: Clearview (Nina Morales) paid the half-a-grand setup fee in full and completed provisioning. Completely wrapped up.",
        {"contact": "Nina Morales", "company": "Clearview", "amount_usd": 500.0, "category": "onboarding", "requires_followup": False},
    ),
    (
        "Regional Expansion Closed: Silverline Transit Systems (Owen Gallagher) signed an order adding eleven thousand two hundred in ACV. Countersigned and booked — no follow-up.",
        {"contact": "Owen Gallagher", "company": "Silverline Transit Systems", "amount_usd": 11200.0, "category": "upsell", "requires_followup": False},
    ),
    (
        "Account Deactivated: Tessa Morales over at Bluebird called to end their three hundred twenty dollar starter tier immediately. De-provisioned; wait for them to re-engage.",
        {"contact": "Tessa Morales", "company": "Bluebird", "amount_usd": 320.0, "category": "cancellation", "requires_followup": False},
    ),
    (
        "License Rollover Sync: Meridian BioLabs (Vikram Shah) wants to keep their $8.75k annual license going if we approve Net-60 terms — finance sign-off pending.",
        {"contact": "Vikram Shah", "company": "Meridian BioLabs", "amount_usd": 8750.0, "category": "renewal", "requires_followup": True},
    ),
    (
        "Late Fee Waived: Gabe Foster at Oakridge questioned a $1,625.25 penalty on bill #442; I already reversed it and sent him the updated receipt.",
        {"contact": "Gabe Foster", "company": "Oakridge", "amount_usd": 1625.25, "category": "invoice_dispute", "requires_followup": False},
    ),
    (
        "Implementation Schedule: Lumen Apparel Co (Claire Beaumont) booked their three grand kickoff package — send her the pre-work checklist this afternoon.",
        {"contact": "Claire Beaumont", "company": "Lumen Apparel Co", "amount_usd": 3000.0, "category": "onboarding", "requires_followup": True},
    ),
    (
        "Dedicated GPU Upgrade: Samira Haddad from Horizon wants to step up to the nine thousand dollar annual GPU add-on once we return their security questionnaire.",
        {"contact": "Samira Haddad", "company": "Horizon", "amount_usd": 9000.0, "category": "upsell", "requires_followup": True},
    ),
    (
        "Outage Escalation: Keystone Industrial Supply (Benny Ortiz) is threatening to walk away from their five and a half grand contract unless CS calls him tomorrow morning.",
        {"contact": "Benny Ortiz", "company": "Keystone Industrial Supply", "amount_usd": 5500.0, "category": "cancellation", "requires_followup": True},
    ),
]


def generate_note_to_json_cases(n: int = 20) -> list[BenchmarkCase]:
    system_prompt = (
        "Extract structured CRM fields from a short unstructured note. "
        "Allowed categories: renewal, invoice_dispute, onboarding, cancellation, upsell. "
        "Convert informal amounts (e.g. '$4.2k', 'twelve hundred bucks') to numeric USD float. "
        "Return ONLY JSON matching:\n"
        '{"contact": "<Full Name>", "company": "<Company Name>", "amount_usd": <float>, '
        '"category": "<category>", "requires_followup": <true|false>}'
    )
    cases: list[BenchmarkCase] = []
    for idx, (note, truth) in enumerate(_NOTE_FIXTURES[:n]):
        prompt = f"Extract the structured CRM record from this note:\n\n{note}"
        cases.append(
            BenchmarkCase(
                task="note_to_json",
                case_id=f"note_to_json_{idx + 1:02d}",
                prompt=prompt,
                system_prompt=system_prompt,
                ground_truth=dict(truth),
                approx_tokens=estimate_tokens(system_prompt + prompt),
                payload={"note": note},
            )
        )
    return cases


# ---------------------------------------------------------------------------
# Task 4: Multi-Document Contradiction Synthesis (~4.5-5.5k tokens, 5 memos)
# ---------------------------------------------------------------------------

_CONTRADICTION_BANK = [
    (
        "budget_scope",
        "The approved Q4 marketing budget cap of $180,000 is strictly all-inclusive and covers the 15% agency retainer fee.",
        "Finance confirmed the $180,000 Q4 media allocation covers direct platform spend only; the 15% agency retainer is billed separately above the cap.",
        "Weekly finance sync reconfirmed that Q4 media spend plus the 15% agency retainer fits inside the approved $180,000 all-inclusive budget ceiling.",
    ),
    (
        "launch_date",
        "All engineering and legal sign-offs are locked for a hard public launch on December 4, 2026 with zero remaining blockers.",
        "Pending third-party penetration testing remediation, the public release has been pushed from December 4 to January 19, 2027.",
        "Release engineering re-verified readiness milestones for the scheduled December 4, 2026 public go-live with no open blockers.",
    ),
    (
        "compliance_status",
        "External auditors closed the SOC 2 Type II review with zero exceptions and issued a clean attestation letter.",
        "Our SOC 2 Type II certification remains on hold due to two unresolved privileged-access logging exceptions flagged by the auditor.",
        "Security leadership distributed the clean SOC 2 Type II attestation report confirming zero audit exceptions for enterprise buyers.",
    ),
    (
        "sla_target",
        "Enterprise customers on the new contract receive a 99.99% uptime SLA backed by automatic 25% monthly service credits.",
        "Under the finalized legal template, Enterprise uptime commitments are capped at 99.9% with service credits limited to 10%.",
        "Customer success updated enterprise onboarding decks to reflect the 99.99% uptime SLA and automatic 25% credit tier.",
    ),
    (
        "headcount_cap",
        "Leadership froze incremental hiring for the analytics pod through Q1, keeping team headcount fixed at 6 FTEs.",
        "Recruiting opened two net-new senior analytics engineer requisitions for immediate Q4 backfill, expanding the pod to 8 FTEs.",
        "People Ops confirmed the analytics pod remains steady at 6 FTEs under the Q1 hiring freeze with no open incremental requisitions.",
    ),
    (
        "pricing_tier",
        "Self-serve annual plans will hold flat at $299/month through 2027 with grandfathering for all active subscribers.",
        "Effective January 1, all self-serve subscribers—including legacy accounts—will migrate to the revised $399/month pricing schedule.",
        "Billing operations verified that active self-serve subscribers remain grandfathered at $299/month through the 2027 renewal cycle.",
    ),
]

_FILLER_PARAGRAPHS = [
    "Weekly telemetry across the core attribution warehouse remained stable with median batch ingestion finishing 14 minutes ahead of the 06:00 EST reporting cutoff. Data quality monitors recorded zero null primary keys across 1.8 million daily session records, and downstream BI dashboards refreshed without cache timeouts.",
    "During Tuesday's cross-functional standup, regional channel managers reviewed creative refresh velocity across paid search, connected TV, and affiliate placements. Asset turnaround averaged 3.2 business days following the migration to shared templates, and brand-safety exclusion lists were synchronized across all active buying desks.",
    "Infrastructure operations completed routine kernel patching on staging nodes and validated automated snapshot restore procedures in us-central1. Cold-start restoration met the internal 12-minute recovery objective, and quarterly disaster-recovery runbooks were updated with current on-call escalation contacts.",
    "Procurement finalized standard hardware lease renewals for analyst workstations and retired seven decommissioned test laptops through certified e-waste disposal. No capital expenditure variance was recorded against the annual IT equipment forecast.",
    "Customer advisory board participants highlighted clearer executive summary exports and automated variance commentary as their most requested workflow improvements. Product design scheduled three usability interviews next sprint to test wireframe layouts prior to engineering estimation.",
    "Legal operations archived signed mutual NDAs for four prospective data-enrichment partners and refreshed the vendor sub-processor registry on the public trust portal. Standard data-retention schedules remain 24 months for raw clickstream logs and 7 years for audited financial ledgers.",
]


def generate_doc_contradictions_cases(n: int = 5, seed: int = 401) -> list[BenchmarkCase]:
    system_prompt = (
        "You are a senior audit analyst reviewing 5 internal project memos (M1..M5). "
        "Identify EVERY direct factual contradiction between two memos. "
        "Use ONLY these canonical topic IDs: "
        "budget_scope, launch_date, compliance_status, sla_target, headcount_cap, pricing_tier. "
        "Return ONLY JSON matching:\n"
        '{"contradictions": [{"topic": "<topic_id>", "memos": ["M1", "M4"], "summary": "<1-line>"}]}'
    )
    cases: list[BenchmarkCase] = []
    # Fixed distant memo pairings so contradictions require full-document context
    contradict_pairings = [("M1", "M4"), ("M2", "M5"), ("M1", "M5"), ("M2", "M4")]
    consistent_pairings = [("M1", "M3"), ("M3", "M5")]
    memo_ids = ["M1", "M2", "M3", "M4", "M5"]

    for idx in range(n):
        rng = random.Random(seed + idx)
        shuffled = list(_CONTRADICTION_BANK)
        rng.shuffle(shuffled)
        chosen_contradict = shuffled[:4]
        chosen_consistent = shuffled[4:]

        memo_claims: dict[str, list[str]] = {m: [] for m in memo_ids}
        truth_items = []
        for (topic, claim_a, claim_b, _cons), (ma, mb) in zip(chosen_contradict, contradict_pairings):
            memo_claims[ma].append(claim_a)
            memo_claims[mb].append(claim_b)
            truth_items.append({"topic": topic, "memos": sorted([ma, mb])})

        # Inject consistent (non-contradictory) statements for the remaining 2 topics
        # so naive keyword co-occurrence cannot distinguish agreement from contradiction.
        for (_topic, claim_a, _claim_b, consistent_restate), (ma, mb) in zip(chosen_consistent, consistent_pairings):
            memo_claims[ma].append(claim_a)
            memo_claims[mb].append(consistent_restate)

        memo_blocks = []
        for mi, mid in enumerate(memo_ids):
            paras = [rng.choice(_FILLER_PARAGRAPHS) for _ in range(10)]
            for claim in memo_claims[mid]:
                insert_at = rng.randint(1, len(paras) - 1)
                paras.insert(insert_at, claim)
            body = "\n\n".join(paras)
            memo_blocks.append(f"=== MEMO {mid} (Week {40 + mi}) ===\n{body}")

        full_bundle = "\n\n".join(memo_blocks)
        prompt = (
            "Read all 5 memos carefully and list every direct factual contradiction "
            "between pairs of memos as JSON.\n\n"
            f"{full_bundle}"
        )
        cases.append(
            BenchmarkCase(
                task="doc_contradictions",
                case_id=f"doc_contradictions_{idx + 1:02d}",
                prompt=prompt,
                system_prompt=system_prompt,
                ground_truth={"contradictions": truth_items},
                approx_tokens=estimate_tokens(system_prompt + prompt),
                payload={"memos_text": full_bundle},
            )
        )
    return cases


# ---------------------------------------------------------------------------
# Track 5A: Context-Length Cliff (500 / 1,500 / 4,000 / 8,000 tokens)
# ---------------------------------------------------------------------------

def generate_context_cliff_cases(
    token_targets: tuple[int, ...] = (500, 1500, 4000, 8000),
    trials_per_length: int = 3,
    seed: int = 501,
) -> list[BenchmarkCase]:
    system_prompt = (
        "Extract ALL exact INCIDENT_ALERT records from the operational digest below. "
        "Return ONLY JSON matching:\n"
        '{"incidents": [{"id": "INC-XXXX", "owner": "<team>", "impact_usd": <int>}]}'
    )
    teams = ["payments", "ad-serving", "data-eng", "checkout", "infra-core", "fraud-ops"]
    cases: list[BenchmarkCase] = []
    counter = 0
    for target_toks in token_targets:
        for t in range(trials_per_length):
            counter += 1
            rng = random.Random(seed + counter * 17)
            incidents = []
            for j in range(4):
                inc_id = f"INC-{rng.randint(1000, 9999)}"
                while any(x["id"] == inc_id for x in incidents):
                    inc_id = f"INC-{rng.randint(1000, 9999)}"
                incidents.append(
                    {
                        "id": inc_id,
                        "owner": rng.choice(teams),
                        "impact_usd": rng.randint(1200, 48000),
                    }
                )

            # Build filler lines to hit target_toks (~4 chars/token)
            target_chars = max(1200, target_toks * 4 - 500)
            filler_lines: list[str] = []
            cur_chars = 0
            seq = 0
            while cur_chars < target_chars:
                seq += 1
                line = (
                    f"[DIGEST-{seq:04d}] svc=telemetry-node-{seq % 12:02d} "
                    f"status=OK throughput_rps={rng.randint(200, 2400)} "
                    f"p95_ms={rng.randint(14, 98)} err_rate=0.000{rng.randint(0, 9)} "
                    f"region=us-east-{(seq % 3) + 1}"
                )
                filler_lines.append(line)
                cur_chars += len(line) + 1

            # Inject 2 incidents early (10%, 22%) and 2 late (55%, 88%)
            fracs = [0.10, 0.22, 0.55, 0.88]
            for inc, frac in zip(incidents, fracs):
                pos = max(0, min(len(filler_lines), int(len(filler_lines) * frac)))
                alert = (
                    f"INCIDENT_ALERT id={inc['id']} owner={inc['owner']} "
                    f"impact_usd={inc['impact_usd']} severity=SEV1"
                )
                filler_lines.insert(pos, alert)

            digest = "\n".join(filler_lines)
            prompt = (
                "Extract every INCIDENT_ALERT record (id, owner, impact_usd) as JSON.\n\n"
                f"{digest}"
            )
            cases.append(
                BenchmarkCase(
                    task="context_cliff",
                    case_id=f"context_cliff_{target_toks}tok_{t + 1:02d}",
                    prompt=prompt,
                    system_prompt=system_prompt,
                    ground_truth={"incidents": incidents, "target_tokens": target_toks},
                    approx_tokens=estimate_tokens(system_prompt + prompt),
                    payload={"digest": digest, "target_tokens": target_toks},
                )
            )
    return cases


# ---------------------------------------------------------------------------
# Track 6: Messy Bank-Statement CSV Categorization & Anomaly Detection
# ---------------------------------------------------------------------------

BANK_STATEMENT_CATEGORIES = (
    "cloud_infra",
    "saas_tools",
    "travel_lodging",
    "meals_dining",
    "office_supplies",
    "shipping_logistics",
)

# Mix of plain-English descriptors (obvious to keyword rules) and abbreviated
# card-processor strings (where naive regex fails and a small local SLM shines).
_BANK_MERCHANT_CATALOG: list[tuple[str, str, bool, tuple[float, float]]] = [
    # (description, category, is_obvious, (min_amt, max_amt))
    ("AWS CLOUD SERVICES US-EAST", "cloud_infra", True, (180.0, 950.0)),
    ("DIGITALOCEAN CLOUD HOSTING", "cloud_infra", True, (45.0, 240.0)),
    ("CF*FLARE PRO PLAN CA 8821", "cloud_infra", False, (25.0, 210.0)),
    ("HETZNER*ONLINE DE 9912", "cloud_infra", False, (38.0, 195.0)),
    ("SLACK TECHNOLOGIES SUB", "saas_tools", True, (65.0, 320.0)),
    ("GITHUB ENTERPRISE BILL", "saas_tools", True, (84.0, 420.0)),
    ("SQSP*WORKSPACE INV441", "saas_tools", False, (28.0, 144.0)),
    ("FGM*FIGMA ORG SEAT SF", "saas_tools", False, (45.0, 225.0)),
    ("LIN*LINEAR APP SANFRAN", "saas_tools", False, (32.0, 160.0)),
    ("DELTA AIR LINES TICKET", "travel_lodging", True, (260.0, 780.0)),
    ("MARRIOTT HOTEL STAY", "travel_lodging", True, (190.0, 620.0)),
    ("UAL*0162418922 CHI", "travel_lodging", False, (240.0, 690.0)),
    ("HTL*COURTYARD BOS 04", "travel_lodging", False, (175.0, 480.0)),
    ("STARBUCKS COFFEE #441", "meals_dining", True, (12.0, 48.0)),
    ("PANERA BREAD CATERING", "meals_dining", True, (68.0, 215.0)),
    ("TST*BLUEBOTTLE NYC 09", "meals_dining", False, (14.0, 52.0)),
    ("SQ *SWEETGREEN 218", "meals_dining", False, (18.0, 64.0)),
    ("STAPLES OFFICE SUPPLY", "office_supplies", True, (35.0, 185.0)),
    ("OFFICE DEPOT ORDER", "office_supplies", True, (42.0, 210.0)),
    ("ULN*ULINE BOX/PACK WI", "office_supplies", False, (85.0, 340.0)),
    ("BLC*BLICK ART MATS 02", "office_supplies", False, (29.0, 135.0)),
    ("FEDEX EXPRESS SHIPPING", "shipping_logistics", True, (28.0, 175.0)),
    ("UPS GROUND FREIGHT", "shipping_logistics", True, (45.0, 260.0)),
    ("PB*PITNEY BOWES POST", "shipping_logistics", False, (34.0, 145.0)),
    ("STMP*ENDICIA LABEL 71", "shipping_logistics", False, (22.0, 118.0)),
]

_REFUND_PREFIXES = ("REFUND ", "CR ", "REV* ")


def generate_bank_statement_cases(n: int = 5, seed: int = 601) -> list[BenchmarkCase]:
    system_prompt = (
        "You are a corporate accounting analyst reviewing a bank-statement CSV "
        "(negative amount_usd = charge, positive amount_usd = refund/credit). "
        "1. Categorize EVERY transaction into one of these exact categories: "
        "cloud_infra, saas_tools, travel_lodging, meals_dining, office_supplies, shipping_logistics. "
        "2. Detect all anomalies:\n"
        "- 'duplicate_charge': two negative charges for the same merchant and exact same amount within 2 days\n"
        "- 'refund_reversal': a negative charge paired with a matching positive credit of the same magnitude for the same merchant\n"
        "3. Compute net_spend_usd = round(-sum(amount_usd across all rows), 2).\n"
        "Return ONLY JSON matching:\n"
        '{"categories": [{"txn_id": "TXN-XXXX", "category": "<category>"}], '
        '"anomalies": [{"type": "duplicate_charge|refund_reversal", "txn_ids": ["TXN-A", "TXN-B"], "amount_usd": <positive_float>}], '
        '"net_spend_usd": <float>}'
    )
    obvious_pool = [m for m in _BANK_MERCHANT_CATALOG if m[2]]
    opaque_pool = [m for m in _BANK_MERCHANT_CATALOG if not m[2]]
    cases: list[BenchmarkCase] = []

    for idx in range(n):
        rng = random.Random(seed + idx)
        # 5 obvious + 5 opaque merchants so deterministic rules score partial credit
        # while a small local SLM can push categorization to 100%.
        chosen = rng.sample(obvious_pool, 5) + rng.sample(opaque_pool, 5)
        rng.shuffle(chosen)

        base_day = date(2026, 10, 1) + timedelta(days=idx * 7)
        txn_seq = 6001 + idx * 100
        used_amounts: set[float] = set()
        raw_entries: list[dict[str, Any]] = []

        for offset, (desc, cat, _is_obv, (lo, hi)) in enumerate(chosen):
            amt = round(rng.uniform(lo, hi), 2)
            while amt in used_amounts:
                amt = round(amt + 0.37, 2)
            used_amounts.add(amt)
            txn_date = base_day + timedelta(days=offset * 2 + rng.randint(0, 1))
            raw_entries.append(
                {
                    "date_obj": txn_date,
                    "description": desc,
                    "amount": -amt,
                    "category": cat,
                }
            )

        # Pick 2 distinct base entries: one gets a duplicate charge, one gets a refund reversal.
        dup_src_idx, ref_src_idx = rng.sample(range(len(raw_entries)), 2)
        dup_src = raw_entries[dup_src_idx]
        ref_src = raw_entries[ref_src_idx]

        dup_entry = {
            "date_obj": dup_src["date_obj"] + timedelta(days=rng.randint(0, 1)),
            "description": dup_src["description"],
            "amount": dup_src["amount"],
            "category": dup_src["category"],
            "pair_kind": "duplicate_charge",
            "pair_src": dup_src_idx,
        }
        ref_prefix = rng.choice(_REFUND_PREFIXES)
        ref_entry = {
            "date_obj": ref_src["date_obj"] + timedelta(days=rng.randint(1, 3)),
            "description": f"{ref_prefix}{ref_src['description']}",
            "amount": round(-ref_src["amount"], 2),
            "category": ref_src["category"],
            "pair_kind": "refund_reversal",
            "pair_src": ref_src_idx,
        }

        all_entries = [
            dict(e, orig_idx=i) for i, e in enumerate(raw_entries)
        ] + [dup_entry, ref_entry]
        all_entries.sort(key=lambda e: (e["date_obj"], e["description"], e["amount"]))

        rows: list[dict[str, Any]] = []
        truth_categories: list[dict[str, str]] = []
        orig_to_txn_id: dict[int, str] = {}
        dup_txn_id = ""
        ref_txn_id = ""
        category_map: dict[str, str] = {}

        for entry in all_entries:
            tid = f"TXN-{txn_seq}"
            txn_seq += 1
            amt_val = round(float(entry["amount"]), 2)
            rows.append(
                {
                    "txn_id": tid,
                    "date": entry["date_obj"].isoformat(),
                    "description": entry["description"],
                    "amount_usd": f"{amt_val:.2f}",
                }
            )
            truth_categories.append({"txn_id": tid, "category": entry["category"]})
            category_map[entry["description"]] = entry["category"]
            if "orig_idx" in entry:
                orig_to_txn_id[entry["orig_idx"]] = tid
            elif entry.get("pair_kind") == "duplicate_charge":
                dup_txn_id = tid
            elif entry.get("pair_kind") == "refund_reversal":
                ref_txn_id = tid

        truth_anomalies = [
            {
                "type": "duplicate_charge",
                "txn_ids": sorted([orig_to_txn_id[dup_src_idx], dup_txn_id]),
                "amount_usd": round(abs(float(dup_src["amount"])), 2),
            },
            {
                "type": "refund_reversal",
                "txn_ids": sorted([orig_to_txn_id[ref_src_idx], ref_txn_id]),
                "amount_usd": round(abs(float(ref_src["amount"])), 2),
            },
        ]
        net_spend = round(-sum(float(r["amount_usd"]) for r in rows), 2)
        stmt_csv = _rows_to_csv(rows, ["txn_id", "date", "description", "amount_usd"])

        prompt = (
            "Categorize each bank-statement line, detect any duplicate charges or "
            "refund reversals, and compute exact net_spend_usd as JSON.\n\n"
            f"BANK_STATEMENT:\n{stmt_csv}"
        )
        cases.append(
            BenchmarkCase(
                task="bank_statement",
                case_id=f"bank_statement_{idx + 1:02d}",
                prompt=prompt,
                system_prompt=system_prompt,
                ground_truth={
                    "categories": truth_categories,
                    "anomalies": truth_anomalies,
                    "net_spend_usd": net_spend,
                    "category_map": category_map,
                },
                approx_tokens=estimate_tokens(system_prompt + prompt),
                payload={"rows": rows},
            )
        )
    return cases


_DEFAULT_TASKS = (
    "log_triage",
    "finance_reconcile",
    "note_to_json",
    "doc_contradictions",
    "context_cliff",
)


def load_all_cases(tasks: list[str] | None = None) -> list[BenchmarkCase]:
    builders = {
        "log_triage": generate_log_triage_cases,
        "finance_reconcile": generate_finance_reconcile_cases,
        "note_to_json": generate_note_to_json_cases,
        "doc_contradictions": generate_doc_contradictions_cases,
        "context_cliff": generate_context_cliff_cases,
        "bank_statement": generate_bank_statement_cases,
    }
    selected = tasks or list(_DEFAULT_TASKS)
    out: list[BenchmarkCase] = []
    for name in selected:
        if name not in builders:
            raise ValueError(f"Unknown task '{name}'. Valid: {list(builders)}")
        out.extend(builders[name]())
    return out
