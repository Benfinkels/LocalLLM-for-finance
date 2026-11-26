"""Deterministic ground-truth verifiers for every benchmark task.

Zero LLM-as-a-Judge: every score is an exact, unit-tested function of structured output.
"""

from __future__ import annotations

import json
import re
from typing import Any


def extract_json_object(raw_text: str) -> tuple[dict[str, Any] | None, bool]:
    """Strip <think> blocks / markdown fences and parse the first valid JSON object."""
    if not raw_text:
        return None, False
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    if "<think>" in cleaned:
        cleaned = cleaned.split("<think>", 1)[0].strip()

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    candidates = [fence.group(1)] if fence else []
    candidates.append(cleaned)

    # Also try outermost {...} substring if surrounded by prose
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidates.append(cleaned[first_brace : last_brace + 1])

    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj, True
        except Exception:
            continue
    return None, False


def _norm_str(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip().lower())


def _safe_float(v: Any) -> float | None:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    if isinstance(v, str):
        s = v.replace("$", "").replace(",", "").strip()
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _f1(tp: int, pred_n: int, truth_n: int) -> tuple[float, float, float]:
    prec = tp / pred_n if pred_n else (1.0 if truth_n == 0 else 0.0)
    rec = tp / truth_n if truth_n else (1.0 if pred_n == 0 else 0.0)
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return round(f1, 4), round(prec, 4), round(rec, 4)


def score_log_triage(pred: dict[str, Any] | None, truth: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(pred, dict) or not isinstance(pred.get("errors"), list):
        return {"score": 0.0, "signature_f1": 0.0, "count_accuracy": 0.0, "hallucinations": 0}

    truth_map = {
        (_norm_str(e["service"]), _norm_str(e["code"])): int(e["count"])
        for e in truth["errors"]
    }
    pred_map: dict[tuple[str, str], int] = {}
    for item in pred["errors"]:
        if not isinstance(item, dict):
            continue
        key = (_norm_str(item.get("service")), _norm_str(item.get("code")))
        if not key[1]:
            continue
        cnt = item.get("count", 0)
        try:
            cnt_i = int(round(float(cnt)))
        except Exception:
            cnt_i = 0
        pred_map[key] = pred_map.get(key, 0) + cnt_i

    tp_keys = set(truth_map) & set(pred_map)
    f1, prec, rec = _f1(len(tp_keys), len(pred_map), len(truth_map))
    exact_counts = sum(1 for k in tp_keys if pred_map[k] == truth_map[k])
    count_acc = round(exact_counts / len(truth_map), 4) if truth_map else 1.0
    hallucinations = len(set(pred_map) - set(truth_map))
    # Composite: 60% signature F1 + 40% exact count accuracy
    score = round(0.6 * f1 + 0.4 * count_acc, 4)
    return {
        "score": score,
        "signature_f1": f1,
        "precision": prec,
        "recall": rec,
        "count_accuracy": count_acc,
        "hallucinations": hallucinations,
    }


def score_finance_reconcile(pred: dict[str, Any] | None, truth: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(pred, dict) or not isinstance(pred.get("anomalies"), list):
        return {"score": 0.0, "anomaly_f1": 0.0, "net_variance_exact": False, "exact_anomaly_math": 0.0}

    truth_anoms = {
        (_norm_str(a["platform_code"]), _norm_str(a["period"]), _norm_str(a["type"])): float(a["variance_usd"])
        for a in truth["anomalies"]
    }
    pred_anoms: dict[tuple[str, str, str], float] = {}
    for a in pred["anomalies"]:
        if not isinstance(a, dict):
            continue
        key = (_norm_str(a.get("platform_code")), _norm_str(a.get("period")), _norm_str(a.get("type")))
        v = _safe_float(a.get("variance_usd"))
        if key[0] and key[1] and key[2]:
            pred_anoms[key] = v if v is not None else float("nan")

    tp_keys = set(truth_anoms) & set(pred_anoms)
    f1, prec, rec = _f1(len(tp_keys), len(pred_anoms), len(truth_anoms))

    exact_anom_math = sum(
        1 for k in tp_keys if abs(pred_anoms[k] - truth_anoms[k]) <= 0.02
    )
    anom_math_acc = round(exact_anom_math / len(truth_anoms), 4) if truth_anoms else 1.0

    pred_net = _safe_float(pred.get("net_variance_usd"))
    truth_net = float(truth["net_variance_usd"])
    net_exact = pred_net is not None and abs(pred_net - truth_net) <= 0.02

    # Finance demands BOTH anomaly detection AND cent-exact arithmetic:
    # 40% anomaly F1 + 35% exact anomaly variance math + 25% exact net variance
    score = round(0.40 * f1 + 0.35 * anom_math_acc + 0.25 * (1.0 if net_exact else 0.0), 4)
    return {
        "score": score,
        "anomaly_f1": f1,
        "precision": prec,
        "recall": rec,
        "exact_anomaly_math": anom_math_acc,
        "net_variance_exact": net_exact,
        "net_variance_err_usd": round(abs(pred_net - truth_net), 2) if pred_net is not None else None,
    }


def score_note_to_json(pred: dict[str, Any] | None, truth: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(pred, dict):
        return {"score": 0.0, "fields_matched": 0, "fields_total": 5}

    checks = {
        "contact": _norm_str(pred.get("contact")) == _norm_str(truth["contact"]),
        "company": _norm_str(pred.get("company")) == _norm_str(truth["company"]),
        "category": _norm_str(pred.get("category")) == _norm_str(truth["category"]),
        "requires_followup": (
            isinstance(pred.get("requires_followup"), bool)
            and pred.get("requires_followup") == truth["requires_followup"]
        ),
    }
    pred_amt = _safe_float(pred.get("amount_usd"))
    checks["amount_usd"] = pred_amt is not None and abs(pred_amt - float(truth["amount_usd"])) <= 0.02

    matched = sum(1 for v in checks.values() if v)
    return {
        "score": round(matched / 5.0, 4),
        "fields_matched": matched,
        "fields_total": 5,
        "field_checks": checks,
    }


def score_doc_contradictions(pred: dict[str, Any] | None, truth: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(pred, dict) or not isinstance(pred.get("contradictions"), list):
        return {"score": 0.0, "exact_f1": 0.0, "topic_f1": 0.0}

    truth_exact = {
        (_norm_str(c["topic"]), tuple(sorted(_norm_str(m) for m in c["memos"])))
        for c in truth["contradictions"]
    }
    truth_topics = {_norm_str(c["topic"]) for c in truth["contradictions"]}

    pred_exact = set()
    pred_topics = set()
    for c in pred["contradictions"]:
        if not isinstance(c, dict):
            continue
        topic = _norm_str(c.get("topic"))
        memos = c.get("memos")
        if topic:
            pred_topics.add(topic)
        if topic and isinstance(memos, list) and len(memos) == 2:
            norm_pair = tuple(sorted(_norm_str(m).replace("memo ", "") for m in memos))
            pred_exact.add((topic, norm_pair))

    exact_f1, prec, rec = _f1(len(truth_exact & pred_exact), len(pred_exact), len(truth_exact))
    topic_f1, _, _ = _f1(len(truth_topics & pred_topics), len(pred_topics), len(truth_topics))
    # 75% exact (topic + memo pair) + 25% topic-only credit
    score = round(0.75 * exact_f1 + 0.25 * topic_f1, 4)
    return {
        "score": score,
        "exact_f1": exact_f1,
        "topic_f1": topic_f1,
        "precision": prec,
        "recall": rec,
    }


def score_context_cliff(pred: dict[str, Any] | None, truth: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(pred, dict) or not isinstance(pred.get("incidents"), list):
        return {"score": 0.0, "recall": 0.0, "precision": 0.0}

    truth_set = {
        (_norm_str(x["id"]), _norm_str(x["owner"]), int(x["impact_usd"]))
        for x in truth["incidents"]
    }
    pred_set = set()
    for x in pred["incidents"]:
        if not isinstance(x, dict):
            continue
        inc_id = _norm_str(x.get("id"))
        owner = _norm_str(x.get("owner"))
        imp = _safe_float(x.get("impact_usd"))
        if inc_id and owner and imp is not None:
            pred_set.add((inc_id, owner, int(round(imp))))

    f1, prec, rec = _f1(len(truth_set & pred_set), len(pred_set), len(truth_set))
    return {
        "score": f1,
        "recall": rec,
        "precision": prec,
        "matched": len(truth_set & pred_set),
        "total": len(truth_set),
    }


def score_bank_statement(pred: dict[str, Any] | None, truth: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(pred, dict):
        return {
            "score": 0.0,
            "category_accuracy": 0.0,
            "anomaly_f1": 0.0,
            "exact_anomaly_math": 0.0,
            "net_spend_exact": False,
        }

    truth_cats = {
        _norm_str(item["txn_id"]): _norm_str(item["category"])
        for item in truth["categories"]
    }
    pred_cats: dict[str, str] = {}
    if isinstance(pred.get("categories"), list):
        for item in pred["categories"]:
            if not isinstance(item, dict):
                continue
            tid = _norm_str(item.get("txn_id"))
            cat = _norm_str(item.get("category"))
            if tid and cat:
                pred_cats[tid] = cat

    matched_cats = sum(1 for tid, cat in truth_cats.items() if pred_cats.get(tid) == cat)
    cat_acc = round(matched_cats / len(truth_cats), 4) if truth_cats else 1.0

    truth_anoms = {
        (_norm_str(a["type"]), tuple(sorted(_norm_str(t) for t in a["txn_ids"]))): float(a["amount_usd"])
        for a in truth["anomalies"]
    }
    pred_anoms: dict[tuple[str, tuple[str, ...]], float] = {}
    if isinstance(pred.get("anomalies"), list):
        for a in pred["anomalies"]:
            if not isinstance(a, dict):
                continue
            atype = _norm_str(a.get("type"))
            tids = a.get("txn_ids")
            amt = _safe_float(a.get("amount_usd"))
            if atype and isinstance(tids, list) and len(tids) == 2:
                key = (atype, tuple(sorted(_norm_str(t) for t in tids)))
                pred_anoms[key] = abs(amt) if amt is not None else float("nan")

    tp_anoms = set(truth_anoms) & set(pred_anoms)
    anom_f1, anom_prec, anom_rec = _f1(len(tp_anoms), len(pred_anoms), len(truth_anoms))
    exact_anom_math = sum(1 for k in tp_anoms if abs(pred_anoms[k] - truth_anoms[k]) <= 0.02)
    anom_math_acc = round(exact_anom_math / len(truth_anoms), 4) if truth_anoms else 1.0

    pred_net = _safe_float(pred.get("net_spend_usd"))
    truth_net = float(truth["net_spend_usd"])
    net_exact = pred_net is not None and abs(pred_net - truth_net) <= 0.02

    # 50% categorization (where messy merchant strings require an SLM) +
    # 20% anomaly pair detection + 15% exact anomaly amount + 15% exact net spend
    score = round(
        0.50 * cat_acc
        + 0.20 * anom_f1
        + 0.15 * anom_math_acc
        + 0.15 * (1.0 if net_exact else 0.0),
        4,
    )
    return {
        "score": score,
        "category_accuracy": cat_acc,
        "categories_matched": matched_cats,
        "categories_total": len(truth_cats),
        "anomaly_f1": anom_f1,
        "anomaly_precision": anom_prec,
        "anomaly_recall": anom_rec,
        "exact_anomaly_math": anom_math_acc,
        "net_spend_exact": net_exact,
        "net_spend_err_usd": round(abs(pred_net - truth_net), 2) if pred_net is not None else None,
    }


def score_case(task: str, raw_output: str, ground_truth: dict[str, Any]) -> dict[str, Any]:
    parsed, json_valid = extract_json_object(raw_output)
    scorers = {
        "log_triage": score_log_triage,
        "finance_reconcile": score_finance_reconcile,
        "note_to_json": score_note_to_json,
        "doc_contradictions": score_doc_contradictions,
        "context_cliff": score_context_cliff,
        "bank_statement": score_bank_statement,
    }
    if task not in scorers:
        raise ValueError(f"No verifier for task '{task}'")
    res = scorers[task](parsed, ground_truth)
    res["json_valid"] = json_valid
    return res
