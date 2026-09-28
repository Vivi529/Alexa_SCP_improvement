# -*- coding: utf-8 -*-
"""
Evidence construction for the A/B/C information-ablation experiment.

Core experiment
---------------
A (static):
    observed review evidence + static ecosystem relations + API knowledge
B (dynamic evolution):
    A + dynamic opinion-evolution facts
C (full / main):
    B + explicit risk chains

Important design rule
---------------------
The observed review cards used by A/B/C are IDENTICAL.  Dynamic information is
not allowed to re-rank the review cards in the main A/B/C experiment, otherwise
B-A would mix two treatments (different review evidence + dynamic facts).

This module therefore builds:
1) pure static review profiles WITHOUT risk_chain_res;
2) static three-network relation profiles from A_api/A_coo/A_user;
3) B-safe dynamic-evolution profiles from R_node/W_eff/dimension parameters.

Risk-chain JSONL is generated separately by risk_chain_export_service.py and is
added only by Condition C inside data_service.py.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import os

# ============================================================
# 0. Utilities
# ============================================================


def _is_missing(x: Any) -> bool:
    return x is None or (isinstance(x, float) and pd.isna(x))


def _clean_str(x: Any) -> str:
    if _is_missing(x):
        return ""
    return str(x).strip()


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.Period):
        return str(obj)
    if isinstance(obj, float) and (pd.isna(obj) or not np.isfinite(obj)):
        return None
    return obj


def save_jsonl(records: Iterable[Dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(_json_safe(row), ensure_ascii=False) + "\n")


def stable_id(prefix: str, *parts: Any, length: int = 12) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length].upper()
    return f"{prefix}-{digest}"


def _split_items(x: Any) -> List[str]:
    if _is_missing(x):
        return []
    if isinstance(x, (list, tuple, set, np.ndarray)):
        vals = list(x)
    elif isinstance(x, str):
        s = x.strip()
        if not s or s.lower() in {"nan", "none", "null", "[]"}:
            return []
        try:
            obj = ast.literal_eval(s)
            vals = list(obj) if isinstance(obj, (list, tuple, set)) else re.split(r"[;；]+", s)
        except Exception:
            vals = re.split(r"[;；]+", s)
    else:
        vals = [x]
    out: List[str] = []
    seen = set()
    for v in vals:
        s = str(v).strip()
        if s and s.lower() not in {"nan", "none", "null"} and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _safe_mean(x: Any) -> float:
    s = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    return float(s.mean()) if len(s) else np.nan


def _minmax(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)
    if len(s) == 0:
        return s
    lo, hi = float(s.min()), float(s.max())
    if hi <= lo + 1e-12:
        return pd.Series(np.ones(len(s)) if hi > 0 else np.zeros(len(s)), index=s.index, dtype=float)
    return (s - lo) / (hi - lo)


def _cutoff_end(cutoff: Optional[str]) -> Optional[pd.Timestamp]:
    if not cutoff:
        return None
    p = pd.Period(str(cutoff), freq="M")
    return p.end_time


# ============================================================
# 1. Pure static review evidence (Condition A base)
# ============================================================


def build_scenario_map(
    scen_map_df: Optional[pd.DataFrame],
    raw_col: str = "scen",
    clean_col: str = "scen_clean",
) -> Dict[str, str]:
    if scen_map_df is None or scen_map_df.empty:
        return {}
    if raw_col not in scen_map_df.columns or clean_col not in scen_map_df.columns:
        return {}
    return {
        _clean_str(r[raw_col]): _clean_str(r[clean_col])
        for _, r in scen_map_df.iterrows()
        if _clean_str(r[raw_col])
    }


def preprocess_review_rows(
    df_raw: pd.DataFrame,
    scen_map_df: Optional[pd.DataFrame] = None,
    analysis_cutoff: Optional[str] = None,
    category_col: str = "Category",
    dimension_col: str = "dimension",
    statement_col: str = "st",
    sentiment_col: str = "sentiment",
    failure_col: str = "failure_type",
    scenario_col: str = "scen",
    review_id_col: str = "review_id",
    review_date_col: str = "review_date",
    ds_set_col: str = "ds_category_list",
) -> pd.DataFrame:
    """Prepare statement-level rows without using risk_chain_res or dynamic model output."""
    for col in [category_col, dimension_col]:
        if col not in df_raw.columns:
            raise ValueError(f"df_raw missing required column: {col}")

    df = df_raw.copy()
    ren = {}
    for src, dst in [
        (category_col, "Category"),
        (dimension_col, "dimension"),
        (sentiment_col, "sentiment"),
        (failure_col, "failure_type"),
        (review_id_col, "review_id"),
        (review_date_col, "review_date"),
    ]:
        if src in df.columns and src != dst:
            ren[src] = dst
    df = df.rename(columns=ren)

    df["Category"] = df["Category"].astype(str)
    df["dimension"] = df["dimension"].astype(str)

    if "review_date" in df.columns:
        df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    else:
        df["review_date"] = pd.NaT

    end = _cutoff_end(analysis_cutoff)
    if end is not None:
        # Missing dates are excluded from strict historical snapshots because their
        # availability at the cutoff cannot be established.
        df = df[df["review_date"].notna() & (df["review_date"] <= end)].copy()

    if statement_col in df.columns:
        df["st_clean"] = df[statement_col].fillna("").astype(str).str.strip()
    elif "review_text" in df.columns:
        df["st_clean"] = df["review_text"].fillna("").astype(str).str.strip()
    else:
        df["st_clean"] = ""

    if "review_id" not in df.columns:
        df["review_id"] = np.arange(len(df)).astype(str)
    else:
        df["review_id"] = df["review_id"].astype(str)

    if "sentiment" not in df.columns:
        df["sentiment"] = np.nan
    df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce")

    if "failure_type" not in df.columns:
        df["failure_type"] = "unspecified"
    df["failure_type"] = df["failure_type"].fillna("").astype(str).str.strip().replace("", "unspecified")

    scen_map = build_scenario_map(scen_map_df)
    if scenario_col in df.columns:
        def _norm_scen(x: Any) -> List[str]:
            vals = [scen_map.get(v, v) for v in _split_items(x)]
            vals = [v for v in vals if v and v != "/"]
            return vals or ["unspecified"]
        df["scenario_list"] = df[scenario_col].apply(_norm_scen)
    else:
        df["scenario_list"] = [["unspecified"] for _ in range(len(df))]

    if ds_set_col in df.columns:
        df["ds_set_list"] = df[ds_set_col].apply(_split_items)
    else:
        df["ds_set_list"] = [[] for _ in range(len(df))]

    return df.reset_index(drop=True)


def _top_statements_by_severity(df_sub: pd.DataFrame, top_n: int = 10) -> List[str]:
    if df_sub.empty:
        return []
    tmp = (
        df_sub[df_sub["st_clean"].astype(str).str.len() > 0]
        .groupby("st_clean", as_index=False)
        .agg(freq=("st_clean", "size"), avg_sent=("sentiment", _safe_mean))
    )
    if tmp.empty:
        return []
    tmp["neg_strength"] = pd.to_numeric(tmp["avg_sent"], errors="coerce").fillna(0.0).map(lambda x: max(-float(x), 0.0))
    tmp["score"] = tmp["freq"] * tmp["neg_strength"]
    return tmp.sort_values(["score", "freq"], ascending=False)["st_clean"].head(top_n).tolist()


def _top_cooccurring(series: pd.Series, top_n: int = 5) -> List[str]:
    cnt: Counter = Counter()
    for xs in series.tolist():
        for x in _split_items(xs):
            cnt[x] += 1
    return [k for k, _ in cnt.most_common(top_n)]


def _top_counted_values(values: Iterable[Any], key_name: str, top_n: int = 3) -> List[Dict[str, Any]]:
    """Return top descriptive values as [{key_name: value, count: n, share: p}, ...]."""
    cnt: Counter = Counter()
    for raw in values:
        vals = _split_items(raw)
        if not vals and not _is_missing(raw):
            vals = [_clean_str(raw)]
        seen = set()
        for v in vals:
            v = _clean_str(v)
            if not v or v == "/" or v.lower() in {"nan", "none", "null"} or v in seen:
                continue
            seen.add(v)
            cnt[v] += 1
    total = sum(cnt.values())
    return [
        {
            key_name: k,
            "count": int(v),
            "share": float(v / total) if total > 0 else 0.0,
        }
        for k, v in cnt.most_common(max(int(top_n), 0))
    ]


def summarize_static_issue_cards(
    df_processed: pd.DataFrame,
    target_categories: Optional[Sequence[str]] = None,
    min_reviews_per_dimension: int = 2,
    recent_months: int = 3,
    top_statements_per_issue: int = 10,
    top_cooccurring_per_issue: int = 5,
    top_failure_types_per_dimension: int = 3,
    top_scenarios_per_dimension: int = 3,
    exclude_general_dimension: bool = True,
    general_dimension_names: Optional[set[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Build pure-static observed evidence at Category × dimension granularity.

    Primary evidence unit
    ---------------------
        Product category × evaluation dimension

    failure_type and scenario are NOT grouping keys.  They are retained as
    descriptive attributes inside each dimension card.  This avoids fragmenting
    sparse historical snapshots into many one-off failure/scenario combinations.

    Minimum support is based on the number of independent reviews rather than
    statement rows.  A dimension card is retained when:
        n_unique_reviews >= min_reviews_per_dimension
    """
    df = df_processed.copy()
    if target_categories is not None:
        cats = set(map(str, target_categories))
        df = df[df["Category"].astype(str).isin(cats)].copy()

    if general_dimension_names is None:
        general_dimension_names = {"general"}
    if exclude_general_dimension:
        general_names = {str(x).strip().lower() for x in general_dimension_names}
        df = df[~df["dimension"].astype(str).str.strip().str.lower().isin(general_names)].copy()

    if df.empty:
        return []

    max_date = df["review_date"].max() if df["review_date"].notna().any() else pd.NaT
    recent_cutoff = None
    if pd.notna(max_date) and recent_months and recent_months > 0:
        recent_cutoff = max_date - pd.DateOffset(months=int(recent_months))

    product_stmt_total = df.groupby("Category", dropna=False).size().to_dict()
    product_review_total = (
        df.groupby("Category", dropna=False)["review_id"].nunique().to_dict()
        if "review_id" in df.columns else {}
    )

    rows: List[Dict[str, Any]] = []
    for (category, dimension), g in df.groupby(["Category", "dimension"], dropna=False):
        g = g.copy()
        category, dimension = str(category), str(dimension)
        n_stmt = int(len(g))
        n_review = int(g["review_id"].nunique()) if "review_id" in g.columns else n_stmt
        if n_review < int(min_reviews_per_dimension):
            continue

        mean_sent = _safe_mean(g["sentiment"])
        if recent_cutoff is not None:
            recent = g[g["review_date"] >= recent_cutoff].copy()
            recent_count = int(len(recent))
            recent_review_count = int(recent["review_id"].nunique()) if "review_id" in recent.columns else recent_count
        else:
            recent_count = 0
            recent_review_count = 0

        failure_values = g["failure_type"].tolist() if "failure_type" in g.columns else []
        top_failures = _top_counted_values(
            failure_values,
            key_name="failure_type",
            top_n=top_failure_types_per_dimension,
        )

        # scenario_list is already normalized by preprocess_review_rows.  Each
        # statement contributes at most once to the same scenario through
        # _top_counted_values' per-row de-duplication.
        scenario_values = g["scenario_list"].tolist() if "scenario_list" in g.columns else []
        top_scenarios = _top_counted_values(
            scenario_values,
            key_name="scenario",
            top_n=top_scenarios_per_dimension,
        )

        top_statements = _top_statements_by_severity(g, top_statements_per_issue)
        coo = _top_cooccurring(g["ds_set_list"], top_cooccurring_per_issue)

        prod_stmt_n = int(product_stmt_total.get(category, 0) or 0)
        prod_review_n = int(product_review_total.get(category, 0) or 0)

        rows.append({
            "Category": category,
            "dimension": dimension,
            "issue_cluster": dimension,
            "evidence_granularity": "category_dimension",
            "severity_proxy": {
                "n_stmt": n_stmt,
                "n_review": n_review,
                "mean_sentiment": None if pd.isna(mean_sent) else float(mean_sent),
                "recent_count": recent_count,
                "recent_review_count": recent_review_count,
                "share_in_product": float(n_stmt / max(prod_stmt_n, 1)),
                "review_share_in_product": float(n_review / max(prod_review_n, 1)),
            },
            "top_failure_types": top_failures,
            "top_scenarios": top_scenarios,
            "representative_statements": top_statements,
            "co_occurring_categories": coo,
            "evidence_type": "observed_review_evidence",
        })

    return _json_safe(rows)


def _static_card_rank_table(cards: List[Dict[str, Any]]) -> pd.DataFrame:
    """Compute static-only evidence utility for dimension-level cards."""
    rows = []
    for idx, c in enumerate(cards):
        sev = c.get("severity_proxy") or {}
        n_stmt = int(sev.get("n_stmt") or 0)
        n_review = int(sev.get("n_review") or 0)
        mean_sent = float(sev.get("mean_sentiment") or 0.0)
        recent_review_count = int(sev.get("recent_review_count") or 0)
        recent_count = int(sev.get("recent_count") or 0)
        recent_share = (
            recent_review_count / max(n_review, 1)
            if "recent_review_count" in sev
            else recent_count / max(n_stmt, 1)
        )
        prevalence = float(
            sev.get("share_in_product")
            if sev.get("share_in_product") is not None
            else (sev.get("share_in_node") or 0.0)
        )
        
        rows.append({
            "idx": idx,
            "dimension": str(c.get("dimension") or ""),
            "severity_raw": max(-mean_sent, 0.0)*n_stmt,
            "recent_share": recent_share,
            "prevalence": prevalence,
            "breadth_raw": math.log1p(n_review),
        })
    x = pd.DataFrame(rows)
    if x.empty:
        return x
    x["severity_norm"] = _minmax(x["severity_raw"])
    x["recency_norm"] = _minmax(x["recent_share"])
    x["prevalence_norm"] = _minmax(x["prevalence"])
    x["breadth_norm"] = _minmax(x["breadth_raw"])
    x["evidence_utility"] = (
        0.50 * x["severity_norm"]
        + 0.20 * x["recency_norm"]
        + 0.20 * x["prevalence_norm"]
        + 0.10 * x["breadth_norm"]
    )
    return x


def select_static_evidence_cards(
    cards: List[Dict[str, Any]],
    max_cards: int = 10,
    diversity_first: bool = True,
) -> List[Dict[str, Any]]:
    """
    Select static-only dimension cards used identically by A/B/C.

    Because each card is already one Category × dimension unit, dimension
    diversity is inherent.  Selection is therefore a direct ranking by static
    evidence utility, with severity as the tie-breaker.  ``diversity_first`` is
    retained only for call compatibility with earlier versions.
    """
    if not cards or max_cards <= 0:
        return []
    rank = _static_card_rank_table(cards)
    if rank.empty:
        return []

    ordered = rank.sort_values(
        ["evidence_utility", "severity_raw", "breadth_raw", "dimension"],
        ascending=[False, False, False, True],
    )
    selected = [int(x) for x in ordered["idx"].head(max_cards).tolist()]

    info = rank.set_index("idx").to_dict("index")
    out = []
    for pos, idx in enumerate(selected, start=1):
        c = dict(cards[idx])
        r = info[idx]
        c["evidence_selection"] = {
            "rank": pos,
            "selection_rule": "dimension_level_static_evidence_utility",
            "evidence_utility": round(float(r["evidence_utility"]), 6),
            "severity_component": round(float(r["severity_norm"]), 6),
            "recency_component": round(float(r["recency_norm"]), 6),
            "prevalence_component": round(float(r["prevalence_norm"]), 6),
            "breadth_component": round(float(r["breadth_norm"]), 6),
        }
        out.append(c)
    return _json_safe(out)


def build_pure_static_profiles(
    df_raw: pd.DataFrame,
    scen_map_df: Optional[pd.DataFrame] = None,
    analysis_cutoff: Optional[str] = None,
    target_categories: Optional[Sequence[str]] = None,
    min_reviews_per_dimension: int = 2,
    recent_months: int = 3,
    top_statements_per_issue: int = 10,
    top_cooccurring_per_issue: int = 5,
    top_failure_types_per_dimension: int = 3,
    top_scenarios_per_dimension: int = 3,
    max_evidence_cards_per_product: int = 10,
    top_n_summary: int = 5,
) -> List[Dict[str, Any]]:
    """
    Pure Condition-A static profiles. No risk_chain_res argument exists by design.

    Observed evidence cards are Product × Dimension units.  Failure types and
    scenarios are descriptive sub-signals, not card-defining grouping keys.
    """
    df = preprocess_review_rows(
        df_raw=df_raw,
        scen_map_df=scen_map_df,
        analysis_cutoff=analysis_cutoff,
    )
    cards = summarize_static_issue_cards(
        df,
        target_categories=target_categories,
        min_reviews_per_dimension=min_reviews_per_dimension,
        recent_months=recent_months,
        top_statements_per_issue=top_statements_per_issue,
        top_cooccurring_per_issue=top_cooccurring_per_issue,
        top_failure_types_per_dimension=top_failure_types_per_dimension,
        top_scenarios_per_dimension=top_scenarios_per_dimension,
    )

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in cards:
        grouped[str(c["Category"])].append(c)

    # Product-level exact unique-review/statements summaries are taken directly
    # from the processed rows rather than by summing dimension-card review counts.
    df_for_summary = df.copy()
    df_for_summary = df_for_summary[
        ~df_for_summary["dimension"].astype(str).str.strip().str.lower().isin({"general"})
    ].copy()

    out: List[Dict[str, Any]] = []
    for cat, cat_cards in grouped.items():
        dim_score: Dict[str, float] = defaultdict(float)
        fail_score: Dict[str, float] = defaultdict(float)
        scen_score: Dict[str, float] = defaultdict(float)
        coo_counter: Counter = Counter()

        for c in cat_cards:
            sev = c.get("severity_proxy") or {}
            n = int(sev.get("n_stmt") or 0)
            sent = float(sev.get("mean_sentiment") or 0.0)
            neg = max(-sent, 0.0)
            score = n * neg
            dim_score[str(c.get("dimension"))] += score

            # Failure/scenario are secondary descriptors.  Weight their observed
            # counts by the dimension's negative strength to form product summaries.
            for item in c.get("top_failure_types") or []:
                if isinstance(item, dict):
                    name = _clean_str(item.get("failure_type"))
                    count = int(item.get("count") or 0)
                else:
                    name, count = _clean_str(item), 1
                if name:
                    fail_score[name] += count * neg

            for item in c.get("top_scenarios") or []:
                if isinstance(item, dict):
                    name = _clean_str(item.get("scenario"))
                    count = int(item.get("count") or 0)
                else:
                    name, count = _clean_str(item), 1
                if name:
                    scen_score[name] += count * neg

            for other in c.get("co_occurring_categories") or []:
                if str(other) != cat:
                    coo_counter[str(other)] += 1

        selected_cards = select_static_evidence_cards(
            cat_cards,
            max_cards=max_evidence_cards_per_product,
        )

        cat_df = df_for_summary[df_for_summary["Category"].astype(str) == str(cat)].copy()
        total_stmt = int(len(cat_df))
        total_review = int(cat_df["review_id"].nunique()) if "review_id" in cat_df.columns else total_stmt
        if total_stmt > 0:
            weighted_sent = _safe_mean(cat_df["sentiment"])
        else:
            weighted_sent = np.nan

        out.append({
            "Category": cat,
            "analysis_cutoff": analysis_cutoff,
            "profile_source": "pure_static_reviews_no_risk_chain_dimension_level",
            "evidence_granularity": "category_dimension",
            "min_reviews_per_dimension": int(min_reviews_per_dimension),
            "dominant_dimensions": [k for k, _ in sorted(dim_score.items(), key=lambda kv: kv[1], reverse=True)[:top_n_summary]],
            "dominant_failure_types": [k for k, _ in sorted(fail_score.items(), key=lambda kv: kv[1], reverse=True)[:top_n_summary]],
            "high_risk_scenarios": [k for k, _ in sorted(scen_score.items(), key=lambda kv: kv[1], reverse=True)[:top_n_summary]],
            "co_occurring_devices": [k for k, _ in coo_counter.most_common(top_n_summary)],
            "dimension_severity_score": dict(sorted(dim_score.items(), key=lambda kv: kv[1], reverse=True)),
            "failure_severity_score": dict(sorted(fail_score.items(), key=lambda kv: kv[1], reverse=True)),
            "scenario_severity_score": dict(sorted(scen_score.items(), key=lambda kv: kv[1], reverse=True)),
            "total_stmt": total_stmt,
            "total_review": total_review,
            "avg_weighted_sentiment": None if pd.isna(weighted_sent) else float(weighted_sent),
            "evidence_card_pool": cat_cards,
            "evidence_cards": selected_cards,
            "evidence_selection_basis": "static_review_information_only_dimension_level",
            "evidence_type": "observed_review_evidence",
        })

    return _json_safe(sorted(out, key=lambda r: str(r["Category"])))


# ============================================================
# 2. Static ecosystem relation cards for A/B/C
# ============================================================


def _to_numpy_matrix(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(float)
    return np.asarray(x, dtype=float)


def _ensure_node_index(node_index: pd.DataFrame) -> pd.DataFrame:
    if not {"Category", "dimension"}.issubset(node_index.columns):
        raise ValueError("node_index must contain Category and dimension")
    meta = node_index.copy().reset_index(drop=True)
    if "node_id" not in meta.columns:
        meta["node_id"] = np.arange(len(meta), dtype=int)
    meta["node_id"] = pd.to_numeric(meta["node_id"], errors="raise").astype(int)
    meta["Category"] = meta["Category"].astype(str)
    meta["dimension"] = meta["dimension"].astype(str)
    return meta


def build_static_network_relation_profiles(
    A_dict: Mapping[str, Any],
    node_index: pd.DataFrame,
    target_categories: Optional[Sequence[str]] = None,
    max_related_per_product: int = 8,
    analysis_cutoff: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Convert the three raw association networks to product-level relation cards.

    Assumption: A[j, i] represents the stored i -> j association used by the
    evolution model.  These raw A matrices are treated only as topology / static
    association, NOT as risk propagation and NOT as causal evidence.

    For a product pair, each channel stores the maximum positive node-level
    association in both directions. Max aggregation avoids mechanically giving
    products with more dimensions larger relation scores.
    """
    meta = _ensure_node_index(node_index)
    mats = {k: np.clip(_to_numpy_matrix(v), 0.0, None) for k, v in A_dict.items() if k in {"api", "coo", "user"}}
    missing = {"api", "coo", "user"} - set(mats)
    if missing:
        raise ValueError(f"A_dict missing channels: {sorted(missing)}")
    n = len(meta)
    if any(m.shape != (n, n) for m in mats.values()):
        raise ValueError("Each A_dict matrix must be [n_nodes, n_nodes] aligned with node_index")

    cat_nodes = {
        cat: g["node_id"].astype(int).tolist()
        for cat, g in meta.groupby("Category", dropna=False)
    }
    cats = sorted(cat_nodes)
    targets = cats if target_categories is None else [str(x) for x in target_categories if str(x) in cat_nodes]

    profiles = []
    for focal in targets:
        src_nodes = cat_nodes[focal]
        cards = []
        for other in cats:
            if other == focal:
                continue
            dst_nodes = cat_nodes[other]
            channel_strengths = {}
            overall = 0.0
            for name, A in mats.items():
                # A[j, i] = i -> j
                forward = float(A[np.ix_(dst_nodes, src_nodes)].max()) if src_nodes and dst_nodes else 0.0
                reverse = float(A[np.ix_(src_nodes, dst_nodes)].max()) if src_nodes and dst_nodes else 0.0
                channel_strengths[name] = {
                    "focal_to_other": forward,
                    "other_to_focal": reverse,
                    "undirected_strength": max(forward, reverse),
                }
                overall = max(overall, forward, reverse)
            if overall <= 0:
                continue
            cards.append({
                "relation_id": stable_id("REL", focal, other, analysis_cutoff, channel_strengths),
                "relation_type": "static_multi_channel_ecosystem_association",
                "product_category": focal,
                "related_product": other,
                "channel_strengths": channel_strengths,
                "overall_relation_strength": overall,
                "direction_note": "Raw A[j,i] is stored as i->j; relation is associative/topological, not causal propagation.",
                "epistemic_status": "observed_or_constructed_static_relation_not_causal",
            })
        cards.sort(key=lambda r: float(r["overall_relation_strength"]), reverse=True)
        profiles.append({
            "Category": focal,
            "analysis_cutoff": analysis_cutoff,
            "relation_cards": cards[:max_related_per_product],
            "relation_source": "raw_api_cooccurrence_user_networks",
        })
    return _json_safe(profiles)


# ============================================================
# 3. Dynamic-evolution facts for Condition B (NO chains)
# ============================================================
def _weighted_mean(v: np.ndarray, w: np.ndarray) -> Optional[float]:
    m = np.isfinite(v) & np.isfinite(w)
    v, w = v[m], w[m]
    if len(v) == 0:
        return None
    if float(w.sum()) <= 1e-12:
        return float(v.mean())
    return float(np.sum(v * w) / np.sum(w))


def build_dynamic_evolution_profiles(
    pred_next_node: Any,
    W_eff_node: Any,
    node_index: pd.DataFrame,
    dimension_params: Optional[pd.DataFrame] = None,
    node_dim_ids: Optional[Any] = None,
    analysis_cutoff: Optional[str] = None,
    target_month: Optional[str] = None,
    target_categories: Optional[Sequence[str]] = None,
    top_dimensions: int = 5,
    exclude_general_dimension: bool = True,
    general_dimension_names: Optional[set[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Build Condition-B information directly from the evolution model state.

    Design rules
    ------------
    1. No chain extraction, chain score, chain role, marginal chain gain, or
       chain-level edge decomposition is used here.
    2. ``General`` can remain in the underlying evolution model, but is excluded
       from LLM-facing B/H evidence by default so that A/B/C use the same
       substantive risk-dimension universe.
    3. ``top_predicted_risk_dimensions`` is retained for the original B/C
       information-ablation baseline.
    4. ``dimension_signals`` stores ALL non-excluded dimensions for Hybrid
       anchor-conditioned lookup. This prevents an A-selected dimension from
       disappearing merely because it was outside B's top-K list.
    """
    meta = _ensure_node_index(node_index)
    R = np.clip(np.asarray(pred_next_node, dtype=float).reshape(-1), 0.0, None)
    W = np.clip(_to_numpy_matrix(W_eff_node), 0.0, None)
    if len(R) != len(meta) or W.shape != (len(meta), len(meta)):
        raise ValueError("pred_next_node/W_eff_node must align with node_index")

    meta = meta.copy()
    meta["predicted_next_risk"] = meta["node_id"].map(lambda i: float(R[int(i)]))
    # W[j,i] = i -> j
    meta["spillover_output"] = meta["node_id"].map(lambda i: float(W[:, int(i)].sum()))
    meta["spillover_exposure"] = meta["node_id"].map(lambda i: float(W[int(i), :].sum()))

    # Keep General in the underlying model, but remove it from exported
    # decision-support evidence. Filtering here also ensures that product-level
    # summaries/quantile labels are not inflated by a generic catch-all node.
    if general_dimension_names is None:
        general_dimension_names = {"general"}
    excluded_dim_names = {
        str(x).strip().lower() for x in general_dimension_names if str(x).strip()
    }
    if exclude_general_dimension and excluded_dim_names:
        meta = meta[ ~meta["dimension"].astype(str).str.strip().str.lower().isin(excluded_dim_names)].copy()

    if meta.empty:
        return []

    param_by_dim: Dict[str, Dict[str, float]] = {}
    if dimension_params is not None and not dimension_params.empty:
        p = dimension_params.copy().reset_index(drop=True)
        need = [c for c in ["rho", "beta_api", "beta_coo", "beta_user"] if c in p.columns]
        if "dimension" in p.columns:
            for _, r in p.iterrows():
                dim = str(r["dimension"])
                if exclude_general_dimension and dim.strip().lower() in excluded_dim_names:
                    continue
                param_by_dim[dim] = {
                    c: float(r[c]) if pd.notna(r[c]) else np.nan for c in need
                }
        elif node_dim_ids is not None:
            dim_ids = np.asarray(node_dim_ids, dtype=int)
            dim_name_to_id = {}
            for _, r in meta.iterrows():
                nid = int(r["node_id"])
                if 0 <= nid < len(dim_ids):
                    dim_name_to_id.setdefault(str(r["dimension"]), int(dim_ids[nid]))
            for dim, did in dim_name_to_id.items():
                if 0 <= did < len(p):
                    param_by_dim[dim] = {
                        c: float(p.iloc[did][c]) if pd.notna(p.iloc[did][c]) else np.nan for c in need
                    }

    categories = sorted(meta["Category"].unique().tolist())
    if target_categories is not None:
        allowed = set(map(str, target_categories))
        categories = [c for c in categories if c in allowed]

    # Product-level quantile labels are computed only among the requested
    # product set and after excluding generic dimensions.
    preliminary = []
    for cat in categories:
        g = meta[meta["Category"] == cat].copy()
        if g.empty:
            continue

        dim_rows = (
            g.groupby("dimension", as_index=False)
            .agg(
                predicted_next_risk=("predicted_next_risk", "mean"),
                max_predicted_next_risk=("predicted_next_risk", "max"),
                spillover_output=("spillover_output", "mean"),
                spillover_exposure=("spillover_exposure", "mean"),
            )
            .sort_values(
                ["predicted_next_risk", "max_predicted_next_risk", "dimension"],
                ascending=[False, False, True],
            ).reset_index(drop=True)
        )

        # Full dimension-level signal table used by the hierarchical/Hybrid
        # pipeline.  Each dimension receives its own persistence/driver
        # parameters rather than only the product-level weighted summary.
        dimension_signals: List[Dict[str, Any]] = []
        for rank, (_, r) in enumerate(dim_rows.iterrows(), start=1):
            dim = str(r["dimension"])
            p_dim = param_by_dim.get(dim, {}) or {}

            driver_vals_dim = {
                "api": p_dim.get("beta_api"),
                "co_occurrence": p_dim.get("beta_coo"),
                "user": p_dim.get("beta_user"),
            }
            finite_driver_dim = {
                k: float(v)
                for k, v in driver_vals_dim.items()
                if v is not None and np.isfinite(v)
            }
            dominant_driver_dim = (
                max(finite_driver_dim, key=finite_driver_dim.get)
                if finite_driver_dim and max(finite_driver_dim.values()) > 0 else None
            )

            dimension_signals.append({
                "dimension": dim,
                "risk_rank": int(rank),
                "predicted_next_risk": float(r["predicted_next_risk"]),
                "max_predicted_next_risk": float(r["max_predicted_next_risk"]),
                "spillover_output": float(r["spillover_output"]),
                "spillover_exposure": float(r["spillover_exposure"]),
                "rho": (
                    float(p_dim["rho"])
                    if p_dim.get("rho") is not None and np.isfinite(p_dim.get("rho"))
                    else None
                ),
                "beta_api": (
                    float(p_dim["beta_api"])
                    if p_dim.get("beta_api") is not None and np.isfinite(p_dim.get("beta_api"))
                    else None
                ),
                "beta_coo": (
                    float(p_dim["beta_coo"])
                    if p_dim.get("beta_coo") is not None and np.isfinite(p_dim.get("beta_coo"))
                    else None
                ),
                "beta_user": (
                    float(p_dim["beta_user"])
                    if p_dim.get("beta_user") is not None and np.isfinite(p_dim.get("beta_user"))
                    else None
                ),
                "dominant_driver": dominant_driver_dim,
                "epistemic_status": "model_inferred_dimension_forecast",
            })

        risk_w = g["predicted_next_risk"].to_numpy(dtype=float)
        params_summary: Dict[str, Optional[float]] = {}
        for param in ["rho", "beta_api", "beta_coo", "beta_user"]:
            vals = np.array([
                (param_by_dim.get(str(dim), {}) or {}).get(param, np.nan)
                for dim in g["dimension"].astype(str).tolist()
            ], dtype=float)
            params_summary[param] = _weighted_mean(vals, risk_w)

        driver_vals = {
            "api": params_summary.get("beta_api"),
            "co_occurrence": params_summary.get("beta_coo"),
            "user": params_summary.get("beta_user"),
        }
        finite_driver = {k: float(v) for k, v in driver_vals.items() if v is not None and np.isfinite(v)}
        dominant_driver = max(finite_driver, key=finite_driver.get) if finite_driver and max(finite_driver.values()) > 0 else None

        preliminary.append({
            "Category": cat,
            "analysis_cutoff": analysis_cutoff,
            "target_month": target_month,
            "dynamic_evolution": {
                "state_signals": {
                    "predicted_next_risk_score": float(g["predicted_next_risk"].mean()),
                    "max_dimension_risk_score": float(g["predicted_next_risk"].max()),
                    "top_predicted_risk_dimensions": [
                        {
                            "dimension": str(r["dimension"]),
                            "predicted_next_risk": float(r["predicted_next_risk"]),
                            "spillover_output": float(r["spillover_output"]),
                            "spillover_exposure": float(r["spillover_exposure"]),
                        }
                        for _, r in dim_rows.head(top_dimensions).iterrows()
                    ],
                },
                # NEW: complete non-General dimension-level dynamic evidence.
                "dimension_signals": dimension_signals,
                "evolution_signals": {"persistence_score": params_summary.get("rho"),},
                "driver_signals": {
                    "beta_api": params_summary.get("beta_api"),
                    "beta_coo": params_summary.get("beta_coo"),
                    "beta_user": params_summary.get("beta_user"),
                    "dominant_driver": dominant_driver,
                },
                "network_position_signals": {
                    "spillover_output_score": float(g["spillover_output"].mean()),
                    "spillover_exposure_score": float(g["spillover_exposure"].mean()),
                    "direction_note": "W_eff[j,i] means i->j; output is column-sum, exposure is row-sum.",
                },
                "dynamic_dimension_priority": dim_rows["dimension"].astype(str).head(top_dimensions).tolist(),
                "dimension_signal_count": int(len(dimension_signals)),
                "general_dimension_excluded": bool(exclude_general_dimension),
                "evidence_type": "model_inferred_dynamic_evolution_no_explicit_chain",
            },
        })

    if preliminary:
        risks = pd.Series([r["dynamic_evolution"]["state_signals"]["predicted_next_risk_score"] for r in preliminary])
        outs = pd.Series([r["dynamic_evolution"]["network_position_signals"]["spillover_output_score"] for r in preliminary])
        exps = pd.Series([r["dynamic_evolution"]["network_position_signals"]["spillover_exposure_score"] for r in preliminary])
        for rows, key in [(risks, "predicted_next_risk_level"), (outs, "spillover_output_level"), (exps, "spillover_exposure_level")]:
            q1, q2 = float(rows.quantile(.33)), float(rows.quantile(.67))
            for i, val in enumerate(rows):
                label = "low" if val <= q1 else ("medium" if val <= q2 else "high")
                if key == "predicted_next_risk_level":
                    preliminary[i]["dynamic_evolution"]["state_signals"][key] = label
                else:
                    preliminary[i]["dynamic_evolution"]["network_position_signals"][key] = label
        for r in preliminary:
            rho = r["dynamic_evolution"]["evolution_signals"].get("persistence_score")
            label = None if rho is None else ("high" if rho >= .67 else ("medium" if rho >= .34 else "low"))
            r["dynamic_evolution"]["evolution_signals"]["persistence"] = label

    return _json_safe(preliminary)


# ============================================================
# 4. Convenience: one-cutoff export
# ============================================================
def export_one_cutoff_evidence_inputs(
    *,
    df_raw: pd.DataFrame,
    static_profiles_path: str | Path,
    scen_map_df: Optional[pd.DataFrame] = None,
    analysis_cutoff: Optional[str] = None,
    target_categories: Optional[Sequence[str]] = None,
    A_dict: Optional[Mapping[str, Any]] = None,
    node_index: Optional[pd.DataFrame] = None,
    static_relations_path: Optional[str | Path] = None,
    pred_next_node: Optional[Any] = None,
    W_eff_node: Optional[Any] = None,
    dimension_params: Optional[pd.DataFrame] = None,
    node_dim_ids: Optional[Any] = None,
    target_month: Optional[str] = None,
    dynamic_profiles_path: Optional[str | Path] = None,
    max_evidence_cards_per_product: int = 10,
) -> Dict[str, Any]:
    """Build the A base plus optional B dynamic profiles for one cutoff."""
    static_profiles = build_pure_static_profiles(
        df_raw=df_raw,
        scen_map_df=scen_map_df,
        analysis_cutoff=analysis_cutoff,
        target_categories=target_categories,
        max_evidence_cards_per_product=max_evidence_cards_per_product,
    )
    save_jsonl(static_profiles, static_profiles_path)

    relation_profiles: List[Dict[str, Any]] = []
    if A_dict is not None and node_index is not None and static_relations_path is not None:
        relation_profiles = build_static_network_relation_profiles(
            A_dict=A_dict,
            node_index=node_index,
            target_categories=target_categories,
            analysis_cutoff=analysis_cutoff,
        )
        save_jsonl(relation_profiles, static_relations_path)

    dynamic_profiles: List[Dict[str, Any]] = []
    if (
        pred_next_node is not None
        and W_eff_node is not None
        and node_index is not None
        and dynamic_profiles_path is not None
    ):
        dynamic_profiles = build_dynamic_evolution_profiles(
            pred_next_node=pred_next_node,
            W_eff_node=W_eff_node,
            node_index=node_index,
            dimension_params=dimension_params,
            node_dim_ids=node_dim_ids,
            analysis_cutoff=analysis_cutoff,
            target_month=target_month,
            target_categories=target_categories,
        )
        save_jsonl(dynamic_profiles, dynamic_profiles_path)

    return {
        "static_profiles": static_profiles,
        "static_relation_profiles": relation_profiles,
        "dynamic_evolution_profiles": dynamic_profiles,
    }


__all__ = [
    "build_pure_static_profiles",
    "build_static_network_relation_profiles",
    "build_dynamic_evolution_profiles",
    "export_one_cutoff_evidence_inputs",
    "preprocess_review_rows",
    "summarize_static_issue_cards",
    "select_static_evidence_cards",
    "save_jsonl",
]

# ============================================================
# 5. Rolling convenience builders for historical A/B/C evaluation
# ============================================================
def build_pure_static_profiles_for_cutoffs(
    df_raw: pd.DataFrame,
    cutoffs: Sequence[str],
    scen_map_df: Optional[pd.DataFrame] = None,
    target_categories: Optional[Sequence[str]] = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """Build leakage-controlled static snapshots for several historical cutoffs."""
    out: List[Dict[str, Any]] = []
    for cutoff in cutoffs:
        out.extend(build_pure_static_profiles(
            df_raw=df_raw,
            scen_map_df=scen_map_df,
            analysis_cutoff=str(cutoff),
            target_categories=target_categories,
            **kwargs,
        ))
    return out


def _effective_matrix_from_params(
    A_dict: Mapping[str, Any],
    node_index: pd.DataFrame,
    param_df: pd.DataFrame,
    positive_beta_only: bool = False,
) -> np.ndarray:
    """Rebuild W_eff[j,i] from raw A matrices and target-node dimension betas."""
    meta = _ensure_node_index(node_index)
    mats = {k: _to_numpy_matrix(v) for k, v in A_dict.items() if k in {"api", "coo", "user"}}
    if set(mats) != {"api", "coo", "user"}:
        raise ValueError("A_dict must contain api/coo/user")
    p = param_df.copy()
    if "dimension" not in p.columns:
        raise ValueError("rolling params_df must contain dimension names for automatic W_eff reconstruction")
    need = ["beta_api", "beta_coo", "beta_user"]
    if any(c not in p.columns for c in need):
        raise ValueError(f"params_df must contain {need}")
    p = p.drop_duplicates("dimension", keep="last").set_index(p["dimension"].astype(str))

    beta_api = np.zeros(len(meta), dtype=float)
    beta_coo = np.zeros(len(meta), dtype=float)
    beta_user = np.zeros(len(meta), dtype=float)
    for _, r in meta.iterrows():
        nid = int(r["node_id"])
        dim = str(r["dimension"])
        if dim not in p.index:
            continue
        pr = p.loc[dim]
        # p.loc can be DataFrame if duplicated despite defensive drop; handle first.
        if isinstance(pr, pd.DataFrame):
            pr = pr.iloc[0]
        vals = [float(pr[c]) if pd.notna(pr[c]) else 0.0 for c in need]
        if positive_beta_only:
            vals = [max(v, 0.0) for v in vals]
        beta_api[nid], beta_coo[nid], beta_user[nid] = vals

    # Receiver/target node j determines dimension-specific beta.
    W = (
        beta_api[:, None] * mats["api"]
        + beta_coo[:, None] * mats["coo"]
        + beta_user[:, None] * mats["user"]
    )
    np.fill_diagonal(W, 0.0)
    return W


def build_dynamic_evolution_profiles_from_res_roll(
    res_roll: Dict[str, Any],
    A_dict: Mapping[str, Any],
    node_index: Optional[pd.DataFrame] = None,
    target_months: Optional[Sequence[str]] = None,
    target_categories: Optional[Sequence[str]] = None,
    seed_aggregation: str = "mean",
    positive_beta_only: bool = False,
    top_dimensions: int = 5,
    exclude_general_dimension: bool = True,
    general_dimension_names: Optional[set[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Build B snapshots from already-computed rolling outputs; no model retraining and
    no risk-chain extraction are required.

    The exported snapshots include the complete non-General ``dimension_signals``
    table (through ``build_dynamic_evolution_profiles``) while retaining the
    original top-K B/C baseline fields.

    Required res_roll fields:
      forecast_df: month, node_id, pred_level (and optionally seed)
      params_df:   month, dimension, beta_api/beta_coo/beta_user/rho (optionally seed)
    """
    forecast = res_roll.get("forecast_df")
    params = res_roll.get("params_df")
    if not isinstance(forecast, pd.DataFrame) or not isinstance(params, pd.DataFrame):
        raise ValueError("res_roll must contain DataFrame forecast_df and params_df")
    if node_index is None:
        node_index = res_roll.get("node_index")
    if not isinstance(node_index, pd.DataFrame):
        raise ValueError("Provide node_index or res_roll['node_index']")

    month_col_f = "month" if "month" in forecast.columns else "target_month"
    month_col_p = "month" if "month" in params.columns else "target_month"
    for c in [month_col_f, "node_id", "pred_level"]:
        if c not in forecast.columns:
            raise ValueError(f"forecast_df missing {c}")
    for c in [month_col_p, "dimension", "rho", "beta_api", "beta_coo", "beta_user"]:
        if c not in params.columns:
            raise ValueError(f"params_df missing {c}")

    available = sorted(set(forecast[month_col_f].astype(str)).intersection(params[month_col_p].astype(str)))
    months = available if target_months is None else [str(m) for m in target_months if str(m) in available]
    meta = _ensure_node_index(node_index)
    n_nodes = len(meta)
    out: List[Dict[str, Any]] = []

    for target_month in months:
        f = forecast[forecast[month_col_f].astype(str) == target_month].copy()
        p = params[params[month_col_p].astype(str) == target_month].copy()

        if seed_aggregation != "mean":
            raise ValueError("Currently supported seed_aggregation: 'mean'")
        f_agg = f.groupby("node_id", as_index=False)["pred_level"].mean()
        p_agg = p.groupby("dimension", as_index=False)[["rho", "beta_api", "beta_coo", "beta_user"]].mean()

        R = np.zeros(n_nodes, dtype=float)
        for _, r in f_agg.iterrows():
            nid = int(r["node_id"])
            if 0 <= nid < n_nodes:
                R[nid] = max(float(r["pred_level"]), 0.0)

        W_eff = _effective_matrix_from_params(
            A_dict=A_dict,
            node_index=meta,
            param_df=p_agg,
            positive_beta_only=positive_beta_only,
        )
        analysis_cutoff = str(pd.Period(target_month, freq="M") - 1)
        profiles = build_dynamic_evolution_profiles(
            pred_next_node=R,
            W_eff_node=W_eff,
            node_index=meta,
            dimension_params=p_agg,
            analysis_cutoff=analysis_cutoff,
            target_month=target_month,
            target_categories=target_categories,
            top_dimensions=top_dimensions,
            exclude_general_dimension=exclude_general_dimension,
            general_dimension_names=general_dimension_names,
        )
        for r in profiles:
            r["seed_aggregation"] = seed_aggregation
            r["profile_source"] = "rolling_forecast_and_params_no_explicit_risk_chain"
        out.extend(profiles)
    return _json_safe(out)



def load_res_roll_csv(save_dir):
    forecast_df = pd.read_csv(os.path.join(save_dir, "forecast_df.csv"))
    params_df = pd.read_csv(os.path.join(save_dir, "params_df.csv"))
    history_df = pd.read_csv(os.path.join(save_dir, "history_df.csv"))
    node_index = pd.read_csv(os.path.join(save_dir, "node_index.csv"))

    with open(os.path.join(save_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    return {
        "forecast_df": forecast_df,
        "params_df": params_df,
        "history_df": history_df,
        "months": meta["months"],
        "target_months_all": meta["target_months_all"],
        "node_index": node_index,
        "feature_names": meta["feature_names"],
    }


__all__.extend([
    "build_pure_static_profiles_for_cutoffs",
    "build_dynamic_evolution_profiles_from_res_roll",
    "load_res_roll_csv",
])
