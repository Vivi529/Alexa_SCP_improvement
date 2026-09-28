from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from data_service import (
    CONDITION_A,
    CONDITION_C,
    build_all_condition_packages,
    build_hybrid_case_package,
    build_profile_index,
    canonical_category_name,
    load_jsonl,
    resolve_profile_row,
    retrieve_api_evidence_static_only,
)
from diagnosis_service import (
    GenerationTraceError,
    generate_dimension_anchor,
    generate_hybrid_diagnosis,
)
from ecosystem_service import (
    EcosystemContextConfig,
    EcosystemGenerationTraceError,
    generate_ecosystem_synthesis,
    select_ecosystem_context,
)
from action_service import (
    ActionGenerationTraceError,
    generate_prospective_actions,
)
from feedback_service import apply_action_feedback, apply_mechanism_feedback
from workflow_store import make_workflow_id, save_named_artifact, save_workflow, utc_now_iso


EXPERIMENT_CONFIG = {
    "max_observed_cards": 6,
    "statements_per_card": 2,
    "max_relations": 5,
    "max_api_cards": 3,
    "max_chains": 4,
    "max_standalone_chains": 1,
    "chain_selection_mode": "dimension_diverse",
    "hybrid_max_promotions": 0,
    "hybrid_max_dynamic_rank": 3,
}


DEFAULT_PROSPECTIVE_CONFIG = {
    "max_observed_cards": 6,
    "statements_per_card": 3,
    "max_relations": 5,
    "max_api_cards": 5,
    "max_chains": 4,
    "max_standalone_chains": 1,
    "chain_selection_mode": "dimension_diverse",
    "hybrid_max_promotions": 0,
    "hybrid_max_dynamic_rank": 3,
}


STATUS_DIAGNOSIS_PENDING = "DIAGNOSIS_PENDING"
STATUS_H1_COMPLETED = "H1_COMPLETED"
STATUS_ROUTING_COMPLETED = "ROUTING_COMPLETED"
STATUS_WAITING_MECHANISM_REVIEW = "WAITING_MECHANISM_REVIEW"
STATUS_BLOCKED_OUTPUT_REVIEW = "BLOCKED_OUTPUT_REVIEW"
STATUS_READY_FOR_ACTION = "READY_FOR_ACTION"
STATUS_WAITING_ACTION_REVIEW = "WAITING_ACTION_REVIEW"
STATUS_BLOCKED_ACTION_OUTPUT_REVIEW = "BLOCKED_ACTION_OUTPUT_REVIEW"
STATUS_COMPLETED = "COMPLETED"
STATUS_NO_ACTION = "NO_ACTION_REQUIRED"


# ---------------------------------------------------------------------------
# Data / preflight
# ---------------------------------------------------------------------------

def load_prospective_data(
    data_dir: str | Path,
    *,
    expected_analysis_cutoff: str | None = None,
    expected_target_month: str | None = None,
) -> Dict[str, Any]:

    root = Path(data_dir)

    required = [
        "product_static_profiles_prospective.jsonl",
        "product_dynamic_evolution_prospective.jsonl",
        "static_network_relations_prospective.jsonl",
        "risk_chains_prospective.jsonl",
        "api_cards.jsonl",
        "category_api_map.json",
    ]

    missing = [
        name
        for name in required
        if not (root / name).exists()
    ]

    if missing:
        raise FileNotFoundError(
            f"Prospective data directory is incomplete: "
            f"missing {missing} under {root}"
        )

    static_rows = load_jsonl(
        root / "product_static_profiles_prospective.jsonl"
    )

    dynamic_rows = load_jsonl(
        root / "product_dynamic_evolution_prospective.jsonl"
    )

    relation_rows = load_jsonl(
        root / "static_network_relations_prospective.jsonl"
    )

    chain_rows = load_jsonl(
        root / "risk_chains_prospective.jsonl"
    )

    api_cards = load_jsonl(
        root / "api_cards.jsonl"
    )

    # --------------------------------------------------------
    # Snapshot validation
    # --------------------------------------------------------
    if expected_analysis_cutoff is not None:
        cutoff = str(expected_analysis_cutoff)

        bad_static = [
            r for r in static_rows
            if str(r.get("analysis_cutoff")) != cutoff
        ]

        bad_dynamic = [
            r for r in dynamic_rows
            if str(r.get("analysis_cutoff")) != cutoff
        ]

        bad_relations = [
            r for r in relation_rows
            if str(r.get("analysis_cutoff")) != cutoff
        ]

        bad_chains = [
            r for r in chain_rows
            if str(r.get("analysis_cutoff")) != cutoff
        ]

        if bad_static:
            raise ValueError(
                "Static prospective evidence does not match "
                f"analysis_cutoff={cutoff}"
            )

        if bad_dynamic:
            raise ValueError(
                "Dynamic prospective evidence does not match "
                f"analysis_cutoff={cutoff}"
            )

        if bad_relations:
            raise ValueError(
                "Static-relation prospective evidence does not match "
                f"analysis_cutoff={cutoff}"
            )

        if bad_chains:
            raise ValueError(
                "Risk-chain prospective evidence does not match "
                f"analysis_cutoff={cutoff}"
            )

    if expected_target_month is not None:
        target = str(expected_target_month)

        bad_dynamic = [
            r for r in dynamic_rows
            if str(r.get("target_month")) != target
        ]

        bad_chains = [
            r for r in chain_rows
            if str(r.get("target_month")) != target
        ]

        if bad_dynamic:
            raise ValueError(
                "Dynamic prospective evidence does not match "
                f"target_month={target}"
            )

        if bad_chains:
            raise ValueError(
                "Risk-chain prospective evidence does not match "
                f"target_month={target}"
            )

    return {
        "data_dir": str(root),
        "static_rows": static_rows,
        "dynamic_rows": dynamic_rows,
        "api_cards": api_cards,
        "relation_rows": relation_rows,
        "static_index": build_profile_index(static_rows),
        "dynamic_index": build_profile_index(dynamic_rows),
        "risk_chain_path": str(root / "risk_chains_prospective.jsonl"),
        "category_api_map_path": str(root / "category_api_map.json"),
    }




def load_risk_selection_bundle(path: str | Path) -> Dict[str, Any]:
    """Load a one-shot risk-chain selection artifact saved with ``torch.save``.

    Accepted shapes:
      1. the raw ``risk_chain_res`` mapping;
      2. a wrapper ``{"risk_chain_res": ..., "analysis_cutoff": ..., "target_month": ...}``.

    When month metadata is absent, it is inferred only if ``chain_df`` contains a
    unique non-empty value for the corresponding column. Ambiguous time metadata
    is never guessed.
    """
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependency
        raise ImportError("torch is required to load a .pt risk-selection artifact") from exc

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    try:
        obj = torch.load(p, map_location="cpu", weights_only=False)
    except TypeError:  # older torch
        obj = torch.load(p, map_location="cpu")
    if not isinstance(obj, Mapping):
        raise TypeError("Risk-selection artifact must contain a mapping")

    if isinstance(obj.get("risk_chain_res"), Mapping):
        risk_selection = obj["risk_chain_res"]
        analysis_cutoff = str(obj.get("analysis_cutoff") or "").strip()
        target_month = str(obj.get("target_month") or "").strip()
    else:
        risk_selection = obj
        analysis_cutoff = str(obj.get("analysis_cutoff") or "").strip()
        target_month = str(obj.get("target_month") or "").strip()

    rank_df = risk_selection.get("selected_rank_df")
    if not isinstance(rank_df, pd.DataFrame):
        raise TypeError("Loaded risk_selection has no pandas selected_rank_df")

    chain_df = risk_selection.get("chain_df")
    if isinstance(chain_df, pd.DataFrame):
        if not analysis_cutoff and "analysis_cutoff" in chain_df.columns:
            values = sorted({str(x).strip() for x in chain_df["analysis_cutoff"].dropna() if str(x).strip()})
            if len(values) == 1:
                analysis_cutoff = values[0]
        if not target_month and "target_month" in chain_df.columns:
            values = sorted({str(x).strip() for x in chain_df["target_month"].dropna() if str(x).strip()})
            if len(values) == 1:
                target_month = values[0]

    return {
        "risk_selection": risk_selection,
        "analysis_cutoff": analysis_cutoff or None,
        "target_month": target_month or None,
        "source_path": str(p),
    }


def preflight_prospective_system(
    *,
    data_context: Dict[str, Any],
    model_name: Optional[str] = None,
) -> None:
    """Fail before the first paid call when local system/configuration is broken."""
    required_keys = {
        "static_index",
        "dynamic_index",
        "api_cards",
        "relation_rows",
        "risk_chain_path",
        "category_api_map_path",
    }
    missing_keys = sorted(required_keys - set(data_context))
    if missing_keys:
        raise KeyError(f"data_context missing required keys: {missing_keys}")

    for path_key in ("risk_chain_path", "category_api_map_path"):
        path = Path(str(data_context[path_key]))
        if not path.exists():
            raise FileNotFoundError(f"{path_key} does not exist: {path}")

    if not callable(generate_dimension_anchor):
        raise TypeError("generate_dimension_anchor is not callable")
    if not callable(generate_hybrid_diagnosis):
        raise TypeError("generate_hybrid_diagnosis is not callable")
    if not callable(generate_prospective_actions):
        raise TypeError("generate_prospective_actions is not callable")

    if not os.getenv("LLM_API_KEY"):
        raise RuntimeError("LLM_API_KEY is not configured")
    if not (model_name or os.getenv("LLM_MODEL")):
        raise RuntimeError("LLM_MODEL is not configured and model_name was not supplied")


def _merge_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = dict(DEFAULT_PROSPECTIVE_CONFIG)
    if config is not None and not isinstance(config, dict):
        raise TypeError("config must be a dict or None")
    if config:

        unknown = sorted(set(config) - set(DEFAULT_PROSPECTIVE_CONFIG))
        if unknown:
            raise KeyError(f"Unknown prospective config keys: {unknown}")
        out.update(config)

    for key in (
        "max_observed_cards",
        "statements_per_card",
        "max_relations",
        "max_api_cards",
        "max_chains",
        "max_standalone_chains",
        "hybrid_max_dynamic_rank",
    ):
        value = int(out[key])
        if value < 0 or (key == "hybrid_max_dynamic_rank" and value < 1):
            raise ValueError(f"{key} has invalid value: {value}")
        out[key] = value

    mode = str(out.get("chain_selection_mode") or "").strip().lower()
    if mode not in {"dimension_diverse", "global_topk"}:
        raise ValueError(
            "chain_selection_mode must be 'dimension_diverse' or 'global_topk'"
        )
    out["chain_selection_mode"] = mode

    out["hybrid_max_promotions"] = int(out["hybrid_max_promotions"])
    if out["hybrid_max_promotions"] != 0:
        raise ValueError(
            "The final prospective H pipeline is frozen to the selected hard-anchor "
            "variant (hybrid_max_promotions=0). Use the experiment runner for "
            "controlled-promotion ablations."
        )
    return out


def _validate_product_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(spec, dict):
        raise TypeError("Each Top-K product specification must be a dict")

    category = canonical_category_name(spec.get("product_category"))
    cutoff = str(spec.get("analysis_cutoff") or "").strip()
    target = str(spec.get("target_month") or "").strip()
    if not category:
        raise ValueError("Top-K product specification requires product_category")
    if not cutoff:
        raise ValueError(f"{category}: analysis_cutoff is required")
    if not target:
        raise ValueError(f"{category}: target_month is required")

    def _parse_month(value: str, field: str) -> tuple[int, int]:
        parts = value.split("-")
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            raise ValueError(f"{category}: {field} must use YYYY-MM; got {value!r}")
        year, month = int(parts[0]), int(parts[1])
        if len(parts[0]) != 4 or len(parts[1]) != 2 or not (1 <= month <= 12):
            raise ValueError(f"{category}: {field} must use YYYY-MM; got {value!r}")
        return year, month

    cutoff_year, cutoff_month = _parse_month(cutoff, "analysis_cutoff")
    target_year, target_m = _parse_month(target, "target_month")
    expected_year = cutoff_year + (1 if cutoff_month == 12 else 0)
    expected_month = 1 if cutoff_month == 12 else cutoff_month + 1
    if (target_year, target_m) != (expected_year, expected_month):
        expected = f"{expected_year:04d}-{expected_month:02d}"
        raise ValueError(
            f"{category}: target_month must be the month immediately after "
            f"analysis_cutoff for the ex-ante H pipeline; expected {expected!r}, "
            f"got {target!r}"
        )

    out = dict(spec)
    out["product_category"] = category
    out["analysis_cutoff"] = cutoff
    out["target_month"] = target
    return out


def _selection_context(spec: Dict[str, Any]) -> Dict[str, Any]:
    metrics = spec.get("selection_metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    return {
        "priority_rank": spec.get("priority_rank"),
        "selection_method": spec.get("selection_method") or spec.get("selection_reason"),
        "selection_score": spec.get("selection_score"),
        # Backward-compatible field for manually supplied legacy specs only.
        "risk_score": spec.get("risk_score"),
        "selection_metrics": copy.deepcopy(metrics),
        "analysis_cutoff": spec.get("analysis_cutoff"),
        "target_month": spec.get("target_month"),
    }


def build_topk_product_specs_from_risk_selection(
    *,
    risk_selection: Mapping[str, Any],
    analysis_cutoff: str,
    target_month: str,
    expected_k: int = 5,
) -> List[Dict[str, Any]]:
    """Convert the upstream one-shot ``risk_chain_res`` into H input specs.

    Product selection is NOT recomputed here. ``selected_rank_df`` is treated as
    authoritative; this function only validates and serializes its chosen set.
    """
    if not isinstance(risk_selection, Mapping):
        raise TypeError("risk_selection must be a mapping")
    rank_df = risk_selection.get("selected_rank_df")
    if not isinstance(rank_df, pd.DataFrame):
        raise TypeError("risk_selection['selected_rank_df'] must be a pandas DataFrame")
    if rank_df.empty:
        raise ValueError("risk_selection['selected_rank_df'] is empty")

    required = {"rank", "Category"}
    missing = sorted(required - set(rank_df.columns))
    if missing:
        raise ValueError(f"selected_rank_df missing required columns: {missing}")

    expected_k = int(expected_k)
    if expected_k < 1:
        raise ValueError("expected_k must be >= 1")

    df = rank_df.copy()
    rank_values = pd.to_numeric(df["rank"], errors="raise")
    if rank_values.isna().any() or ((rank_values % 1) != 0).any():
        raise ValueError("selected_rank_df.rank must contain finite integer ranks")
    df["_rank"] = rank_values.astype(int)
    df["_category"] = df["Category"].map(canonical_category_name)
    if (df["_category"].astype(str).str.len() == 0).any():
        raise ValueError("selected_rank_df contains an empty Category")
    if df["_category"].duplicated().any():
        dups = df.loc[df["_category"].duplicated(keep=False), "_category"].tolist()
        raise ValueError(f"selected_rank_df contains duplicate categories: {dups}")
    if df["_rank"].duplicated().any():
        raise ValueError("selected_rank_df contains duplicate rank values")

    df = df.sort_values("_rank").reset_index(drop=True)
    expected_ranks = list(range(1, len(df) + 1))
    if df["_rank"].tolist() != expected_ranks:
        raise ValueError(
            "selected_rank_df rank must be consecutive and 1-based; "
            f"got {df['_rank'].tolist()}"
        )
    if len(df) != expected_k:
        raise ValueError(
            f"Expected exactly {expected_k} focal products, but selected_rank_df has {len(df)}"
        )

    summary = risk_selection.get("summary")
    if isinstance(summary, dict) and summary.get("selected_categories"):
        summary_cats = [canonical_category_name(x) for x in summary.get("selected_categories") or []]
        if set(summary_cats) != set(df["_category"].tolist()):
            raise ValueError(
                "risk_selection summary.selected_categories disagrees with selected_rank_df"
            )

    def _num(row: pd.Series, key: str, *, integer: bool = False) -> Any:
        if key not in row.index or pd.isna(row[key]):
            return None
        return int(row[key]) if integer else float(row[key])

    specs: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        metrics = {
            "marginal_chain_value": _num(row, "marginal_chain_value"),
            "captured_chain_count": _num(row, "captured_chain_count", integer=True),
            "captured_chain_weight_raw": _num(row, "captured_chain_weight_raw"),
        }
        specs.append({
            "product_category": str(row["_category"]),
            "priority_rank": int(row["_rank"]),
            "selection_method": "risk_chain_coverage_objective",
            "selection_score": metrics["marginal_chain_value"],
            "selection_metrics": metrics,
            "analysis_cutoff": str(analysis_cutoff),
            "target_month": str(target_month),
        })

    # Reuse the product-spec time/category validator before any model call.
    return [_validate_product_spec(x) for x in specs]


def _persist(record: Dict[str, Any], workflow_dir: Optional[str | Path]) -> None:
    if workflow_dir is not None:
        save_workflow(record, workflow_dir)


def _first_attempt(trace: Dict[str, Any]) -> Dict[str, Any]:
    attempts = trace.get("attempts") if isinstance(trace, dict) else None
    if isinstance(attempts, list) and attempts and isinstance(attempts[0], dict):
        return copy.deepcopy(attempts[0])
    return {}


def _has_completed_llm_attempt(exc: Exception) -> bool:
    """True only when the model/provider returned a completed response to audit.

    diagnosis_service uses GenerationTraceError for both API-call failures and
    post-response parse/validation failures. Empty attempts therefore mean a
    systemic/provider failure and must be re-raised rather than converted into
    a human-review artifact.
    """
    trace = getattr(exc, "llm_trace", None)
    attempts = trace.get("attempts") if isinstance(trace, dict) else None
    return bool(isinstance(attempts, list) and attempts)


def _stage_from_success(result: Dict[str, Any]) -> Dict[str, Any]:
    attempt = _first_attempt(result.get("llm_trace") or {})
    return {
        "status": "GENERATED",
        "input_payload": copy.deepcopy(attempt.get("input_payload")),
        "raw_output": result.get("raw_output", ""),
        "parsed_output": copy.deepcopy(result.get("parsed")),
        "parse_error": attempt.get("parse_error"),
        "validation_error": attempt.get("validation_error"),
        "usage": copy.deepcopy(result.get("usage") or {}),
        "request_meta": copy.deepcopy(attempt.get("request_meta") or {}),
        "llm_trace": copy.deepcopy(result.get("llm_trace") or {}),
    }


def _stage_from_output_error(exc: Exception) -> Dict[str, Any]:
    trace = copy.deepcopy(getattr(exc, "llm_trace", None) or {"attempts": []})
    attempt = _first_attempt(trace)
    return {
        "status": "BLOCKED_OUTPUT_CONTRACT",
        "input_payload": copy.deepcopy(attempt.get("input_payload")),
        "raw_output": attempt.get("raw_output", ""),
        "parsed_output": copy.deepcopy(attempt.get("parsed_output")),
        "parse_error": attempt.get("parse_error"),
        "validation_error": attempt.get("validation_error") or str(exc),
        "usage": copy.deepcopy(attempt.get("usage") or {}),
        "request_meta": copy.deepcopy(attempt.get("request_meta") or {}),
        "llm_trace": trace,
    }


# ---------------------------------------------------------------------------
# H diagnosis stage
# ---------------------------------------------------------------------------

def run_h_product_diagnosis(
    *,
    product_spec: Dict[str, Any],
    data_context: Dict[str, Any],
    model_name: Optional[str] = None,
    temperature: float = 0.2,
    config: Optional[Dict[str, Any]] = None,
    workflow_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Run one H product diagnosis with output-error preservation and fail-fast system errors.

    Only GenerationTraceError is converted into a BLOCKED_OUTPUT_REVIEW artifact.
    API/runtime/import/data errors are not swallowed and immediately propagate.
    """
    spec = _validate_product_spec(product_spec)
    cfg = _merge_config(config)
    category = spec["product_category"]
    cutoff = spec["analysis_cutoff"]
    target_month = spec["target_month"]
    # Local configuration errors are detected before any workflow/cache file is created.
    preflight_prospective_system(data_context=data_context, model_name=model_name)

    workflow_id = make_workflow_id(
        product_category=category,
        analysis_cutoff=cutoff,
        target_month=target_month,
    )

    record: Dict[str, Any] = {
        "workflow": "H_prospective_v2_audited",
        "workflow_id": workflow_id,
        "status": STATUS_DIAGNOSIS_PENDING,
        "selection_context": _selection_context(spec),
        "prospective_config": copy.deepcopy(cfg),
        "requested_case": {
            "product_category": category,
            "analysis_cutoff": cutoff,
            "target_month": target_month,
        },
        "stage_outputs": {},
        "created_at": utc_now_iso(),
    }

    # System/data failures here intentionally propagate. No cache is written yet.
    static_row = resolve_profile_row(data_context["static_index"], category, cutoff)
    dynamic_row = resolve_profile_row(data_context["dynamic_index"], category, cutoff)

    dynamic_target = str(dynamic_row.get("target_month") or "").strip()
    if dynamic_target and dynamic_target != target_month:
        raise ValueError(
            f"{category}: dynamic profile target_month={dynamic_target!r} "
            f"does not match requested target_month={target_month!r}"
        )

    packages = build_all_condition_packages(
        category=category,
        static_row=static_row,
        dynamic_row=dynamic_row,
        api_cards=data_context["api_cards"],
        static_relation_rows=data_context["relation_rows"],
        risk_chain_path=data_context["risk_chain_path"],
        analysis_cutoff=cutoff,
        target_month=target_month,
        max_observed_cards=int(cfg["max_observed_cards"]),
        statements_per_card=int(cfg["statements_per_card"]),
        max_relations=int(cfg["max_relations"]),
        max_api_cards=int(cfg["max_api_cards"]),
        max_chains=int(cfg["max_chains"]),
        max_standalone_chains=int(cfg["max_standalone_chains"]),
        chain_selection_mode=str(cfg["chain_selection_mode"]),
        category_api_map_path=data_context.get("category_api_map_path"),
    )

    # H1: model-output failures are reviewable; system/API failures propagate.
    try:
        core_result = generate_dimension_anchor(
            packages[CONDITION_A],
            model_name=model_name,
            temperature=temperature,
        )
    except GenerationTraceError as exc:
        if not _has_completed_llm_attempt(exc):
            raise
        record["status"] = STATUS_BLOCKED_OUTPUT_REVIEW
        record["failed_stage"] = "dimension_anchor"
        record["stage_outputs"]["dimension_anchor"] = _stage_from_output_error(exc)
        record["case_meta"] = copy.deepcopy(packages[CONDITION_A].get("case_meta") or {})
        record["case_package"] = copy.deepcopy(packages[CONDITION_A])
        _persist(record, workflow_dir)
        return record

    core_anchor = core_result["parsed"]
    record["stage_outputs"]["dimension_anchor"] = _stage_from_success(core_result)
    record["core_dimension_anchor"] = copy.deepcopy(core_anchor)
    record["status"] = STATUS_H1_COMPLETED
    _persist(record, workflow_dir)

    # H2/H3 deterministic routing. Any exception is a system/data/program error.
    hybrid_package = build_hybrid_case_package(
        base_c_package=packages[CONDITION_C],
        dynamic_row=dynamic_row,
        risk_chain_path=data_context["risk_chain_path"],
        anchor_dimensions=core_anchor.get("risk_dimensions") or [],
        max_chains=int(cfg["max_chains"]),
        max_standalone_chains=int(cfg["max_standalone_chains"]),
        chain_selection_mode=str(cfg["chain_selection_mode"]),
        max_promotions=int(cfg["hybrid_max_promotions"]),
        max_dynamic_rank=int(cfg["hybrid_max_dynamic_rank"]),
    )
    final_anchor = hybrid_package.get("dimension_anchor")
    if not isinstance(final_anchor, dict):
        raise ValueError("Hybrid package is missing final dimension_anchor")

    record["case_meta"] = copy.deepcopy(hybrid_package.get("case_meta") or {})
    record["final_dimension_anchor"] = copy.deepcopy(final_anchor)
    record["case_package"] = hybrid_package
    record["status"] = STATUS_ROUTING_COMPLETED
    _persist(record, workflow_dir)

    # H4: same output-quality/system-error split as H1.
    try:
        diagnosis_result = generate_hybrid_diagnosis(
            case_package=hybrid_package,
            dimension_anchor=final_anchor,
            model_name=model_name,
            temperature=temperature,
        )
    except GenerationTraceError as exc:
        if not _has_completed_llm_attempt(exc):
            raise
        record["status"] = STATUS_BLOCKED_OUTPUT_REVIEW
        record["failed_stage"] = "diagnosis"
        record["stage_outputs"]["diagnosis"] = _stage_from_output_error(exc)
        _persist(record, workflow_dir)
        return record

    record.update({
        "status": STATUS_WAITING_MECHANISM_REVIEW,
        "diagnosis": copy.deepcopy(diagnosis_result["parsed"]),
        "llm_trace": {
            "dimension_anchor": copy.deepcopy(core_result.get("llm_trace") or {}),
            "diagnosis": copy.deepcopy(diagnosis_result.get("llm_trace") or {}),
        },
        "usage": {
            "dimension_anchor": copy.deepcopy(core_result.get("usage") or {}),
            "diagnosis": copy.deepcopy(diagnosis_result.get("usage") or {}),
        },
    })
    record["stage_outputs"]["diagnosis"] = _stage_from_success(diagnosis_result)
    _persist(record, workflow_dir)
    return record


def run_h_topk_diagnoses(
    *,
    topk_products: Sequence[Dict[str, Any]],
    data_context: Dict[str, Any],
    model_name: Optional[str] = None,
    temperature: float = 0.2,
    config: Optional[Dict[str, Any]] = None,
    workflow_dir: Optional[str | Path] = None,
) -> List[Dict[str, Any]]:
    """Run H for Top-K.

    Output-contract blocks are returned as explicit records and the next product
    may proceed. Any systemic exception propagates immediately and aborts the
    batch. Completed earlier products remain durably persisted when workflow_dir
    is supplied.
    """
    preflight_prospective_system(data_context=data_context, model_name=model_name)
    rows = [_validate_product_spec(dict(x)) for x in topk_products]
    categories = [x["product_category"] for x in rows]
    if len(categories) != len(set(categories)):
        raise ValueError("Top-K input contains duplicate product categories")
    time_pairs = {(x["analysis_cutoff"], x["target_month"]) for x in rows}
    if len(time_pairs) > 1:
        raise ValueError("All Top-K products must share the same analysis_cutoff/target_month")
    ranks = [x.get("priority_rank") for x in rows if x.get("priority_rank") is not None]
    if len(ranks) != len({str(x) for x in ranks}):
        raise ValueError("Top-K input contains duplicate priority_rank values")
    rows.sort(key=lambda x: (
        int(x.get("priority_rank")) if str(x.get("priority_rank") or "").isdigit() else 10**9,
        x["product_category"],
    ))

    records: List[Dict[str, Any]] = []
    for row in rows:
        # No broad try/except here: system failures MUST abort immediately.
        record = run_h_product_diagnosis(
            product_spec=row,
            data_context=data_context,
            model_name=model_name,
            temperature=temperature,
            config=config,
            workflow_dir=workflow_dir,
        )
        records.append(record)
    return records


def run_h_from_risk_selection(
    *,
    risk_selection: Mapping[str, Any],
    analysis_cutoff: str,
    target_month: str,
    data_context: Dict[str, Any],
    model_name: Optional[str] = None,
    temperature: float = 0.2,
    config: Optional[Dict[str, Any]] = None,
    workflow_dir: Optional[str | Path] = None,
    expected_k: int = 5,
) -> List[Dict[str, Any]]:
    """Formal entry point from one-shot ``risk_chain_res`` to product-level H."""
    specs = build_topk_product_specs_from_risk_selection(
        risk_selection=risk_selection,
        analysis_cutoff=analysis_cutoff,
        target_month=target_month,
        expected_k=expected_k,
    )
    return run_h_topk_diagnoses(
        topk_products=specs,
        data_context=data_context,
        model_name=model_name,
        temperature=temperature,
        config=config,
        workflow_dir=workflow_dir,
    )


# ---------------------------------------------------------------------------
# Human Gate 1 persistence
# ---------------------------------------------------------------------------

def record_mechanism_review(
    *,
    diagnosis_record: Dict[str, Any],
    mechanism_feedback: Dict[str, Any],
    workflow_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Persist Gate 1 immediately, without generating actions."""
    if diagnosis_record.get("status") != STATUS_WAITING_MECHANISM_REVIEW:
        raise ValueError(
            f"Workflow is not waiting for mechanism review: "
            f"status={diagnosis_record.get('status')!r}"
        )
    diagnosis = diagnosis_record.get("diagnosis")
    if not isinstance(diagnosis, dict):
        raise ValueError("diagnosis_record must contain diagnosis")

    out = copy.deepcopy(diagnosis_record)
    out["mechanism_feedback"] = copy.deepcopy(mechanism_feedback)
    out["reviewed_diagnosis"] = apply_mechanism_feedback(diagnosis, mechanism_feedback)
    out["status"] = STATUS_READY_FOR_ACTION
    out["mechanism_reviewed_at"] = utc_now_iso()
    _persist(out, workflow_dir)
    return out


# ---------------------------------------------------------------------------
# Human Gate 1 -> action
# ---------------------------------------------------------------------------

def generate_product_actions_after_review(
    *,
    diagnosis_record: Dict[str, Any],
    mechanism_feedback: Dict[str, Any],
    decision_context: Dict[str, Any],
    model_name: Optional[str] = None,
    temperature: float = 0.2,
    workflow_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Human Gate 1 -> prospective action generation with partial salvage.

    A completed model response may contain a mixture of valid and invalid action
    items. Invalid items are discarded by action_service, fully audited, and do
    not block valid siblings. Root-level contract failures or zero surviving
    actions remain BLOCKED_ACTION_OUTPUT_REVIEW. Provider/runtime errors fail fast.
    """
    if diagnosis_record.get("status") not in {
        STATUS_WAITING_MECHANISM_REVIEW,
        STATUS_READY_FOR_ACTION,
    }:
        raise ValueError(
            "Workflow is not ready for mechanism review/action generation: "
            f"status={diagnosis_record.get('status')!r}"
        )

    diagnosis = diagnosis_record.get("diagnosis")
    case_package = diagnosis_record.get("case_package")
    if not isinstance(diagnosis, dict) or not isinstance(case_package, dict):
        raise ValueError("diagnosis_record must contain diagnosis and case_package")

    if diagnosis_record.get("status") == STATUS_WAITING_MECHANISM_REVIEW:
        out = record_mechanism_review(
            diagnosis_record=diagnosis_record,
            mechanism_feedback=mechanism_feedback,
            workflow_dir=workflow_dir,
        )
    else:
        out = copy.deepcopy(diagnosis_record)
        persisted_feedback = out.get("mechanism_feedback")
        if persisted_feedback is not None and mechanism_feedback != persisted_feedback:
            raise ValueError("mechanism_feedback differs from the persisted Gate-1 review")

    reviewed_diagnosis = out.get("reviewed_diagnosis")
    if not isinstance(reviewed_diagnosis, dict):
        reviewed_diagnosis = apply_mechanism_feedback(diagnosis, mechanism_feedback)
        out["reviewed_diagnosis"] = reviewed_diagnosis

    out["decision_context"] = copy.deepcopy(decision_context)
    out["status"] = STATUS_READY_FOR_ACTION
    _persist(out, workflow_dir)

    try:
        action_result = generate_prospective_actions(
            reviewed_diagnosis=reviewed_diagnosis,
            case_package=case_package,
            decision_context=decision_context,
            model_name=model_name,
            temperature=temperature,
        )
    except ActionGenerationTraceError as exc:
        if not _has_completed_llm_attempt(exc):
            raise
        out["status"] = STATUS_BLOCKED_ACTION_OUTPUT_REVIEW
        out["failed_stage"] = "action"
        out.setdefault("stage_outputs", {})["action"] = _stage_from_output_error(exc)
        out.setdefault("llm_trace", {})["action"] = copy.deepcopy(exc.llm_trace)
        out["action_generated_at"] = utc_now_iso()
        _persist(out, workflow_dir)
        return out

    action_output = copy.deepcopy(action_result.get("parsed") or {})
    discarded_actions = copy.deepcopy(action_result.get("discarded_actions") or [])
    n_generated = int(action_result.get("n_generated") or 0)
    n_valid = int(action_result.get("n_valid") or len(action_output.get("actions") or []))
    n_discarded = int(action_result.get("n_discarded") or len(discarded_actions))

    out.update({
        "action_output": action_output,
        "action_raw_output": action_result.get("raw_output", ""),
        "action_original_parsed_output": copy.deepcopy(
            action_result.get("original_parsed") or {}
        ),
        "action_normalization_notes": copy.deepcopy(
            action_result.get("normalization_notes") or []
        ),
        "action_discarded_actions": discarded_actions,
        "action_id_map": copy.deepcopy(action_result.get("action_id_map") or []),
        "action_n_generated": n_generated,
        "action_n_valid": n_valid,
        "action_n_discarded": n_discarded,
        "action_generation_status": action_result.get("generation_status"),
        "action_context_mode": action_result.get("action_context_mode"),
        "action_context_stats": copy.deepcopy(
            action_result.get("action_context_stats") or {}
        ),
        "action_generated_at": utc_now_iso(),
    })
    out.pop("failed_stage", None)

    out.setdefault("llm_trace", {})["action"] = copy.deepcopy(
        action_result.get("llm_trace") or {}
    )
    out.setdefault("usage", {})["action"] = copy.deepcopy(
        action_result.get("usage") or {}
    )

    stage = _stage_from_success(action_result)
    stage.update({
        "status": (
            "GENERATED_PARTIAL_VALID"
            if n_discarded > 0
            else "GENERATED"
        ),
        "generation_status": action_result.get("generation_status"),
        "original_parsed_output": copy.deepcopy(
            action_result.get("original_parsed") or {}
        ),
        "accepted_output": copy.deepcopy(action_output),
        "discarded_actions": discarded_actions,
        "action_id_map": copy.deepcopy(action_result.get("action_id_map") or []),
        "normalization_notes": copy.deepcopy(
            action_result.get("normalization_notes") or []
        ),
        "n_generated": n_generated,
        "n_valid": n_valid,
        "n_discarded": n_discarded,
    })
    out.setdefault("stage_outputs", {})["action"] = stage

    if action_result.get("generation_status") in {
        "no_human_approved_mechanism",
        "no_allowed_intervention_layer",
    }:
        out["status"] = STATUS_NO_ACTION
    else:
        # partial_valid is a successful action stage as long as >=1 item survives.
        if n_valid < 1:
            raise AssertionError(
                "generate_prospective_actions returned a non-blocked result with no valid actions"
            )
        out["status"] = STATUS_WAITING_ACTION_REVIEW

    _persist(out, workflow_dir)
    return out

# ---------------------------------------------------------------------------
# Human Gate 2
# ---------------------------------------------------------------------------

def finalize_product_decision(
    *,
    action_stage_record: Dict[str, Any],
    action_feedback: Dict[str, Any],
    workflow_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Human Gate 2 -> final product decision. No LLM call."""
    if action_stage_record.get("status") != STATUS_WAITING_ACTION_REVIEW:
        raise ValueError(
            f"Workflow is not waiting for action review: "
            f"status={action_stage_record.get('status')!r}"
        )
    reviewed_actions = apply_action_feedback(
        action_stage_record.get("action_output") or {},
        action_feedback,
    )
    out = copy.deepcopy(action_stage_record)
    out["action_feedback"] = copy.deepcopy(action_feedback)
    out["reviewed_action_output"] = reviewed_actions
    out["approved_actions"] = copy.deepcopy(reviewed_actions.get("approved_actions") or [])
    out["deferred_actions"] = copy.deepcopy(reviewed_actions.get("deferred_actions") or [])
    out["rejected_actions"] = copy.deepcopy(reviewed_actions.get("rejected_actions") or [])
    out["status"] = STATUS_COMPLETED
    _persist(out, workflow_dir)
    return out


def _record_category(record: Mapping[str, Any]) -> str:
    meta = record.get("case_meta") if isinstance(record.get("case_meta"), dict) else {}
    requested = record.get("requested_case") if isinstance(record.get("requested_case"), dict) else {}
    return canonical_category_name(
        meta.get("product_category") or requested.get("product_category")
    )



def _portfolio_action_output(record: Dict[str, Any]) -> Dict[str, Any]:
    """Expose only Gate-2 approved action fields needed for ecosystem synthesis."""
    category = _record_category(record)
    actions: List[Dict[str, Any]] = []

    for action in record.get("approved_actions") or []:
        if not isinstance(action, dict):
            continue

        final_action = str(
            action.get("final_action")
            or action.get("action")
            or ""
        ).strip()

        actions.append({
            "action_id": action.get("action_id"),
            "product_category": category,
            "action_subfamily": action.get("action_subfamily"),
            "action_level": action.get("action_level"),
            "improvement_direction": action.get("improvement_direction"),
            "action": final_action,
            "linked_mechanism_ids": copy.deepcopy(
                action.get("linked_mechanism_ids") or []
            ),
            "source_refs": copy.deepcopy(
                action.get("source_refs") or []
            ),
            "target_products": copy.deepcopy(
                action.get("target_products") or []
            ),
            "target_layers": copy.deepcopy(
                action.get("target_layers") or []
            ),
        })

    return {
        "case_id": (record.get("case_meta") or {}).get("case_id"),
        "product_category": category,
        "actions": actions,
    }


def _coerce_ecosystem_config(
    value: Optional[EcosystemContextConfig | Dict[str, Any]],
) -> EcosystemContextConfig:
    if value is None:
        return EcosystemContextConfig()
    if isinstance(value, EcosystemContextConfig):
        return value
    if not isinstance(value, dict):
        raise TypeError("ecosystem_config must be EcosystemContextConfig, dict, or None")
    known = set(EcosystemContextConfig.__dataclass_fields__)
    unknown = sorted(set(value) - known)
    if unknown:
        raise KeyError(f"Unknown ecosystem config keys: {unknown}")
    return EcosystemContextConfig(**value)


def _portfolio_h_config(
    records: Sequence[Dict[str, Any]],
    override: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if override is not None:
        cfg = _merge_config(override)
    else:
        cfg = None
    seen: List[Dict[str, Any]] = []
    for record in records:
        raw = record.get("prospective_config")
        if isinstance(raw, dict):
            seen.append(_merge_config(raw))
    if seen:
        first = seen[0]
        if any(x != first for x in seen[1:]):
            raise ValueError("Product workflows use inconsistent prospective_config values")
        if cfg is not None and cfg != first:
            raise ValueError("prospective_config override disagrees with persisted product workflows")
        return first
    return cfg if cfg is not None else _merge_config(None)


def _build_c_case_package_for_context(
    *,
    category: str,
    analysis_cutoff: str,
    target_month: str,
    data_context: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Build structural B+C context only; no product-level H/LLM call occurs."""
    static_row = resolve_profile_row(data_context["static_index"], category, analysis_cutoff)
    dynamic_row = resolve_profile_row(data_context["dynamic_index"], category, analysis_cutoff)
    dynamic_target = str(dynamic_row.get("target_month") or "").strip()
    if dynamic_target and dynamic_target != target_month:
        raise ValueError(
            f"{category}: dynamic profile target_month={dynamic_target!r} does not "
            f"match ecosystem target_month={target_month!r}"
        )
    packages = build_all_condition_packages(
        category=category,
        static_row=static_row,
        dynamic_row=dynamic_row,
        api_cards=data_context["api_cards"],
        static_relation_rows=data_context["relation_rows"],
        risk_chain_path=data_context["risk_chain_path"],
        analysis_cutoff=analysis_cutoff,
        target_month=target_month,
        max_observed_cards=int(config["max_observed_cards"]),
        statements_per_card=int(config["statements_per_card"]),
        max_relations=int(config["max_relations"]),
        max_api_cards=int(config["max_api_cards"]),
        max_chains=int(config["max_chains"]),
        max_standalone_chains=int(config["max_standalone_chains"]),
        chain_selection_mode=str(config["chain_selection_mode"]),
        category_api_map_path=data_context.get("category_api_map_path"),
    )
    return packages[CONDITION_C]


def _build_ecosystem_case_pool(
    *,
    records: Sequence[Dict[str, Any]],
    risk_selection: Mapping[str, Any],
    data_context: Dict[str, Any],
    prospective_config: Dict[str, Any],
    ecosystem_config: EcosystemContextConfig,
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Build only the evidence packages required by deterministic ecosystem routing."""
    selection = select_ecosystem_context(
        risk_selection=risk_selection,
        config=ecosystem_config,
    )
    focal_products = [canonical_category_name(x) for x in selection["focal_products"]]
    record_map = {_record_category(r): r for r in records}
    record_map = {k: v for k, v in record_map.items() if k}

    if set(record_map) != set(focal_products):
        missing = [x for x in focal_products if x not in record_map]
        extra = [x for x in record_map if x not in set(focal_products)]
        raise ValueError(
            "final_product_records must match the upstream focal Top-K exactly; "
            f"missing={missing}, extra={extra}"
        )

    time_pairs = {
        (
            str((r.get("requested_case") or {}).get("analysis_cutoff") or ""),
            str((r.get("requested_case") or {}).get("target_month") or ""),
        )
        for r in records
    }
    if len(time_pairs) != 1:
        raise ValueError("All focal product workflows must use the same analysis_cutoff/target_month")
    analysis_cutoff, target_month = next(iter(time_pairs))
    if not analysis_cutoff or not target_month:
        raise ValueError("Focal workflows are missing analysis_cutoff/target_month")

    pool: Dict[str, Dict[str, Any]] = {}
    # Focal H packages are retained for shared API/time metadata. Their reviewed
    # mechanisms/actions are supplied separately and remain the primary evidence.
    for product in focal_products:
        case = record_map[product].get("case_package")
        if not isinstance(case, dict):
            raise ValueError(f"Focal workflow {product!r} is missing case_package")
        pool[product] = copy.deepcopy(case)

    context_products = list(selection.get("chain_context_products") or [])
    context_products.extend(
        row.get("product_category")
        for row in (selection.get("residual_context_rows") or [])
        if isinstance(row, dict)
    )
    missing_context_profiles: List[str] = []
    for raw_product in context_products:
        product = canonical_category_name(raw_product)
        if not product or product in pool:
            continue
        try:
            pool[product] = _build_c_case_package_for_context(
                category=product,
                analysis_cutoff=analysis_cutoff,
                target_month=target_month,
                data_context=data_context,
                config=prospective_config,
            )
        except KeyError:
            # Non-focal context is allowed to degrade to chain-only evidence.
            # Focal products were already validated above and never use this path.
            missing_context_profiles.append(product)

    selection = copy.deepcopy(selection)
    selection["context_products_without_profile_package"] = sorted(
        set(missing_context_profiles)
    )
    return pool, selection


def _build_ecosystem_api_candidates(
    *,
    case_pool: Mapping[str, Dict[str, Any]],
    selection: Mapping[str, Any],
    data_context: Dict[str, Any],
    ecosystem_config: EcosystemContextConfig,
) -> Dict[str, List[Dict[str, Any]]]:
    """Build a broader static API candidate pool for the shared-API gate.

    This does not alter the product-level H package. It only prevents a shared
    ecosystem API from disappearing because it ranked fourth or lower within an
    individual product's three-card H package.
    """
    products = []
    for x in list(selection.get("focal_products") or []) + list(selection.get("chain_context_products") or []):
        cat = canonical_category_name(x)
        if cat and cat not in products:
            products.append(cat)

    out: Dict[str, List[Dict[str, Any]]] = {}
    for product in products:
        case = case_pool.get(product)
        if not isinstance(case, dict):
            continue
        meta = case.get("case_meta") if isinstance(case.get("case_meta"), dict) else {}
        cutoff = str(meta.get("analysis_cutoff") or "").strip()
        static_row = resolve_profile_row(data_context["static_index"], product, cutoff)
        out[product] = retrieve_api_evidence_static_only(
            category=product,
            static_row=static_row,
            static_relations=case.get("ecosystem_relations") or [],
            api_cards=data_context["api_cards"],
            analysis_cutoff=cutoff or None,
            max_cards=int(ecosystem_config.max_api_candidates_per_product),
            category_api_map_path=data_context.get("category_api_map_path"),
        )
    return out


# ---------------------------------------------------------------------------
# Ecosystem stage
# ---------------------------------------------------------------------------

def generate_topk_ecosystem_synthesis(
    *,
    final_product_records: Sequence[Dict[str, Any]],
    risk_selection: Mapping[str, Any],
    data_context: Dict[str, Any],
    model_name: Optional[str] = None,
    temperature: float = 0.2,
    prospective_config: Optional[Dict[str, Any]] = None,
    ecosystem_config: Optional[EcosystemContextConfig | Dict[str, Any]] = None,
    workflow_dir: Optional[str | Path] = None,
    expected_k: int = 5,
) -> Dict[str, Any]:
    """Synthesize the reviewed Top-K using the same upstream ``risk_chain_res``.

    Top-K membership is immutable here. Non-focal chain/residual context products
    are explanatory evidence only and are never promoted to intervention targets.
    """
    records = [copy.deepcopy(x) for x in final_product_records if isinstance(x, dict)]
    if not records:
        raise ValueError("final_product_records is empty")

    bad = [
        x.get("workflow_id")
        for x in records
        if x.get("status") not in {STATUS_COMPLETED, STATUS_NO_ACTION}
    ]
    if bad:
        raise ValueError(
            "Ecosystem synthesis requires every focal workflow to be terminal "
            f"(COMPLETED or NO_ACTION_REQUIRED); non-terminal={bad}"
        )

    preflight_prospective_system(data_context=data_context, model_name=model_name)
    h_cfg = _portfolio_h_config(records, prospective_config)
    eco_cfg = _coerce_ecosystem_config(ecosystem_config)
    case_pool, deterministic_selection = _build_ecosystem_case_pool(
        records=records,
        risk_selection=risk_selection,
        data_context=data_context,
        prospective_config=h_cfg,
        ecosystem_config=eco_cfg,
    )
    expected_k = int(expected_k)
    if expected_k < 1:
        raise ValueError("expected_k must be >= 1")
    focal_products = list(deterministic_selection.get("focal_products") or [])
    if len(focal_products) != expected_k:
        raise ValueError(
            f"Prospective ecosystem stage expects exactly {expected_k} focal products; "
            f"risk_selection contains {len(focal_products)}"
        )
    api_candidates = _build_ecosystem_api_candidates(
        case_pool=case_pool,
        selection=deterministic_selection,
        data_context=data_context,
        ecosystem_config=eco_cfg,
    )

    # Preserve upstream focal order/rank rather than record/list order.
    rank_df = risk_selection.get("selected_rank_df")
    rank_order: Dict[str, int] = {}
    if isinstance(rank_df, pd.DataFrame) and "Category" in rank_df.columns:
        for _, row in rank_df.iterrows():
            cat = canonical_category_name(row.get("Category"))
            if cat:
                try:
                    rank_order[cat] = int(row.get("rank"))
                except Exception:
                    rank_order[cat] = 10**9
    records.sort(key=lambda r: (rank_order.get(_record_category(r), 10**9), _record_category(r)))

    workflow_ids = [str(x.get("workflow_id") or "") for x in records]
    if any(not x for x in workflow_ids):
        raise ValueError("All focal product records require workflow_id")
    portfolio_digest = hashlib.sha1("|".join(workflow_ids).encode("utf-8")).hexdigest()[:12]
    portfolio_id = f"ECO-{portfolio_digest}"

    reviewed_diagnoses: List[Dict[str, Any]] = []
    for record in records:
        diag = record.get("reviewed_diagnosis")
        if not isinstance(diag, dict):
            raise ValueError(
                f"Focal workflow {_record_category(record)!r} has no persisted reviewed_diagnosis"
            )
        reviewed_diagnoses.append(copy.deepcopy(diag))

    action_outputs = [_portfolio_action_output(x) for x in records]

    try:
        result = generate_ecosystem_synthesis(
            risk_selection=risk_selection,
            reviewed_diagnoses=reviewed_diagnoses,
            action_outputs=action_outputs,
            all_case_packages=case_pool,
            model_name=model_name,
            temperature=temperature,
            config=eco_cfg,
            api_candidates_by_product=api_candidates,
        )
    except EcosystemGenerationTraceError as exc:
        if not _has_completed_llm_attempt(exc):
            raise
        blocked_stage = _stage_from_output_error(exc)
        out = {
            "workflow": "H_prospective_ecosystem_v3_risk_selection_audited",
            "portfolio_id": portfolio_id,
            "source_workflow_ids": workflow_ids,
            "status": STATUS_BLOCKED_OUTPUT_REVIEW,
            "failed_stage": "ecosystem_synthesis",
            "n_focal_products": len(records),
            "focal_products": list(deterministic_selection.get("focal_products") or []),
            "chain_context_products": list(deterministic_selection.get("chain_context_products") or []),
            "context_products_without_profile_package": list(
                deterministic_selection.get("context_products_without_profile_package") or []
            ),
            "selection_metrics": copy.deepcopy(deterministic_selection.get("metrics") or {}),
            "stage_output": blocked_stage,
            "raw_output": blocked_stage.get("raw_output", ""),
            "llm_trace": copy.deepcopy(exc.llm_trace),
        }
        if workflow_dir is not None:
            save_named_artifact(portfolio_id, out, Path(workflow_dir) / "ecosystem")
        return out

    stage = _stage_from_success(result)
    out = {
        "workflow": "H_prospective_ecosystem_v3_risk_selection_audited",
        "portfolio_id": portfolio_id,
        "source_workflow_ids": workflow_ids,
        "status": "COMPLETED",
        "n_focal_products": len(records),
        "focal_products": list(deterministic_selection.get("focal_products") or []),
        "chain_context_products": list(deterministic_selection.get("chain_context_products") or []),
        "context_products_without_profile_package": list(
            deterministic_selection.get("context_products_without_profile_package") or []
        ),
        "residual_chain_ids": list(deterministic_selection.get("residual_chain_ids") or []),
        "selection_metrics": copy.deepcopy(deterministic_selection.get("metrics") or {}),
        "parsed": copy.deepcopy(result.get("parsed") or {}),
        "raw_output": result.get("raw_output", ""),
        "usage": copy.deepcopy(result.get("usage") or {}),
        "ecosystem_audit": copy.deepcopy(result.get("ecosystem_audit") or {}),
        "ecosystem_input": copy.deepcopy(result.get("ecosystem_input") or {}),
        "stage_output": stage,
        "llm_trace": copy.deepcopy(result.get("llm_trace") or {}),
    }
    if workflow_dir is not None:
        save_named_artifact(portfolio_id, out, Path(workflow_dir) / "ecosystem")
    return out


def reset_blocked_action_for_retry(
    *,
    action_stage_record: Dict[str, Any],
    workflow_dir: str | Path,
) -> Dict[str, Any]:
    """Rollback only a blocked action stage to READY_FOR_ACTION.

    Gate-1 review, reviewed diagnosis, case package, and decision context are
    preserved. The failed action attempt is archived before current action-state
    fields are cleared.
    """
    if not isinstance(action_stage_record, dict):
        raise TypeError("action_stage_record must be a dict")

    status = str(action_stage_record.get("status") or "").strip()
    if status != STATUS_BLOCKED_ACTION_OUTPUT_REVIEW:
        raise ValueError(
            "reset_blocked_action_for_retry requires "
            "STATUS_BLOCKED_ACTION_OUTPUT_REVIEW"
        )

    failed_stage = str(action_stage_record.get("failed_stage") or "").strip()
    if failed_stage != "action":
        raise ValueError(f"Expected failed_stage='action', got {failed_stage!r}")

    if not isinstance(action_stage_record.get("reviewed_diagnosis"), dict):
        raise ValueError("Blocked action workflow has no reviewed_diagnosis")
    if not isinstance(action_stage_record.get("mechanism_feedback"), dict):
        raise ValueError("Blocked action workflow has no mechanism_feedback")

    updated = copy.deepcopy(action_stage_record)
    stage_outputs = updated.get("stage_outputs")
    if not isinstance(stage_outputs, dict):
        stage_outputs = {}
        updated["stage_outputs"] = stage_outputs

    failed_action_stage = copy.deepcopy(stage_outputs.get("action"))
    llm_trace = updated.get("llm_trace")
    failed_action_trace = (
        copy.deepcopy(llm_trace.get("action"))
        if isinstance(llm_trace, dict)
        else None
    )
    usage = updated.get("usage")
    failed_action_usage = (
        copy.deepcopy(usage.get("action"))
        if isinstance(usage, dict)
        else None
    )

    history = updated.get("action_attempt_history")
    if not isinstance(history, list):
        history = []
    history.append({
        "attempt_no": len(history) + 1,
        "archived_at": utc_now_iso(),
        "terminal_status": STATUS_BLOCKED_ACTION_OUTPUT_REVIEW,
        "failed_stage": "action",
        "stage_output": failed_action_stage,
        "llm_trace": failed_action_trace,
        "usage": failed_action_usage,
    })
    updated["action_attempt_history"] = history

    stage_outputs.pop("action", None)
    if isinstance(llm_trace, dict):
        llm_trace.pop("action", None)
    if isinstance(usage, dict):
        usage.pop("action", None)

    for key in [
        "action_output",
        "action_raw_output",
        "action_original_parsed_output",
        "action_normalization_notes",
        "action_discarded_actions",
        "action_id_map",
        "action_n_generated",
        "action_n_valid",
        "action_n_discarded",
        "action_generation_status",
        "action_context_mode",
        "action_context_stats",
        "action_feedback",
        "reviewed_action_output",
        "approved_actions",
        "deferred_actions",
        "rejected_actions",
        "action_generated_at",
        "action_reviewed_at",
        "completed_at",
    ]:
        updated.pop(key, None)

    updated.pop("failed_stage", None)
    updated["status"] = STATUS_READY_FOR_ACTION
    updated["action_retry_count"] = len(history)
    updated["action_retry_reset_at"] = utc_now_iso()
    save_workflow(updated, workflow_dir)
    return updated


def retry_blocked_action_generation(
    *,
    action_stage_record: Dict[str, Any],
    model_name: Optional[str] = None,
    temperature: float = 0.2,
    workflow_dir: str | Path,
) -> Dict[str, Any]:
    """Archive/reset only the blocked action stage and immediately regenerate it.

    The persisted Gate-1 review and decision_context are reused exactly, so retry
    does not rerun H diagnosis and does not silently alter human constraints.
    """
    if not isinstance(action_stage_record, dict):
        raise TypeError("action_stage_record must be a dict")

    mechanism_feedback = action_stage_record.get("mechanism_feedback")
    decision_context = action_stage_record.get("decision_context")
    if not isinstance(mechanism_feedback, dict):
        raise ValueError("Blocked action workflow has no persisted mechanism_feedback")
    if not isinstance(decision_context, dict):
        raise ValueError("Blocked action workflow has no persisted decision_context")

    reset_record = reset_blocked_action_for_retry(
        action_stage_record=action_stage_record,
        workflow_dir=workflow_dir,
    )
    return generate_product_actions_after_review(
        diagnosis_record=reset_record,
        mechanism_feedback=copy.deepcopy(mechanism_feedback),
        decision_context=copy.deepcopy(decision_context),
        model_name=model_name,
        temperature=temperature,
        workflow_dir=workflow_dir,
    )

__all__ = [
    "DEFAULT_PROSPECTIVE_CONFIG",
    "STATUS_DIAGNOSIS_PENDING",
    "STATUS_H1_COMPLETED",
    "STATUS_ROUTING_COMPLETED",
    "STATUS_WAITING_MECHANISM_REVIEW",
    "STATUS_BLOCKED_OUTPUT_REVIEW",
    "STATUS_READY_FOR_ACTION",
    "STATUS_WAITING_ACTION_REVIEW",
    "STATUS_BLOCKED_ACTION_OUTPUT_REVIEW",
    "STATUS_COMPLETED",
    "STATUS_NO_ACTION",
    "load_prospective_data",
    "load_risk_selection_bundle",
    "preflight_prospective_system",
    "build_topk_product_specs_from_risk_selection",
    "run_h_product_diagnosis",
    "run_h_topk_diagnoses",
    "run_h_from_risk_selection",
    "record_mechanism_review",
    "generate_product_actions_after_review",
    "finalize_product_decision",
    "generate_topk_ecosystem_synthesis",
    "reset_blocked_action_for_retry",
    "retry_blocked_action_generation",
]
