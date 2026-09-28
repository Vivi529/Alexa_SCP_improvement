from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from llm_service import call_json


class EcosystemGenerationTraceError(ValueError):
    """Ecosystem-generation error carrying any completed raw LLM attempt."""

    def __init__(self, message: str, *, llm_trace: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.llm_trace = llm_trace or {"attempts": []}


def _ecosystem_attempt_trace(
    *,
    result: Dict[str, Any],
    parsed: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    input_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "attempts": [{
            "attempt": 1,
            "input_payload": input_payload,
            "raw_output": result.get("raw_output", "") if isinstance(result, dict) else "",
            "parsed_output": parsed,
            "parse_error": result.get("parse_error") if isinstance(result, dict) else None,
            "validation_error": error,
            "usage": dict(result.get("usage") or {}) if isinstance(result, dict) else {},
            "request_meta": dict(result.get("request_meta") or {}) if isinstance(result, dict) else {},
        }]
    }


# ============================================================================
# Prompt
# ============================================================================

ECOSYSTEM_SYNTHESIS_PROMPT = r"""
You synthesize reviewed product-level diagnoses and approved actions into an ecosystem-level prospective diagnosis for smart connected products.

Use only the supplied input.
Return exactly one valid JSON object and no prose outside JSON.


EVIDENCE ROLES

- FOCAL_PRODUCTS:
  Authoritative intervention targets. Their reviewed mechanisms and approved actions are the primary diagnostic and decision evidence.

- CHAIN_CONTEXT_PRODUCTS:
  Non-focal explanatory context only. Use them to interpret dependencies,
  propagation pathways, or shared structures involving focal products.
  Never treat them as intervention targets or create product-level mechanisms/actions.

- SYSTEMIC_API_CONTEXT:
  Candidate implementation-structure knowledge. Shared API use may support a systemic hypothesis but is not proof of a shared implementation defect.

- RESIDUAL_CHAIN_SIGNALS:
  Coverage-awareness context only. Use them to identify important uncovered ecosystem regions, not to revise focal diagnoses or create focal actions.


CORE RULES

1. CLOSED-WORLD SYNTHESIS
   Do not invent products, dimensions, mechanisms, actions, chains, relations, APIs, or evidence IDs. Do not create new product-level mechanisms or actions.

2. SYSTEMIC CLAIM DISCIPLINE
   Repeated risks or symptoms alone do not establish a shared mechanism.

   Evidence classes are:
   A = reviewed focal-product mechanisms
   B = relevant risk chains or ecosystem relations
   C = shared/chain-conditioned API implementation context

   Any finding that asserts a shared mechanism, dependency, propagation pathway, implementation structure, technical gap, or governance problem requires at least TWO distinct evidence classes.

   If this requirement is not met, use validation_priorities instead.

   Exception: recurring_cross_product_risk may describe a repeated focal-product risk pattern without claiming a shared underlying mechanism.

3. CAUSAL DISCIPLINE
   Risk chains and ecosystem relations are predictive/associational, not causal proof. Phrase systemic explanations as hypotheses, not confirmed root causes.

4. ACTION CONSOLIDATION
   Consolidate only supplied approved focal-product actions that address the same supported shared issue. Preserve all source_action_ids.

5. NON-REDUNDANCY
   Prefer one well-supported finding over several findings describing the same issue.


EVIDENCE STRENGTH

For systemic explanations:
- strong = evidence classes A + B + C are all present and consistent
- moderate = two distinct evidence classes provide consistent support
- weak = two distinct evidence classes support the finding indirectly, incompletely, or with some inconsistency

For recurring_cross_product_risk, strength reflects only the breadth and consistency of the repeated risk pattern, not evidence of a shared mechanism.


FINDING TYPES

Use exactly one:
recurring_cross_product_risk |
shared_dependency |
propagation_pathway |
shared_implementation_structure |
capability_gap |
state_consistency |
dependency_resilience |
compatibility_or_lifecycle |
privacy_or_security |
governance_gap |
other

Key distinctions:
- recurring_cross_product_risk: repeated risks without a demonstrated shared mechanism
- shared_dependency: common service/interface/platform dependency
- propagation_pathway: supported by risk-chain structure
- shared_implementation_structure: common API/interface/state/capability architecture
- governance_gap: primarily requires ecosystem-level coordination or standards


OUTPUT RULES

Return at most:
- 5 systemic_findings
- 4 portfolio_consolidations
- 4 validation_priorities
- 3 governance_priorities

Keep free-text concise.
Do not invent numerical validation targets unless supplied in the input.
main_uncertainty states the unresolved uncertainty, not the validation task.

Required JSON shape:

{
  "systemic_findings": [
    {
      "finding_id": "SF1",
      "finding_type": "",
      "affected_focal_products": [],
      "context_products": [],
      "risk_dimensions": [],
      "supporting_mechanism_ids": [],
      "risk_chain_refs": [],
      "relation_refs": [],
      "api_refs": [],
      "statement": "",
      "evidence_strength": "strong|moderate|weak",
      "main_uncertainty": ""
    }
  ],
  "portfolio_consolidations": [
    {
      "portfolio_id": "PA1",
      "portfolio_level": "interface|cross_product|ecosystem",
      "source_action_ids": [],
      "supported_finding_ids": [],
      "affected_focal_products": [],
      "recommendation": "",
      "validation_metric": ""
    }
  ],
  "validation_priorities": [
    {
      "validation_id": "EV1",
      "supported_finding_ids": [],
      "products": [],
      "question": "",
      "required_evidence": []
    }
  ],
  "governance_priorities": [
    {
      "priority_id": "G1",
      "supported_finding_ids": [],
      "source_action_ids": [],
      "priority": ""
    }
  ],
  "ecosystem_summary": ""
}
""".strip()

# ============================================================================
# Config
# ============================================================================

@dataclass(frozen=True)
class EcosystemContextConfig:
    # Explain the focal set using only the most valuable focal-associated chains.
    focal_chain_cumulative_mass: float = 0.85
    max_focal_context_chains: int = 16

    # A non-focal product enters CHAIN_CONTEXT_PRODUCTS only if it provides
    # material incremental explanatory mass.
    min_context_mass_share: float = 0.04
    min_shared_focal_chain_count: int = 2
    min_linked_focal_products: int = 2
    critical_single_chain_share: float = 0.10
    max_chain_context_products: int = 4

    # Residual context is a safeguard, not a routine evidence source.
    residual_trigger_coverage: float = 0.85
    ecosystem_information_coverage_target: float = 0.92
    min_residual_chain_share: float = 0.02
    max_residual_chains: int = 4

    # Product-level medium residual context is off by default to avoid noise.
    include_residual_product_context: bool = False
    min_residual_product_mass_share: float = 0.06
    max_residual_context_products: int = 2

    # Package-size controls.
    max_observed_dimension_rows: int = 4
    max_dynamic_rows: int = 4
    max_relation_rows: int = 5
    max_api_cards_per_api: int = 1
    max_shared_api_items: int = 8
    # Candidate pool can be broader than the per-product H package (which keeps
    # max_api_cards=3); the systemic gate below still emits shared APIs only.
    max_api_candidates_per_product: int = 12

    # API gate: ecosystem API context is shared-by-default.
    min_api_product_support: int = 2

    def __post_init__(self) -> None:
        share_fields = (
            "focal_chain_cumulative_mass", "min_context_mass_share",
            "critical_single_chain_share", "residual_trigger_coverage",
            "ecosystem_information_coverage_target", "min_residual_chain_share",
            "min_residual_product_mass_share",
        )
        for name in share_fields:
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]; got {value}")
        if self.focal_chain_cumulative_mass <= 0:
            raise ValueError("focal_chain_cumulative_mass must be > 0")
        int_fields = (
            "max_focal_context_chains", "min_shared_focal_chain_count",
            "min_linked_focal_products", "max_chain_context_products",
            "max_residual_chains", "max_residual_context_products",
            "max_observed_dimension_rows", "max_dynamic_rows", "max_relation_rows",
            "max_api_cards_per_api", "max_shared_api_items",
            "max_api_candidates_per_product",
        )
        for name in int_fields:
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be >= 0")
        if int(self.min_api_product_support) < 1:
            raise ValueError("min_api_product_support must be >= 1")


# ============================================================================
# Generic helpers
# ============================================================================

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


def _norm_text(value: Any) -> str:
    return str(value or "").strip()


def _norm_category(value: Any) -> str:
    # Keep the project's existing category labels. Only trim whitespace here;
    # callers may canonicalize before constructing all_case_packages.
    return _norm_text(value)


def _unique(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        text = _norm_text(value)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _list_of_dicts(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [x for x in value if isinstance(x, dict)]


def _case_category(case: Mapping[str, Any]) -> str:
    meta = case.get("case_meta") if isinstance(case.get("case_meta"), dict) else {}
    return _norm_category(meta.get("product_category"))


def _index_case_packages(
    case_packages: Mapping[str, Dict[str, Any]] | Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    if isinstance(case_packages, Mapping):
        out: Dict[str, Dict[str, Any]] = {}
        for key, value in case_packages.items():
            if not isinstance(value, dict):
                continue
            cat = _case_category(value) or _norm_category(key)
            if cat:
                out[cat] = value
        return out

    out = {}
    for case in case_packages or []:
        if not isinstance(case, dict):
            continue
        cat = _case_category(case)
        if cat:
            out[cat] = case
    return out


def _index_by_product(
    rows: Sequence[Dict[str, Any]],
    product_keys: Sequence[str] = ("product_category", "Category"),
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        obj = row.get("parsed") if isinstance(row.get("parsed"), dict) else row
        cat = ""
        for key in product_keys:
            cat = _norm_category(obj.get(key))
            if cat:
                break
        if cat:
            out[cat] = obj
    return out


def _chain_products(chain: Mapping[str, Any]) -> List[str]:
    products: List[str] = []
    products.extend([
        chain.get("source_category"),
        chain.get("target_category"),
    ])

    for edge in _list_of_dicts(chain.get("edges")):
        products.extend([
            edge.get("from_category"),
            edge.get("to_category"),
        ])

    # Some chain tables do not contain edges. path_text is generated in the
    # project as "Category::dimension -> Category::dimension".
    path_text = _norm_text(chain.get("path_text"))
    if path_text:
        for part in path_text.split("->"):
            token = part.strip()
            if "::" in token:
                products.append(token.split("::", 1)[0].strip())

    return _unique(products)


def _chain_id(chain: Mapping[str, Any], fallback_index: int) -> str:
    return _norm_text(chain.get("chain_id")) or f"CHAIN_ROW_{fallback_index}"


def _relation_products(rel: Mapping[str, Any]) -> List[str]:
    values: List[Any] = []
    for key in (
        "source_category", "target_category",
        "from_category", "to_category",
        "source_product", "target_product",
        "from_product", "to_product",
        "product_category", "related_product",
    ):
        values.append(rel.get(key))
    raw_products = rel.get("products")
    if isinstance(raw_products, list):
        values.extend(raw_products)
    return _unique(values)


def _relation_relevant(
    rel: Mapping[str, Any],
    product: str,
    linked_products: set[str],
) -> bool:
    products = set(_relation_products(rel))
    if product not in products:
        return False
    return not linked_products or bool(products & linked_products)


def _api_name(card: Mapping[str, Any]) -> str:
    return _norm_text(
        card.get("api_name")
        or card.get("name")
        or card.get("interface")
    )


def _api_key(name: Any) -> str:
    return "".join(ch.lower() for ch in _norm_text(name) if ch.isalnum() or ch in "._-")


def _compact_api_card(card: Mapping[str, Any]) -> Dict[str, Any]:
    # Deliberately emphasize implementation structure rather than pre-written
    # root-cause suggestions, which can anchor the ecosystem LLM too strongly.
    keep = (
        "knowledge_id",
        "api_name",
        "functionality",
        "device_context",
        "core_directives_or_events",
        "reportable_properties",
        "dependencies_or_related_APIs",
        "dependencies",
        "dependency_api_names",
        "design_constraints",
        "interaction_behavior_model",
        "design_insight",
        "retrieval_score",
        "retrieval_basis",
        "category_match",
    )
    return {k: card.get(k) for k in keep if card.get(k) not in (None, "", [], {})}


def _compact_chain(chain: Mapping[str, Any], idx: int) -> Dict[str, Any]:
    out = {
        "chain_id": _chain_id(chain, idx),
        "dimension": chain.get("dimension"),
        "path_text": chain.get("path_text"),
        "path_length": chain.get("path_length"),
        "source_category": chain.get("source_category"),
        "target_category": chain.get("target_category"),
        "chain_score": chain.get("chain_score"),
        "source_risk_raw": chain.get("source_risk_raw"),
        "target_risk_raw": chain.get("target_risk_raw"),
        "epistemic_status": chain.get(
            "epistemic_status",
            "model_inferred_risk_chain_not_causal",
        ),
    }
    edges = []
    for e in _list_of_dicts(chain.get("edges")):
        edges.append({
            "edge_id": e.get("edge_id") or e.get("edge_ref"),
            "from_category": e.get("from_category"),
            "to_category": e.get("to_category"),
            "dimension": e.get("dimension") or chain.get("dimension"),
            "dominant_channel": e.get("dominant_channel"),
            "w_eff_score": e.get("w_eff_score") or e.get("w_eff"),
        })
    if edges:
        out["edges"] = edges
    return {k: v for k, v in out.items() if v not in (None, "", [], {})}


# ============================================================================
# Risk-chain selection / expansion
# ============================================================================

def _validate_risk_selection(risk_selection: Mapping[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if not isinstance(risk_selection, Mapping):
        raise TypeError("risk_selection must be a mapping returned by select_final_topk_products_by_risk_chains")

    chain_df = risk_selection.get("chain_df")
    coverage_obj = risk_selection.get("coverage_obj")
    if not isinstance(chain_df, pd.DataFrame):
        raise TypeError("risk_selection['chain_df'] must be a pandas DataFrame")
    if not isinstance(coverage_obj, dict):
        raise TypeError("risk_selection['coverage_obj'] must be a dict")
    if chain_df.empty:
        raise ValueError("risk_selection['chain_df'] is empty")

    C = np.asarray(coverage_obj.get("C"), dtype=float)
    omega = np.asarray(coverage_obj.get("omega"), dtype=float).reshape(-1)
    categories = [_norm_category(x) for x in (coverage_obj.get("categories") or [])]
    cat2idx_raw = dict(coverage_obj.get("cat2idx") or {})
    cat2idx = {_norm_category(k): v for k, v in cat2idx_raw.items() if _norm_category(k)}

    if C.ndim != 2:
        raise ValueError("coverage_obj['C'] must be a 2D product x chain matrix")
    if C.shape[1] != len(chain_df):
        raise ValueError(
            f"coverage matrix/chain_df mismatch: C has {C.shape[1]} chains but chain_df has {len(chain_df)} rows"
        )
    if len(omega) != len(chain_df):
        raise ValueError(
            f"omega/chain_df mismatch: omega has {len(omega)} values but chain_df has {len(chain_df)} rows"
        )
    if C.shape[0] != len(categories):
        raise ValueError("coverage_obj categories length does not match C rows")
    if not categories or any(not x for x in categories):
        raise ValueError("coverage_obj categories must be non-empty strings")
    if len(categories) != len(set(categories)):
        raise ValueError("coverage_obj categories contains duplicates")
    if any(c not in cat2idx for c in categories):
        raise ValueError("coverage_obj cat2idx is incomplete")
    for expected_i, cat in enumerate(categories):
        try:
            actual_i = int(cat2idx[cat])
        except Exception as exc:
            raise ValueError(f"coverage_obj cat2idx[{cat!r}] must be an integer") from exc
        if actual_i != expected_i:
            raise ValueError(
                "coverage_obj cat2idx must map each category to its C row; "
                f"{cat!r}: expected {expected_i}, got {actual_i}"
            )
        cat2idx[cat] = actual_i

    if not np.isfinite(C).all():
        raise ValueError("coverage_obj['C'] contains non-finite values")
    if np.any(C < 0) or np.any(C > 1):
        raise ValueError("coverage_obj['C'] must lie within [0, 1]")
    if not np.isfinite(omega).all():
        raise ValueError("coverage_obj['omega'] contains non-finite values")
    if np.any(omega < 0):
        raise ValueError("coverage_obj['omega'] must be non-negative")

    normalized_chain_df = chain_df.reset_index(drop=True)
    chain_ids = [
        _chain_id(normalized_chain_df.iloc[i].to_dict(), i)
        for i in range(len(normalized_chain_df))
    ]
    if len(chain_ids) != len(set(chain_ids)):
        raise ValueError("risk_selection chain_df contains duplicate chain IDs")

    return normalized_chain_df, {
        **coverage_obj,
        "C": C,
        "omega": omega,
        "categories": categories,
        "cat2idx": cat2idx,
    }


def _selected_categories(risk_selection: Mapping[str, Any]) -> List[str]:
    # selected_rank_df is authoritative for final focal membership AND order.
    # summary.selected_categories is retained only as a backward-compatible fallback.
    rank_df = risk_selection.get("selected_rank_df")
    if isinstance(rank_df, pd.DataFrame) and "Category" in rank_df.columns:
        df = rank_df.copy()
        if "rank" in df.columns:
            ranks = pd.to_numeric(df["rank"], errors="coerce")
            df = df.assign(_rank=ranks).sort_values(
                ["_rank", "Category"], na_position="last"
            )
        cats = _unique(df["Category"].astype(str).tolist())
        if cats:
            return cats

    summary = risk_selection.get("summary") if isinstance(risk_selection.get("summary"), dict) else {}
    return _unique(summary.get("selected_categories") or [])


def _take_cumulative_mass_indices(
    values: np.ndarray,
    target_share: float,
    max_items: int,
) -> List[int]:
    values = np.asarray(values, dtype=float).reshape(-1)
    positive_idx = np.where(values > 0)[0]
    if len(positive_idx) == 0 or max_items <= 0:
        return []

    order = positive_idx[np.argsort(-values[positive_idx])]
    total = float(values[positive_idx].sum())
    chosen: List[int] = []
    cumulative = 0.0
    for idx in order:
        chosen.append(int(idx))
        cumulative += float(values[idx])
        if len(chosen) >= max_items:
            break
        if total > 0 and cumulative / total >= target_share:
            break
    return chosen


def select_ecosystem_context(
    *,
    risk_selection: Mapping[str, Any],
    config: EcosystemContextConfig = EcosystemContextConfig(),
) -> Dict[str, Any]:
    """
    Deterministically expand the Top-K intervention set into a compact ecosystem
    context set.

    Important distinctions:
    - focal_products: intervention-priority products chosen upstream.
    - chain_context_products: non-focal products that add material explanatory
      mass on high-value focal-associated chains.
    - residual_chains: coverage safeguard only; included only when the focal-set
      chain-mass coverage is below config.residual_trigger_coverage.
    - residual_context_products: optional and OFF by default.
    """
    chain_df, cov = _validate_risk_selection(risk_selection)
    C = cov["C"]
    omega = np.clip(cov["omega"], 0.0, None)
    categories = cov["categories"]
    cat2idx = cov["cat2idx"]

    focal_products = _selected_categories(risk_selection)
    if not focal_products:
        raise ValueError("No focal products found in risk_selection")
    missing = [c for c in focal_products if c not in cat2idx]
    if missing:
        raise ValueError(f"Focal products missing from coverage matrix: {missing}")

    focal_idx = [cat2idx[c] for c in focal_products]
    focal_capture = 1.0 - np.prod(1.0 - np.clip(C[focal_idx, :], 0.0, 1.0), axis=0)

    total_mass = float(omega.sum())
    captured_mass_by_chain = omega * focal_capture
    captured_mass = float(captured_mass_by_chain.sum())
    focal_coverage = captured_mass / total_mass if total_mass > 0 else 0.0

    focal_chain_indices = _take_cumulative_mass_indices(
        captured_mass_by_chain,
        target_share=float(config.focal_chain_cumulative_mass),
        max_items=int(config.max_focal_context_chains),
    )
    focal_chain_mass = float(captured_mass_by_chain[focal_chain_indices].sum()) if focal_chain_indices else 0.0

    # Build chain -> products from the global chain table, not from per-product
    # truncated case packages.
    chain_products: Dict[int, List[str]] = {
        i: _chain_products(chain_df.iloc[i].to_dict())
        for i in range(len(chain_df))
    }

    # Context candidate ranking.
    candidate_rows: List[Dict[str, Any]] = []
    focal_set = set(focal_products)

    for cat in categories:
        if cat in focal_set:
            continue
        i = cat2idx[cat]

        if not focal_chain_indices or focal_chain_mass <= 0:
            context_mass = 0.0
            mass_share = 0.0
            shared_count = 0
            linked_focals: List[str] = []
            largest_single_share = 0.0
        else:
            contributions = (
                omega[focal_chain_indices]
                * np.clip(C[i, focal_chain_indices], 0.0, 1.0)
            )
            context_mass = float(contributions.sum())
            mass_share = context_mass / focal_chain_mass if focal_chain_mass > 0 else 0.0
            shared_count = int(np.sum(C[i, focal_chain_indices] > 0))

            linked: List[str] = []
            for l_idx in focal_chain_indices:
                if C[i, l_idx] <= 0:
                    continue
                products = set(chain_products[l_idx])
                linked.extend([f for f in focal_products if f in products])
            linked_focals = _unique(linked)

            largest_single = float(contributions.max()) if len(contributions) else 0.0
            largest_single_share = largest_single / focal_chain_mass if focal_chain_mass > 0 else 0.0

        eligible = bool(
            mass_share >= config.min_context_mass_share
            and (
                shared_count >= config.min_shared_focal_chain_count
                or len(linked_focals) >= config.min_linked_focal_products
                or largest_single_share >= config.critical_single_chain_share
            )
        )

        candidate_rows.append({
            "product_category": cat,
            "eligible": eligible,
            "context_mass": context_mass,
            "context_mass_share": mass_share,
            "shared_focal_chain_count": shared_count,
            "linked_focal_products": linked_focals,
            "largest_single_chain_share": largest_single_share,
        })

    candidate_rows.sort(
        key=lambda x: (
            x["eligible"],
            x["context_mass_share"],
            x["shared_focal_chain_count"],
            len(x["linked_focal_products"]),
        ),
        reverse=True,
    )

    chain_context_rows = [x for x in candidate_rows if x["eligible"]][
        : config.max_chain_context_products
    ]
    chain_context_products = [x["product_category"] for x in chain_context_rows]

    # Relevant focal-chain IDs for each selected context product.
    for row in chain_context_rows:
        i = cat2idx[row["product_category"]]
        relevant = [l for l in focal_chain_indices if C[i, l] > 0]
        relevant.sort(key=lambda l: captured_mass_by_chain[l] * C[i, l], reverse=True)
        row["relevant_focal_chain_indices"] = relevant
        row["relevant_focal_chain_ids"] = [
            _chain_id(chain_df.iloc[l].to_dict(), l)
            for l in relevant
        ]

    # Residual safeguard.
    residual_mass_by_chain = omega * (1.0 - focal_capture)
    residual_total = float(residual_mass_by_chain.sum())
    residual_share = residual_total / total_mass if total_mass > 0 else 0.0

    residual_chain_indices: List[int] = []
    information_coverage = focal_coverage
    if focal_coverage < config.residual_trigger_coverage and total_mass > 0:
        order = np.argsort(-residual_mass_by_chain)
        for l_idx in order:
            l_idx = int(l_idx)
            share = float(residual_mass_by_chain[l_idx] / total_mass)
            if residual_mass_by_chain[l_idx] <= 0:
                break
            if share < config.min_residual_chain_share:
                continue
            residual_chain_indices.append(l_idx)
            information_coverage += share
            if len(residual_chain_indices) >= config.max_residual_chains:
                break
            if information_coverage >= config.ecosystem_information_coverage_target:
                break

    residual_context_rows: List[Dict[str, Any]] = []
    if config.include_residual_product_context and residual_chain_indices:
        denom = float(residual_mass_by_chain[residual_chain_indices].sum())
        excluded = focal_set | set(chain_context_products)
        for cat in categories:
            if cat in excluded:
                continue
            i = cat2idx[cat]
            mass = float(np.sum(
                residual_mass_by_chain[residual_chain_indices]
                * np.clip(C[i, residual_chain_indices], 0.0, 1.0)
            ))
            share = mass / denom if denom > 0 else 0.0
            if share >= config.min_residual_product_mass_share:
                relevant = [l for l in residual_chain_indices if C[i, l] > 0]
                residual_context_rows.append({
                    "product_category": cat,
                    "residual_context_mass": mass,
                    "residual_context_mass_share": share,
                    "relevant_residual_chain_indices": relevant,
                    "relevant_residual_chain_ids": [
                        _chain_id(chain_df.iloc[l].to_dict(), l)
                        for l in relevant
                    ],
                })
        residual_context_rows.sort(
            key=lambda x: x["residual_context_mass_share"],
            reverse=True,
        )
        residual_context_rows = residual_context_rows[: config.max_residual_context_products]

    return {
        "focal_products": focal_products,
        "focal_chain_indices": focal_chain_indices,
        "focal_chain_ids": [
            _chain_id(chain_df.iloc[i].to_dict(), i)
            for i in focal_chain_indices
        ],
        "chain_context_products": chain_context_products,
        "chain_context_rows": chain_context_rows,
        "residual_chain_indices": residual_chain_indices,
        "residual_chain_ids": [
            _chain_id(chain_df.iloc[i].to_dict(), i)
            for i in residual_chain_indices
        ],
        "residual_context_rows": residual_context_rows,
        "metrics": {
            "n_total_products": len(categories),
            "n_focal_products": len(focal_products),
            "n_chain_context_products": len(chain_context_products),
            "n_residual_context_products": len(residual_context_rows),
            "n_total_chains": len(chain_df),
            "focal_chain_mass_coverage": round(float(focal_coverage), 6),
            "residual_chain_mass_share": round(float(residual_share), 6),
            "information_coverage_after_residual_signals": round(
                min(float(information_coverage), 1.0), 6
            ),
        },
        "config": asdict(config),
        "candidate_context_ranking": candidate_rows,
    }


# ============================================================================
# Context package construction
# ============================================================================

def _dimension_rows_for_dims(
    case: Mapping[str, Any],
    dims: set[str],
    max_rows: int,
) -> List[Dict[str, Any]]:
    rows = []
    for x in _list_of_dicts(case.get("observed_dimension_catalog")):
        dim = _norm_text(x.get("dimension"))
        if dims and dim not in dims:
            continue
        rows.append({
            k: x.get(k)
            for k in (
                # Current observed-dimension schema.
                "evidence_id", "product_category", "dimension", "n_review",
                "mean_sentiment", "recent_review_count", "review_share_in_product",
                "evidence_utility", "epistemic_status",
                # Backward-compatible legacy fields.
                "statement_count", "review_count", "weighted_sentiment",
                "recent_weighted_sentiment", "negative_share", "risk_score", "rank",
            )
            if x.get(k) not in (None, "", [], {})
        })
        if len(rows) >= max_rows:
            break
    return rows


def _dynamic_rows_for_dims(
    case: Mapping[str, Any],
    dims: set[str],
    max_rows: int,
) -> List[Dict[str, Any]]:
    """Compact current B facts for ecosystem context.

    The production data_service emits ``predicted_risk_state``,
    ``evolution_persistence``, ``dynamic_network_drivers`` and
    ``dynamic_network_position``.  Older aliases are still accepted so saved
    packages remain readable, but current names are authoritative.
    """
    max_rows = max(0, int(max_rows))
    if max_rows == 0:
        return []

    rows: List[Dict[str, Any]] = []
    for x in _list_of_dicts(case.get("model_facts")):
        fact_type = _norm_text(x.get("fact_type"))
        fact_id = x.get("fact_id")
        epistemic = x.get("epistemic_status")

        if fact_type in {"predicted_risk_state", "risk_state"}:
            signals = x.get("top_predicted_risk_dimensions") or []
            if isinstance(signals, list) and signals:
                for sig in signals:
                    if not isinstance(sig, dict):
                        continue
                    dim = _norm_text(sig.get("dimension"))
                    if dims and dim not in dims:
                        continue
                    row = {
                        "fact_id": fact_id,
                        "fact_type": "predicted_risk_state",
                        "dimension": dim,
                        "risk_rank": sig.get("risk_rank"),
                        "predicted_next_risk": sig.get("predicted_next_risk"),
                        "dominant_driver": sig.get("dominant_driver"),
                        "spillover_output": sig.get("spillover_output"),
                        "spillover_exposure": sig.get("spillover_exposure"),
                        "epistemic_status": sig.get("epistemic_status") or epistemic,
                    }
                    rows.append({k: v for k, v in row.items() if v not in (None, "", [], {})})
                    if len(rows) >= max_rows:
                        return rows
            else:
                dim = _norm_text(x.get("dimension"))
                if not dims or not dim or dim in dims:
                    rows.append({
                        k: x.get(k)
                        for k in (
                            "fact_id", "fact_type", "dimension", "risk_rank", "risk_score",
                            "predicted_risk", "predicted_next_risk_score", "trend", "statement",
                            "value", "epistemic_status",
                        )
                        if x.get(k) not in (None, "", [], {})
                    })

        elif fact_type in {"dynamic_network_drivers", "driver_pattern"}:
            rows.append({
                k: x.get(k)
                for k in (
                    "fact_id", "fact_type", "beta_api", "beta_coo", "beta_user",
                    "dominant_driver", "statement", "epistemic_status",
                )
                if x.get(k) not in (None, "", [], {})
            })

        elif fact_type in {"dynamic_network_position", "ecosystem_position"}:
            rows.append({
                k: x.get(k)
                for k in (
                    "fact_id", "fact_type", "spillover_output_score", "spillover_output_level",
                    "spillover_exposure_score", "spillover_exposure_level", "statement",
                    "epistemic_status",
                )
                if x.get(k) not in (None, "", [], {})
            })

        elif fact_type in {"evolution_persistence", "anchored_dimension_forecast"}:
            dim = _norm_text(x.get("dimension"))
            if dims and dim and dim not in dims:
                continue
            rows.append({
                k: x.get(k)
                for k in (
                    "fact_id", "fact_type", "dimension", "persistence_score", "persistence",
                    "risk_rank", "predicted_risk", "trend", "statement", "epistemic_status",
                )
                if x.get(k) not in (None, "", [], {})
            })

        if len(rows) >= max_rows:
            return rows[:max_rows]

    return rows[:max_rows]


def _relations_for_context(
    case: Mapping[str, Any],
    product: str,
    linked_products: set[str],
    max_rows: int,
) -> List[Dict[str, Any]]:
    rows = []
    for rel in _list_of_dicts(case.get("ecosystem_relations")):
        if not _relation_relevant(rel, product, linked_products):
            continue
        rows.append({
            k: rel.get(k)
            for k in (
                # Current static-relation schema.
                "relation_id", "relation_type", "product_category", "related_product",
                "channel_strengths", "overall_relation_strength", "direction_note",
                "epistemic_status",
                # Backward-compatible legacy alternatives.
                "source_category", "target_category", "from_category", "to_category",
                "source_product", "target_product", "from_product", "to_product",
                "dimension", "weight", "score", "statement",
            )
            if rel.get(k) not in (None, "", [], {})
        })
        if len(rows) >= max_rows:
            break
    return rows


def build_chain_context_product_packages(
    *,
    all_case_packages: Mapping[str, Dict[str, Any]] | Sequence[Dict[str, Any]],
    risk_selection: Mapping[str, Any],
    selection: Mapping[str, Any],
    config: EcosystemContextConfig = EcosystemContextConfig(),
) -> List[Dict[str, Any]]:
    case_idx = _index_case_packages(all_case_packages)
    chain_df, _ = _validate_risk_selection(risk_selection)

    packages: List[Dict[str, Any]] = []
    for row in selection.get("chain_context_rows", []) or []:
        product = _norm_category(row.get("product_category"))
        case = case_idx.get(product)
        case_available = isinstance(case, dict)
        case_view: Mapping[str, Any] = case if case_available else {}

        relevant_indices = list(row.get("relevant_focal_chain_indices") or [])
        relevant_chains = [
            _compact_chain(chain_df.iloc[int(i)].to_dict(), int(i))
            for i in relevant_indices
        ]
        dims = {
            _norm_text(ch.get("dimension"))
            for ch in relevant_chains
            if _norm_text(ch.get("dimension"))
        }
        linked_focals = set(row.get("linked_focal_products") or [])

        packages.append({
            "product_category": product,
            "context_role": "chain_context",
            "selection_reason": {
                "context_mass_share": round(_safe_float(row.get("context_mass_share")), 6),
                "shared_focal_chain_count": int(row.get("shared_focal_chain_count") or 0),
                "linked_focal_products": list(row.get("linked_focal_products") or []),
                "largest_single_chain_share": round(
                    _safe_float(row.get("largest_single_chain_share")), 6
                ),
            },
            "supporting_profile_available": bool(case_available),
            "observed_dimension_context": _dimension_rows_for_dims(
                case_view, dims, config.max_observed_dimension_rows
            ),
            "dynamic_context": _dynamic_rows_for_dims(
                case_view, dims, config.max_dynamic_rows
            ),
            "relevant_relations": _relations_for_context(
                case_view, product, linked_focals, config.max_relation_rows
            ),
            "relevant_risk_chains": relevant_chains,
            "evidence_availability_note": (
                "Product profile package available."
                if case_available
                else "No product profile package available; use supplied risk-chain structure only."
            ),
            "interpretation_constraint": (
                "Secondary explanatory context only; do not infer a new product-level "
                "mechanism or treat this product as an intervention target solely from inclusion."
            ),
        })

    return packages


def build_residual_chain_signals(
    *,
    risk_selection: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    chain_df, cov = _validate_risk_selection(risk_selection)
    omega = np.clip(cov["omega"], 0.0, None)
    total_mass = float(omega.sum())

    signals = []
    for i in selection.get("residual_chain_indices", []) or []:
        i = int(i)
        chain = chain_df.iloc[i].to_dict()
        compact = _compact_chain(chain, i)
        compact["total_chain_mass_share"] = (
            round(float(omega[i] / total_mass), 6) if total_mass > 0 else 0.0
        )
        compact["context_role"] = "residual_coverage_signal"
        signals.append(compact)
    return signals


def build_residual_context_product_packages(
    *,
    all_case_packages: Mapping[str, Dict[str, Any]] | Sequence[Dict[str, Any]],
    risk_selection: Mapping[str, Any],
    selection: Mapping[str, Any],
    config: EcosystemContextConfig = EcosystemContextConfig(),
) -> List[Dict[str, Any]]:
    if not config.include_residual_product_context:
        return []

    case_idx = _index_case_packages(all_case_packages)
    chain_df, _ = _validate_risk_selection(risk_selection)
    out = []
    for row in selection.get("residual_context_rows", []) or []:
        product = _norm_category(row.get("product_category"))
        case = case_idx.get(product)
        case_available = isinstance(case, dict)
        case_view: Mapping[str, Any] = case if case_available else {}
        indices = list(row.get("relevant_residual_chain_indices") or [])
        chains = [
            _compact_chain(chain_df.iloc[int(i)].to_dict(), int(i))
            for i in indices
        ]
        dims = {
            _norm_text(ch.get("dimension"))
            for ch in chains
            if _norm_text(ch.get("dimension"))
        }
        out.append({
            "product_category": product,
            "context_role": "residual_medium_context",
            "selection_reason": {
                "residual_context_mass_share": round(
                    _safe_float(row.get("residual_context_mass_share")), 6
                )
            },
            "supporting_profile_available": bool(case_available),
            "observed_dimension_context": _dimension_rows_for_dims(
                case_view, dims, config.max_observed_dimension_rows
            ),
            "dynamic_context": _dynamic_rows_for_dims(
                case_view, dims, config.max_dynamic_rows
            ),
            "relevant_risk_chains": chains,
            "evidence_availability_note": (
                "Product profile package available."
                if case_available
                else "No product profile package available; use supplied residual chains only."
            ),
            "interpretation_constraint": (
                "Coverage calibration only. Do not use this product to override focal-product "
                "diagnoses or convert it into an intervention target."
            ),
        })
    return out


# ============================================================================
# Focal reviewed context
# ============================================================================

def _qualified_id(product: str, local_id: Any) -> str:
    local = _norm_text(local_id)
    return f"{product}::{local}" if product and local else local


def build_focal_product_contexts(
    *,
    focal_products: Sequence[str],
    reviewed_diagnoses: Sequence[Dict[str, Any]],
    action_outputs: Sequence[Dict[str, Any]],
    risk_selection: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    diag_idx = _index_by_product(reviewed_diagnoses)

    # Action service commonly returns {parsed:{product_category,...}}. Accept
    # either the parsed object or the direct object.
    action_idx: Dict[str, Dict[str, Any]] = {}
    for raw in action_outputs or []:
        if not isinstance(raw, dict):
            continue
        obj = raw.get("parsed") if isinstance(raw.get("parsed"), dict) else raw
        cat = _norm_category(obj.get("product_category"))
        if cat:
            action_idx[cat] = obj

    selected_rank_df = risk_selection.get("selected_rank_df")
    rank_map: Dict[str, Dict[str, Any]] = {}
    if isinstance(selected_rank_df, pd.DataFrame) and not selected_rank_df.empty:
        for _, row in selected_rank_df.iterrows():
            cat = _norm_category(row.get("Category"))
            if cat:
                rank_map[cat] = row.to_dict()

    out: List[Dict[str, Any]] = []
    for product in focal_products:
        diag = diag_idx.get(product)
        if not diag:
            raise ValueError(
                f"Focal product {product!r} has no reviewed diagnosis in ecosystem input"
            )
        actions_obj = action_idx.get(product, {"actions": []})

        mechanisms = []
        for m in _list_of_dicts(diag.get("mechanisms")):
            human_status = _norm_text(
                m.get("human_assessment") or m.get("assessment_status")
            ).lower()
            if human_status not in {"supported", "plausible", "uncertain"}:
                continue
            local_mid = _norm_text(m.get("mechanism_id"))
            mechanisms.append({
                "mechanism_id": _qualified_id(product, local_mid),
                "local_mechanism_id": local_mid,
                "mechanism_family": m.get("mechanism_family"),
                "mechanism_scope": m.get("mechanism_scope"),
                "statement": m.get("human_revision") or m.get("statement"),
                "human_assessment": human_status or "not_assessed",
                "human_validation_note": m.get("human_validation_note") or m.get("validation_note"),
                "source_refs": list(m.get("source_refs") or []),
                "confidence": m.get("confidence"),
                "causal_status": m.get("causal_status"),
            })

        actions = []
        for a in _list_of_dicts(actions_obj.get("actions")):
            decision = _norm_text(a.get("human_decision") or a.get("decision")).lower()
            if decision not in {"accept", "revise", "approved"}:
                continue
            local_aid = _norm_text(a.get("action_id"))
            local_mids = list(a.get("linked_mechanism_ids") or [])
            actions.append({
                "action_id": _qualified_id(product, local_aid),
                "local_action_id": local_aid,
                "priority_rank": a.get("priority_rank"),
                "action_subfamily": a.get("action_subfamily"),
                "action_level": a.get("action_level"),
                "action": a.get("human_revision") or a.get("revision") or a.get("action"),
                "human_decision": decision or "not_assessed",
                "linked_mechanism_ids": [
                    _qualified_id(product, mid) for mid in local_mids
                ],
                "source_refs": list(a.get("source_refs") or []),
                "target_products": list(a.get("target_products") or []),
                "target_layers": list(a.get("target_layers") or []),
                "validation_metric": a.get("validation_metric"),
            })

        rank = rank_map.get(product, {})
        out.append({
            "product_category": product,
            "context_role": "focal_intervention_target",
            "selection_metadata": {
                "rank": rank.get("rank"),
                "marginal_chain_value": rank.get("marginal_chain_value"),
                "captured_chain_count": rank.get("captured_chain_count"),
                "captured_chain_weight_raw": rank.get("captured_chain_weight_raw"),
            },
            "risk_dimensions": list(diag.get("risk_dimensions") or []),
            "symptom_families": list(diag.get("symptom_families") or []),
            "reviewed_mechanisms": mechanisms,
            "validation_tasks": _list_of_dicts(diag.get("validation_tasks"))[:3],
            "abstentions": _list_of_dicts(diag.get("abstentions"))[:3],
            "reviewed_actions": actions,
        })
    return out


# ============================================================================
# API aggregation / systemic API gate
# ============================================================================

def build_systemic_api_context(
    *,
    all_case_packages: Mapping[str, Dict[str, Any]] | Sequence[Dict[str, Any]],
    focal_products: Sequence[str],
    chain_context_products: Sequence[str],
    config: EcosystemContextConfig = EcosystemContextConfig(),
    api_candidates_by_product: Optional[Mapping[str, Sequence[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """
    Shared-by-default API gate.

    Only APIs present in at least config.min_api_product_support included products
    are sent to the ecosystem LLM. This intentionally excludes standalone API
    cards from ecosystem synthesis; they remain available in product-level H.
    """
    case_idx = _index_case_packages(all_case_packages)
    focal_set = set(focal_products)
    included_products = _unique(list(focal_products) + list(chain_context_products))

    buckets: Dict[str, Dict[str, Any]] = {}
    for product in included_products:
        case = case_idx.get(product)
        if not case and not api_candidates_by_product:
            continue
        if api_candidates_by_product is not None and product in api_candidates_by_product:
            cards = [x for x in (api_candidates_by_product.get(product) or []) if isinstance(x, dict)]
        else:
            cards = _list_of_dicts((case or {}).get("api_knowledge"))
        for card in cards:
            name = _api_name(card)
            key = _api_key(name)
            if not key:
                continue
            bucket = buckets.setdefault(key, {
                "api_name": name,
                "products": [],
                "focal_products": [],
                "context_products": [],
                "knowledge_refs": [],
                "cards": [],
            })
            bucket["products"].append(product)
            if product in focal_set:
                bucket["focal_products"].append(product)
            else:
                bucket["context_products"].append(product)
            kid = _norm_text(card.get("knowledge_id"))
            if kid:
                bucket["knowledge_refs"].append(kid)
            if len(bucket["cards"]) < config.max_api_cards_per_api:
                bucket["cards"].append(_compact_api_card(card))

    out = []
    for bucket in buckets.values():
        bucket["products"] = _unique(bucket["products"])
        bucket["focal_products"] = _unique(bucket["focal_products"])
        bucket["context_products"] = _unique(bucket["context_products"])
        bucket["knowledge_refs"] = _unique(bucket["knowledge_refs"])

        if len(bucket["products"]) < config.min_api_product_support:
            continue

        # Shared API gets highest priority when it crosses focal products, then
        # when it bridges focal and context products.
        if len(bucket["focal_products"]) >= 2:
            role = "shared_across_focal_products"
            priority = 2
        elif bucket["focal_products"] and bucket["context_products"]:
            role = "shared_across_focal_and_chain_context"
            priority = 1
        else:
            role = "shared_context_only"
            priority = 0

        out.append({
            **bucket,
            "ecosystem_role": role,
            "product_support_count": len(bucket["products"]),
            "epistemic_constraint": (
                "Implementation-structure knowledge only; shared use does not prove a shared defect."
            ),
            "_priority": priority,
        })

    out.sort(
        key=lambda x: (x["_priority"], x["product_support_count"]),
        reverse=True,
    )
    out = out[: config.max_shared_api_items]
    for row in out:
        row.pop("_priority", None)
    return out


# ============================================================================
# Full payload builder
# ============================================================================

def build_ecosystem_synthesis_payload(
    *,
    risk_selection: Mapping[str, Any],
    all_case_packages: Mapping[str, Dict[str, Any]] | Sequence[Dict[str, Any]],
    reviewed_diagnoses: Sequence[Dict[str, Any]],
    action_outputs: Sequence[Dict[str, Any]],
    config: EcosystemContextConfig = EcosystemContextConfig(),
    api_candidates_by_product: Optional[Mapping[str, Sequence[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    selection = select_ecosystem_context(
        risk_selection=risk_selection,
        config=config,
    )

    focal_products = list(selection["focal_products"])
    chain_context_products = list(selection["chain_context_products"])

    focal_contexts = build_focal_product_contexts(
        focal_products=focal_products,
        reviewed_diagnoses=reviewed_diagnoses,
        action_outputs=action_outputs,
        risk_selection=risk_selection,
    )

    chain_context_packages = build_chain_context_product_packages(
        all_case_packages=all_case_packages,
        risk_selection=risk_selection,
        selection=selection,
        config=config,
    )

    residual_signals = build_residual_chain_signals(
        risk_selection=risk_selection,
        selection=selection,
    )

    residual_products = build_residual_context_product_packages(
        all_case_packages=all_case_packages,
        risk_selection=risk_selection,
        selection=selection,
        config=config,
    )

    api_context = build_systemic_api_context(
        all_case_packages=all_case_packages,
        focal_products=focal_products,
        chain_context_products=chain_context_products,
        config=config,
        api_candidates_by_product=api_candidates_by_product,
    )

    # Time metadata: take it from any available current package.
    case_idx = _index_case_packages(all_case_packages)
    time_meta: Dict[str, Any] = {}
    for product in focal_products + chain_context_products:
        case = case_idx.get(product)
        if not case:
            continue
        meta = case.get("case_meta") if isinstance(case.get("case_meta"), dict) else {}
        time_meta = {
            "analysis_cutoff": meta.get("analysis_cutoff"),
            "target_month": meta.get("target_month"),
        }
        break

    return {
        "TIME_CONTEXT": time_meta,
        "SELECTION_METADATA": {
            **selection["metrics"],
            "focal_products": focal_products,
            "chain_context_products": chain_context_products,
            "residual_chain_ids": list(selection["residual_chain_ids"]),
            "selection_policy": (
                "Top-K focal products are chosen upstream by risk-chain mass capture. "
                "Chain-context products are selected deterministically for incremental "
                "explanatory mass. Residual signals are included only as a coverage safeguard."
            ),
        },
        "FOCAL_PRODUCTS": focal_contexts,
        "CHAIN_CONTEXT_PRODUCTS": chain_context_packages,
        "SYSTEMIC_API_CONTEXT": api_context,
        "RESIDUAL_CHAIN_SIGNALS": residual_signals,
        "RESIDUAL_CONTEXT_PRODUCTS": residual_products,
    }


# ============================================================================
# Output audit (no repair / no second model call)
# ============================================================================

def _collect_input_ids(payload: Mapping[str, Any]) -> Dict[str, set[str]]:
    mechanism_ids: set[str] = set()
    action_ids: set[str] = set()
    source_refs: set[str] = set()

    for product in payload.get("FOCAL_PRODUCTS", []) or []:
        if not isinstance(product, dict):
            continue
        for m in product.get("reviewed_mechanisms", []) or []:
            if not isinstance(m, dict):
                continue
            if m.get("mechanism_id"):
                mechanism_ids.add(str(m["mechanism_id"]))
            source_refs.update(str(x) for x in (m.get("source_refs") or []) if str(x).strip())
        for a in product.get("reviewed_actions", []) or []:
            if not isinstance(a, dict):
                continue
            if a.get("action_id"):
                action_ids.add(str(a["action_id"]))
            source_refs.update(str(x) for x in (a.get("source_refs") or []) if str(x).strip())

    for section in ("CHAIN_CONTEXT_PRODUCTS", "RESIDUAL_CONTEXT_PRODUCTS"):
        for product in payload.get(section, []) or []:
            if not isinstance(product, dict):
                continue
            for rel in product.get("relevant_relations", []) or []:
                if isinstance(rel, dict) and rel.get("relation_id"):
                    source_refs.add(str(rel["relation_id"]))
            for fact in product.get("dynamic_context", []) or []:
                if isinstance(fact, dict) and fact.get("fact_id"):
                    source_refs.add(str(fact["fact_id"]))
            for obs in product.get("observed_dimension_context", []) or []:
                if isinstance(obs, dict) and obs.get("evidence_id"):
                    source_refs.add(str(obs["evidence_id"]))
            for ch in product.get("relevant_risk_chains", []) or []:
                if isinstance(ch, dict) and ch.get("chain_id"):
                    source_refs.add(str(ch["chain_id"]))

    for ch in payload.get("RESIDUAL_CHAIN_SIGNALS", []) or []:
        if isinstance(ch, dict) and ch.get("chain_id"):
            source_refs.add(str(ch["chain_id"]))

    for api in payload.get("SYSTEMIC_API_CONTEXT", []) or []:
        if not isinstance(api, dict):
            continue
        source_refs.update(str(x) for x in (api.get("knowledge_refs") or []) if str(x).strip())

    return {
        "mechanism_ids": mechanism_ids,
        "action_ids": action_ids,
        "source_refs": source_refs,
    }


def _ecosystem_allowed_dimensions(payload: Mapping[str, Any]) -> set[str]:
    dims: set[str] = set()
    for product in payload.get("FOCAL_PRODUCTS", []) or []:
        if isinstance(product, dict):
            dims.update(_norm_text(x) for x in (product.get("risk_dimensions") or []) if _norm_text(x))
    for section in ("CHAIN_CONTEXT_PRODUCTS", "RESIDUAL_CONTEXT_PRODUCTS"):
        for product in payload.get(section, []) or []:
            if not isinstance(product, dict):
                continue
            for row in product.get("observed_dimension_context", []) or []:
                if isinstance(row, dict) and _norm_text(row.get("dimension")):
                    dims.add(_norm_text(row.get("dimension")))
            for row in product.get("dynamic_context", []) or []:
                if isinstance(row, dict) and _norm_text(row.get("dimension")):
                    dims.add(_norm_text(row.get("dimension")))
            for row in product.get("relevant_risk_chains", []) or []:
                if isinstance(row, dict) and _norm_text(row.get("dimension")):
                    dims.add(_norm_text(row.get("dimension")))
    for row in payload.get("RESIDUAL_CHAIN_SIGNALS", []) or []:
        if isinstance(row, dict) and _norm_text(row.get("dimension")):
            dims.add(_norm_text(row.get("dimension")))
    return dims


def validate_ecosystem_output(
    parsed: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> None:
    """Validate the ecosystem output contract without any repair/model retry."""
    if not isinstance(parsed, Mapping):
        raise TypeError("ecosystem output must be a JSON object")

    list_fields = {
        "systemic_findings": 5,
        "portfolio_consolidations": 4,
        "validation_priorities": 4,
        "governance_priorities": 3,
    }
    for field, max_items in list_fields.items():
        value = parsed.get(field)
        if not isinstance(value, list):
            raise ValueError(f"root.{field} must be a list")
        if len(value) > max_items:
            raise ValueError(f"root.{field} exceeds maximum {max_items} items")
        if any(not isinstance(x, dict) for x in value):
            raise ValueError(f"root.{field} must contain JSON objects only")

    if not isinstance(parsed.get("ecosystem_summary"), str):
        raise ValueError("root.ecosystem_summary must be a string")

    finding_types = {
        "shared_dependency", "propagation_pathway", "recurring_cross_product_risk",
        "shared_implementation_structure", "capability_gap", "state_consistency",
        "dependency_resilience", "compatibility_or_lifecycle", "privacy_or_security",
        "governance_gap", "other",
    }
    evidence_strengths = {"strong", "moderate", "weak"}
    portfolio_levels = {"interface", "cross_product", "ecosystem"}

    focal_products = {
        _norm_category(x.get("product_category"))
        for x in payload.get("FOCAL_PRODUCTS", []) or []
        if isinstance(x, dict) and _norm_category(x.get("product_category"))
    }
    context_products = {
        _norm_category(x.get("product_category"))
        for section in ("CHAIN_CONTEXT_PRODUCTS", "RESIDUAL_CONTEXT_PRODUCTS")
        for x in (payload.get(section, []) or [])
        if isinstance(x, dict) and _norm_category(x.get("product_category"))
    }
    allowed_dims = _ecosystem_allowed_dimensions(payload)

    def _require_list_field(item: Mapping[str, Any], field: str, where: str) -> List[Any]:
        value = item.get(field)
        if not isinstance(value, list):
            raise ValueError(f"{where}.{field} must be a list")
        return value

    def _require_string_field(item: Mapping[str, Any], field: str, where: str) -> str:
        value = item.get(field)
        if not isinstance(value, str):
            raise ValueError(f"{where}.{field} must be a string")
        return value

    finding_ids: List[str] = []
    for i, item in enumerate(parsed.get("systemic_findings") or []):
        where = f"root.systemic_findings[{i}]"
        for field in (
            "affected_focal_products", "context_products", "risk_dimensions",
            "supporting_mechanism_ids", "risk_chain_refs", "relation_refs", "api_refs",
        ):
            _require_list_field(item, field, where)
        _require_string_field(item, "statement", where)
        _require_string_field(item, "main_uncertainty", where)
        fid = _norm_text(item.get("finding_id"))
        if not fid:
            raise ValueError(f"{where}.finding_id must be non-empty")
        finding_ids.append(fid)
        ftype = _norm_text(item.get("finding_type"))
        if ftype not in finding_types:
            raise ValueError(f"{where}.finding_type is invalid: {ftype!r}")
        strength = _norm_text(item.get("evidence_strength"))
        if strength not in evidence_strengths:
            raise ValueError(f"{where}.evidence_strength is invalid: {strength!r}")

        affected = [_norm_category(x) for x in (item.get("affected_focal_products") or []) if _norm_category(x)]
        invalid_focal = [x for x in affected if x not in focal_products]
        if invalid_focal:
            raise ValueError(f"{where}.affected_focal_products contains non-focal products: {invalid_focal}")
        if ftype == "recurring_cross_product_risk" and len(set(affected)) < 2:
            raise ValueError(
                f"{where}: recurring_cross_product_risk requires at least two focal products"
            )
        ctx = [_norm_category(x) for x in (item.get("context_products") or []) if _norm_category(x)]
        invalid_ctx = [x for x in ctx if x not in context_products]
        if invalid_ctx:
            raise ValueError(f"{where}.context_products contains unavailable context products: {invalid_ctx}")
        dims = [_norm_text(x) for x in (item.get("risk_dimensions") or []) if _norm_text(x)]
        invalid_dims = [x for x in dims if x not in allowed_dims]
        if invalid_dims:
            raise ValueError(f"{where}.risk_dimensions contains unseen dimensions: {invalid_dims}")

        # Shared/systemic explanations require >=2 heterogeneous evidence classes.
        if ftype != "recurring_cross_product_risk":
            evidence_classes = 0
            if item.get("supporting_mechanism_ids"):
                evidence_classes += 1  # A
            if item.get("risk_chain_refs") or item.get("relation_refs"):
                evidence_classes += 1  # B
            if item.get("api_refs"):
                evidence_classes += 1  # C
            if evidence_classes < 2:
                raise ValueError(
                    f"{where} violates heterogeneous-evidence requirement: "
                    f"found {evidence_classes} evidence classes, need at least 2"
                )
            if f == "strong" and evidence_classes < 3:
                raise ValueError(f"{where}: strong finding requires evidence classes A+B+C")

    if len(finding_ids) != len(set(finding_ids)):
        raise ValueError("root.systemic_findings contains duplicate finding_id values")
    finding_id_set = set(finding_ids)

    def _check_supported_ids(items: Sequence[Dict[str, Any]], field_name: str) -> None:
        for i, item in enumerate(items):
            refs = [_norm_text(x) for x in (item.get("supported_finding_ids") or []) if _norm_text(x)]
            invalid = [x for x in refs if x not in finding_id_set]
            if invalid:
                raise ValueError(f"root.{field_name}[{i}].supported_finding_ids contains unknown IDs: {invalid}")

    consolidations = parsed.get("portfolio_consolidations") or []
    portfolio_ids: List[str] = []
    for i, item in enumerate(consolidations):
        where = f"root.portfolio_consolidations[{i}]"
        for field in ("source_action_ids", "supported_finding_ids", "affected_focal_products"):
            _require_list_field(item, field, where)
        _require_string_field(item, "recommendation", where)
        _require_string_field(item, "validation_metric", where)
        pid = _norm_text(item.get("portfolio_id"))
        if not pid:
            raise ValueError(f"{where}.portfolio_id must be non-empty")
        portfolio_ids.append(pid)
        if _norm_text(item.get("portfolio_level")) not in portfolio_levels:
            raise ValueError(f"{where}.portfolio_level is invalid")
        if not (item.get("source_action_ids") or []):
            raise ValueError(f"{where}.source_action_ids must contain at least one supplied action")
        affected = [_norm_category(x) for x in (item.get("affected_focal_products") or []) if _norm_category(x)]
        invalid = [x for x in affected if x not in focal_products]
        if invalid:
            raise ValueError(f"{where}.affected_focal_products contains non-focal products: {invalid}")
    if len(portfolio_ids) != len(set(portfolio_ids)):
        raise ValueError("root.portfolio_consolidations contains duplicate portfolio_id values")

    validation_ids = [_norm_text(x.get("validation_id")) for x in (parsed.get("validation_priorities") or [])]
    if any(not x for x in validation_ids) or len(validation_ids) != len(set(validation_ids)):
        raise ValueError("root.validation_priorities requires unique non-empty validation_id values")
    for i, item in enumerate(parsed.get("validation_priorities") or []):
        where = f"root.validation_priorities[{i}]"
        for field in ("supported_finding_ids", "products", "required_evidence"):
            _require_list_field(item, field, where)
        _require_string_field(item, "question", where)
        products = [_norm_category(x) for x in (item.get("products") or []) if _norm_category(x)]
        invalid = [x for x in products if x not in focal_products | context_products]
        if invalid:
            raise ValueError(f"root.validation_priorities[{i}].products contains unknown products: {invalid}")

    governance_items = parsed.get("governance_priorities") or []
    governance_ids = [_norm_text(x.get("priority_id")) for x in governance_items]
    if any(not x for x in governance_ids) or len(governance_ids) != len(set(governance_ids)):
        raise ValueError("root.governance_priorities requires unique non-empty priority_id values")
    for i, item in enumerate(governance_items):
        where = f"root.governance_priorities[{i}]"
        _require_list_field(item, "supported_finding_ids", where)
        _require_list_field(item, "source_action_ids", where)
        _require_string_field(item, "priority", where)

    _check_supported_ids(consolidations, "portfolio_consolidations")
    _check_supported_ids(parsed.get("validation_priorities") or [], "validation_priorities")
    _check_supported_ids(parsed.get("governance_priorities") or [], "governance_priorities")


def audit_ecosystem_output(
    parsed: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    allowed = _collect_input_ids(payload)
    used_mechanisms: List[str] = []
    used_actions: List[str] = []
    used_refs: List[str] = []

    # Current schema: systemic_findings.  Keep no legacy silent fallback: if the
    # prompt/schema changes again, validate_ecosystem_output should be updated too.
    for item in parsed.get("systemic_findings", []) or []:
        if not isinstance(item, dict):
            continue
        used_mechanisms.extend(item.get("supporting_mechanism_ids") or [])
        used_refs.extend(item.get("risk_chain_refs") or [])
        used_refs.extend(item.get("relation_refs") or [])
        used_refs.extend(item.get("api_refs") or [])

    for item in parsed.get("portfolio_consolidations", []) or []:
        if not isinstance(item, dict):
            continue
        used_actions.extend(item.get("source_action_ids") or [])

    # validation_priorities.required_evidence is free text, not an evidence-ID
    # field, so it is intentionally not audited as a reference.
    for item in parsed.get("governance_priorities", []) or []:
        if not isinstance(item, dict):
            continue
        used_actions.extend(item.get("source_action_ids") or [])

    used_mechanisms = _unique(used_mechanisms)
    used_actions = _unique(used_actions)
    used_refs = _unique(used_refs)

    invalid_mechanisms = [x for x in used_mechanisms if x not in allowed["mechanism_ids"]]
    invalid_actions = [x for x in used_actions if x not in allowed["action_ids"]]
    invalid_refs = [x for x in used_refs if x not in allowed["source_refs"]]

    return {
        "n_used_mechanism_ids": len(used_mechanisms),
        "n_used_action_ids": len(used_actions),
        "n_used_source_refs": len(used_refs),
        "invalid_mechanism_ids": invalid_mechanisms,
        "invalid_action_ids": invalid_actions,
        "invalid_source_refs": invalid_refs,
        "mechanism_reference_validity": (
            1.0 - len(invalid_mechanisms) / len(used_mechanisms)
            if used_mechanisms else 1.0
        ),
        "action_reference_validity": (
            1.0 - len(invalid_actions) / len(used_actions)
            if used_actions else 1.0
        ),
        "source_reference_validity": (
            1.0 - len(invalid_refs) / len(used_refs)
            if used_refs else 1.0
        ),
        "reference_contract_ok": not (
            invalid_mechanisms or invalid_actions or invalid_refs
        ),
    }


# ============================================================================
# One-shot generation
# ============================================================================

def generate_ecosystem_synthesis(
    *,
    risk_selection: Mapping[str, Any],
    reviewed_diagnoses: Sequence[Dict[str, Any]],
    action_outputs: Sequence[Dict[str, Any]],
    all_case_packages: Mapping[str, Dict[str, Any]] | Sequence[Dict[str, Any]],
    model_name: Optional[str] = None,
    temperature: float = 0.2,
    config: EcosystemContextConfig = EcosystemContextConfig(),
    use_json_mode: Optional[bool] = None,
    api_candidates_by_product: Optional[Mapping[str, Sequence[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """Generate one audited ecosystem synthesis.

    System/provider failures from ``call_json`` propagate immediately.  A
    completed model response that cannot satisfy the local JSON/output contract
    is raised as ``EcosystemGenerationTraceError`` with the raw attempt attached,
    allowing the orchestration layer to persist it for human inspection without
    misclassifying API/runtime failures as reviewable model output.
    """
    payload = build_ecosystem_synthesis_payload(
        risk_selection=risk_selection,
        all_case_packages=all_case_packages,
        reviewed_diagnoses=reviewed_diagnoses,
        action_outputs=action_outputs,
        config=config,
        api_candidates_by_product=api_candidates_by_product,
    )

    result = call_json(
        system_prompt=ECOSYSTEM_SYNTHESIS_PROMPT,
        user_payload=payload,
        model_name=model_name,
        temperature=float(temperature),
        use_json_mode=use_json_mode,
    )

    if not isinstance(result, dict):
        raise TypeError("call_json must return a dict")

    parsed = result.get("parsed")
    if not isinstance(parsed, dict):
        error = (
            "ecosystem synthesis JSON parse failed: "
            f"{result.get('parse_error')!r}"
        )
        raise EcosystemGenerationTraceError(
            error,
            llm_trace=_ecosystem_attempt_trace(
                result=result,
                parsed=None,
                error=error,
                input_payload=payload,
            ),
        )

    try:
        validate_ecosystem_output(parsed, payload)
        audit = audit_ecosystem_output(parsed, payload)
        if not audit.get("reference_contract_ok"):
            raise ValueError(
                "ecosystem synthesis contains references not present in the input: "
                f"mechanisms={audit.get('invalid_mechanism_ids')}, "
                f"actions={audit.get('invalid_action_ids')}, "
                f"refs={audit.get('invalid_source_refs')}"
            )
    except Exception as exc:
        raise EcosystemGenerationTraceError(
            str(exc),
            llm_trace=_ecosystem_attempt_trace(
                result=result,
                parsed=parsed,
                error=str(exc),
                input_payload=payload,
            ),
        ) from exc

    trace = _ecosystem_attempt_trace(
        result=result,
        parsed=parsed,
        error=None,
        input_payload=payload,
    )
    return {
        "raw_output": result.get("raw_output", ""),
        "parsed": parsed,
        "usage": dict(result.get("usage") or {}),
        "request_meta": dict(result.get("request_meta") or {}),
        "ecosystem_input": payload,
        "ecosystem_audit": audit,
        "llm_trace": trace,
    }


__all__ = [
    "ECOSYSTEM_SYNTHESIS_PROMPT",
    "EcosystemGenerationTraceError",
    "EcosystemContextConfig",
    "select_ecosystem_context",
    "build_chain_context_product_packages",
    "build_residual_chain_signals",
    "build_residual_context_product_packages",
    "build_focal_product_contexts",
    "build_systemic_api_context",
    "build_ecosystem_synthesis_payload",
    "validate_ecosystem_output",
    "audit_ecosystem_output",
    "generate_ecosystem_synthesis",
]
