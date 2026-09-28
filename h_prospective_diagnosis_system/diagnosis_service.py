from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from data_service import all_reference_ids, stable_id, build_llm_case_view
from llm_service import call_json

MECHANISM_FAMILIES = [
    'local_hardware_or_device',
    'connectivity_or_state_sync',
    'setup_or_configuration',
    'api_or_interface',
    'automation_or_coordination',
    'capability_or_compatibility',
    'privacy_or_security',
    'ecosystem_dependency',
    'user_context',
    'unclear',
]

SYMPTOM_FAMILIES = [
    'setup_failure',
    'connectivity_failure',
    'unresponsive_or_outage',
    'automation_failure',
    'compatibility_failure',
    'privacy_security_issue',
    'hardware_failure',
    'performance_or_response_issue',
    'media_failure',
    'sensor_failure',
    'other',
]


DIAGNOSIS_PROMPT = f"""
You diagnose risk in smart connected product ecosystems.
Use only CASE_PACKAGE. Return one valid JSON object and no prose outside JSON.

Rules:
1. Separate observed evidence, model-inferred facts/relations, risk chains, and API knowledge.
2. Across its linked claims and source_refs, each mechanism should have at least one observed-evidence reference and at least one genuinely relevant contextual reference from model facts, ecosystem relations, risk chains, or API knowledge.
3. If no mechanism satisfies Rule 2, return mechanisms=[]. Do not invent or over-specify a mechanism merely to avoid an empty mechanism list. Use abstentions and validation_tasks to state what additional evidence would be needed.
4. Symptoms are not mechanisms. Do not classify a mechanism merely by repeating the observed symptom.
5. A standalone_risk record indicates elevated risk, not propagation. Only when relations or propagation chains support an upstream/shared dependency should mechanism_family or mechanism_scope reflect that pathway.
6. Model relations are predictive/associational, not causal proof. API knowledge is candidate mechanism knowledge, not proof of an implementation defect.
7. If evidence cannot locate an exact root cause, abstain and propose a validation task. Link the task to a retained mechanism only when that task actually validates that mechanism; a task investigating a separate unresolved issue may remain unlinked even when other mechanisms are present.
8. Return at most 4 claims, 3 mechanisms, and 3 validation tasks. Keep statements concise.
9. risk_dimensions must reuse dimension names from CASE_PACKAGE.observed_dimension_catalog or detailed observed_evidence; never output General.
10. mechanism_family must be one of: {MECHANISM_FAMILIES}
11. symptom_family must be one of: {SYMPTOM_FAMILIES}
12. confidence must be between 0 and 1.

Required JSON shape:
{{
  "case_id": "",
  "product_category": "",
  "risk_dimensions": [],
  "symptom_families": [],
  "claims": [
    {{"claim_id":"CL1","claim_type":"observed|model_pattern|knowledge_context","statement":"","source_refs":[]}}
  ],
  "mechanisms": [
    {{
      "mechanism_id":"M1",
      "mechanism_family":"",
      "mechanism_scope":"local|interface|cross_product|ecosystem",
      "statement":"",
      "supporting_claim_ids":[],
      "source_refs":[],
      "causal_status":"unverified_hypothesis",
      "confidence":0.0,
      "main_uncertainty":""
    }}
  ],
  "validation_tasks": [
    {{"validation_id":"V1","linked_mechanism_ids":[],"task":"","required_data":[]}}
  ],
  "abstentions": [
    {{"scope":"exact_root_cause|implementation_defect|other","reason":""}}
  ],
}}
""".strip()


DIMENSION_ANCHOR_PROMPT = """
Identify observed risk dimensions only from pre-event review evidence.
Return one valid JSON object and no prose outside JSON.

Use only CASE_META, OBSERVED_DIMENSION_CATALOG, and DETAILED_OBSERVED_EVIDENCE.
Do not use dynamic forecasts, risk chains, API knowledge, ecosystem relations,
external events, or post-event information.

Rules:
1. Select only dimensions present in OBSERVED_DIMENSION_CATALOG.
2. Never output General.
3. This stage identifies WHAT users are experiencing, not root causes.
4. Use the full compact catalog to avoid top-K censoring; use detailed cards only
   to strengthen interpretation of dimensions for which details are supplied.
5. Prefer coherent evidence with broader independent-review support, stronger
   negative sentiment, recency, and consistent failure/scenario descriptions.
6. Return at most 5 dimensions. 
7. Every selected dimension must preserve at least one observed evidence_id that
   belongs to that same dimension.
8. confidence must be between 0 and 1.

Required JSON shape:
{
  "case_id":"",
  "product_category":"",
  "risk_dimensions":[],
  "dimension_support":[
    {"dimension":"","source_refs":[],"confidence":0.0}
  ]
}
""".strip()


HYBRID_DIAGNOSIS_PROMPT = f"""
You perform prospective smart-product risk diagnosis using a FIXED, pre-routed dimension anchor. Return one valid JSON object and no prose outside JSON.

DIMENSION_ANCHOR has already been constructed before this diagnosis stage:

1. core_risk_dimensions were selected from pre-event observed review evidence;
2. promoted_risk_dimensions, if any, were selected deterministically from dimensions that were already present in the observed dimension catalog but were not selected into the core anchor;
3. dynamic information was allowed only to promote such already-observed dimensions under the predefined routing rule.

You must NOT perform dimension selection or promotion yourself.
Use CASE_PACKAGE to reason about plausible mechanisms, dependency pathways, and validation needs for the fixed final dimensions.

Rules:
1. Copy DIMENSION_ANCHOR.risk_dimensions exactly. Do not add, delete, replace, or re-rank dimensions.
2. Treat core_risk_dimensions and promoted_risk_dimensions as different provenance classes:
   - core dimensions were selected by observed-evidence salience;
   - promoted dimensions were already observed but were elevated because of prospective dynamic evidence.
   Neither class constitutes causal proof.
3. Dynamic facts may calibrate target-month risk for the final dimensions. Do not infer any dimension outside DIMENSION_ANCHOR.risk_dimensions.
4. Risk chains may explain possible propagation only when present.
5. Ecosystem relations and risk chains are predictive/associational, not causal proof. API knowledge is candidate mechanism knowledge, not implementation proof.
6. Every mechanism that you output must be supported, across its own source_refs and its supporting claims, by:(a) at least one observed-evidence reference; (b) at least one genuinely relevant contextual reference from MF, REL, CHN, or KB.
7. If no mechanism satisfies Rule 6, return mechanisms=[].
   Do not invent a mechanism merely to avoid an empty mechanism list.
   You may use abstentions and validation_tasks to explain what additional evidence would be needed.
   Link a validation task to a retained mechanism only when it actually validates that mechanism; a task investigating a separate unresolved issue may remain unlinked.
8. Because risk_dimensions are fixed and non-empty, return at least one evidence-bearing claim grounded in OBS evidence.
   
9. Return at most 4 claims, 3 mechanisms, and 3 validation tasks.
10. mechanism_family must be one of: {MECHANISM_FAMILIES}
11. symptom_family must be one of: {SYMPTOM_FAMILIES}
12. confidence must be between 0 and 1.

Required JSON shape:
{{
  "case_id": "",
  "product_category": "",
  "risk_dimensions": [],
  "symptom_families": [],
  "claims": [
    {{"claim_id":"CL1","claim_type":"observed|model_pattern|knowledge_context","statement":"","source_refs":[]}}
  ],
  "mechanisms": [
    {{
      "mechanism_id":"M1",
      "mechanism_family":"",
      "mechanism_scope":"local|interface|cross_product|ecosystem",
      "statement":"",
      "supporting_claim_ids":[],
      "source_refs":[],
      "causal_status":"unverified_hypothesis",
      "confidence":0.0,
      "main_uncertainty":""
    }}
  ],
  "validation_tasks": [
    {{"validation_id":"V1","linked_mechanism_ids":[],"task":"","required_data":[]}}
  ],
  "abstentions": [
    {{"scope":"exact_root_cause|implementation_defect|other","reason":""}}
  ]
}}
""".strip()


ECOSYSTEM_PROMPT = """
Synthesize product-level reviewed diagnoses and actions into an ecosystem-level diagnosis.
Use only the supplied evidence. Return one JSON object and no prose outside JSON.
Do not invent new mechanisms or evidence references.
Prioritize patterns that recur across products, connect products through shared relations/risk chains, or imply ecosystem governance.
Return at most 4 ecosystem patterns and 3 governance priorities.

Required JSON shape:
{
  "ecosystem_patterns":[
    {
      "pattern_id":"P1",
      "products":[],
      "risk_dimensions":[],
      "mechanism_families":[],
      "source_refs":[],
      "interpretation":""
    }
  ],
  "action_portfolio":{
    "product_action_ids":[],
    "interface_action_ids":[],
    "cross_product_action_ids":[],
    "ecosystem_action_ids":[]
  },
  "governance_priorities":[
    {"priority_id":"G1","priority":"","supported_action_ids":[],"source_refs":[]}
  ],
  "ecosystem_summary":""
}
""".strip()


def _normalize_str_list(value: Any) -> List[str]:
    """Normalize scalar/list string fields without splitting strings into characters."""
    if value is None:
        return []

    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []

    if not isinstance(value, list):
        return []

    out: List[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            out.append(text)

    return list(dict.fromkeys(out))


def _normalize_refs(refs: Any) -> List[str]:
    return _normalize_str_list(refs)


def _observed_dimension_ref_map(
    case_package: Dict[str, Any],
) -> Dict[str, set[str]]:
    out: Dict[str, set[str]] = {}
    for section in ("observed_dimension_catalog", "observed_evidence"):
        for row in case_package.get(section, []) or []:
            if not isinstance(row, dict):
                continue
            dim = str(row.get("dimension") or "").strip()
            ref = str(row.get("evidence_id") or "").strip()
            if dim and ref:
                out.setdefault(dim, set()).add(ref)
    return out


def _observed_ref_set(case_package: Dict[str, Any]) -> set[str]:
    return set().union(*_observed_dimension_ref_map(case_package).values()) if _observed_dimension_ref_map(case_package) else set()


def normalize_dimension_anchor(
    obj: Dict[str, Any],
    case_package: Dict[str, Any],
) -> Dict[str, Any]:
    meta = case_package.get("case_meta") if isinstance(case_package.get("case_meta"), dict) else {}
    out = dict(obj or {})
    out["case_id"] = str(meta.get("case_id") or "")
    out["product_category"] = str(meta.get("product_category") or "")
    # Preserve the model's dimension labels so hard dimension-contract checks
    # remain observable. Do not silently drop General here.
    dims = _normalize_str_list(out.get("risk_dimensions"))[:5]
    out["risk_dimensions"] = dims

    raw_support = out.get("dimension_support") or []
    support: List[Dict[str, Any]] = []
    for raw in raw_support:
        if not isinstance(raw, dict):
            continue
        dim = str(raw.get("dimension") or "").strip()
        if not dim or dim.lower() == "general":
            continue
        try:
            conf = min(1.0, max(0.0, float(raw.get("confidence", 0.5))))
        except Exception:
            conf = 0.5
        support.append({
            "dimension": dim,
            "source_refs": _normalize_refs(raw.get("source_refs")),
            "confidence": round(conf, 4),
        })
    # one support row per dimension; keep first occurrence
    by_dim: Dict[str, Dict[str, Any]] = {}
    for row in support:
        by_dim.setdefault(row["dimension"], row)
    out["dimension_support"] = [by_dim[d] for d in dims if d in by_dim]
    return out


def validate_dimension_anchor_output(
    anchor: Dict[str, Any],
    case_package: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate only experiment-defining H1 anchor contracts.

    Dimension-support citation quality is deliberately *not* a hard-failure
    condition. H1 can still route on a valid observed dimension set when the
    model omits/misstates a support row; that quality can be audited from the
    preserved raw/parsed response instead of deleting the H run.
    """
    if not isinstance(anchor, dict):
        raise TypeError("dimension anchor must be a dict")

    meta = (
        case_package.get("case_meta")
        if isinstance(case_package.get("case_meta"), dict)
        else {}
    )
    expected_case = str(meta.get("case_id") or "")
    expected_cat = str(meta.get("product_category") or "")

    if str(anchor.get("case_id") or "") != expected_case:
        raise ValueError("dimension_anchor.case_id does not match CASE_META")
    if str(anchor.get("product_category") or "") != expected_cat:
        raise ValueError("dimension_anchor.product_category does not match CASE_META")

    dims = _normalize_str_list(anchor.get("risk_dimensions"))
    if not dims:
        raise ValueError(
            "dimension_anchor.risk_dimensions must contain at least one observed dimension"
        )
    if len(dims) > 5:
        raise ValueError("dimension_anchor.risk_dimensions has more than 5 items")
    if any(d.lower() == "general" for d in dims):
        raise ValueError("dimension_anchor.risk_dimensions must not contain General")

    allowed_dims = set(_observed_dimension_ref_map(case_package))
    invalid_dims = [d for d in dims if d not in allowed_dims]
    if invalid_dims:
        raise ValueError(
            "dimension_anchor contains dimensions absent from observed catalog: "
            f"{invalid_dims}"
        )

    # Soft/audit-only here:
    # - missing dimension_support rows
    # - support refs missing or not belonging to the selected dimension
    # - support confidence formatting/range
    return anchor

def validate_final_hybrid_anchor(
    anchor: Dict[str, Any],
    case_package: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Validate the final routed Hybrid dimension anchor.

    Unlike the A-stage dimension anchor, this object may contain
    dimensions deterministically promoted by B. Promotion does not
    invent dimensions: every final dimension must already exist in
    the observed dimension catalog.
    """

    if not isinstance(anchor, dict):
        raise TypeError(
            "final hybrid anchor must be a dict"
        )

    meta = (
        case_package.get("case_meta") if isinstance(case_package.get("case_meta"),dict,) else {}
    )

    expected_case = str(
        meta.get("case_id") or ""
    )

    expected_cat = str(
        meta.get("product_category") or ""
    )

    if (
        str(anchor.get("case_id") or "")
        != expected_case
    ):
        raise ValueError(
            "final_dimension_anchor.case_id "
            "does not match CASE_META"
        )

    if (
        str(
            anchor.get(
                "product_category"
            )
            or ""
        )
        != expected_cat
    ):
        raise ValueError(
            "final_dimension_anchor.product_category "
            "does not match CASE_META"
        )

    final_dims = _normalize_str_list(
        anchor.get("risk_dimensions")
    )

    core_dims = _normalize_str_list(
        anchor.get(
            "core_risk_dimensions"
        )
    )

    promoted_dims = _normalize_str_list(
        anchor.get(
            "promoted_risk_dimensions"
        )
    )

    if not final_dims:
        raise ValueError(
            "final hybrid anchor must contain at least one risk dimension"
        )

    if any(
        d.lower() == "general"
        for d in final_dims
    ):
        raise ValueError(
            "final hybrid anchor must not "
            "contain General"
        )

    # Current design:
    # A anchor <= 5 and at most one promotion,
    # therefore final H may legitimately contain 6.
    if len(final_dims) > 6:
        raise ValueError(
            "final hybrid anchor contains more than 6 dimensions"
        )

    ref_map = _observed_dimension_ref_map(
        case_package
    )

    observed_dims = set(ref_map)

    invalid = [
        d
        for d in final_dims
        if d not in observed_dims
    ]

    if invalid:
        raise ValueError(
            "final hybrid anchor contains "
            "dimensions absent from observed "
            f"catalog: {invalid}"
        )

    # ----------------------------------------------------
    # Routing-contract checks
    # ----------------------------------------------------

    invalid_core = [
        d
        for d in core_dims
        if d not in final_dims
    ]

    if invalid_core:
        raise ValueError(
            "A core dimensions disappeared from "
            f"final anchor: {invalid_core}"
        )

    invalid_promoted = [
        d
        for d in promoted_dims
        if d not in final_dims
    ]

    if invalid_promoted:
        raise ValueError(
            "promoted dimensions missing from "
            f"final anchor: {invalid_promoted}"
        )

    overlap = (
        set(core_dims)
        & set(promoted_dims)
    )

    if overlap:
        raise ValueError(
            "core and promoted dimensions "
            f"overlap: {sorted(overlap)}"
        )

    expected_final = list(
        dict.fromkeys(
            core_dims
            + promoted_dims
        )
    )

    if final_dims != expected_final:
        raise ValueError(
            "final hybrid anchor must equal "
            "core dimensions followed by "
            "promoted dimensions; "
            f"got={final_dims}, "
            f"expected={expected_final}"
        )

    return anchor


def normalize_diagnosis(obj: Dict[str, Any], case_package: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(obj or {})
    # case_id/product_category are metadata, not prediction targets.
    # Make them authoritative from CASE_PACKAGE to avoid wasting a run on copy errors.
    out['case_id'] = str(case_package['case_meta']['case_id'])
    out['product_category'] = str(case_package['case_meta']['product_category'])
    out['risk_dimensions'] = _normalize_str_list(out.get('risk_dimensions'))[:5]
    # Preserve explicit model labels so the validator can detect taxonomy
    # violations instead of silently deleting them.
    out['symptom_families'] = _normalize_str_list(out.get('symptom_families'))[:5]

    claims = []
    for i, c in enumerate(out.get('claims') or [], start=1):
        if not isinstance(c, dict):
            continue
        claims.append({
            'claim_id': str(c.get('claim_id') or f'CL{i}'),
            'claim_type': str(c.get('claim_type') or 'observed'),
            'statement': str(c.get('statement') or ''),
            'source_refs': _normalize_refs(c.get('source_refs')),
        })
    out['claims'] = claims[:4]

    mechs = []
    for i, m in enumerate(out.get('mechanisms') or [], start=1):
        if not isinstance(m, dict):
            continue
        # Missing family defaults to the valid catch-all 'unclear', but an
        # explicitly invalid family is preserved so validation can catch it.
        fam = str(m.get('mechanism_family') or 'unclear').strip()
        try:
            conf = min(1.0, max(0.0, float(m.get('confidence', 0.5))))
        except Exception:
            conf = 0.5
        mechs.append({
            'mechanism_id': str(m.get('mechanism_id') or f'M{i}'),
            'mechanism_family': fam,
            'mechanism_scope': str(m.get('mechanism_scope') or 'local'),
            'statement': str(m.get('statement') or ''),
            'supporting_claim_ids': _normalize_refs(m.get('supporting_claim_ids')),
            'source_refs': _normalize_refs(m.get('source_refs')),
            # Preserve an explicit model status so overclaiming_rate remains
            # measurable; missing values still use the conservative default.
            'causal_status': str(m.get('causal_status') or 'unverified_hypothesis').strip(),
            'confidence': round(conf, 4),
            'main_uncertainty': str(m.get('main_uncertainty') or ''),
        })
    out['mechanisms'] = mechs[:3]

    tasks = []
    for i, v in enumerate(out.get('validation_tasks') or [], start=1):
        if not isinstance(v, dict):
            continue
        tasks.append({
            'validation_id': str(v.get('validation_id') or f'V{i}'),
            'linked_mechanism_ids': _normalize_refs(v.get('linked_mechanism_ids')),
            'task': str(v.get('task') or ''),
            'required_data': _normalize_str_list(v.get('required_data'))[:6],
        })
    out['validation_tasks'] = tasks[:3]
    out['abstentions'] = [x for x in (out.get('abstentions') or []) if isinstance(x, dict)][:4]
    return out


def validate_diagnosis_output(
    diagnosis: Dict[str, Any],
    case_package: Dict[str, Any],
    required_risk_dimensions: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Validate only experiment-defining hard contracts.

    IMPORTANT
    ---------
    This validator intentionally does *not* hard-fail on quality dimensions
    that are measured downstream, including citation validity, taxonomy
    compliance, mechanism traceability, evidence grounding, or validation-task
    linkage. Those belong in evaluation metrics; filtering them here would
    create survivorship/selection bias across A/B/C/H.
    """
    if not isinstance(diagnosis, dict):
        raise TypeError("diagnosis must be a dict")

    meta = (
        case_package.get("case_meta")
        if isinstance(case_package.get("case_meta"), dict)
        else {}
    )
    expected_case = str(meta.get("case_id") or "")
    expected_cat = str(meta.get("product_category") or "")

    # These are normally enforced by normalize_diagnosis(), but keep the
    # assertions as protection against programmatic misuse.
    if str(diagnosis.get("case_id") or "") != expected_case:
        raise ValueError("diagnosis.case_id does not match CASE_PACKAGE")
    if str(diagnosis.get("product_category") or "") != expected_cat:
        raise ValueError("diagnosis.product_category does not match CASE_PACKAGE")

    # --------------------------------------------------------
    # Hard dimension contract
    # --------------------------------------------------------
    ref_map = _observed_dimension_ref_map(case_package)
    allowed_dimensions = set(ref_map)
    pred_dimensions = _normalize_str_list(diagnosis.get("risk_dimensions"))

    if not pred_dimensions:
        raise ValueError(
            "diagnosis.risk_dimensions must contain at least one observed risk dimension"
        )
    if any(x.lower() == "general" for x in pred_dimensions):
        raise ValueError("diagnosis.risk_dimensions must not contain General")

    invalid_dims = [x for x in pred_dimensions if x not in allowed_dimensions]
    if invalid_dims:
        raise ValueError(
            "diagnosis.risk_dimensions contains dimensions not supported by "
            f"observed evidence: {invalid_dims}"
        )

    if required_risk_dimensions is not None:
        required = [
            str(x).strip()
            for x in required_risk_dimensions
            if str(x).strip() and str(x).strip().lower() != "general"
        ]
        required = list(dict.fromkeys(required))
        if pred_dimensions != required:
            raise ValueError(
                "Hybrid diagnosis must preserve DIMENSION_ANCHOR.risk_dimensions exactly; "
                f"got={pred_dimensions}, required={required}"
            )

    # --------------------------------------------------------
    # Hard structural contract
    # --------------------------------------------------------
    claims = diagnosis.get("claims")
    mechanisms = diagnosis.get("mechanisms")
    tasks = diagnosis.get("validation_tasks")
    abstentions = diagnosis.get("abstentions")

    if not isinstance(claims, list):
        raise ValueError("diagnosis.claims must be a list")
    if not isinstance(mechanisms, list):
        raise ValueError("diagnosis.mechanisms must be a list")
    if not isinstance(tasks, list):
        raise ValueError("diagnosis.validation_tasks must be a list")
    if not isinstance(abstentions, list):
        raise ValueError("diagnosis.abstentions must be a list")

    if not claims:
        raise ValueError(
            "diagnosis.claims must contain at least one evidence-bearing claim"
        )

    claim_ids = [
        str(c.get("claim_id") or "").strip()
        for c in claims
        if isinstance(c, dict)
    ]
    mechanism_ids = [
        str(m.get("mechanism_id") or "").strip()
        for m in mechanisms
        if isinstance(m, dict)
    ]

    if len(claim_ids) != len(claims) or any(not x for x in claim_ids):
        raise ValueError("diagnosis contains a missing claim_id")
    if len(set(claim_ids)) != len(claim_ids):
        raise ValueError("diagnosis contains duplicate claim_id values")

    if len(mechanism_ids) != len(mechanisms) or any(not x for x in mechanism_ids):
        raise ValueError("diagnosis contains a missing mechanism_id")
    if len(set(mechanism_ids)) != len(mechanism_ids):
        raise ValueError("diagnosis contains duplicate mechanism_id values")

    # Deliberately soft (evaluated downstream, never hard-failed here):
    # - symptom/mechanism taxonomy and mechanism scope
    # - claim_type validity
    # - source-ref validity / observed-claim grounding
    # - supporting_claim_ids validity / mechanism traceability
    # - OBS/context/mixed mechanism support
    # - validation-task linkage
    # - causal-status overclaiming
    return diagnosis

def _usage_dict(result: Dict[str, Any]) -> Dict[str, Any]:
    usage = result.get("usage") if isinstance(result, dict) else None
    return dict(usage) if isinstance(usage, dict) else {}


class GenerationTraceError(ValueError):
    """Validation/generation error that preserves all completed LLM attempts."""

    def __init__(
        self,
        message: str,
        *,
        llm_trace: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.llm_trace = llm_trace or {"attempts": []}


def _attempt_record(
    *,
    attempt: int,
    result: Dict[str, Any],
    parsed: Optional[Dict[str, Any]],
    validation_error: Optional[str],
) -> Dict[str, Any]:
    return {
        "attempt": int(attempt),
        "raw_output": result.get("raw_output", "") if isinstance(result, dict) else "",
        "parsed_output": parsed,
        "parse_error": result.get("parse_error") if isinstance(result, dict) else None,
        "validation_error": validation_error,
        "usage": _usage_dict(result),
        "request_meta": (
            dict(result.get("request_meta"))
            if isinstance(result, dict) and isinstance(result.get("request_meta"), dict)
            else {}
        ),
    }


def _generation_audit(
    *,
    usage_calls: List[Dict[str, Any]],
    n_attempts: int,
    validation_errors: List[str],
) -> Dict[str, Any]:
    def _sum(key: str):
        values = [
            x.get(key)
            for x in usage_calls
            if isinstance(x, dict) and x.get(key) is not None
        ]
        if not values:
            return None
        return int(sum(int(v) for v in values))

    return {
        "n_attempts": int(n_attempts),
        "repair_triggered": bool(n_attempts > 1),
        "validation_errors": list(validation_errors),
        "usage_calls": usage_calls,
        "prompt_tokens": _sum("prompt_tokens"),
        "cached_prompt_tokens": _sum("cached_prompt_tokens"),
        "completion_tokens": _sum("completion_tokens"),
        "total_tokens": _sum("total_tokens"),
    }

def generate_dimension_anchor(
    case_package: Dict[str, Any],
    model_name: Optional[str] = None,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """Stage H1: one-shot observed-evidence dimension anchoring.

    No LLM repair/retry is performed here. A parse/hard-contract failure is
    surfaced immediately with the completed raw response preserved in
    GenerationTraceError.llm_trace.
    """
    view = build_llm_case_view(case_package, max_statements=2)
    payload: Dict[str, Any] = {
        "CASE_META": view.get("case_meta", {}),
        "OBSERVED_DIMENSION_CATALOG": view.get("observed_dimension_catalog", []),
        "DETAILED_OBSERVED_EVIDENCE": view.get("observed_evidence", []),
    }

    try:
        result = call_json(
            DIMENSION_ANCHOR_PROMPT,
            payload,
            model_name=model_name,
            temperature=temperature,
            use_json_mode=None,
        )
    except Exception as exc:
        raise GenerationTraceError(
            f"dimension anchor API call failed: {exc!r}",
            llm_trace={"attempts": []},
        ) from exc

    usage_calls = [{"attempt": 1, **_usage_dict(result)}]
    parsed_obj = result.get("parsed") if isinstance(result, dict) else None
    parsed = normalize_dimension_anchor(
        parsed_obj if isinstance(parsed_obj, dict) else {},
        case_package,
    )

    parse_error = result.get("parse_error") if isinstance(result, dict) else None
    if parse_error:
        error = f"dimension anchor JSON parse failed: {parse_error}"
        attempt = _attempt_record(
            attempt=1,
            result=result,
            parsed=parsed,
            validation_error=error,
        )
        raise GenerationTraceError(
            error,
            llm_trace={"attempts": [attempt]},
        )

    try:
        validate_dimension_anchor_output(parsed, case_package)
    except (TypeError, ValueError) as exc:
        attempt = _attempt_record(
            attempt=1,
            result=result,
            parsed=parsed,
            validation_error=str(exc),
        )
        raise GenerationTraceError(
            str(exc),
            llm_trace={"attempts": [attempt]},
        ) from exc

    attempt = _attempt_record(
        attempt=1,
        result=result,
        parsed=parsed,
        validation_error=None,
    )
    return {
        "raw_output": result.get("raw_output", ""),
        "parsed": parsed,
        "usage": _usage_dict(result),
        "generation_audit": _generation_audit(
            usage_calls=usage_calls,
            n_attempts=1,
            validation_errors=[],
        ),
        "llm_trace": {"attempts": [attempt]},
    }

def generate_hybrid_diagnosis(
    *,
    case_package: Dict[str, Any],
    dimension_anchor: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """One-shot Hybrid diagnosis after deterministic routing."""
    if dimension_anchor is None:
        dimension_anchor = case_package.get("dimension_anchor")
    if not isinstance(dimension_anchor, dict):
        raise ValueError(
            "Hybrid case package does not contain a valid final dimension anchor"
        )

    validate_final_hybrid_anchor(dimension_anchor, case_package)
    required_dims = _normalize_str_list(dimension_anchor.get("risk_dimensions"))
    llm_case_package = _build_hybrid_llm_case_view(case_package, required_dims)
    llm_anchor = _compact_hybrid_anchor_for_llm(dimension_anchor)
    payload: Dict[str, Any] = {
        "DIMENSION_ANCHOR": llm_anchor,
        "CASE_PACKAGE": llm_case_package,
    }

    try:
        result = call_json(
            HYBRID_DIAGNOSIS_PROMPT,
            payload,
            model_name=model_name,
            temperature=temperature,
            use_json_mode=None,
        )
    except Exception as exc:
        raise GenerationTraceError(
            f"hybrid diagnosis API call failed: {exc!r}",
            llm_trace={"attempts": []},
        ) from exc

    usage_calls = [{"attempt": 1, **_usage_dict(result)}]
    parsed_obj = result.get("parsed") if isinstance(result, dict) else None
    parsed = normalize_diagnosis(
        parsed_obj if isinstance(parsed_obj, dict) else {},
        case_package,
    )

    # H dimensions are deterministic routing output, not an LLM prediction.
    parsed["risk_dimensions"] = list(required_dims)

    parse_error = result.get("parse_error") if isinstance(result, dict) else None
    if parse_error:
        error = f"hybrid diagnosis JSON parse failed: {parse_error}"
        attempt = _attempt_record(
            attempt=1,
            result=result,
            parsed=parsed,
            validation_error=error,
        )
        raise GenerationTraceError(
            error,
            llm_trace={"attempts": [attempt]},
        )

    try:
        validate_diagnosis_output(
            parsed,
            case_package,
            required_risk_dimensions=required_dims,
        )
    except (TypeError, ValueError) as exc:
        attempt = _attempt_record(
            attempt=1,
            result=result,
            parsed=parsed,
            validation_error=str(exc),
        )
        raise GenerationTraceError(
            str(exc),
            llm_trace={"attempts": [attempt]},
        ) from exc

    attempt = _attempt_record(
        attempt=1,
        result=result,
        parsed=parsed,
        validation_error=None,
    )
    return {
        "raw_output": result.get("raw_output", ""),
        "parsed": parsed,
        "usage": _usage_dict(result),
        "generation_audit": _generation_audit(
            usage_calls=usage_calls,
            n_attempts=1,
            validation_errors=[],
        ),
        "llm_trace": {"attempts": [attempt]},
    }

def _build_hybrid_llm_case_view(
    case_package: Dict[str, Any],
    required_dimensions: List[str],
) -> Dict[str, Any]:
    """Compact H evidence using the same field-level representation as A/B/C."""
    return build_llm_case_view(
        case_package,
        required_dimensions=required_dimensions,
        max_statements=2,
    )

def _compact_hybrid_anchor_for_llm(
    dimension_anchor: Dict[str, Any],
) -> Dict[str, Any]:

    return {
        "core_risk_dimensions":
            list(
                dimension_anchor.get(
                    "core_risk_dimensions"
                )
                or []
            ),

        "promoted_risk_dimensions":
            list(
                dimension_anchor.get(
                    "promoted_risk_dimensions"
                )
                or []
            ),

        "risk_dimensions":
            list(
                dimension_anchor.get(
                    "risk_dimensions"
                )
                or []
            ),
    }
        
        
def generate_diagnosis(
    case_package: Dict[str, Any],
    model_name: Optional[str] = None,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """One-shot A/B/C diagnosis with complete raw-response audit trace."""
    llm_case_package = build_llm_case_view(case_package, max_statements=2)
    payload: Dict[str, Any] = {"CASE_PACKAGE": llm_case_package}

    try:
        result = call_json(
            DIAGNOSIS_PROMPT,
            payload,
            model_name=model_name,
            temperature=temperature,
            use_json_mode=None,
        )
    except Exception as exc:
        raise GenerationTraceError(
            f"diagnosis API call failed: {exc!r}",
            llm_trace={"attempts": []},
        ) from exc

    usage_calls = [{"attempt": 1, **_usage_dict(result)}]
    parsed_obj = result.get("parsed") if isinstance(result, dict) else None
    parsed = normalize_diagnosis(
        parsed_obj if isinstance(parsed_obj, dict) else {},
        case_package,
    )

    parse_error = result.get("parse_error") if isinstance(result, dict) else None
    if parse_error:
        error = f"diagnosis JSON parse failed: {parse_error}"
        attempt = _attempt_record(
            attempt=1,
            result=result,
            parsed=parsed,
            validation_error=error,
        )
        raise GenerationTraceError(
            error,
            llm_trace={"attempts": [attempt]},
        )

    try:
        validate_diagnosis_output(parsed, case_package)
    except (TypeError, ValueError) as exc:
        attempt = _attempt_record(
            attempt=1,
            result=result,
            parsed=parsed,
            validation_error=str(exc),
        )
        raise GenerationTraceError(
            str(exc),
            llm_trace={"attempts": [attempt]},
        ) from exc

    attempt = _attempt_record(
        attempt=1,
        result=result,
        parsed=parsed,
        validation_error=None,
    )
    return {
        "raw_output": result.get("raw_output", ""),
        "parsed": parsed,
        "usage": _usage_dict(result),
        "generation_audit": _generation_audit(
            usage_calls=usage_calls,
            n_attempts=1,
            validation_errors=[],
        ),
        "llm_trace": {"attempts": [attempt]},
    }

def apply_mechanism_feedback(diagnosis: Dict[str, Any], feedback: Dict[str, Any]) -> Dict[str, Any]:
    by_id = {str(x.get('mechanism_id')): x for x in feedback.get('mechanism_feedback', []) if isinstance(x, dict)}
    reviewed = json.loads(json.dumps(diagnosis, ensure_ascii=False))
    for m in reviewed.get('mechanisms', []):
        fb = by_id.get(str(m.get('mechanism_id')), {})
        m['human_assessment'] = fb.get('assessment_status', 'not_assessed')
        m['human_revision'] = fb.get('revision', '')
        m['human_validation_note'] = fb.get('validation_note', '')
    reviewed['respondent_profile'] = feedback.get('respondent_profile', {})
    return reviewed


def generate_ecosystem_synthesis(
    reviewed_diagnoses: List[Dict[str, Any]],
    action_outputs: List[Dict[str, Any]],
    case_packages: Optional[List[Dict[str, Any]]] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    ecosystem_context = []
    for case in case_packages or []:
        ecosystem_context.append({
            'product_category': case.get('case_meta', {}).get('product_category'),
            'ecosystem_evidence_mode': case.get('case_meta', {}).get('ecosystem_evidence_mode'),
            'ecosystem_relations': case.get('ecosystem_relations', []),
            'risk_chains': case.get('risk_chains', []),
            'network_model_facts': [
                x for x in case.get('model_facts', [])
                if x.get('fact_type') in {'driver_pattern', 'ecosystem_position', 'risk_state'}
            ],
        })
    payload = {
        'ECOSYSTEM_CONTEXT': ecosystem_context,
        'REVIEWED_PRODUCT_DIAGNOSES': reviewed_diagnoses,
        'PRODUCT_ACTIONS': action_outputs,
    }
    result = call_json(ECOSYSTEM_PROMPT, payload, model_name=model_name, temperature=temperature, use_json_mode= None,)
    parsed = result.get('parsed') if isinstance(result, dict) else None
    if not isinstance(parsed, dict):
        error = (
            f"ecosystem synthesis JSON parse failed: {result.get('parse_error')!r}"
            if isinstance(result, dict)
            else "ecosystem synthesis returned a non-dict result"
        )
        raise GenerationTraceError(
            error,
            llm_trace={
                'attempts': [_attempt_record(
                    attempt=1,
                    result=result if isinstance(result, dict) else {},
                    parsed=None,
                    validation_error=error,
                )]
            },
        )
    return {
        'raw_output': result.get('raw_output', '') if isinstance(result, dict) else '',
        'parsed': parsed,
        'llm_trace': {
            'attempts': [_attempt_record(
                attempt=1,
                result=result if isinstance(result, dict) else {},
                parsed=parsed if isinstance(parsed, dict) else None,
                validation_error=(
                    f"JSON parse failed: {result.get('parse_error')}"
                    if isinstance(result, dict) and result.get('parse_error')
                    else None
                ),
            )]
        },
    }


__all__ = [
    "MECHANISM_FAMILIES", "SYMPTOM_FAMILIES", "DIAGNOSIS_PROMPT", 
    "DIMENSION_ANCHOR_PROMPT", "HYBRID_DIAGNOSIS_PROMPT",
    "normalize_diagnosis", "validate_diagnosis_output",
    "normalize_dimension_anchor", "validate_dimension_anchor_output",
    "generate_diagnosis", "generate_dimension_anchor", "generate_hybrid_diagnosis",
    "apply_mechanism_feedback", "generate_ecosystem_synthesis",
    "validate_final_hybrid_anchor", "GenerationTraceError"
]
