# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Optional, Set, Tuple

from llm_service import call_json


class ActionGenerationTraceError(ValueError):
    """Action-generation error carrying the completed raw LLM attempt."""

    def __init__(self, message: str, *, llm_trace: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.llm_trace = llm_trace or {"attempts": []}


def _action_attempt_trace(
    *,
    result: Dict[str, Any],
    parsed: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "attempts": [{
            "attempt": 1,
            "raw_output": result.get("raw_output", "") if isinstance(result, dict) else "",
            "parsed_output": parsed,
            "parse_error": result.get("parse_error") if isinstance(result, dict) else None,
            "validation_error": error,
            "usage": dict(result.get("usage") or {}) if isinstance(result, dict) else {},
        }]
    }
from data_service import all_reference_ids, build_llm_case_view


# ============================================================
# 0. Frozen experimental action taxonomy
# ============================================================

ACTION_SUBFAMILIES = [
    "software_or_firmware_remediation",
    "hardware_design_or_repair",
    "feature_containment_or_service_recovery",
    "security_or_access_hardening",
    "interface_or_dependency_resilience",
    "user_guidance_or_notification",
    "customer_remedy_or_recovery",
    "cross_product_or_third_party_coordination",
    "observability_or_validation",
    "privacy_or_data_governance",
    "lifecycle_or_platform_governance",
    "compatibility_or_migration_management",
]

ACTION_LEVELS = {
    "product",
    "interface",
    "cross_product",
    "ecosystem",
}

ACTION_SUBFAMILY_JSON = json.dumps(
    ACTION_SUBFAMILIES,
    ensure_ascii=False,
)


# ============================================================
# 1. Experimental prompt
# ============================================================

ACTION_EXPERIMENT_PROMPT = f"""
You generate prospective product-ecosystem interventions from a pre-event risk diagnosis.

Use only DIAGNOSIS and CASE_PACKAGE.
OUTPUT_CONSTRAINTS is derived only from these inputs and prevents identifier/reference hallucination.

This is strictly ex-ante. Do not use later incidents, recalls, patches, outages,
company responses, or other post-event information.

Return exactly one valid JSON object and no prose outside JSON.

IMPORTANT TAXONOMY RULE:
DIAGNOSIS.mechanisms[*].mechanism_family is a diagnostic label describing the risk mechanism.
action_subfamily is a separate intervention label describing what the action directly changes.
Never copy, reuse, or paraphrase mechanism_family as action_subfamily.
action_subfamily is closed-set and must exactly match one value from:
{ACTION_SUBFAMILY_JSON}

Rules:

1. Generate actions only for mechanisms explicitly listed in DIAGNOSIS.
   Every linked_mechanism_id must appear in OUTPUT_CONSTRAINTS.eligible_mechanism_ids.
   Do not introduce or infer a new mechanism.

2. Every action must:
   - address at least one linked mechanism;
   - describe one concrete intervention;
   - use exactly one valid action_subfamily;
   - use source_refs only from OUTPUT_CONSTRAINTS.references_by_mechanism
     for its linked mechanisms;
   - preserve at least one valid reference when available;
   - otherwise return source_refs as [];
   - return target_products and target_layers as JSON arrays of strings;
   - return validation_metric as a string.

3. Classify action_subfamily by what the intervention directly changes:

   - software/firmware implementation -> software_or_firmware_remediation
   - physical hardware/design -> hardware_design_or_repair
   - containment, rollback, isolation, or service restoration -> feature_containment_or_service_recovery
   - authentication, authorization, permissions, or access control -> security_or_access_hardening
   - API/interface behavior, state consistency, retry/failover, graceful degradation, or dependency resilience -> interface_or_dependency_resilience
   - user warning, configuration instruction, or usage guidance -> user_guidance_or_notification
   - refund, replacement, account recovery, or formal customer remedy -> customer_remedy_or_recovery
   - coordination across products, teams, vendors, or platforms -> cross_product_or_third_party_coordination
   - monitoring, logging, diagnostics, testing, or validation -> observability_or_validation
   - data collection, retention, deletion, sharing, or privacy rules -> privacy_or_data_governance
   - platform support, maintenance, deprecation, or lifecycle governance -> lifecycle_or_platform_governance
   - protocol/API/version compatibility, migration, transition, or fallback -> compatibility_or_migration_management

4. A mechanism category does NOT determine an action category.
   For example, mechanism_family="setup_or_configuration" is not a valid action_subfamily. Classify the actual intervention instead:
   user instructions -> user_guidance_or_notification;
   software setup logic -> software_or_firmware_remediation;
   diagnostic checking -> observability_or_validation.

5. One action represents one atomic intervention.
   Split substantively different interventions, but do not create actions merely to cover more taxonomy labels.

6. If evidence does not support a direct remediation, prefer observability_or_validation over a speculative fix.

7. Return at most 5 distinct actions, ordered by relevance. Use action_id A1, A2, ... and priority_rank 1, 2, ... in the same order.

8. Copy case_id and product_category exactly from OUTPUT_CONSTRAINTS.

Required JSON:
{{
  "case_id": "",
  "product_category": "",
  "actions": [
    {{
      "action_id": "A1",
      "priority_rank": 1,
      "action_subfamily": "",
      "action_level": "product|interface|cross_product|ecosystem",
      "action": "",
      "linked_mechanism_ids": [],
      "source_refs": [],
      "target_products": [],
      "target_layers": [],
      "validation_metric": ""
    }}
  ]
}}
""".strip()




PROSPECTIVE_ACTION_LAYERS = [
    "app_or_configuration",
    "cloud_service",
    "firmware",
    "alexa_skill_or_api_implementation",
    "support_or_user_guidance",
    "hardware",
    "ecosystem_governance",
]


PROSPECTIVE_ACTION_PROMPT = f"""
You generate implementation-oriented prospective actions for a smart connected product after a human review of an ex-ante diagnosis.

Use only REVIEWED_DIAGNOSIS, ACTION_CASE_PACKAGE, DECISION_CONTEXT, and OUTPUT_CONSTRAINTS.
Do not use later incidents, patches, recalls, outages, company responses, or other post-event information.
Return exactly one valid JSON object and no prose outside JSON.

The human review is authoritative:
- supported: May support mechanism-specific corrective remediation, as well as robustness improvements, validation, and observability. 
- plausible: May support conservative, low-regret product improvements and validation.
  A product change is allowed when it remains justified under the unresolved mechanism uncertainty. Do not make a corrective change whose justification depends on the candidate mechanism or a specific implementation defect being true.
- uncertain: Must not independently justify mechanism-specific corrective remediation. It may support validation, instrumentation, diagnostic testing, or observability.
  In a multi-mechanism corrective action, an uncertain mechanism may be included only when the intervention is independently justified by supported or plausible mechanisms and the uncertain mechanism is only a secondary beneficiary.
- rejected or not_assessed: Must not be used.
- when human_revision is non-empty, use the revised statement instead of the original wording.

Decision constraints are authoritative:
- every target_layers value must be in OUTPUT_CONSTRAINTS.allowed_layers;
- if hardware is not allowed, do not propose hardware changes;
- if third-party change is not allowed, the action may adapt the focal product to an external dependency,
  but must not require a third party to modify its own product/platform as a prerequisite;
- pursue DECISION_CONTEXT.objective without inventing unsupported mechanisms.

Evidence and taxonomy rules:
1. Every linked_mechanism_id must appear in OUTPUT_CONSTRAINTS.eligible_mechanism_ids.
2. source_refs may contain only references allowed for the linked mechanisms.
3. action_subfamily must exactly match one value from: {ACTION_SUBFAMILY_JSON}
4. Each action is one concrete intervention, not a bundle of unrelated changes.
5. Return at most 5 actions using A1, A2, ... and matching priority_rank. Fewer well-grounded actions are preferred to redundant or weakly justified actions.
6. For a corrective action with mixed human assessments, its justification must be traceable to the supported or plausible mechanism(s) that independently justify the intervention. When valid source_refs are available for those mechanisms, include at least one of them. Evidence linked only to an uncertain mechanism must not be the sole evidence used to justify remediation.
7. For validation/observability actions, supported, plausible, and uncertain mechanisms may be linked together when the same validation task materially informs them.
8. improvement_direction states the desired product/experience change concisely.
9. intervention_logic explains how the action addresses the linked reviewed mechanism(s).
10. validation_metric is a concrete observable criterion for checking the intervention.

Required JSON:
{{
  "case_id": "",
  "product_category": "",
  "actions": [
    {{
      "action_id": "A1",
      "priority_rank": 1,
      "action_subfamily": "",
      "action_level": "product|interface|cross_product|ecosystem",
      "improvement_direction": "",
      "action": "",
      "intervention_logic": "",
      "linked_mechanism_ids": [],
      "source_refs": [],
      "target_products": [],
      "target_layers": [],
      "validation_metric": ""
    }}
  ]
}}
""".strip()


ACTION_SUBFAMILY_PREDICTION_PROMPT = f"""
You predict prospective intervention categories from a pre-event smart connected product risk diagnosis and its condition-specific evidence package.

Use only DIAGNOSIS, CASE_PACKAGE, and OUTPUT_CONSTRAINTS. This is strictly ex-ante.

Return exactly one valid JSON object and no prose outside JSON.

IMPORTANT:
mechanism_family describes the diagnosed risk mechanism.
action_subfamily describes what type of intervention would directly address the diagnosed risk.
They are different taxonomies.

Allowed action_subfamilies:
{ACTION_SUBFAMILY_JSON}

Rules:

1. Use the information available in DIAGNOSIS and CASE_PACKAGE to determine which intervention categories are most appropriate for this case.

2. Generate predictions only for mechanisms explicitly present in DIAGNOSIS.
   Every linked_mechanism_id must appear in OUTPUT_CONSTRAINTS.eligible_mechanism_ids.

3. Do not introduce a new mechanism.

4. Do not mechanically map mechanism_family to action_subfamily.
   Select the intervention category according to what would need to change to address the mechanism.
   
   - software/firmware implementation -> software_or_firmware_remediation
   - physical hardware/design -> hardware_design_or_repair
   - containment, rollback, isolation, or service restoration -> feature_containment_or_service_recovery
   - authentication, authorization, permissions, or access control -> security_or_access_hardening
   - API/interface behavior, state consistency, retry/failover, graceful degradation, or dependency resilience -> interface_or_dependency_resilience
   - user warning, configuration instruction, or usage guidance -> user_guidance_or_notification
   - refund, replacement, account recovery, or formal customer remedy -> customer_remedy_or_recovery
   - coordination across products, teams, vendors, or platforms -> cross_product_or_third_party_coordination
   - monitoring, logging, diagnostics, testing, or validation -> observability_or_validation
   - data collection, retention, deletion, sharing, or privacy rules -> privacy_or_data_governance
   - platform support, maintenance, deprecation, or lifecycle governance -> lifecycle_or_platform_governance
   - protocol/API/version compatibility, migration, transition, or fallback -> compatibility_or_migration_management

5. Dynamic facts, ecosystem relations, API knowledge, and risk chains may inform the intervention category only when they are present in CASE_PACKAGE.

6. If available evidence does not justify a direct remediation, observability_or_validation may be preferred.

7. Return at most 5 distinct action_subfamilies, ordered by relevance. Use priority_rank 1, 2, ... in the same order.

8. Copy case_id and product_category exactly from OUTPUT_CONSTRAINTS.

Required JSON:
{{
  "case_id": "",
  "product_category": "",
  "actions": [
    {{
      "priority_rank": 1,
      "action_subfamily": "",
      "linked_mechanism_ids": []
    }}
  ]
}}
""".strip()

# ============================================================
# 2. Small validation helpers
# ============================================================

def _clean_required_str(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{where} must be a string")
    value = value.strip()
    if not value:
        raise ValueError(f"{where} must be a non-empty string")
    return value


def _require_str(
    obj: Dict[str, Any],
    key: str,
    where: str,
) -> str:
    return _clean_required_str(
        obj.get(key),
        f"{where}.{key}",
    )


def _require_str_list(
    obj: Dict[str, Any],
    key: str,
    where: str,
    *,
    allow_empty: bool = True,
) -> List[str]:
    value = obj.get(key)

    if not isinstance(value, list):
        raise ValueError(
            f"{where}.{key} must be a list of strings"
        )

    out: List[str] = []
    for j, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                f"{where}.{key}[{j}] must be a non-empty string"
            )
        out.append(item.strip())

    if not allow_empty and not out:
        raise ValueError(
            f"{where}.{key} must not be empty"
        )

    if len(out) != len(set(out)):
        raise ValueError(
            f"{where}.{key} contains duplicate values: {out}"
        )

    return out


def _optional_nonempty_str(
    obj: Dict[str, Any],
    key: str,
    where: str,
) -> Optional[str]:
    if key not in obj:
        return None

    value = obj.get(key)
    if not isinstance(value, str):
        raise ValueError(
            f"{where}.{key} must be a string"
        )

    return value.strip()


# ============================================================
# 3. Diagnosis / reference contract extraction
# ============================================================

def _diagnosis_mechanisms(
    diagnosis: Dict[str, Any],
) -> List[Dict[str, Any]]:
    mechanisms = diagnosis.get("mechanisms")

    if mechanisms is None:
        mechanisms = []

    if not isinstance(mechanisms, list):
        raise ValueError(
            "DIAGNOSIS.mechanisms must be a list"
        )

    out: List[Dict[str, Any]] = []
    for i, mech in enumerate(mechanisms):
        if not isinstance(mech, dict):
            raise ValueError(
                f"DIAGNOSIS.mechanisms[{i}] must be an object"
            )

        mechanism_id = mech.get("mechanism_id")
        if not isinstance(mechanism_id, str) or not mechanism_id.strip():
            raise ValueError(
                f"DIAGNOSIS.mechanisms[{i}].mechanism_id "
                "must be a non-empty string"
            )

        out.append(mech)

    ids = [
        str(m["mechanism_id"]).strip()
        for m in out
    ]

    if len(ids) != len(set(ids)):
        raise ValueError(
            f"DIAGNOSIS contains duplicate mechanism_id values: {ids}"
        )

    return out


def _claim_index(
    diagnosis: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    claims = diagnosis.get("claims") or []

    if not isinstance(claims, list):
        raise ValueError(
            "DIAGNOSIS.claims must be a list"
        )

    out: Dict[str, Dict[str, Any]] = {}

    for i, claim in enumerate(claims):
        if not isinstance(claim, dict):
            raise ValueError(
                f"DIAGNOSIS.claims[{i}] must be an object"
            )

        claim_id = claim.get("claim_id")
        if claim_id is None:
            continue

        claim_id = str(claim_id).strip()
        if not claim_id:
            continue

        if claim_id in out:
            raise ValueError(
                f"DIAGNOSIS contains duplicate claim_id={claim_id!r}"
            )

        out[claim_id] = claim

    return out


def _valid_refs_for_mechanism(
    mechanism: Dict[str, Any],
    *,
    claims_by_id: Dict[str, Dict[str, Any]],
    valid_case_refs: Set[str],
) -> Set[str]:
    """
    References eligible for an action are inherited from the linked mechanism
    itself and from claims explicitly supporting that mechanism.

    They are intersected with case-package reference IDs so the action cannot
    cite a hallucinated or stale identifier.
    """
    refs: Set[str] = set()

    for ref in mechanism.get("source_refs") or []:
        if isinstance(ref, str) and ref.strip():
            refs.add(ref.strip())

    for claim_id in mechanism.get("supporting_claim_ids") or []:
        cid = str(claim_id).strip()
        claim = claims_by_id.get(cid)
        if not claim:
            continue

        for ref in claim.get("source_refs") or []:
            if isinstance(ref, str) and ref.strip():
                refs.add(ref.strip())

    return refs & valid_case_refs


def _build_action_constraints(
    *,
    diagnosis: Dict[str, Any],
    case_package: Dict[str, Any],
) -> Dict[str, Any]:
    mechanisms = _diagnosis_mechanisms(diagnosis)
    claims_by_id = _claim_index(diagnosis)

    valid_case_refs = {
        str(x).strip()
        for x in all_reference_ids(case_package)
        if str(x).strip()
    }

    references_by_mechanism: Dict[str, List[str]] = {}

    for mech in mechanisms:
        mid = str(mech["mechanism_id"]).strip()

        references_by_mechanism[mid] = sorted(
            _valid_refs_for_mechanism(
                mech,
                claims_by_id=claims_by_id,
                valid_case_refs=valid_case_refs,
            )
        )

    case_meta = (
        case_package.get("case_meta")
        if isinstance(case_package.get("case_meta"), dict)
        else {}
    )

    diagnosis_case_id = str(
        diagnosis.get("case_id") or ""
    ).strip()

    package_case_id = str(
        case_meta.get("case_id")
        or case_package.get("case_id")
        or ""
    ).strip()

    if (
        diagnosis_case_id
        and package_case_id
        and diagnosis_case_id != package_case_id
    ):
        raise ValueError(
            "DIAGNOSIS.case_id and CASE_PACKAGE.case_id disagree: "
            f"{diagnosis_case_id!r} != {package_case_id!r}"
        )

    case_id = diagnosis_case_id or package_case_id

    diagnosis_category = str(
        diagnosis.get("product_category") or ""
    ).strip()

    package_category = str(
        case_meta.get("product_category")
        or case_package.get("product_category")
        or case_package.get("category")
        or case_package.get("Category")
        or ""
    ).strip()

    if (
        diagnosis_category
        and package_category
        and diagnosis_category != package_category
    ):
        raise ValueError(
            "DIAGNOSIS.product_category and CASE_PACKAGE product category disagree: "
            f"{diagnosis_category!r} != {package_category!r}"
        )

    product_category = (
        diagnosis_category
        or package_category
    )

    if not product_category:
        raise ValueError(
            "Cannot determine product_category from DIAGNOSIS or CASE_PACKAGE"
        )

    return {
        "case_id": case_id,
        "product_category": product_category,
        "eligible_mechanism_ids": [
            str(m["mechanism_id"]).strip()
            for m in mechanisms
        ],
        "references_by_mechanism": references_by_mechanism,
    
        # NEW
        "allowed_action_subfamilies": list(ACTION_SUBFAMILIES),
    }


import re

def normalize_experimental_action_output(
    action_output: Dict[str, Any],
    *,
    constraints: Dict[str, Any],
) -> tuple[Dict[str, Any], list[str]]:
    """
    Normalize only structurally equivalent LLM output variants.

    This function does NOT:
    - invent mechanism IDs;
    - invent source refs;
    - remap invalid IDs;
    - infer new actions.

    All normalized values are still checked by the strict validator.
    """

    if not isinstance(action_output, dict):
        raise TypeError(
            "Experimental action output must be a dict"
        )

    out = dict(action_output)

    actions = out.get("actions")

    if not isinstance(actions, list):
        # Do not silently repair a missing/wrong actions object.
        raise ValueError(
            "root.actions must be a list"
        )

    eligible_mechanism_ids = set(
        constraints.get(
            "eligible_mechanism_ids"
        )
        or []
    )


    notes = []
    normalized_actions = []

    for i, action in enumerate(actions):

        where = f"root.actions[{i}]"

        if not isinstance(action, dict):
            raise TypeError(
                f"{where} must be an object"
            )

        action = dict(action)

        # ====================================================
        # 1. linked_mechanism_ids
        # ====================================================

        raw_linked = action.get(
            "linked_mechanism_ids"
        )

        normalized_linked = []

        # Case A:
        # "M1"
        if isinstance(raw_linked, str):

            value = raw_linked.strip()

            if not value:
                raise ValueError(
                    f"{where}.linked_mechanism_ids "
                    "contains an empty string"
                )

            normalized_linked = [value]

            notes.append(
                f"{where}.linked_mechanism_ids: "
                "scalar string -> list[str]"
            )

        # Case B:
        # {"mechanism_id": "M1"}
        elif isinstance(raw_linked, dict):

            value = raw_linked.get(
                "mechanism_id"
            )

            if (
                not isinstance(value, str)
                or not value.strip()
            ):
                raise ValueError(
                    f"{where}.linked_mechanism_ids "
                    "object must contain a non-empty "
                    "'mechanism_id'"
                )

            normalized_linked = [
                value.strip()
            ]

            notes.append(
                f"{where}.linked_mechanism_ids: "
                "mechanism object -> list[str]"
            )

        # Case C:
        # ["M1", "M2"]
        # or
        # [{"mechanism_id": "M1"}]
        elif isinstance(raw_linked, list):

            for j, item in enumerate(
                raw_linked
            ):

                if isinstance(item, str):

                    value = item.strip()

                    if not value:
                        raise ValueError(
                            f"{where}."
                            f"linked_mechanism_ids[{j}] "
                            "must not be empty"
                        )

                    normalized_linked.append(
                        value
                    )

                elif isinstance(item, dict):

                    value = item.get(
                        "mechanism_id"
                    )

                    if (
                        not isinstance(
                            value,
                            str,
                        )
                        or not value.strip()
                    ):
                        raise ValueError(
                            f"{where}."
                            f"linked_mechanism_ids[{j}] "
                            "object must contain "
                            "'mechanism_id'"
                        )

                    normalized_linked.append(
                        value.strip()
                    )

                    notes.append(
                        f"{where}."
                        f"linked_mechanism_ids[{j}]: "
                        "mechanism object -> string"
                    )

                else:

                    raise ValueError(
                        f"{where}."
                        f"linked_mechanism_ids[{j}] "
                        "must be a string or "
                        "mechanism object; "
                        f"got {type(item).__name__}"
                    )

        else:

            raise ValueError(
                f"{where}.linked_mechanism_ids "
                "must be a string, mechanism object, "
                "or list; "
                f"got {type(raw_linked).__name__}"
            )

        # ----------------------------------------------------
        # Normalize accidentally concatenated mechanism IDs
        #
        # Examples:
        #   ["M1, M2"] -> ["M1", "M2"]
        #   ["M1，M2"] -> ["M1", "M2"]
        #   ["M1; M2"] -> ["M1", "M2"]
        #
        # IMPORTANT:
        # split only when EVERY resulting token is already
        # an eligible mechanism ID.
        # ----------------------------------------------------

        expanded_linked = []

        for mid in normalized_linked:

            # First keep exact valid IDs unchanged.
            if mid in eligible_mechanism_ids:
                expanded_linked.append(mid)
                continue

            # Try common delimiters used by LLMs.
            candidate_parts = None
            parts = [
                part.strip()
                for part in re.split(
                    r"[,，;；]",
                    mid,
                )
                if part.strip()
            ]
            
            if (
                len(parts) >= 2
                and all(
                    part in eligible_mechanism_ids
                    for part in parts
                )
            ):
                candidate_parts = parts

            if candidate_parts is not None:

                expanded_linked.extend(
                    candidate_parts
                )

                notes.append(
                    f"{where}.linked_mechanism_ids: "
                    f"concatenated mechanism IDs "
                    f"{mid!r} -> {candidate_parts!r}"
                )

            else:

                # Leave invalid values untouched.
                # The strict validator below will reject them.
                expanded_linked.append(mid)

        normalized_linked = expanded_linked

        # ----------------------------------------------------
        # De-duplicate while preserving order
        # ----------------------------------------------------

        normalized_linked = list(
            dict.fromkeys(
                normalized_linked
            )
        )

        # ----------------------------------------------------
        # Important:
        # normalization must NEVER invent a mechanism ID.
        # ----------------------------------------------------

        invalid_mids = [
            mid
            for mid in normalized_linked
            if mid
            not in eligible_mechanism_ids
        ]

        if invalid_mids:

            raise ValueError(
                f"{where}.linked_mechanism_ids "
                "contains mechanism IDs not present "
                f"in DIAGNOSIS: {invalid_mids}; "
                f"eligible="
                f"{sorted(eligible_mechanism_ids)}"
            )

        action[
            "linked_mechanism_ids"
        ] = normalized_linked
        

        # ====================================================
        # 2. source_refs
        # ====================================================

        raw_refs = action.get(
            "source_refs"
        )
        if isinstance(raw_refs, str):

            ref = raw_refs.strip()
        
            # ------------------------------------------------
            # Empty string -> []
            # ------------------------------------------------
            if not ref:
        
                action["source_refs"] = []
        
                notes.append(
                    f"{where}.source_refs: "
                    "empty string -> []"
                )
        
            # ------------------------------------------------
            # JSON array serialized as a string
            #
            # Examples:
            #   "[]"                  -> []
            #   '["REF1"]'            -> ["REF1"]
            #   '["REF1", "REF2"]'    -> ["REF1", "REF2"]
            #
            # Important:
            # We only parse syntactically valid JSON arrays.
            # Semantic validity is still checked later against
            # references_by_mechanism.
            # ------------------------------------------------
            elif (
                ref.startswith("[")
                and ref.endswith("]")
            ):
        
                try:
                    decoded = json.loads(ref)
                except Exception:
                    decoded = None
        
                if not isinstance(decoded, list):
                    raise ValueError(
                        f"{where}.source_refs looks like "
                        "a serialized JSON array but could "
                        f"not be parsed as list[str]: {ref!r}"
                    )
        
                normalized_refs = []
        
                for j, item in enumerate(decoded):
        
                    if not isinstance(item, str):
                        raise ValueError(
                            f"{where}.source_refs[{j}] "
                            "decoded from serialized JSON "
                            "must be a string; "
                            f"got {type(item).__name__}"
                        )
        
                    item = item.strip()
        
                    if not item:
                        raise ValueError(
                            f"{where}.source_refs[{j}] "
                            "decoded from serialized JSON "
                            "must not be empty"
                        )
        
                    normalized_refs.append(item)
        
                action["source_refs"] = list(
                    dict.fromkeys(normalized_refs)
                )
        
                notes.append(
                    f"{where}.source_refs: "
                    "serialized JSON array -> list[str]"
                )
        
            # ------------------------------------------------
            # Ordinary scalar reference
            # ------------------------------------------------
            else:
        
                action["source_refs"] = [ref]
        
                notes.append(
                    f"{where}.source_refs: "
                    "scalar string -> list[str]"
                )

        elif raw_refs is None:

            action["source_refs"] = []

            notes.append(
                f"{where}.source_refs: "
                "null -> []"
            )

        elif isinstance(raw_refs, list):

            refs = []

            for j, ref in enumerate(
                raw_refs
            ):

                if not isinstance(ref, str):

                    raise ValueError(
                        f"{where}.source_refs[{j}] "
                        "must be a string; "
                        f"got {type(ref).__name__}"
                    )

                ref = ref.strip()

                if not ref:
                    raise ValueError(
                        f"{where}.source_refs[{j}] "
                        "must not be empty"
                    )

                refs.append(ref)

            action["source_refs"] = list(
                dict.fromkeys(refs)
            )

        else:

            raise ValueError(
                f"{where}.source_refs must be "
                "a string, list[str], or null; "
                f"got {type(raw_refs).__name__}"
            )

        # ====================================================
        # 3. target_products / target_layers
        # ====================================================

        for key in (
            "target_products",
            "target_layers",
        ):

            raw_value = action.get(key)

            if raw_value is None:

                action[key] = []

                notes.append(
                    f"{where}.{key}: "
                    "null/missing -> []"
                )

            elif isinstance(raw_value, str):

                value = raw_value.strip()
            
                if not value:
            
                    action[key] = []
            
                elif key == "target_layers":
            
                    values = [
                        x.strip()
                        for x in re.split(
                            r"[,，;；]",
                            value,
                        )
                        if x.strip()
                    ]
            
                    action[key] = list(
                        dict.fromkeys(values)
                    )
            
                else:
            
                    # target_products:
                    # do not split product names automatically
                    action[key] = [value]
            
                notes.append(
                    f"{where}.{key}: "
                    "scalar string -> list[str]"
                )

            elif isinstance(raw_value, list):
                values = []
                for j, item in enumerate(raw_value):
                    if not isinstance(item, str,):
                        raise ValueError(
                            f"{where}.{key}[{j}] "
                            "must be a string; "
                            f"got "
                            f"{type(item).__name__}"
                        )
            
                    item = item.strip()
            
                    if not item:
                        raise ValueError(
                            f"{where}.{key}[{j}] "
                            "must not be empty"
                        )
            
                    # target_layers may contain accidentally
                    # concatenated descriptive layer names.
                    if key == "target_layers":
            
                        parts = [
                            x.strip()
                            for x in re.split(
                                r"[,，;；]",
                                item,
                            )
                            if x.strip()
                        ]
            
                        values.extend(parts)
            
                    else:
            
                        # target_products should not be split:
                        # product names may legitimately contain
                        # punctuation or compound names.
                        values.append(item)
            
                action[key] = list(
                    dict.fromkeys(values)
                )

            else:

                raise ValueError(
                    f"{where}.{key} must be "
                    "a string, list[str], or null; "
                    f"got "
                    f"{type(raw_value).__name__}"
                )

        normalized_actions.append(
            action
        )

    out["actions"] = normalized_actions

    return out, notes


# ============================================================
# 4. Strict local action-output validation
# ============================================================

def validate_experimental_action_output(
    action_output: Dict[str, Any],
    *,
    diagnosis: Optional[Dict[str, Any]] = None,
    case_package: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Strict local validation.

    llm_service.call_json currently provides JSON-object mode rather than
    provider-side JSON Schema enforcement, so all research-critical constraints
    are checked here.
    """
    if not isinstance(action_output, dict):
        raise TypeError(
            "Experimental action output must be a dict"
        )

    expected_constraints: Optional[Dict[str, Any]] = None

    if diagnosis is not None or case_package is not None:
        if not isinstance(diagnosis, dict):
            raise TypeError(
                "diagnosis must be a dict when supplied"
            )
        if not isinstance(case_package, dict):
            raise TypeError(
                "case_package must be a dict when supplied"
            )

        expected_constraints = _build_action_constraints(
            diagnosis=diagnosis,
            case_package=case_package,
        )

    # --------------------------------------------------------
    # Root metadata
    # --------------------------------------------------------

    case_id_value = action_output.get("case_id")

    if not isinstance(case_id_value, str):
        raise ValueError(
            "root.case_id must be a string"
        )

    case_id_value = case_id_value.strip()

    product_category = _require_str(
        action_output,
        "product_category",
        "root",
    )

    if expected_constraints is not None:
        expected_case_id = expected_constraints["case_id"]
        expected_category = expected_constraints["product_category"]

        if case_id_value != expected_case_id:
            raise ValueError(
                "root.case_id does not match the input case: "
                f"got {case_id_value!r}, expected {expected_case_id!r}"
            )

        if product_category != expected_category:
            raise ValueError(
                "root.product_category does not match the input case: "
                f"got {product_category!r}, expected {expected_category!r}"
            )

    # --------------------------------------------------------
    # Actions array
    # --------------------------------------------------------

    actions = action_output.get("actions")

    if not isinstance(actions, list):
        raise ValueError(
            "root.actions must be a list"
        )

    if len(actions) > 5:
        raise ValueError(
            f"root.actions contains {len(actions)} items; maximum is 5"
        )

    eligible_mechanism_ids: Set[str] = set()

    references_by_mechanism: Dict[str, List[str]] = {}

    if expected_constraints is not None:
        eligible_mechanism_ids = set(
            expected_constraints["eligible_mechanism_ids"]
        )
        references_by_mechanism = dict(
            expected_constraints["references_by_mechanism"]
        )

        # If the diagnosis contains mechanisms, silently returning no action
        # violates the experiment prompt because uncertain cases should yield
        # an observability/validation action rather than disappear.
        if eligible_mechanism_ids and not actions:
            raise ValueError(
                "DIAGNOSIS contains mechanisms but root.actions is empty"
            )

        if not eligible_mechanism_ids and actions:
            raise ValueError(
                "DIAGNOSIS contains no mechanisms, so root.actions must be empty"
            )

    seen_action_ids: Set[str] = set()
    seen_ranks: Set[int] = set()

    for i, action in enumerate(actions, start=1):
        where = f"root.actions[{i - 1}]"

        if not isinstance(action, dict):
            raise TypeError(
                f"{where} must be an object"
            )

        # Deterministic action identifiers.
        action_id = _require_str(
            action,
            "action_id",
            where,
        )

        expected_action_id = f"A{i}"
        if action_id != expected_action_id:
            raise ValueError(
                f"{where}.action_id must be {expected_action_id!r}; "
                f"got {action_id!r}"
            )

        if action_id in seen_action_ids:
            raise ValueError(
                f"Duplicate action_id: {action_id}"
            )
        seen_action_ids.add(action_id)

        # Deterministic ranking.
        rank = action.get("priority_rank")

        if (
            not isinstance(rank, int)
            or isinstance(rank, bool)
            or rank < 1
        ):
            raise ValueError(
                f"{where}.priority_rank must be a positive integer"
            )

        if rank != i:
            raise ValueError(
                f"{where}.priority_rank must be {i}; got {rank}"
            )

        if rank in seen_ranks:
            raise ValueError(
                f"Duplicate priority_rank: {rank}"
            )
        seen_ranks.add(rank)

        # Frozen taxonomy.
        subfamily = _require_str(
            action,
            "action_subfamily",
            where,
        )

        if subfamily not in ACTION_SUBFAMILIES:
            raise ValueError(
                f"{where}.action_subfamily={subfamily!r} is not in frozen ACTION_SUBFAMILIES"
            )

        level = _require_str(
            action,
            "action_level",
            where,
        )

        if level not in ACTION_LEVELS:
            raise ValueError(
                f"{where}.action_level={level!r} "
                f"must be one of {sorted(ACTION_LEVELS)}"
            )

        _require_str(
            action,
            "action",
            where,
        )

        # ----------------------------------------------------
        # Mechanism linkage: must be a true subset of DIAGNOSIS
        # ----------------------------------------------------

        linked = _require_str_list(
            action,
            "linked_mechanism_ids",
            where,
            allow_empty=False,
        )

        if expected_constraints is not None:
            invalid_mids = [
                mid
                for mid in linked
                if mid not in eligible_mechanism_ids
            ]

            if invalid_mids:
                raise ValueError(
                    f"{where}.linked_mechanism_ids contains IDs not present "
                    f"in DIAGNOSIS: {invalid_mids}; "
                    f"eligible={sorted(eligible_mechanism_ids)}"
                )

        # ----------------------------------------------------
        # Reference linkage
        # ----------------------------------------------------

        refs = _require_str_list(
            action,
            "source_refs",
            where,
            allow_empty=True,
        )

        if expected_constraints is not None:
            allowed_refs: Set[str] = set()

            for mid in linked:
                allowed_refs.update(
                    references_by_mechanism.get(mid, [])
                )

            invalid_refs = [
                ref
                for ref in refs
                if ref not in allowed_refs
            ]

            if invalid_refs:
                raise ValueError(
                    f"{where}.source_refs contains references not inherited "
                    f"from the linked diagnosis mechanisms: {invalid_refs}; "
                    f"allowed={sorted(allowed_refs)}"
                )

            if allowed_refs and not refs:
                raise ValueError(
                    f"{where}.source_refs is empty although valid references "
                    f"exist for its linked mechanisms: {sorted(allowed_refs)}"
                )

            if not allowed_refs and refs:
                raise ValueError(
                    f"{where}.source_refs must be [] because no valid "
                    "reference is available for its linked mechanisms"
                )

        # Optional list fields must still have valid shape.
        for key in (
            "target_products",
            "target_layers",
        ):
            if key in action:
                _require_str_list(
                    action,
                    key,
                    where,
                    allow_empty=True,
                )

        _optional_nonempty_str(
            action,
            "validation_metric",
            where,
        )

    return action_output


# ============================================================
# 5. Experimental action generation
# ============================================================

def generate_experimental_actions(
    *,
    diagnosis: Dict[str, Any],
    case_package: Dict[str, Any],
    model_name: Optional[str] = None,
    temperature: float = 0.2,
    action_context_mode: str = "auto",
) -> Dict[str, Any]:
    """Generate detailed prospective actions with task-routed evidence.

    Default routing
    ---------------
    ``action_context_mode="auto"`` preserves the historical FULL action input for A/B/C, but routes H_hierarchical through H_AB:

        H diagnosis:  A + B + C  (built upstream; unchanged here)
        H action:     A + B      (risk_chains removed only for this stage)

    Explicit ``full`` / ``h_ab`` / ``h_a`` remain available for controlled
    ablations.  Gold labels are never used by this routing rule.
    """
    if not isinstance(diagnosis, dict):
        raise TypeError(
            f"diagnosis must be a dict, got {type(diagnosis).__name__}"
        )
    if not isinstance(case_package, dict):
        raise TypeError(
            f"case_package must be a dict, got {type(case_package).__name__}"
        )
    if (
        not isinstance(temperature, (int, float))
        or isinstance(temperature, bool)
    ):
        raise TypeError("temperature must be numeric")

    resolved_mode = resolve_action_context_mode(
        case_package=case_package,
        action_context_mode=action_context_mode,
    )

    # IMPORTANT: create a stage-local copy.  The upstream H diagnosis package
    # remains intact and can still contain C risk chains for mechanism diagnosis.
    action_case_package = build_routed_action_case_package(
        case_package=case_package,
        action_context_mode=resolved_mode,
    )
    action_context_stats = summarize_action_case_view(action_case_package)

    # For final detailed actions, constraints are derived from the evidence
    # actually visible at the action stage.  Therefore H_AB cannot cite CHN refs
    # that have deliberately been removed from its action package.
    constraints = _build_action_constraints(
        diagnosis=diagnosis,
        case_package=action_case_package,
    )

    if not constraints["eligible_mechanism_ids"]:
        parsed = {
            "case_id": constraints["case_id"],
            "product_category": constraints["product_category"],
            "actions": [],
        }
        validate_experimental_action_output(
            parsed,
            diagnosis=diagnosis,
            case_package=action_case_package,
        )
        return {
            "parsed": parsed,
            "raw_output": "",
            "generation_status": "no_eligible_mechanism",
            "usage": {
                "prompt_tokens": 0,
                "cached_prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "llm_trace": {"attempts": []},
            "requested_action_context_mode": str(
                action_context_mode or "auto"
            ).strip().lower(),
            "action_context_mode": resolved_mode,
            "action_context_stats": action_context_stats,
        }

    payload = {
        "DIAGNOSIS": diagnosis,
        "CASE_PACKAGE": action_case_package,
        "OUTPUT_CONSTRAINTS": constraints,
        "TASK": (
            "Generate prospective product-ecosystem interventions using only "
            "the supplied pre-event diagnosis and action-stage case package. "
            "Do not introduce mechanisms or evidence references that are not "
            "allowed by OUTPUT_CONSTRAINTS."
        ),
    }

    result = call_json(
        system_prompt=ACTION_EXPERIMENT_PROMPT,
        user_payload=payload,
        model_name=model_name,
        temperature=float(temperature),
    )
    if not isinstance(result, dict):
        raise TypeError(
            f"call_json must return a dict, got {type(result).__name__}"
        )

    parsed = result.get("parsed")
    if not isinstance(parsed, dict):
        error = (
            "call_json returned no parsed JSON object; "
            f"parse_error={result.get('parse_error')!r}"
        )
        raise ActionGenerationTraceError(
            error,
            llm_trace=_action_attempt_trace(
                result=result,
                parsed=None,
                error=error,
            ),
        )

    try:
        parsed, normalization_notes = normalize_experimental_action_output(
            parsed,
            constraints=constraints,
        )
        validate_experimental_action_output(
            parsed,
            diagnosis=diagnosis,
            case_package=action_case_package,
        )
    except Exception as exc:
        raise ActionGenerationTraceError(
            str(exc),
            llm_trace=_action_attempt_trace(
                result=result,
                parsed=parsed if isinstance(parsed, dict) else None,
                error=str(exc),
            ),
        ) from exc

    result["parsed"] = parsed
    result["normalization_notes"] = normalization_notes
    result["llm_trace"] = _action_attempt_trace(
        result=result,
        parsed=parsed,
        error=None,
    )
    result["generation_status"] = "generated"
    result["requested_action_context_mode"] = str(
        action_context_mode or "auto"
    ).strip().lower()
    result["action_context_mode"] = resolved_mode
    result["action_context_stats"] = action_context_stats
    return result



# ==================================================
# compare
# ==================================================
def build_compact_action_case_package(
    *,
    case_package: Dict[str, Any],
    diagnosis: Dict[str, Any],
) -> Dict[str, Any]:
    """Compact condition-specific evidence for action-subfamily prediction.

    Uses the same field-level compactor as diagnosis so A/B/C/H retain the
    same information treatment without repeatedly sending verbose audit fields.
    """
    if not isinstance(case_package, dict):
        raise TypeError("case_package must be a dict")
    if not isinstance(diagnosis, dict):
        raise TypeError("diagnosis must be a dict")

    risk_dimensions = [
        str(x).strip()
        for x in (diagnosis.get("risk_dimensions") or [])
        if str(x).strip() and str(x).strip().lower() != "general"
    ]

    compact = build_llm_case_view(
        case_package,
        required_dimensions=(risk_dimensions if risk_dimensions else None),
        max_statements=2,
    )

    anchor = case_package.get("dimension_anchor")
    if isinstance(anchor, dict):
        compact["dimension_anchor"] = {
            "core_risk_dimensions": list(anchor.get("core_risk_dimensions") or []),
            "promoted_risk_dimensions": list(anchor.get("promoted_risk_dimensions") or []),
            "risk_dimensions": list(anchor.get("risk_dimensions") or []),
        }

    return compact



# ============================================================
# 5B. Action-stage evidence routing
# ============================================================

# ``auto`` is the production default:
#   A/B/C -> full
#   H     -> h_ab
# Explicit modes remain available for ablation/reproducibility.
ACTION_CONTEXT_MODES = {
    "auto",
    "full",
    "h_a",
    "h_ab",
}
DEFAULT_H_ACTION_CONTEXT_MODE = "h_ab"
ACTION_ROUTING_VERSION = "h_ab_default_v1"
_H_CONDITION_NAME = "H_hierarchical"


def _action_case_condition(
    case_package: Dict[str, Any],
) -> str:
    if not isinstance(case_package, dict):
        return ""
    meta = (
        case_package.get("case_meta")
        if isinstance(case_package.get("case_meta"), dict)
        else {}
    )
    return str(
        meta.get("condition")
        or case_package.get("condition")
        or ""
    ).strip()


def resolve_action_context_mode(
    *,
    case_package: Dict[str, Any],
    action_context_mode: str = "auto",
) -> str:
    """Resolve production/ablation action routing without using gold labels.

    ``auto`` is intentionally condition-specific:
      - H_hierarchical -> h_ab
      - A/B/C/other    -> full

    This lets the existing experiment runner call the action generator exactly
    as before while replacing only H's downstream action evidence route.
    """
    mode = str(action_context_mode or "auto").strip().lower()
    if mode not in ACTION_CONTEXT_MODES:
        raise ValueError(
            f"Unknown action_context_mode={mode!r}; "
            f"allowed={sorted(ACTION_CONTEXT_MODES)}"
        )
    if mode != "auto":
        return mode
    return (
        DEFAULT_H_ACTION_CONTEXT_MODE
        if _action_case_condition(case_package) == _H_CONDITION_NAME else "full"
    )


def build_routed_action_case_package(
    *,
    case_package: Dict[str, Any],
    action_context_mode: str = "auto",
) -> Dict[str, Any]:
    """Return the raw stage-local action package for the requested route.

    The input object is never mutated.  This is the production routing helper
    used by detailed action generation.  H_AB removes only C risk-chain exposure;
    H_A additionally removes B dynamic facts.  Static OBS/REL/API and metadata
    are preserved exactly.
    """
    if not isinstance(case_package, dict):
        raise TypeError("case_package must be a dict")

    mode = resolve_action_context_mode(
        case_package=case_package,
        action_context_mode=action_context_mode,
    )
    routed = copy.deepcopy(case_package)

    if mode == "full":
        return routed
    if mode == "h_ab":
        routed["risk_chains"] = []
        return routed
    if mode == "h_a":
        routed["model_facts"] = []
        routed["risk_chains"] = []
        return routed

    raise AssertionError(f"Unhandled resolved action context mode: {mode}")


def _action_view_row_reference_id(
    row: Dict[str, Any],
) -> Optional[str]:
    """Return the traceable evidence/reference ID carried by one compact row."""
    if not isinstance(row, dict):
        return None
    for key in (
        "evidence_id",
        "relation_id",
        "knowledge_id",
        "fact_id",
        "chain_id",
    ):
        value = row.get(key)
        if value is None:
            continue
        value = str(value).strip()
        if value:
            return value
    return None


def mechanism_linked_reference_ids(
    *,
    diagnosis: Dict[str, Any],
    case_package: Dict[str, Any],
) -> Set[str]:
    """Return refs inherited by final diagnosis mechanisms."""
    constraints = _build_action_constraints(
        diagnosis=diagnosis,
        case_package=case_package,
    )
    refs: Set[str] = set()
    for values in (
        constraints.get("references_by_mechanism") or {}
    ).values():
        for value in values or []:
            value = str(value).strip()
            if value:
                refs.add(value)
    return refs


def build_h_action_case_view(
    *,
    case_package: Dict[str, Any],
    diagnosis: Dict[str, Any],
    action_context_mode: str = "auto",
) -> Dict[str, Any]:
    """Build the compact LLM-visible action view used by subfamily prediction.

    Production default is H_AB for H and FULL for A/B/C.  Explicit modes keep
    the previous ablation interface available.
    """
    if not isinstance(case_package, dict):
        raise TypeError("case_package must be a dict")
    if not isinstance(diagnosis, dict):
        raise TypeError("diagnosis must be a dict")

    mode = resolve_action_context_mode(
        case_package=case_package,
        action_context_mode=action_context_mode,
    )
    routed = build_routed_action_case_package(
        case_package=case_package,
        action_context_mode=mode,
    )
    return build_compact_action_case_package(
        case_package=routed,
        diagnosis=diagnosis,
    )


def summarize_action_case_view(
    view: Dict[str, Any],
) -> Dict[str, Any]:
    """Return compact, gold-blind context-size diagnostics for one action view."""
    if not isinstance(view, dict):
        raise TypeError("view must be a dict")

    def _n(field: str) -> int:
        value = view.get(field) or []
        return len(value) if isinstance(value, list) else 0

    payload_chars = len(
        json.dumps(
            view,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
    )

    ref_ids: Set[str] = set()
    for field in (
        "observed_dimension_catalog",
        "observed_evidence",
        "ecosystem_relations",
        "api_knowledge",
        "model_facts",
        "risk_chains",
    ):
        for row in view.get(field) or []:
            if not isinstance(row, dict):
                continue
            ref_id = _action_view_row_reference_id(row)
            if ref_id:
                ref_ids.add(ref_id)

    return {
        "n_action_catalog_rows": _n("observed_dimension_catalog"),
        "n_action_observed_evidence": _n("observed_evidence"),
        "n_action_relations": _n("ecosystem_relations"),
        "n_action_api_knowledge": _n("api_knowledge"),
        "n_action_model_facts": _n("model_facts"),
        "n_action_risk_chains": _n("risk_chains"),
        "n_action_unique_reference_ids": len(ref_ids),
        "action_context_chars": int(payload_chars),
    }


# ============================================================
# 5C. Final prospective action generation after human review
# ============================================================

_ACTIONABLE_HUMAN_ASSESSMENTS = {
    "supported",
    "plausible",
    "uncertain",
}


def _prepare_reviewed_diagnosis_for_actions(
    reviewed_diagnosis: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, str]]:
    """Keep only reviewed mechanisms that are eligible for downstream action.

    Human revisions replace the LLM statement only in this stage-local copy;
    the persisted original diagnosis remains unchanged.
    """
    if not isinstance(reviewed_diagnosis, dict):
        raise TypeError("reviewed_diagnosis must be a dict")

    out = copy.deepcopy(reviewed_diagnosis)
    mechanisms = _diagnosis_mechanisms(out)
    kept: List[Dict[str, Any]] = []
    status_by_id: Dict[str, str] = {}

    for mech in mechanisms:
        status = str(mech.get("human_assessment") or "not_assessed").strip().lower()
        if status not in _ACTIONABLE_HUMAN_ASSESSMENTS:
            continue

        row = copy.deepcopy(mech)
        mid = str(row["mechanism_id"]).strip()
        revision = str(row.get("human_revision") or "").strip()
        if revision:
            row["original_statement"] = str(row.get("statement") or "")
            row["statement"] = revision
            
        row.pop("human_validation_note", None)
        
        kept.append(row)
        status_by_id[mid] = status

    out["mechanisms"] = kept
    return out, status_by_id


def _normalize_decision_context(decision_context: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(decision_context, dict):
        raise TypeError("decision_context must be a dict")

    objective = str(decision_context.get("objective") or "").strip()
    if not objective:
        raise ValueError("decision_context.objective must be a non-empty string")

    raw_layers = decision_context.get("allowed_layers")
    if not isinstance(raw_layers, list):
        raise ValueError("decision_context.allowed_layers must be a list")

    known_layers = set(PROSPECTIVE_ACTION_LAYERS)
    layers: List[str] = []
    for i, value in enumerate(raw_layers):
        layer = str(value or "").strip()
        if not layer:
            raise ValueError(f"decision_context.allowed_layers[{i}] must be non-empty")
        if layer not in known_layers:
            raise ValueError(
                f"Unknown decision_context layer {layer!r}; "
                f"allowed={PROSPECTIVE_ACTION_LAYERS}"
            )
        if layer not in layers:
            layers.append(layer)

    allow_hardware = bool(decision_context.get("allow_hardware_change", False))
    if not allow_hardware:
        layers = [x for x in layers if x != "hardware"]

    return {
        "context_source": str(decision_context.get("context_source") or "").strip(),
        "product_category": str(decision_context.get("product_category") or "").strip(),
        "objective": objective,
        "allowed_layers": layers,
        "allow_hardware_change": allow_hardware,
        "allow_third_party_change": bool(
            decision_context.get("allow_third_party_change", False)
        ),
    }


def _validate_prospective_action_output(
    action_output: Dict[str, Any],
    *,
    diagnosis: Dict[str, Any],
    case_package: Dict[str, Any],
    decision_context: Dict[str, Any],
    status_by_mechanism: Dict[str, str],
) -> Dict[str, Any]:
    """Apply the experimental traceability contract plus deployment constraints."""
    validate_experimental_action_output(
        action_output,
        diagnosis=diagnosis,
        case_package=case_package,
    )

    allowed_layers = set(decision_context.get("allowed_layers") or [])
    allow_hardware = bool(decision_context.get("allow_hardware_change", False))

    for i, action in enumerate(action_output.get("actions") or []):
        where = f"root.actions[{i}]"
        _require_str(action, "improvement_direction", where)
        _require_str(action, "intervention_logic", where)
        _require_str(action, "validation_metric", where)

        target_layers = _require_str_list(
            action,
            "target_layers",
            where,
            allow_empty=False,
        )
        invalid_layers = [x for x in target_layers if x not in allowed_layers]
        if invalid_layers:
            raise ValueError(
                f"{where}.target_layers contains layers outside decision_context: {invalid_layers}; allowed={sorted(allowed_layers)}"
            )
        if not allow_hardware and "hardware" in target_layers:
            raise ValueError(f"{where} proposes hardware although hardware change is disabled")
        if action.get("action_subfamily") == "hardware_design_or_repair":
            if not allow_hardware or "hardware" not in target_layers:
                raise ValueError(
                    f"{where}.action_subfamily='hardware_design_or_repair' requires hardware to be explicitly allowed and included in target_layers"
                )

        linked = _require_str_list(action, "linked_mechanism_ids", where, allow_empty=False)
        
        linked_statuses = [status_by_mechanism.get(mid, "") for mid in linked]
        #if "uncertain" in linked_statuses:
        if linked_statuses and all(status == "uncertain" for status in linked_statuses):
            if action.get("action_subfamily") != "observability_or_validation":
                raise ValueError(f"{where} links only human-uncertain mechanisms and therefore must use action_subfamily='observability_or_validation'.")
       
    return action_output



def _validate_prospective_root_for_salvage(
    action_output: Dict[str, Any],
    *,
    constraints: Dict[str, Any],
) -> List[Any]:
    """Validate only root-level invariants before per-action salvage.

    Root errors are never salvaged because they make the identity or structure
    of the complete response ambiguous. Action-local errors are handled later.
    """
    if not isinstance(action_output, dict):
        raise TypeError("Prospective action output must be a dict")

    case_id = action_output.get("case_id")
    if not isinstance(case_id, str):
        raise ValueError("root.case_id must be a string")
    case_id = case_id.strip()
    if case_id != str(constraints.get("case_id") or ""):
        raise ValueError(
            "root.case_id does not match the input case: "
            f"got {case_id!r}, expected {constraints.get('case_id')!r}"
        )

    product_category = _require_str(action_output, "product_category", "root")
    expected_category = str(constraints.get("product_category") or "")
    if product_category != expected_category:
        raise ValueError(
            "root.product_category does not match the input case: "
            f"got {product_category!r}, expected {expected_category!r}"
        )

    actions = action_output.get("actions")
    if not isinstance(actions, list):
        raise ValueError("root.actions must be a list")
    if len(actions) > 5:
        raise ValueError(
            f"root.actions contains {len(actions)} items; maximum is 5"
        )
    if constraints.get("eligible_mechanism_ids") and not actions:
        raise ValueError(
            "Eligible reviewed mechanisms exist but the model returned no actions"
        )
    return actions


def _validate_prospective_action_item(
    action: Dict[str, Any],
    *,
    where: str,
    constraints: Dict[str, Any],
    decision_context: Dict[str, Any],
    status_by_mechanism: Dict[str, str],
) -> Dict[str, Any]:
    """Strictly validate one action without checking array position/id/rank.

    This is intentionally semantic: failures here discard only this action.
    It never invents mechanism IDs, evidence references, or intervention logic.
    """
    if not isinstance(action, dict):
        raise TypeError(f"{where} must be an object")

    subfamily = _require_str(action, "action_subfamily", where)
    if subfamily not in ACTION_SUBFAMILIES:
        raise ValueError(
            f"{where}.action_subfamily={subfamily!r} is not in frozen ACTION_SUBFAMILIES"
        )

    level = _require_str(action, "action_level", where)
    if level not in ACTION_LEVELS:
        raise ValueError(
            f"{where}.action_level={level!r} must be one of {sorted(ACTION_LEVELS)}"
        )

    _require_str(action, "improvement_direction", where)
    _require_str(action, "action", where)
    _require_str(action, "intervention_logic", where)
    _require_str(action, "validation_metric", where)

    linked = _require_str_list(
        action,
        "linked_mechanism_ids",
        where,
        allow_empty=False,
    )
    eligible = set(constraints.get("eligible_mechanism_ids") or [])
    invalid_mids = [mid for mid in linked if mid not in eligible]
    if invalid_mids:
        raise ValueError(
            f"{where}.linked_mechanism_ids contains IDs not present in the "
            f"reviewed diagnosis: {invalid_mids}; eligible={sorted(eligible)}"
        )

    refs = _require_str_list(action, "source_refs", where, allow_empty=True)
    references_by_mechanism = dict(
        constraints.get("references_by_mechanism") or {}
    )
    allowed_refs: Set[str] = set()
    for mid in linked:
        allowed_refs.update(references_by_mechanism.get(mid, []) or [])

    invalid_refs = [ref for ref in refs if ref not in allowed_refs]
    if invalid_refs:
        raise ValueError(
            f"{where}.source_refs contains references not inherited from the "
            f"linked diagnosis mechanisms: {invalid_refs}; "
            f"allowed={sorted(allowed_refs)}"
        )
    if allowed_refs and not refs:
        raise ValueError(
            f"{where}.source_refs is empty although valid references exist for "
            f"its linked mechanisms: {sorted(allowed_refs)}"
        )
    if not allowed_refs and refs:
        raise ValueError(
            f"{where}.source_refs must be [] because no valid reference is "
            "available for its linked mechanisms"
        )

    _require_str_list(action, "target_products", where, allow_empty=True)
    target_layers = _require_str_list(
        action,
        "target_layers",
        where,
        allow_empty=False,
    )

    allowed_layers = set(decision_context.get("allowed_layers") or [])
    invalid_layers = [x for x in target_layers if x not in allowed_layers]
    if invalid_layers:
        raise ValueError(
            f"{where}.target_layers contains layers outside decision_context: "
            f"{invalid_layers}; allowed={sorted(allowed_layers)}"
        )

    allow_hardware = bool(decision_context.get("allow_hardware_change", False))
    if not allow_hardware and "hardware" in target_layers:
        raise ValueError(
            f"{where} proposes hardware although hardware change is disabled"
        )
    if subfamily == "hardware_design_or_repair":
        if not allow_hardware or "hardware" not in target_layers:
            raise ValueError(
                f"{where}.action_subfamily='hardware_design_or_repair' requires "
                "hardware to be explicitly allowed and included in target_layers"
            )

    linked_statuses = [status_by_mechanism.get(mid, "") for mid in linked]
    if linked_statuses and all(status == "uncertain" for status in linked_statuses):
    #if "uncertain" in linked_statuses:
        if subfamily != "observability_or_validation":
            raise ValueError(
                f"{where} links only human-uncertain mechanisms and therefore must use action_subfamily='observability_or_validation'"
            )
    return action


def salvage_prospective_action_output(
    action_output: Dict[str, Any],
    *,
    diagnosis: Dict[str, Any],
    case_package: Dict[str, Any],
    decision_context: Dict[str, Any],
    status_by_mechanism: Dict[str, str],
    constraints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Keep valid prospective actions and deterministically discard invalid ones.

    Policy:
      * root-level identity/shape error -> raise and block the whole response;
      * action-local contract error -> discard that action and retain audit data;
      * >=1 valid action -> canonicalize A1..Ak and continue to Human Gate 2;
      * 0 valid actions -> raise so the workflow is BLOCKED_ACTION_OUTPUT_REVIEW.

    No semantic repair is performed. Only structurally equivalent normalization
    and positional action-id/rank canonicalization are allowed.
    """
    if constraints is None:
        constraints = _build_action_constraints(
            diagnosis=diagnosis,
            case_package=case_package,
        )

    raw_actions = _validate_prospective_root_for_salvage(
        action_output,
        constraints=constraints,
    )

    accepted: List[Dict[str, Any]] = []
    discarded: List[Dict[str, Any]] = []
    normalization_notes: List[str] = []
    action_id_map: List[Dict[str, Any]] = []

    for original_index, raw_action in enumerate(raw_actions, start=1):
        original_id = (
            str(raw_action.get("action_id") or "").strip()
            if isinstance(raw_action, dict)
            else ""
        )
        original_rank = (
            raw_action.get("priority_rank")
            if isinstance(raw_action, dict)
            else None
        )

        try:
            wrapper = {
                "case_id": constraints["case_id"],
                "product_category": constraints["product_category"],
                "actions": [copy.deepcopy(raw_action)],
            }
            normalized_wrapper, notes = normalize_experimental_action_output(
                wrapper,
                constraints=constraints,
            )
            action = normalized_wrapper["actions"][0]
            action = _validate_prospective_action_item(
                action,
                where=f"root.actions[{original_index - 1}]",
                constraints=constraints,
                decision_context=decision_context,
                status_by_mechanism=status_by_mechanism,
            )
            accepted.append(copy.deepcopy(action))
            normalization_notes.extend(
                f"original_action[{original_index}]: {note}" for note in notes
            )
        except Exception as exc:
            discarded.append({
                "original_index": original_index,
                "original_action_id": original_id or None,
                "original_priority_rank": original_rank,
                "validation_error": str(exc),
                "raw_action": copy.deepcopy(raw_action),
            })

    if not accepted:
        reasons = [
            f"#{x['original_index']}({x.get('original_action_id') or 'no-id'}): "
            f"{x['validation_error']}"
            for x in discarded
        ]
        raise ValueError(
            "No valid prospective action remains after per-action validation. "
            + " | ".join(reasons)
        )

    final_actions: List[Dict[str, Any]] = []
    for new_rank, action in enumerate(accepted, start=1):
        row = copy.deepcopy(action)
        old_id = str(row.get("action_id") or "").strip() or None
        old_rank = row.get("priority_rank")
        row["action_id"] = f"A{new_rank}"
        row["priority_rank"] = new_rank
        final_actions.append(row)
        action_id_map.append({
            "original_action_id": old_id,
            "original_priority_rank": old_rank,
            "final_action_id": row["action_id"],
            "final_priority_rank": new_rank,
        })

    sanitized = {
        "case_id": constraints["case_id"],
        "product_category": constraints["product_category"],
        "actions": final_actions,
    }

    # Final whole-output assertion after canonical re-numbering.
    _validate_prospective_action_output(
        sanitized,
        diagnosis=diagnosis,
        case_package=case_package,
        decision_context=decision_context,
        status_by_mechanism=status_by_mechanism,
    )

    return {
        "parsed": sanitized,
        "original_parsed": copy.deepcopy(action_output),
        "discarded_actions": discarded,
        "normalization_notes": normalization_notes,
        "action_id_map": action_id_map,
        "n_generated": len(raw_actions),
        "n_valid": len(final_actions),
        "n_discarded": len(discarded),
        "generation_status": "partial_valid" if discarded else "generated",
    }


def generate_prospective_actions(
    *,
    reviewed_diagnosis: Dict[str, Any],
    case_package: Dict[str, Any],
    decision_context: Dict[str, Any],
    model_name: Optional[str] = None,
    temperature: float = 0.2,
    action_context_mode: str = "auto",
) -> Dict[str, Any]:
    """Generate final detailed actions after Human Gate 1.

    Historical experimental action generation remains strict and unchanged.
    This prospective-only path salvages valid individual actions from a completed
    model response while preserving every discarded action for audit.

    API/runtime/provider failures fail fast. Root-level output-contract failures,
    or a response for which every action is invalid, become
    ActionGenerationTraceError and are persisted by the workflow as blocked.
    """
    if not isinstance(case_package, dict):
        raise TypeError("case_package must be a dict")
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        raise TypeError("temperature must be numeric")

    normalized_context = _normalize_decision_context(decision_context)
    diagnosis, status_by_mechanism = _prepare_reviewed_diagnosis_for_actions(
        reviewed_diagnosis
    )

    resolved_mode = resolve_action_context_mode(
        case_package=case_package,
        action_context_mode=action_context_mode,
    )
    action_case_package = build_routed_action_case_package(
        case_package=case_package,
        action_context_mode=resolved_mode,
    )
    action_context_stats = summarize_action_case_view(action_case_package)
    constraints = _build_action_constraints(
        diagnosis=diagnosis,
        case_package=action_case_package,
    )
    constraints["allowed_layers"] = list(normalized_context["allowed_layers"])
    constraints["allow_hardware_change"] = bool(
        normalized_context["allow_hardware_change"]
    )
    constraints["allow_third_party_change"] = bool(
        normalized_context["allow_third_party_change"]
    )
    constraints["human_assessment_by_mechanism"] = dict(status_by_mechanism)

    context_category = normalized_context.get("product_category")
    if context_category and context_category != constraints["product_category"]:
        raise ValueError(
            "decision_context.product_category does not match diagnosis/package: "
            f"{context_category!r} != {constraints['product_category']!r}"
        )

    if not constraints["eligible_mechanism_ids"]:
        parsed = {
            "case_id": constraints["case_id"],
            "product_category": constraints["product_category"],
            "actions": [],
        }
        return {
            "parsed": parsed,
            "original_parsed": copy.deepcopy(parsed),
            "raw_output": "",
            "normalization_notes": [],
            "discarded_actions": [],
            "action_id_map": [],
            "n_generated": 0,
            "n_valid": 0,
            "n_discarded": 0,
            "generation_status": "no_human_approved_mechanism",
            "usage": {
                "prompt_tokens": 0,
                "cached_prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "llm_trace": {"attempts": []},
            "requested_action_context_mode": str(action_context_mode or "auto").strip().lower(),
            "action_context_mode": resolved_mode,
            "action_context_stats": action_context_stats,
        }

    if not constraints["allowed_layers"]:
        parsed = {
            "case_id": constraints["case_id"],
            "product_category": constraints["product_category"],
            "actions": [],
        }
        return {
            "parsed": parsed,
            "original_parsed": copy.deepcopy(parsed),
            "raw_output": "",
            "normalization_notes": [],
            "discarded_actions": [],
            "action_id_map": [],
            "n_generated": 0,
            "n_valid": 0,
            "n_discarded": 0,
            "generation_status": "no_allowed_intervention_layer",
            "usage": {
                "prompt_tokens": 0,
                "cached_prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "llm_trace": {"attempts": []},
            "requested_action_context_mode": str(action_context_mode or "auto").strip().lower(),
            "action_context_mode": resolved_mode,
            "action_context_stats": action_context_stats,
        }

    payload = {
        "REVIEWED_DIAGNOSIS": diagnosis,
        "ACTION_CASE_PACKAGE": action_case_package,
        "DECISION_CONTEXT": normalized_context,
        "OUTPUT_CONSTRAINTS": constraints,
        "TASK": (
            "Generate only independently justified, concrete, traceable product-"
            "improvement actions consistent with the human-reviewed mechanisms "
            "and product-team constraints. Do not aim for a fixed action count."
        ),
    }

    # Deliberately no try/except: API/runtime/provider failures are systemic.
    result = call_json(
        system_prompt=PROSPECTIVE_ACTION_PROMPT,
        user_payload=payload,
        model_name=model_name,
        temperature=float(temperature),
    )
    if not isinstance(result, dict):
        raise TypeError(f"call_json must return a dict, got {type(result).__name__}")

    original_parsed = result.get("parsed")
    if not isinstance(original_parsed, dict):
        error = (
            "call_json returned no parsed JSON object; "
            f"parse_error={result.get('parse_error')!r}"
        )
        raise ActionGenerationTraceError(
            error,
            llm_trace=_action_attempt_trace(
                result=result,
                parsed=None,
                error=error,
            ),
        )

    try:
        salvage = salvage_prospective_action_output(
            original_parsed,
            diagnosis=diagnosis,
            case_package=action_case_package,
            decision_context=normalized_context,
            status_by_mechanism=status_by_mechanism,
            constraints=constraints,
        )
    except Exception as exc:
        raise ActionGenerationTraceError(
            str(exc),
            llm_trace=_action_attempt_trace(
                result=result,
                parsed=copy.deepcopy(original_parsed),
                error=str(exc),
            ),
        ) from exc

    result["parsed"] = salvage["parsed"]
    result["original_parsed"] = salvage["original_parsed"]
    result["normalization_notes"] = salvage["normalization_notes"]
    result["discarded_actions"] = salvage["discarded_actions"]
    result["action_id_map"] = salvage["action_id_map"]
    result["n_generated"] = salvage["n_generated"]
    result["n_valid"] = salvage["n_valid"]
    result["n_discarded"] = salvage["n_discarded"]
    result["generation_status"] = salvage["generation_status"]

    # The LLM trace records what the model actually returned; accepted/sanitized
    # output is stored separately in result["parsed"].
    result["llm_trace"] = _action_attempt_trace(
        result=result,
        parsed=copy.deepcopy(original_parsed),
        error=None,
    )
    result["requested_action_context_mode"] = str(
        action_context_mode or "auto"
    ).strip().lower()
    result["action_context_mode"] = resolved_mode
    result["action_context_stats"] = action_context_stats
    return result

def generate_action_subfamily_predictions(
    *,
    diagnosis: Dict[str, Any],
    case_package: Dict[str, Any],
    model_name: Optional[str] = None,
    temperature: float = 0.2,
    action_context_mode: str = "auto",
) -> Dict[str, Any]:
    """Predict ranked action subfamilies with production H_AB routing by default."""
    if not isinstance(diagnosis, dict):
        raise TypeError("diagnosis must be a dict")
    if not isinstance(case_package, dict):
        raise TypeError(
            f"case_package must be a dict, got {type(case_package).__name__}"
        )
    if (
        not isinstance(temperature, (int, float))
        or isinstance(temperature, bool)
    ):
        raise TypeError("temperature must be numeric")

    resolved_mode = resolve_action_context_mode(
        case_package=case_package,
        action_context_mode=action_context_mode,
    )
    action_case_package = build_routed_action_case_package(
        case_package=case_package,
        action_context_mode=resolved_mode,
    )
    compact_case_package = build_compact_action_case_package(
        case_package=action_case_package,
        diagnosis=diagnosis,
    )
    action_context_stats = summarize_action_case_view(
        compact_case_package
    )

    constraints = _build_action_constraints(
        diagnosis=diagnosis,
        case_package=action_case_package,
    )
    eligible_mechanism_ids = set(
        constraints["eligible_mechanism_ids"]
    )

    if not eligible_mechanism_ids:
        return {
            "parsed": {
                "case_id": constraints["case_id"],
                "product_category": constraints["product_category"],
                "actions": [],
            },
            "raw_output": "",
            "generation_status": "no_eligible_mechanism",
            "usage": {
                "prompt_tokens": 0,
                "cached_prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "llm_trace": {"attempts": []},
            "requested_action_context_mode": str(
                action_context_mode or "auto"
            ).strip().lower(),
            "action_context_mode": resolved_mode,
            "action_context_stats": action_context_stats,
        }

    compact_diagnosis = {
        "case_id": diagnosis.get("case_id"),
        "product_category": diagnosis.get("product_category"),
        "risk_dimensions": diagnosis.get("risk_dimensions", []),
        "symptom_families": diagnosis.get("symptom_families", []),
        "claims": diagnosis.get("claims", []),
        "mechanisms": diagnosis.get("mechanisms", []),
        "validation_tasks": [
            {
                "validation_id": x.get("validation_id"),
                "linked_mechanism_ids": x.get("linked_mechanism_ids") or [],
                "task": x.get("task"),
            }
            for x in (diagnosis.get("validation_tasks") or [])
            if isinstance(x, dict)
        ],
        "abstentions": diagnosis.get("abstentions", []),
    }

    # Subfamily-only output contains no source_refs.  Keep the deterministic
    # mechanism/taxonomy contract, but do not expose unnecessary reference maps.
    subfamily_constraints = {
        "case_id": constraints["case_id"],
        "product_category": constraints["product_category"],
        "eligible_mechanism_ids": constraints["eligible_mechanism_ids"],
        "allowed_action_subfamilies": constraints["allowed_action_subfamilies"],
    }

    payload = {
        "DIAGNOSIS": compact_diagnosis,
        "CASE_PACKAGE": compact_case_package,
        "OUTPUT_CONSTRAINTS": subfamily_constraints,
        "TASK": "Predict ranked prospective action_subfamilies for this case.",
    }

    result = call_json(
        system_prompt=ACTION_SUBFAMILY_PREDICTION_PROMPT,
        user_payload=payload,
        model_name=model_name,
        temperature=float(temperature),
    )
    if not isinstance(result, dict):
        raise TypeError("call_json must return a dict")

    obj = result.get("parsed")
    if not isinstance(obj, dict):
        error = (
            "call_json returned no parsed JSON object; "
            f"parse_error={result.get('parse_error')!r}"
        )
        raise ActionGenerationTraceError(
            error,
            llm_trace=_action_attempt_trace(
                result=result,
                parsed=None,
                error=error,
            ),
        )

    try:
        raw_predictions = obj.get("actions")
        if not isinstance(raw_predictions, list):
            raise ValueError("root.actions must be a list")

        actions: List[Dict[str, Any]] = []
        seen_subfamilies: Set[str] = set()
        linkage_warnings: List[str] = []

        for i, raw in enumerate(raw_predictions[:5], start=1):
            if not isinstance(raw, dict):
                raise ValueError(
                    f"root.actions[{i - 1}] must be an object"
                )

            subfamily = str(
                raw.get("action_subfamily") or ""
            ).strip()
            if not subfamily:
                raise ValueError(
                    f"root.actions[{i - 1}].action_subfamily "
                    "must be a non-empty string"
                )
            if subfamily in seen_subfamilies:
                continue
            seen_subfamilies.add(subfamily)

            raw_linked = raw.get("linked_mechanism_ids")
            if isinstance(raw_linked, str):
                linked = [
                    raw_linked.strip()
                ] if raw_linked.strip() else []
            elif isinstance(raw_linked, list):
                linked = [
                    str(x).strip()
                    for x in raw_linked
                    if str(x).strip()
                ]
            else:
                linked = []
            linked = list(dict.fromkeys(linked))

            invalid_mids = [
                mid
                for mid in linked
                if mid not in eligible_mechanism_ids
            ]
            if invalid_mids:
                linkage_warnings.append(
                    f"prediction={subfamily!r}: invalid linked "
                    f"mechanism IDs {invalid_mids}"
                )

            # Keep invalid subfamily labels in ranking so evaluator cannot gain
            # artificially from silently dropping an invalid prediction.
            actions.append({
                "priority_rank": len(actions) + 1,
                "action_subfamily": subfamily,
                "linked_mechanism_ids": linked,
            })

        parsed = {
            "case_id": constraints["case_id"],
            "product_category": constraints["product_category"],
            "actions": actions,
        }
    except Exception as exc:
        raise ActionGenerationTraceError(
            str(exc),
            llm_trace=_action_attempt_trace(
                result=result,
                parsed=obj,
                error=str(exc),
            ),
        ) from exc

    return {
        "parsed": parsed,
        "raw_output": result.get("raw_output", ""),
        "generation_status": "generated",
        "linkage_warnings": linkage_warnings,
        "usage": result.get("usage", {}) if isinstance(result, dict) else {},
        "llm_trace": _action_attempt_trace(
            result=result,
            parsed=parsed,
            error=None,
        ),
        "requested_action_context_mode": str(
            action_context_mode or "auto"
        ).strip().lower(),
        "action_context_mode": resolved_mode,
        "action_context_stats": action_context_stats,
    }

        
   
        
__all__ = [
    "ACTION_SUBFAMILIES",
    "ACTION_LEVELS",
    "ACTION_EXPERIMENT_PROMPT",
    "PROSPECTIVE_ACTION_PROMPT",
    "PROSPECTIVE_ACTION_LAYERS",
    "ACTION_SUBFAMILY_PREDICTION_PROMPT",
    "ACTION_CONTEXT_MODES",
    "DEFAULT_H_ACTION_CONTEXT_MODE",
    "ACTION_ROUTING_VERSION",
    "ActionGenerationTraceError",
    "validate_experimental_action_output",
    "generate_experimental_actions",
    "generate_prospective_actions",
    "salvage_prospective_action_output",
    "generate_action_subfamily_predictions",
    "build_compact_action_case_package",
    "build_routed_action_case_package",
    "build_h_action_case_view",
    "resolve_action_context_mode",
    "summarize_action_case_view",
    "mechanism_linked_reference_ids",
]
