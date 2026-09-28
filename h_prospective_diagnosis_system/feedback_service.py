from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

try:
    import streamlit as st
except Exception:  # Allows non-UI services/tests to import this module.
    st = None


DEFAULT_LAYERS = [
    "app_or_configuration",
    "cloud_service",
    "firmware",
    "alexa_skill_or_api_implementation",
    "support_or_user_guidance",
    "hardware",
    "ecosystem_governance",
]

MECHANISM_REVIEW_STATUSES = [
    "not_assessed",
    "supported",
    "plausible",
    "uncertain",
    "rejected",
]

ACTION_DECISIONS = [
    "not_assessed",
    "accept",
    "revise",
    "defer",
    "reject",
]


# ============================================================
# Pure feedback application helpers
# ============================================================

def apply_mechanism_feedback(
    diagnosis: Dict[str, Any],
    feedback: Dict[str, Any],
) -> Dict[str, Any]:
    """Attach human epistemic review without overwriting the original diagnosis."""
    reviewed = copy.deepcopy(diagnosis)
    rows = feedback.get("mechanism_feedback") if isinstance(feedback, dict) else []
    by_id = {
        str(x.get("mechanism_id")): x
        for x in (rows or [])
        if isinstance(x, dict) and x.get("mechanism_id")
    }

    for mech in reviewed.get("mechanisms") or []:
        if not isinstance(mech, dict):
            continue
        mid = str(mech.get("mechanism_id") or "")
        fb = by_id.get(mid, {})
        status = str(fb.get("assessment_status") or "not_assessed").strip().lower()
        if status not in MECHANISM_REVIEW_STATUSES:
            status = "not_assessed"
        mech["human_assessment"] = status
        mech["human_revision"] = str(fb.get("revision") or "").strip()
        mech["human_validation_note"] = str(fb.get("validation_note") or "").strip()

    reviewed["respondent_profile"] = copy.deepcopy(
        feedback.get("respondent_profile") if isinstance(feedback, dict) else {}
    )
    return reviewed


def mechanism_review_summary(reviewed_diagnosis: Dict[str, Any]) -> Dict[str, int]:
    counts = {k: 0 for k in MECHANISM_REVIEW_STATUSES}
    for mech in reviewed_diagnosis.get("mechanisms") or []:
        if not isinstance(mech, dict):
            continue
        status = str(mech.get("human_assessment") or "not_assessed").strip().lower()
        if status not in counts:
            status = "not_assessed"
        counts[status] += 1
    return counts


def apply_action_feedback(
    action_output: Dict[str, Any],
    feedback: Dict[str, Any],
) -> Dict[str, Any]:
    """Attach product-team decisions and expose reviewed action portfolios.

    Decision semantics:
    - accept: accept the original generated action; revision must be empty.
    - revise: accept with human revision; revision must be non-empty.
    - defer: retain as a deferred candidate, but not as an approved action.
    - reject: reject the action; management value is not assessed.
    - not_assessed: no final Gate-2 decision.

    No second LLM rewrite is performed. Human revision is used as final_action
    only when decision == "revise".
    """
    reviewed = copy.deepcopy(action_output)

    rows = (
        feedback.get("action_feedback")
        if isinstance(feedback, dict)
        else []
    )

    by_id = {
        str(x.get("action_id") or "").strip(): x
        for x in (rows or [])
        if isinstance(x, dict)
        and str(x.get("action_id") or "").strip()
    }

    allowed_management_values = {
        "not_assessed",
        "high",
        "medium",
        "low",
    }

    approved: List[Dict[str, Any]] = []
    deferred: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for action in reviewed.get("actions") or []:
        if not isinstance(action, dict):
            continue

        aid = str(action.get("action_id") or "").strip()
        original_action = str(action.get("action") or "").strip()
        fb = by_id.get(aid, {})

        # --------------------------------------------------
        # 1. Final decision
        # --------------------------------------------------
        decision = str(
            fb.get("decision") or "not_assessed"
        ).strip().lower()

        if decision not in ACTION_DECISIONS:
            decision = "not_assessed"

        # --------------------------------------------------
        # 2. Human revision
        # --------------------------------------------------
        revision = str(
            fb.get("revision") or ""
        ).strip()

        # Gate-2 semantic consistency:
        # accept = original action accepted as-is
        # revise = accepted with revision
        if decision == "accept" and revision:
            raise ValueError(
                f"{aid}: decision='accept' requires an empty revision. "
                "Use decision='revise' for an accepted revised action."
            )

        if decision == "revise" and not revision:
            raise ValueError(
                f"{aid}: decision='revise' requires a non-empty revision."
            )

        # --------------------------------------------------
        # 3. Management value
        # --------------------------------------------------
        management_value = str(
            fb.get("management_value") or "not_assessed"
        ).strip().lower()

        if management_value not in allowed_management_values:
            management_value = "not_assessed"

        # Rejected / unassessed actions do not receive a management-value
        # rating in the final reviewed portfolio.
        if decision in {"reject", "not_assessed"}:
            management_value = "not_assessed"

        # --------------------------------------------------
        # 4. Persist reviewed fields
        # --------------------------------------------------
        action["human_decision"] = decision
        action["management_value"] = management_value
        action["human_revision"] = revision
        action["human_decision_note"] = str(
            fb.get("decision_note") or ""
        ).strip()

        if decision == "revise":
            action["final_action"] = revision
        else:
            action["final_action"] = original_action

        # --------------------------------------------------
        # 5. Route reviewed action
        # --------------------------------------------------
        if decision in {"accept", "revise"}:
            approved.append(copy.deepcopy(action))

        elif decision == "defer":
            deferred.append(copy.deepcopy(action))

        elif decision == "reject":
            rejected.append(copy.deepcopy(action))

    reviewed["approved_actions"] = approved
    reviewed["deferred_actions"] = deferred
    reviewed["rejected_actions"] = rejected

    reviewed["respondent_profile"] = copy.deepcopy(
        feedback.get("respondent_profile")
        if isinstance(feedback, dict)
        else {}
    )

    return reviewed


# ============================================================
# Streamlit rendering helpers
# ============================================================

def _require_streamlit() -> None:
    if st is None:
        raise RuntimeError("Streamlit is required for UI rendering functions")


def render_respondent_profile(key_prefix: str = "respondent") -> Dict[str, Any]:
    _require_streamlit()
    with st.expander("人工审核者信息", expanded=False):
        role = st.selectbox(
            "角色",
            ["product_practitioner", "software_engineer", "domain_expert", "researcher", "other"],
            index=0,
            key=f"{key_prefix}_role",
        )
        smart_home = st.slider("智能产品/生态熟悉度", 1, 5, 3, key=f"{key_prefix}_smart_home")
        technical = st.slider("技术实现熟悉度", 1, 5, 3, key=f"{key_prefix}_technical")
    return {
        "respondent_role": role,
        "self_reported_expertise": {
            "smart_product_ecosystem": smart_home,
            "technical": technical,
        },
        "feedback_source": "prospective_human_review",
    }


def render_mechanism_feedback_form(
    diagnosis: Dict[str, Any],
    respondent_profile: Dict[str, Any],
    key_prefix: str = "mechanism_feedback",
) -> Optional[Dict[str, Any]]:
    _require_streamlit()
    mechanisms = [m for m in (diagnosis.get("mechanisms") or []) if isinstance(m, dict)]
    if not mechanisms:
        st.info("当前没有足够证据支持机制假设；保留 diagnosis 中的 validation_tasks 作为下一步。")
        return {
            "case_id": diagnosis.get("case_id"),
            "product_category": diagnosis.get("product_category"),
            "respondent_profile": respondent_profile,
            "mechanism_feedback": [],
        }

    with st.form(f"{key_prefix}_{diagnosis.get('case_id', 'case')}"):
        rows: List[Dict[str, Any]] = []
        for i, mech in enumerate(mechanisms, start=1):
            mid = str(mech.get("mechanism_id") or f"M{i}")
            st.markdown(f"**{mid} · {mech.get('mechanism_family', 'unclear')}**")
            st.write(mech.get("statement", ""))
            st.caption(
                f"Confidence: {mech.get('confidence', '')} | "
                f"Sources: {', '.join(mech.get('source_refs') or []) or '—'}"
            )
            if mech.get("main_uncertainty"):
                st.caption(f"主要不确定性: {mech.get('main_uncertainty')}")

            assessment = st.selectbox(
                "机制判断",
                MECHANISM_REVIEW_STATUSES,
                index=0,
                key=f"{key_prefix}_{mid}_assessment",
            )
            revision = st.text_input(
                "必要时修订机制表述",
                key=f"{key_prefix}_{mid}_revision",
            )
            validation_note = st.text_input(
                "必要时补充验证说明",
                key=f"{key_prefix}_{mid}_validation",
            )
            rows.append({
                "mechanism_id": mid,
                "assessment_status": assessment,
                "revision": revision,
                "validation_note": validation_note,
            })
            st.divider()
        submitted = st.form_submit_button("保存机制审核")

    if not submitted:
        return None
    return {
        "case_id": diagnosis.get("case_id"),
        "product_category": diagnosis.get("product_category"),
        "respondent_profile": respondent_profile,
        "mechanism_feedback": rows,
    }


def render_decision_context(
    product_category: str,
    key_prefix: str = "context",
) -> Dict[str, Any]:
    """Collect product-team constraints for downstream action generation.

    Hardware permission is controlled by a single checkbox. The hardware layer
    is excluded from the general layer selector and is added deterministically
    when hardware changes are allowed.
    """
    _require_streamlit()

    with st.expander("Action generation constraints", expanded=False):
        objective = st.text_input(
            "Improvement objective",
            value="reduce prospective user-experience risk and ecosystem propagation risk",
            key=f"{key_prefix}_{product_category}_objective",
        )

        non_hardware_layers = [
            layer for layer in DEFAULT_LAYERS
            if layer != "hardware"
        ]

        default_layers = [
            "app_or_configuration",
            "cloud_service",
            "firmware",
            "alexa_skill_or_api_implementation",
            "support_or_user_guidance",
            "ecosystem_governance",
        ]
        default_layers = [
            layer for layer in default_layers
            if layer in non_hardware_layers
        ]

        allowed_layers = st.multiselect(
            "Allowed intervention layers",
            non_hardware_layers,
            default=default_layers,
            key=f"{key_prefix}_{product_category}_layers",
        )

        allow_hardware = st.checkbox(
            "Allow hardware changes",
            value=False,
            key=f"{key_prefix}_{product_category}_hardware",
        )

        allow_third_party = st.checkbox(
            "Allow actions requiring third-party changes",
            value=False,
            key=f"{key_prefix}_{product_category}_third_party",
        )

    final_layers = list(allowed_layers)
    if allow_hardware and "hardware" not in final_layers:
        final_layers.append("hardware")

    return {
        "context_source": "product_team_input",
        "product_category": product_category,
        "objective": objective,
        "allowed_layers": final_layers,
        "allow_hardware_change": bool(allow_hardware),
        "allow_third_party_change": bool(allow_third_party),
    }


def render_action_feedback_form(
    action_output: Dict[str, Any],
    respondent_profile: Optional[Dict[str, Any]] = None,
    key_prefix: str = "action_feedback",
) -> Optional[Dict[str, Any]]:
    _require_streamlit()
    actions = [a for a in (action_output.get("actions") or []) if isinstance(a, dict)]
    if not actions:
        st.info("当前没有生成直接改进行动。")
        return {
            "case_id": action_output.get("case_id"),
            "respondent_profile": respondent_profile or {},
            "action_feedback": [],
        }

    with st.form(f"{key_prefix}_{action_output.get('case_id', 'case')}"):
        rows: List[Dict[str, Any]] = []
        for i, action in enumerate(actions, start=1):
            aid = str(action.get("action_id") or f"A{i}")
            st.markdown(f"**{aid} · {action.get('action_subfamily', '')}**")
            if action.get("improvement_direction"):
                st.caption(f"改进方向: {action.get('improvement_direction')}")
            st.write(action.get("action", ""))
            if action.get("intervention_logic"):
                st.caption(f"行动逻辑: {action.get('intervention_logic')}")
            st.caption(
                f"Level: {action.get('action_level', '')} | "
                f"Layers: {', '.join(action.get('target_layers') or []) or '—'} | "
                f"Validation: {action.get('validation_metric', '') or '—'}"
            )

            decision = st.selectbox(
                "行动判断",
                ACTION_DECISIONS,
                index=0,
                key=f"{key_prefix}_{aid}_decision",
            )
            expected_value = st.selectbox(
                "预期价值",
                ["not_assessed", "high", "medium", "low"],
                index=0,
                key=f"{key_prefix}_{aid}_value",
            )
            feasibility = st.selectbox(
                "可行性",
                ["not_assessed", "high", "medium", "low"],
                index=0,
                key=f"{key_prefix}_{aid}_feasibility",
            )
            revision = st.text_input(
                "必要时修订行动",
                key=f"{key_prefix}_{aid}_revision",
            )
            decision_note = st.text_input(
                "决策说明（可选）",
                key=f"{key_prefix}_{aid}_note",
            )
            rows.append({
                "action_id": aid,
                "decision": decision,
                "expected_value": expected_value,
                "feasibility": feasibility,
                "revision": revision,
                "decision_note": decision_note,
            })
            st.divider()
        submitted = st.form_submit_button("保存行动审核")

    if not submitted:
        return None
    return {
        "case_id": action_output.get("case_id"),
        "respondent_profile": respondent_profile or {},
        "action_feedback": rows,
    }



def build_evidence_index(
    case_package: dict,
) -> dict:
    index = {}

    for row in case_package.get(
        "observed_evidence", []
    ):
        if isinstance(row, dict):
            ref = str(
                row.get("evidence_id") or ""
            ).strip()
            if ref:
                index[ref] = {
                    "source_type": "observed",
                    "section": "observed_evidence",
                    "data": row,
                }

    for row in case_package.get(
        "ecosystem_relations", []
    ):
        if isinstance(row, dict):
            ref = str(
                row.get("relation_id") or ""
            ).strip()
            if ref:
                index[ref] = {
                    "source_type": "relation",
                    "section": "ecosystem_relations",
                    "data": row,
                }

    for row in case_package.get(
        "api_knowledge", []
    ):
        if isinstance(row, dict):
            ref = str(
                row.get("knowledge_id") or ""
            ).strip()
            if ref:
                index[ref] = {
                    "source_type": "api_knowledge",
                    "section": "api_knowledge",
                    "data": row,
                }

    for row in case_package.get(
        "model_facts", []
    ):
        if isinstance(row, dict):
            ref = str(
                row.get("fact_id") or ""
            ).strip()
            if ref:
                index[ref] = {
                    "source_type": "model_fact",
                    "section": "model_facts",
                    "data": row,
                }

    for row in case_package.get(
        "risk_chains", []
    ):
        if isinstance(row, dict):
            ref = str(
                row.get("chain_id") or ""
            ).strip()
            if ref:
                index[ref] = {
                    "source_type": "risk_chain",
                    "section": "risk_chains",
                    "data": row,
                }

    return index

__all__ = [
    "DEFAULT_LAYERS",
    "MECHANISM_REVIEW_STATUSES",
    "ACTION_DECISIONS",
    "apply_mechanism_feedback",
    "mechanism_review_summary",
    "apply_action_feedback",
    "render_respondent_profile",
    "render_mechanism_feedback_form",
    "render_decision_context",
    "render_action_feedback_form",
]
