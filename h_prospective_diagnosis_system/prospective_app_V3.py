from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()


from typing import Any, Dict, List, Optional

import streamlit as st

from feedback_service import (
    render_decision_context,
    render_respondent_profile,
)
from prospective_service import (
    STATUS_BLOCKED_ACTION_OUTPUT_REVIEW,
    STATUS_BLOCKED_OUTPUT_REVIEW,
    STATUS_COMPLETED,
    STATUS_NO_ACTION,
    STATUS_READY_FOR_ACTION,
    STATUS_WAITING_ACTION_REVIEW,
    STATUS_WAITING_MECHANISM_REVIEW,
    finalize_product_decision,
    generate_product_actions_after_review,
    generate_topk_ecosystem_synthesis,
    load_prospective_data,
    load_risk_selection_bundle,
    record_mechanism_review,
    run_h_from_risk_selection,
    reset_blocked_action_for_retry,
    retry_blocked_action_generation,
)
from workflow_store import list_workflows


# ---------------------------------------------------------------------------
# Workflow/session helpers
# ---------------------------------------------------------------------------


def _case_key(record: Dict[str, Any]) -> str:
    return str(
        record.get("workflow_id")
        or (record.get("case_meta") or {}).get("case_id")
        or (record.get("requested_case") or {}).get("product_category")
        or "case"
    )


def _replace_record(updated: Dict[str, Any]) -> None:
    records = list(st.session_state.get("h_prospective_records") or [])
    key = _case_key(updated)
    replaced = False
    for i, record in enumerate(records):
        if _case_key(record) == key:
            records[i] = updated
            replaced = True
            break
    if not replaced:
        records.append(updated)
    st.session_state["h_prospective_records"] = records


def _require_selection_bundle(path: str) -> Dict[str, Any]:
    bundle = load_risk_selection_bundle(path)
    if not isinstance(bundle, dict):
        raise TypeError("load_risk_selection_bundle() must return a dict")
    if not isinstance(bundle.get("risk_selection"), dict):
        raise TypeError("selection bundle is missing dict risk_selection")

    cutoff = str(bundle.get("analysis_cutoff") or "").strip()
    target = str(bundle.get("target_month") or "").strip()
    if not cutoff or not target:
        raise ValueError(
            "risk selection artifact must contain, or allow unique inference of, "
            "analysis_cutoff and target_month"
        )
    return bundle


def _load_current_context(
    *,
    data_dir: str,
    selection_bundle: Dict[str, Any],
) -> Dict[str, Any]:
    return load_prospective_data(
        data_dir,
        expected_analysis_cutoff=str(selection_bundle["analysis_cutoff"]),
        expected_target_month=str(selection_bundle["target_month"]),
    )


# ---------------------------------------------------------------------------
# Evidence resolution
# ---------------------------------------------------------------------------


def build_evidence_index(case_package: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Resolve stable source IDs to the full audit evidence stored in a case package.

    Detailed OBS cards take precedence over the compact OBS catalog. Other
    evidence classes are indexed by their native stable ID without changing the
    underlying workflow/audit representation.
    """
    if not isinstance(case_package, dict):
        return {}

    index: Dict[str, Dict[str, Any]] = {}

    def add_rows(
        section: str,
        id_field: str,
        evidence_type: str,
        label: str,
        *,
        overwrite: bool = True,
    ) -> None:
        for row in case_package.get(section, []) or []:
            if not isinstance(row, dict):
                continue
            ref = str(row.get(id_field) or "").strip()
            if not ref:
                continue
            if overwrite or ref not in index:
                index[ref] = {
                    "type": evidence_type,
                    "label": label,
                    "section": section,
                    "data": row,
                }

    # Detailed observed evidence is human-review facing; catalog is fallback.
    add_rows(
        "observed_evidence",
        "evidence_id",
        "OBS",
        "Observed review evidence",
        overwrite=True,
    )
    add_rows(
        "observed_dimension_catalog",
        "evidence_id",
        "OBS",
        "Observed dimension evidence",
        overwrite=False,
    )
    add_rows(
        "ecosystem_relations",
        "relation_id",
        "REL",
        "Static ecosystem relation",
    )
    add_rows(
        "api_knowledge",
        "knowledge_id",
        "KB",
        "API / capability knowledge",
    )
    add_rows(
        "model_facts",
        "fact_id",
        "MF",
        "Dynamic model fact",
    )
    add_rows(
        "risk_chains",
        "chain_id",
        "CHN",
        "Model-inferred risk chain",
    )
    return index


def resolve_mechanism_evidence(
    mechanism: Dict[str, Any],
    diagnosis: Dict[str, Any],
    case_package: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Return direct mechanism refs plus refs inherited through supporting claims.

    When the same source is referenced both directly and through one or more
    supporting claims, all provenance routes are retained instead of silently
    dropping the later route.
    """
    evidence_index = build_evidence_index(case_package)
    claims_by_id = {
        str(c.get("claim_id")): c
        for c in diagnosis.get("claims", []) or []
        if isinstance(c, dict) and c.get("claim_id")
    }

    ref_routes: Dict[str, List[str]] = {}

    def add_ref(raw_ref: Any, via: str) -> None:
        ref = str(raw_ref or "").strip()
        if not ref:
            return
        routes = ref_routes.setdefault(ref, [])
        if via not in routes:
            routes.append(via)

    for ref in mechanism.get("source_refs", []) or []:
        add_ref(ref, "direct")

    for claim_id in mechanism.get("supporting_claim_ids", []) or []:
        cid = str(claim_id)
        claim = claims_by_id.get(cid)
        if not isinstance(claim, dict):
            continue
        for ref in claim.get("source_refs", []) or []:
            add_ref(ref, f"claim:{cid}")

    return [
        {
            "ref_id": ref,
            "via": routes,
            "evidence": evidence_index.get(ref),
        }
        for ref, routes in ref_routes.items()
    ]


# ---------------------------------------------------------------------------
# Human-readable evidence rendering
# ---------------------------------------------------------------------------


def _metric_value(value: Any, *, digits: int = 3) -> Any:
    if value is None or value == "":
        return "—"
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return str(value)


def _render_string_list(title: str, values: Any, *, limit: int = 8) -> None:
    rows = values if isinstance(values, list) else []
    rows = [x for x in rows if x not in (None, "")][:limit]
    if not rows:
        return
    st.markdown(f"**{title}**")
    for item in rows:
        if isinstance(item, dict):
            st.json(item)
        else:
            st.markdown(f"- {item}")


def _render_obs(row: Dict[str, Any]) -> None:
    st.write("**Dimension:**", row.get("dimension") or "—")
    severity = row.get("severity") if isinstance(row.get("severity"), dict) else {}

    n_review = severity.get("n_review", row.get("n_review"))
    mean_sentiment = severity.get("mean_sentiment", row.get("mean_sentiment"))
    recent_review = severity.get(
        "recent_review_count",
        row.get("recent_review_count"),
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Reviews", n_review if n_review is not None else "—")
    c2.metric("Mean sentiment", _metric_value(mean_sentiment))
    c3.metric(
        "Recent reviews",
        recent_review if recent_review is not None else "—",
    )

    selection = row.get("evidence_selection")
    if isinstance(selection, dict):
        st.caption(
            "Evidence selection: "
            f"rank={selection.get('rank', '—')} | "
            f"utility={_metric_value(selection.get('evidence_utility'))}"
        )

    statements = row.get("representative_statements") or []
    if statements:
        st.markdown("**Representative statements**")
        for text in statements[:8]:
            st.markdown(f"- {text}")

    failures = row.get("top_failure_types") or []
    if failures:
        st.markdown("**Top failure types**")
        for item in failures[:6]:
            if isinstance(item, dict):
                share = item.get("share")
                try:
                    share_text = "" if share is None else f" ({_metric_value(100 * float(share), digits=1)}%)"
                except (TypeError, ValueError):
                    share_text = ""
                st.markdown(
                    f"- {item.get('failure_type', '—')}: "
                    f"{item.get('count', '—')}{share_text}"
                )

    scenarios = row.get("top_scenarios") or []
    if scenarios:
        st.markdown("**Top scenarios**")
        for item in scenarios[:5]:
            if isinstance(item, dict):
                st.markdown(
                    f"- {item.get('scenario', '—')}: {item.get('count', '—')}"
                )


def _render_kb(row: Dict[str, Any]) -> None:
    st.info(
        "Candidate mechanism knowledge; it can support a hypothesis but is not "
        "proof that this product contains the implementation defect."
    )
    title = (
        row.get("api_name")
        or row.get("capability_name")
        or row.get("name")
        or row.get("knowledge_type")
        or "API knowledge"
    )
    st.write("**Knowledge item:**", title)

    description = row.get("description") or row.get("summary")
    if description:
        st.write(description)

    _render_string_list("Design constraints", row.get("design_constraints"), limit=8)
    _render_string_list("Common root causes", row.get("common_root_causes"), limit=8)
    _render_string_list("Diagnostic keywords", row.get("diagnostic_keywords"), limit=10)

    dependencies = row.get("dependencies") or []
    if dependencies:
        st.markdown("**Dependencies / related capabilities**")
        for dep in dependencies[:6]:
            if isinstance(dep, dict):
                name = dep.get("api_name") or dep.get("name") or "—"
                relation = dep.get("relation_type") or ""
                trigger = dep.get("trigger_condition") or ""
                line = f"- {name}"
                if relation:
                    line += f" · {relation}"
                if trigger:
                    line += f" — {trigger}"
                st.markdown(line)
            else:
                st.markdown(f"- {dep}")

    note = row.get("notes_for_root_cause_mapping")
    if note:
        st.markdown("**Root-cause mapping note**")
        st.write(note)

    st.caption(
        f"Retrieval score={_metric_value(row.get('retrieval_score'))} | "
        f"basis={row.get('retrieval_basis') or '—'}"
    )


def _render_mf(row: Dict[str, Any]) -> None:
    st.info("Model-inferred forecast/context; not observed evidence or causal proof.")
    st.write("**Fact type:**", row.get("fact_type") or "—")

    dim = row.get("dimension")
    if dim:
        st.write("**Dimension:**", dim)

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Predicted next risk",
        _metric_value(
            row.get("predicted_next_risk", row.get("predicted_next_risk_score"))
        ),
    )
    c2.metric(
        "Spillover output",
        _metric_value(
            row.get("spillover_output", row.get("spillover_output_score"))
        ),
    )
    c3.metric(
        "Spillover exposure",
        _metric_value(
            row.get("spillover_exposure", row.get("spillover_exposure_score"))
        ),
    )

    if row.get("persistence") is not None or row.get("persistence_score") is not None:
        st.write(
            "**Persistence:**",
            row.get("persistence") or "—",
            "| score=",
            _metric_value(row.get("persistence_score")),
        )

    driver_parts = []
    for field, label in (
        ("beta_api", "api"),
        ("beta_coo", "co-occurrence"),
        ("beta_user", "user"),
    ):
        if row.get(field) is not None:
            driver_parts.append(f"{label}={_metric_value(row.get(field))}")
    if driver_parts or row.get("dominant_driver"):
        st.write(
            "**Network drivers:**",
            ", ".join(driver_parts) or "—",
            "| dominant=",
            row.get("dominant_driver") or "—",
        )

    top_dims = row.get("top_predicted_risk_dimensions") or []
    if top_dims:
        st.markdown("**Top predicted risk dimensions**")
        for item in top_dims[:5]:
            if isinstance(item, dict):
                st.markdown(
                    f"- {item.get('dimension', '—')}: "
                    f"risk={_metric_value(item.get('predicted_next_risk'))}, "
                    f"driver={item.get('dominant_driver') or '—'}"
                )

    st.caption(
        f"analysis_cutoff={row.get('analysis_cutoff') or '—'} | "
        f"target_month={row.get('target_month') or '—'}"
    )


def _render_chain(row: Dict[str, Any]) -> None:
    st.info("Model-inferred risk chain; it is a propagation hypothesis, not causal proof.")
    st.write("**Path:**", row.get("path_text") or "—")
    st.write(
        "**Endpoints:**",
        f"{row.get('source_category') or '—'} → {row.get('target_category') or '—'}",
    )

    dims = row.get("matched_dimensions") or []
    if not dims and row.get("dimension"):
        dims = [row.get("dimension")]
    if dims:
        st.write("**Matched dimensions:**", dims)

    c1, c2, c3 = st.columns(3)
    c1.metric("Chain score", _metric_value(row.get("chain_score")))
    c2.metric("Source risk", _metric_value(row.get("source_risk_score", row.get("source_risk_raw"))))
    c3.metric("Target risk", _metric_value(row.get("target_risk_score", row.get("target_risk_raw"))))

    retrieval = row.get("retrieval_selection")
    if isinstance(retrieval, dict):
        st.caption(
            "Retrieval: "
            f"mode={retrieval.get('selection_mode') or '—'} | "
            f"rank={retrieval.get('retrieval_rank', '—')} | "
            f"order={retrieval.get('selection_order', '—')}"
        )


def _render_relation(row: Dict[str, Any]) -> None:
    st.info("Static ecosystem association; not causal propagation evidence.")
    st.write("**Related product:**", row.get("related_product") or "—")
    st.metric("Overall relation strength", _metric_value(row.get("overall_relation_strength")))

    channels = row.get("channel_strengths")
    if isinstance(channels, dict) and channels:
        display = []
        for name in ("api", "coo", "user"):
            obj = channels.get(name)
            if not isinstance(obj, dict):
                continue
            value = obj.get("undirected_strength")
            if value is None:
                value = obj.get("focal_to_other")
            display.append((name, value))
        if display:
            cols = st.columns(len(display))
            for col, (name, value) in zip(cols, display):
                col.metric(name, _metric_value(value))

    note = row.get("direction_note")
    if note:
        st.caption(note)


def render_evidence_card(evidence_type: str, row: Dict[str, Any]) -> None:
    """Render a compact human-facing card while retaining raw JSON on demand."""
    if not isinstance(row, dict):
        st.error("Evidence payload is not a JSON object.")
        return

    if evidence_type == "OBS":
        _render_obs(row)
    elif evidence_type == "KB":
        _render_kb(row)
    elif evidence_type == "MF":
        _render_mf(row)
    elif evidence_type == "CHN":
        _render_chain(row)
    elif evidence_type == "REL":
        _render_relation(row)
    else:
        st.warning(f"Unknown evidence type: {evidence_type}")

    status = row.get("epistemic_status")
    if status:
        st.caption(f"Epistemic status: {status}")

    with st.expander("View raw evidence JSON", expanded=False):
        st.json(row)


def _render_resolved_evidence(
    mechanism: Dict[str, Any],
    diagnosis: Dict[str, Any],
    case_package: Dict[str, Any],
    *,
    expanded: bool,
) -> None:
    resolved = resolve_mechanism_evidence(mechanism, diagnosis, case_package)
    with st.expander(f"View supporting evidence ({len(resolved)})", expanded=expanded):
        if not resolved:
            st.warning("The mechanism has no resolvable reference source.")
            return

        for i, item in enumerate(resolved, start=1):
            ref = item["ref_id"]
            evidence = item.get("evidence")
            routes = ", ".join(item.get("via") or []) or "—"

            if evidence is None:
                st.error(f"{ref}: Could not resolve this reference in the case package.")
                continue

            st.markdown(
                f"**{ref} — {evidence['label']}**  \n"
                f"Provenance: `{routes}`"
            )
            render_evidence_card(evidence["type"], evidence["data"])
            if i < len(resolved):
                st.divider()


# ---------------------------------------------------------------------------
# Diagnosis / human-review rendering
# ---------------------------------------------------------------------------


def _render_mechanism_header(mechanism: Dict[str, Any], *, reviewed: bool) -> None:
    mid = mechanism.get("mechanism_id") or "—"
    family = mechanism.get("mechanism_family") or "unclear"
    st.markdown(f"**{mid} · {family}**")
    st.write(mechanism.get("statement") or "")
    st.caption(
        f"Causal status: {mechanism.get('causal_status') or '—'} | "
        f"Confidence: {mechanism.get('confidence') if mechanism.get('confidence') is not None else '—'}"
    )

    uncertainty = mechanism.get("main_uncertainty")
    if uncertainty:
        st.warning(f"Main uncertainty: {uncertainty}")

    if reviewed:
        assessment = mechanism.get("human_assessment") or "not_assessed"
        st.write("**Human assessment:**", assessment)
        revision = str(mechanism.get("human_revision") or "").strip()
        validation_note = str(mechanism.get("human_validation_note") or "").strip()
        if revision:
            st.success(f"Human revision: {revision}")
        if validation_note:
            st.info(f"Human validation note: {validation_note}")


def render_diagnosis_summary(
    diagnosis: Dict[str, Any],
    case_package: Dict[str, Any],
    *,
    reviewed: bool,
) -> None:
    st.markdown("#### Reviewed mechanism diagnosis" if reviewed else "#### Mechanism diagnosis")
    st.write("Risk dimensions:", diagnosis.get("risk_dimensions") or [])
    st.write("Symptoms:", diagnosis.get("symptom_families") or [])

    mechanisms = [
        m for m in (diagnosis.get("mechanisms") or []) if isinstance(m, dict)
    ]
    if not mechanisms:
        st.info("The current diagnosis makes no mechanistic assumptions.")
        return

    for i, mechanism in enumerate(mechanisms, start=1):
        _render_mechanism_header(mechanism, reviewed=reviewed)
        _render_resolved_evidence(
            mechanism,
            diagnosis,
            case_package,
            expanded=False,
        )
        if i < len(mechanisms):
            st.divider()


def render_mechanism_review_form(
    diagnosis: Dict[str, Any],
    case_package: Dict[str, Any],
    respondent_profile: Dict[str, Any],
    *,
    key_prefix: str,
) -> Optional[Dict[str, Any]]:
    """Gate 1 form with evidence resolved inline for each mechanism."""
    mechanisms = [
        m for m in (diagnosis.get("mechanisms") or []) if isinstance(m, dict)
    ]

    if not mechanisms:
        st.info("The current diagnosis makes no mechanistic assumptions. Upon confirmation, it will proceed to a subsequent route with no action branch.")
        if st.button("Confirm no mechanism and save Gate 1", key=f"{key_prefix}_empty_submit"):
            return {
                "case_id": diagnosis.get("case_id"),
                "product_category": diagnosis.get("product_category"),
                "respondent_profile": respondent_profile,
                "mechanism_feedback": [],
            }
        return None

    with st.form(f"{key_prefix}_{diagnosis.get('case_id', 'case')}"):
        st.markdown("### Human Gate 1 · Mechanism review")
        st.caption(
            "Assess each mechanism using the presented evidence. "
            "KB, MF, CHN, and REL cannot independently establish an implementation defect or causal relationship."
        )
        rows: List[Dict[str, Any]] = []

        for i, mechanism in enumerate(mechanisms, start=1):
            mid = str(mechanism.get("mechanism_id") or f"M{i}")
            _render_mechanism_header(mechanism, reviewed=False)
            _render_resolved_evidence(
                mechanism,
                diagnosis,
                case_package,
                expanded=True,
            )

            revision = st.text_area(
                "Mechanism revision (optional)",
                key=f"{key_prefix}_{mid}_revision",
                height=80,
            )
            assessment = st.selectbox(
                "Mechanism judgment",
                ["not_assessed", "supported", "plausible", "uncertain", "rejected"],
                index=0,
                key=f"{key_prefix}_{mid}_assessment",
            )
            st.caption(
                "If revised, assess the revised statement. "
            )
            
            validation_note = st.text_area(
                "Additional validation needed",
                key=f"{key_prefix}_{mid}_validation",
                height=80,
            )
            rows.append(
                {
                    "mechanism_id": mid,
                    "assessment_status": assessment,
                    "revision": revision,
                    "validation_note": validation_note,
                }
            )
            if i < len(mechanisms):
                st.divider()

        submitted = st.form_submit_button("Save mechanism feedback")

    if not submitted:
        return None
    return {
        "case_id": diagnosis.get("case_id"),
        "product_category": diagnosis.get("product_category"),
        "respondent_profile": respondent_profile,
        "mechanism_feedback": rows,
    }


def _reviewed_mechanisms_by_id(reviewed_diagnosis: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(m.get("mechanism_id")): m
        for m in (reviewed_diagnosis.get("mechanisms") or [])
        if isinstance(m, dict) and m.get("mechanism_id")
    }


def resolve_action_support(
    action: Dict[str, Any],
    reviewed_diagnosis: Dict[str, Any],
    case_package: Dict[str, Any],
) -> Dict[str, Any]:
    """Resolve one action to reviewed mechanisms and human-readable evidence.
    The action service already validates IDs.  
    This function is display/audit only: it never invents or repairs a reference.
    """
    evidence_index = build_evidence_index(case_package)
    mechanisms_by_id = _reviewed_mechanisms_by_id(reviewed_diagnosis)

    linked_mechanisms: List[Dict[str, Any]] = []
    inherited_refs: Dict[str, List[str]] = {}

    for mid_raw in action.get("linked_mechanism_ids") or []:
        mid = str(mid_raw).strip()
        if not mid:
            continue
        mechanism = mechanisms_by_id.get(mid)
        if mechanism is None:
            linked_mechanisms.append({
                "mechanism_id": mid,
                "mechanism": None,
                "resolved_evidence": [],
            })
            continue

        resolved = resolve_mechanism_evidence(
            mechanism,
            reviewed_diagnosis,
            case_package,
        )
        linked_mechanisms.append({
            "mechanism_id": mid,
            "mechanism": mechanism,
            "resolved_evidence": resolved,
        })
        for item in resolved:
            ref = str(item.get("ref_id") or "").strip()
            if not ref:
                continue
            inherited_refs.setdefault(ref, [])
            inherited_refs[ref].append(f"linked_mechanism:{mid}")

    direct_evidence: List[Dict[str, Any]] = []
    direct_refs = []
    for ref_raw in action.get("source_refs") or []:
        ref = str(ref_raw).strip()
        if not ref or ref in direct_refs:
            continue
        direct_refs.append(ref)
        direct_evidence.append({
            "ref_id": ref,
            "evidence": evidence_index.get(ref),
            "also_supported_via": sorted(set(inherited_refs.get(ref) or [])),
        })

    linked_ids = [x["mechanism_id"] for x in linked_mechanisms]
    unresolved_mechanisms = [
        x["mechanism_id"] for x in linked_mechanisms if x.get("mechanism") is None
    ]
    unresolved_refs = [
        x["ref_id"] for x in direct_evidence if x.get("evidence") is None
    ]

    # A useful reviewer-side traceability signal.  This does not replace the
    # strict action-service validator; it makes the relationship visible.
    direct_ref_set = set(direct_refs)
    inherited_ref_set = set(inherited_refs)
    refs_outside_linked_mechanisms = sorted(direct_ref_set - inherited_ref_set)

    return {
        "linked_mechanisms": linked_mechanisms,
        "direct_evidence": direct_evidence,
        "linked_mechanism_ids": linked_ids,
        "unresolved_mechanism_ids": unresolved_mechanisms,
        "unresolved_source_refs": unresolved_refs,
        "refs_outside_linked_mechanisms": refs_outside_linked_mechanisms,
    }


def _render_action_support(
    action: Dict[str, Any],
    reviewed_diagnosis: Dict[str, Any],
    case_package: Dict[str, Any],
) -> None:
    support = resolve_action_support(action, reviewed_diagnosis, case_package)

    with st.expander("View action rationale and evidence", expanded=True):
        linked = support["linked_mechanisms"]

        if not linked:
            st.error("This action has no linked mechanism.")
        else:
            st.markdown("**1. Reviewed mechanisms**")

            for i, item in enumerate(linked, start=1):
                mid = item["mechanism_id"]
                mechanism = item.get("mechanism")

                if mechanism is None:
                    st.error(
                        f"{mid}: Could not resolve this mechanism "
                        "in the reviewed diagnosis."
                    )
                    continue

                assessment = mechanism.get("human_assessment") or "not_assessed"
                family = mechanism.get("mechanism_family") or "unclear"
                st.markdown(f"**{mid} · {family} · review={assessment}**")

                statement = str(mechanism.get("human_revision") or "").strip()
                if statement:
                    st.write(statement)
                    st.caption("Human-revised mechanism statement")
                else:
                    st.write(mechanism.get("statement") or "")

                uncertainty = mechanism.get("main_uncertainty")
                if uncertainty:
                    st.warning(f"Main uncertainty: {uncertainty}")

                validation_note = str(
                    mechanism.get("human_validation_note") or ""
                ).strip()
                if validation_note:
                    st.info(f"Human validation note: {validation_note}")

                if assessment in {"rejected", "not_assessed"}:
                    st.error(
                        f"Traceability warning: action links to mechanism {mid} "
                        f"with human_assessment={assessment}."
                    )

                mech_evidence = item.get("resolved_evidence") or []
                st.markdown(f"**Evidence for {mid} ({len(mech_evidence)})**")

                if not mech_evidence:
                    st.warning(
                        "No resolvable evidence is available for this mechanism."
                    )

                for j, ev_item in enumerate(mech_evidence, start=1):
                    ref = ev_item["ref_id"]
                    evidence = ev_item.get("evidence")
                    routes = ", ".join(ev_item.get("via") or []) or "—"

                    if evidence is None:
                        st.error(
                            f"{ref}: Could not resolve this reference "
                            "in the case package."
                        )
                        continue

                    st.markdown(
                        f"**{ref} — {evidence['label']}**  \n"
                        f"Provenance: `{routes}`"
                    )
                    render_evidence_card(evidence["type"], evidence["data"])

                    if j < len(mech_evidence):
                        st.markdown("---")

                if i < len(linked):
                    st.divider()

        st.markdown("**2. Evidence directly cited by the action**")

        direct = support["direct_evidence"]
        if not direct:
            st.warning("This action has no direct evidence reference.")

        for i, item in enumerate(direct, start=1):
            ref = item["ref_id"]
            evidence = item.get("evidence")
            via_mechanisms = item.get("also_supported_via") or []

            if evidence is None:
                st.error(
                    f"{ref}: Could not resolve this reference in the case package."
                )
                continue

            suffix = (
                " · also inherited through " + ", ".join(via_mechanisms)
                if via_mechanisms
                else ""
            )

            st.markdown(f"**{ref} — {evidence['label']}**{suffix}")
            render_evidence_card(evidence["type"], evidence["data"])

            if i < len(direct):
                st.divider()

        outside = support["refs_outside_linked_mechanisms"]
        if outside:
            st.warning(
                "Some action references are not traceable to the linked mechanisms: "
                + ", ".join(outside)
                + ". Review whether these references legitimately support the action."
            )

        if support["unresolved_mechanism_ids"] or support["unresolved_source_refs"]:
            st.error(
                "Unresolved references detected: "
                f"mechanisms={support['unresolved_mechanism_ids']}, "
                f"source_refs={support['unresolved_source_refs']}"
            )


def render_action_review_form(
    action_output: Dict[str, Any],
    reviewed_diagnosis: Dict[str, Any],
    case_package: Dict[str, Any],
    decision_context: Optional[Dict[str, Any]] = None,
    *,
    key_prefix: str,
) -> Optional[Dict[str, Any]]:
    """Gate 2 with action-to-mechanism-to-evidence traceability."""
    decision_context = (
        decision_context
        if isinstance(decision_context, dict)
        else {}
    )

    hardware_allowed = (
        decision_context.get("allow_hardware_change")
        if "allow_hardware_change" in decision_context
        else None
    )
    third_party_allowed = (
        decision_context.get("allow_third_party_change")
        if "allow_third_party_change" in decision_context
        else None
    )

    def display_permission(value: Any) -> str:
        if value is True:
            return "Yes"
        if value is False:
            return "No"
        return "Not recorded"

    actions = [
        a for a in (action_output.get("actions") or [])
        if isinstance(a, dict)
    ]
    if not actions:
        st.warning("No actions are available for review.")
        return None

    st.markdown("### Human Gate 2 · Action review")
    st.caption(
        "Review each action for consistency with the reviewed mechanisms, "
        "evidence traceability, intervention constraints, dependency assumptions, "
        "and validation criteria."
    )

    c1, c2 = st.columns(2)
    c1.write(
        "**Hardware changes allowed:** "
        + display_permission(hardware_allowed)
    )
    c2.write(
        "**Third-party changes allowed:** "
        + display_permission(third_party_allowed)
    )

    for i, action in enumerate(actions, start=1):
        aid = str(action.get("action_id") or f"A{i}")
        family = (
            action.get("action_subfamily")
            or action.get("action_family")
            or action.get("improvement_direction")
            or ""
        )

        st.markdown(f"#### {aid} · {family}")
        st.write(action.get("action") or "")

        targets = [
            str(x).strip()
            for x in (action.get("target_products") or [])
            if str(x).strip()
        ]
        layers = [
            str(x).strip()
            for x in (action.get("target_layers") or [])
            if str(x).strip()
        ]
        subfamily = str(action.get("action_subfamily") or "").strip()

        hardware_involved = (
            "hardware" in layers
            or subfamily == "hardware_design_or_repair"
        )

        a1, a2 = st.columns(2)
        a1.write(f"**Level:** {action.get('action_level') or '—'}")
        a2.write(
            "**Hardware involved:** "
            + ("Yes" if hardware_involved else "No")
        )

        if hardware_involved and hardware_allowed is False:
            st.error(
                "Constraint inconsistency: this action involves hardware, "
                "but hardware changes were not allowed."
            )

        if targets:
            st.write("**Target products:**", targets)

        if layers:
            st.write("**Target layers:**", layers)

        logic = action.get("intervention_logic")
        if logic:
            st.write("**Intervention logic:**", logic)

        metric = action.get("validation_metric")
        if metric:
            st.info(f"Validation metric: {metric}")
        else:
            st.warning("No validation metric is specified.")

        _render_action_support(action, reviewed_diagnosis, case_package)

        if i < len(actions):
            st.divider()

    with st.form(f"{key_prefix}_{action_output.get('case_id', 'case')}"):
        st.markdown("#### Gate 2 decision")
        rows: List[Dict[str, Any]] = []

        for i, action in enumerate(actions, start=1):
            aid = str(action.get("action_id") or f"A{i}")
            st.markdown(f"**{aid}** — {action.get('action') or ''}")
            
            revision = st.text_area(
                "Action revision (optional)",
                key=f"{key_prefix}_{aid}_revision",
                height=80,
            )

            decision = st.selectbox(
                "Action decision",
                ["not_assessed", "accept", "revise", "defer", "reject"],
                index=0,
                key=f"{key_prefix}_{aid}_decision",
            )

            management_value = st.selectbox(
                "Management value",
                ["not_assessed", "high", "medium", "low"],
                index=0,
                key=f"{key_prefix}_{aid}_value",
            )

            rows.append({
                "action_id": aid,
                "decision": decision,
                "management_value": management_value,
                "revision": revision,
            })

            if i < len(actions):
                st.divider()

        submitted = st.form_submit_button("Save action review")

    if not submitted:
        return None

    return {
        "case_id": action_output.get("case_id"),
        "action_feedback": rows,
    }


def render_reviewed_action_summary(
    record: Dict[str, Any],
    reviewed_diagnosis: Dict[str, Any],
    case_package: Dict[str, Any],
) -> None:
    """Render the persisted Human Gate 2 result after review completion.

    This is display-only. It reads the durable approved/deferred/rejected action
    lists written by finalize_product_decision() and never re-runs or mutates the
    review.
    """
    feedback_rows = (
        (record.get("action_feedback") or {}).get("action_feedback")
        or []
    )
    
    feedback_by_id = {
        str(x.get("action_id") or "").strip(): x
        for x in feedback_rows
        if isinstance(x, dict)
    }


    reviewed_output = (
        record.get("reviewed_action_output")
        if isinstance(record.get("reviewed_action_output"), dict)
        else {}
    )
    groups = [
        (
            "Approved actions",
            record.get("approved_actions")
            or reviewed_output.get("approved_actions")
            or [],
            "accept",
        ),
        (
            "Deferred actions",
            record.get("deferred_actions")
            or reviewed_output.get("deferred_actions")
            or [],
            "defer",
        ),
        (
            "Rejected actions",
            record.get("rejected_actions")
            or reviewed_output.get("rejected_actions")
            or [],
            "reject",
        ),
    ]

    n_total = sum(
        len(items) for _, items, _ in groups
        if isinstance(items, list)
    )
    if n_total == 0:
        st.info("No persisted reviewed actions are available to display.")
        return

    st.markdown("### Human Gate 2 · Reviewed actions")
    st.caption(
        "These are the persisted Gate 2 decisions. For revised actions, the "
        "final reviewed wording is shown as the primary action; the original "
        "generated wording remains available for audit."
    )

    for group_title, raw_items, fallback_decision in groups:
        items = [a for a in raw_items if isinstance(a, dict)]
        if not items:
            continue

        st.markdown(f"#### {group_title} ({len(items)})")

        for i, action in enumerate(items, start=1):
            aid = str(action.get("action_id") or f"A{i}")
            family = (
                action.get("action_subfamily")
                or action.get("action_family")
                or action.get("improvement_direction")
                or ""
            )

            original_action = str(action.get("action") or "").strip()
            
            revision = str(
                action.get("human_revision")
                or action.get("revision")
                or ""
            ).strip()
            
            final_action = (
                revision
                if revision
                else str(
                    action.get("final_action")
                    or original_action
                ).strip()
            )

            decision = str(
                action.get("human_decision")
                or action.get("decision")
                or fallback_decision
            ).strip()

            feedback_row = feedback_by_id.get(aid, {})
            management_value = str(
                action.get("management_value")
                or feedback_row.get("management_value")
                or "not_assessed"
            ).strip()
            

            st.markdown(f"**{aid} · {family}**")
            if final_action:
                st.write(final_action)
            else:
                st.warning("The reviewed action has no displayable action text.")

            c1, c2 = st.columns(2)
            c1.write(f"**Final decision:** {decision or '—'}")
            c2.write(f"**Management value:** {management_value or '—'}")

            if final_action and original_action and final_action != original_action:
                with st.expander("View original generated action", expanded=False):
                    st.write(original_action)

            level = action.get("action_level")
            if level:
                st.write("**Level:**", level)

            targets = [
                str(x).strip()
                for x in (action.get("target_products") or [])
                if str(x).strip()
            ]
            layers = [
                str(x).strip()
                for x in (action.get("target_layers") or [])
                if str(x).strip()
            ]
            if targets:
                st.write("**Target products:**", targets)
            if layers:
                st.write("**Target layers:**", layers)

            logic = action.get("intervention_logic")
            if logic:
                st.write("**Intervention logic:**", logic)

            metric = action.get("validation_metric")
            if metric:
                st.info(f"Validation metric: {metric}")

            # Reuse the same mechanism/evidence traceability view used during Gate 2.
            _render_action_support(
                action,
                reviewed_diagnosis,
                case_package,
            )

            if i < len(items):
                st.divider()

        st.divider()
        
        
def _render_action_salvage_summary(record: Dict[str, Any]) -> None:
    discarded = record.get("action_discarded_actions") or []
    n_generated = record.get("action_n_generated")
    n_valid = record.get("action_n_valid")
    n_discarded = record.get("action_n_discarded")

    if n_generated is None and not discarded:
        return

    try:
        generated = int(n_generated or 0)
    except Exception:
        generated = 0
    try:
        valid = int(n_valid or 0)
    except Exception:
        valid = 0
    try:
        rejected = int(n_discarded or len(discarded))
    except Exception:
        rejected = len(discarded)

    if rejected > 0:
        st.warning(
            f"Action generation: generated={generated}, accepted={valid}, "
            f"discarded automatically={rejected}. "
            "Only accepted actions enter Human Gate 2; discarded items remain audited."
        )

        with st.expander(
            f"View discarded actions ({rejected})",
            expanded=False,
        ):
            for i, item in enumerate(discarded, start=1):
                if not isinstance(item, dict):
                    continue

                original_id = item.get("original_action_id") or f"item-{i}"
                st.markdown(f"**{original_id}**")
                st.error(
                    item.get("validation_error")
                    or "Unknown validation error"
                )

                raw_action = item.get("raw_action")
                if raw_action is not None:
                    st.json(raw_action)

                if i < len(discarded):
                    st.divider()

    elif generated:
        st.success(
            f"Action generation contract check passed: "
            f"{valid}/{generated} accepted."
        )


def _show_blocked_stage(record: Dict[str, Any]) -> None:
    stage = str(record.get("failed_stage") or "unknown")
    stage_outputs = record.get("stage_outputs")
    stage_outputs = stage_outputs if isinstance(stage_outputs, dict) else {}

    stage_obj = stage_outputs.get(stage)
    if not isinstance(stage_obj, dict):
        candidate = record.get("stage_output")
        stage_obj = candidate if isinstance(candidate, dict) else {}

    raw_output = stage_obj.get("raw_output")
    if raw_output is None:
        raw_output = record.get("raw_output") or ""

    st.error(
        f"{stage} output failed parsing or contract validation. "
        "The raw model response has been preserved for review."
    )
    st.caption(
        "This is an output-validation failure, not a system or API failure."
    )

    st.write("parse_error:", stage_obj.get("parse_error"))
    st.write("validation_error:", stage_obj.get("validation_error"))

    with st.expander("Raw output", expanded=False):
        st.code(raw_output or "", language="json")

    with st.expander("Parsed output", expanded=False):
        st.json(
            stage_obj.get("parsed_output")
            or record.get("parsed")
            or {}
        )


def main() -> None:
    st.set_page_config(page_title="SCP Prospective Decision", layout="wide")
    st.title("Prospective Diagnosis and Action Decision")
    #st.caption(
    #    "Top-K → H diagnosis → async mechanism review → actions → async action review → ecosystem synthesis. Each stage is persisted; systemic errors fail fast."
   # )

    with st.sidebar:
        data_dir = st.text_input("Data directory", value="data")
        risk_selection_path = st.text_input(
            "Risk selection artifact (.pt)",
            value="data/RBCC_final_selection.pt",
        )
        workflow_dir = st.text_input("Workflow directory", value="prospective_runs")
        model_name = st.text_input("Model name", value="") or None
        temperature = st.number_input(
            "temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1
        )
        #st.caption("Final H is fixed to hard-anchor (max promotions = 0).")

        if st.button("Load saved workflows"):
            st.session_state["h_prospective_records"] = list_workflows(workflow_dir)
            st.session_state.pop("h_ecosystem_output", None)
            st.rerun()

        if st.button("Run Top-K H diagnosis", type="primary"):
            # No broad exception handler: missing files/config/API errors fail fast.
            selection_bundle = _require_selection_bundle(risk_selection_path)
            context = _load_current_context(
                data_dir=data_dir,
                selection_bundle=selection_bundle,
            )
            records = run_h_from_risk_selection(
                risk_selection=selection_bundle["risk_selection"],
                analysis_cutoff=str(selection_bundle["analysis_cutoff"]),
                target_month=str(selection_bundle["target_month"]),
                data_context=context,
                model_name=model_name,
                temperature=float(temperature),
                workflow_dir=workflow_dir,
                expected_k=5,
            )
            st.session_state["h_prospective_records"] = records
            st.session_state["h_risk_selection_bundle"] = selection_bundle
            st.session_state.pop("h_ecosystem_output", None)
            st.rerun()

    records = st.session_state.get("h_prospective_records") or []
    if not records:
        st.info("Run Top-K diagnostics first, or load saved workflows.")
        return

    reviewer = render_respondent_profile("prospective_reviewer")

    for record in records:
        if not isinstance(record, dict):
            continue

        case_key = _case_key(record)
        meta = record.get("case_meta") or record.get("requested_case") or {}
        selection_context = record.get("selection_context") or {}
        status = str(record.get("status") or "")
        title = (
            f"#{selection_context.get('priority_rank') or '-'} · "
            f"{meta.get('product_category', '')} · "
            #f"target {meta.get('target_month', '')} · {status}"
            f"Analysis cutoff {meta.get('analysis_cutoff', '')} · {status}"
        )

        with st.expander(title, expanded=True):
            if status == STATUS_BLOCKED_OUTPUT_REVIEW:
                _show_blocked_stage(record)
                st.info(
                    "This failure occurred outside the action stage. "
                    "Action-stage retry is not applicable."
                )
                continue

            if status == STATUS_BLOCKED_ACTION_OUTPUT_REVIEW:
                _show_blocked_stage(record)
                st.warning(
                    "The Gate 1 review and decision context remain valid. "
                    "Retry only the failed action stage; rerunning the H diagnosis or mechanism review is unnecessary."
                )

                c1, c2 = st.columns(2)
                if c1.button(
                    "Reset action stage",
                    key=f"reset_action_{case_key}",
                ):
                    updated = reset_blocked_action_for_retry(
                        action_stage_record=record,
                        workflow_dir=workflow_dir,
                    )
                    _replace_record(updated)
                    st.rerun()

                can_retry = (
                    isinstance(record.get("mechanism_feedback"), dict)
                    and isinstance(record.get("reviewed_diagnosis"), dict)
                    and isinstance(record.get("decision_context"), dict)
                )
                if c2.button(
                    "Retry action generation",
                    key=f"retry_action_{case_key}",
                    disabled=not can_retry,
                ):
                    updated = retry_blocked_action_generation(
                        action_stage_record=record,
                        model_name=model_name,
                        temperature=float(temperature),
                        workflow_dir=workflow_dir,
                    )
                    _replace_record(updated)
                    st.rerun()

                if not can_retry:
                    st.caption("The workflow is missing the persisted mechanism review, reviewed diagnosis, or decision context.")
                continue

            raw_diag = record.get("diagnosis")
            raw_diag = raw_diag if isinstance(raw_diag, dict) else {}
            reviewed_diag = record.get("reviewed_diagnosis")
            reviewed_diag = reviewed_diag if isinstance(reviewed_diag, dict) else None
            case_package = record.get("case_package")
            case_package = case_package if isinstance(case_package, dict) else {}

            if status == STATUS_WAITING_MECHANISM_REVIEW:
                st.markdown("#### Risk diagnosis")
                st.write("Risk dimensions:", raw_diag.get("risk_dimensions") or [])
                st.write("Symptoms:", raw_diag.get("symptom_families") or [])

                feedback = render_mechanism_review_form(
                    raw_diag,
                    case_package,
                    reviewer,
                    key_prefix=f"mf_{case_key}",
                )
                if feedback is not None:
                    updated = record_mechanism_review(
                        diagnosis_record=record,
                        mechanism_feedback=feedback,
                        workflow_dir=workflow_dir,
                    )
                    _replace_record(updated)
                    st.rerun()
                continue

            # From Gate 1 onward, show the persisted reviewed diagnosis if present.
            display_diag = reviewed_diag or raw_diag
            render_diagnosis_summary(
                display_diag,
                case_package,
                reviewed=reviewed_diag is not None,
            )

            if status == STATUS_READY_FOR_ACTION:
                st.success("The mechanism review has been saved. Actions may be generated now or later.")
                
                saved_context = record.get("decision_context")
                
                if isinstance(saved_context, dict):
                    decision_context = saved_context
                    st.caption("Using the previously persisted decision context.")
                else:
                    decision_context = render_decision_context(
                        str(meta.get("product_category") or ""),
                        key_prefix=f"ctx_{case_key}",
                    )
                    

                if st.button("Generate actions", key=f"gen_action_{case_key}"):
                    updated = generate_product_actions_after_review(
                        diagnosis_record=record,
                        mechanism_feedback=record.get("mechanism_feedback") or {},
                        decision_context=decision_context,
                        model_name=model_name,
                        temperature=float(temperature),
                        workflow_dir=workflow_dir,
                    )
                    _replace_record(updated)
                    st.rerun()

            elif status == STATUS_NO_ACTION:
                st.info("No eligible reviewed mechanism or intervention layer is available; the action model was not called.")

            elif status == STATUS_WAITING_ACTION_REVIEW:
                _render_action_salvage_summary(record)
                action_output = record.get("action_output")
                action_output = action_output if isinstance(action_output, dict) else {}
                action_feedback = render_action_review_form(
                    action_output,
                    reviewed_diag or raw_diag,
                    case_package,
                    decision_context=(
                        record.get("decision_context") if isinstance(record.get("decision_context"), dict)
                        else {}
                    ),
                    key_prefix=f"af_{case_key}",
                )
                
                if action_feedback is not None:
                    updated = finalize_product_decision(
                        action_stage_record=record,
                        action_feedback=action_feedback,
                        workflow_dir=workflow_dir,
                    )
                    _replace_record(updated)
                    st.rerun()

            elif status == STATUS_COMPLETED:
                st.success(
                    f"Action review completed: approved={len(record.get('approved_actions') or [])}, "
                    f"deferred={len(record.get('deferred_actions') or [])}, "
                    f"rejected={len(record.get('rejected_actions') or [])}"
                )
                
                render_reviewed_action_summary(
                    record,
                    reviewed_diag or raw_diag,
                    case_package,
                )

# =============================================================================
#     # -----------------------------------------------------------------------
#     # Ecosystem stage: all and only the upstream five focal workflows must be
#     # terminal. NO_ACTION_REQUIRED remains a focal product with actions=[].
#     # -----------------------------------------------------------------------
#     current_records = [
#         r
#         for r in (st.session_state.get("h_prospective_records") or [])
#         if isinstance(r, dict)
#     ]
#     final_records = [
#         r
#         for r in current_records
#         if r.get("status") in {STATUS_COMPLETED, STATUS_NO_ACTION}
#     ]
# 
#     if current_records and len(current_records) == 5 and len(final_records) == 5:
#         st.divider()
#         st.subheader("Top-K Ecosystem Synthesis (Optional)")
#         st.caption(
#             "Retains all Top-5 focal products and uses only reviewed mechanisms and approved actions."
#         )
#         if st.button("Generate ecosystem synthesis", key="generate_ecosystem"):
#             selection_bundle = _require_selection_bundle(risk_selection_path)
#             context = _load_current_context(
#                 data_dir=data_dir,
#                 selection_bundle=selection_bundle,
#             )
#             st.session_state["h_risk_selection_bundle"] = selection_bundle
#             st.session_state["h_ecosystem_output"] = generate_topk_ecosystem_synthesis(
#                 final_product_records=final_records,
#                 risk_selection=selection_bundle["risk_selection"],
#                 data_context=context,
#                 model_name=model_name,
#                 temperature=float(temperature),
#                 workflow_dir=workflow_dir,
#                 expected_k=5,
#             )
# 
#         eco = st.session_state.get("h_ecosystem_output")
#         if isinstance(eco, dict):
#             if eco.get("status") == STATUS_BLOCKED_OUTPUT_REVIEW:
#                 st.error("Ecosystem synthesis failed output validation; the raw response has been preserved.")
#                 with st.expander("Ecosystem raw output"):
#                     st.code(eco.get("raw_output") or "", language="json")
#                 validation_error = (eco.get("stage_output") or {}).get(
#                     "validation_error"
#                 )
#                 if validation_error:
#                     st.write("validation_error:", validation_error)
#             else:
#                 st.json(eco.get("parsed") or {})
# 
#     elif current_records:
#         terminal = len(final_records)
#         st.info(
#             "Ecosystem synthesis becomes available after all five focal-product workflows reach "
#             f"COMPLETED or NO_ACTION_REQUIRED ({terminal}/5 complete)."
#         )
# =============================================================================


if __name__ == "__main__":
    main()
