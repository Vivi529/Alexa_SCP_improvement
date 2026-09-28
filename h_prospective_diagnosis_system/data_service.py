from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import pandas as pd

CONDITION_A = "A_static"
CONDITION_B = "B_dynamic_evolution"
CONDITION_C = "C_full_risk_chain"
CONDITION_H = "H_hierarchical"

# Keep ALL_CONDITIONS frozen to the original A/B/C information ablation.
# Hybrid is a separate task-routing treatment and must not silently redefine
# the original nested ablation.
ALL_CONDITIONS = (CONDITION_A, CONDITION_B, CONDITION_C)
ALL_EXPERIMENT_CONDITIONS = (*ALL_CONDITIONS, CONDITION_H)
GENERAL_DIMENSION_NAMES = {"general"}

# ---------------------------------------------------------------------------
# Product-category compatibility layer
# ---------------------------------------------------------------------------
# Canonical names intentionally follow the shorter labels used by the ABC
# evaluation manifest/event-gold files.  Historical/raw profile labels are
# mapped conservatively by exact alias only; no fuzzy substring remapping is
# performed because that can silently merge genuinely different categories.
CATEGORY_ALIASES = {
    "安防-摄像头/门铃": "摄像头/门铃",
    "摄像头/门铃": "摄像头/门铃",
    "安防-报警系统/传感器": "报警系统/传感器",
    "报警系统/传感器": "报警系统/传感器",
    "播放HDMI同步盒": "HDMI同步盒",
    "HDMI同步盒": "HDMI同步盒",
}

# Soft lexical hints are retained only as a ranking/fallback layer.  When a
# category->API capability map is available, that map is the hard candidate
# filter and these hints no longer decide eligibility.
CATEGORY_API_HINTS = {
    "智能排插/插头/插座": ["power", "plug", "outlet", "switch"],
    "智能音箱": ["speaker", "audio", "volume", "playback"],
    "摄像头/门铃": ["camera", "doorbell", "stream", "snapshot", "rtc", "video"],
    "智能温控器": ["thermostat", "temperature", "hvac", "schedule"],
    "智能门锁": ["lock", "keypad", "door"],
    "灯带/灯泡/灯具": ["brightness", "color", "power", "light"],
    "智能开关/运动传感器/墙插": ["power", "toggle", "motion", "contact", "switch"],
    "窗帘/遮阳帘/百叶": ["range", "mode", "percentage", "blind"],
    "风扇/吊扇": ["mode", "power", "range", "percentage", "fan"],
    "HDMI同步盒": ["input", "channel", "playback", "video", "hdmi"],
    "流媒体播放器/电视棒": ["playback", "launcher", "channel", "video", "seek", "remote"],
    "报警系统/传感器": ["security", "contact", "motion", "keypad", "sensor"],
    "空气质量监测器": ["humidity", "temperature", "sensor", "data"],
    "空气净化器/香薰机/加湿器": ["mode", "range", "percentage", "power", "humidity"],
    "遥控器": ["input", "channel", "remote", "playback"],
    "耳机/眼镜": ["speaker", "audio", "playback"],
    "智能屏/家庭中控": ["speaker", "video", "input", "launcher", "playback", "camera"],
    "吸尘器": ["mode", "power", "range", "percentage"],
    "集线器/桥接器": ["commission", "automation", "scene", "power", "discovery"],
    "打印机": ["inventory", "notification", "power", "printer"],
    "汽车智能配件": ["speaker", "audio", "playback", "voice"],
}

DEFAULT_CATEGORY_API_MAP_PATH = Path(__file__).resolve().parent / "data" / "category_api_map.json"


def canonical_category_name(x: Any) -> str:
    """Return a conservative canonical category name used across all joins."""
    if x is None:
        return ""
    s = str(x).strip().replace("／", "/")
    if not s:
        return ""
    return CATEGORY_ALIASES.get(s, s)


def category_aliases(category: Any) -> List[str]:
    """Return exact known aliases for safe matching in free-text path labels."""
    canonical = canonical_category_name(category)
    aliases = [canonical]
    for alias, target in CATEGORY_ALIASES.items():
        if target == canonical and alias not in aliases:
            aliases.append(alias)
    return aliases


def normalize_api_name(x: Any) -> str:
    """Normalize API names only for matching, not for display."""
    if x is None:
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"^Alexa\s*\.\s*", "Alexa.", s, flags=re.I)
    return s.lower()


def _extract_api_name(card: Mapping[str, Any]) -> str:
    raw = card.get("api_name")
    if isinstance(raw, dict):
        raw = raw.get("value")
    return str(raw or "").strip()


def load_category_api_map(
    path: Optional[str | Path] = None,
) -> Dict[str, List[str]]:
    """
    Load the explicit ProductCategory -> candidate Alexa API map generated from
    the old MAP_21/T_cate_interfaces procedure.

    The JSON file should be a dict such as:
        {"智能排插/插头/插座": ["Alexa.PowerController", ...], ...}

    Category aliases are canonicalized and duplicate API names are merged.
    Missing files are allowed: retrieval then falls back to CATEGORY_API_HINTS.
    """
    map_path = Path(path) if path else DEFAULT_CATEGORY_API_MAP_PATH
    if not map_path.exists():
        return {}
    with map_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"category_api_map must be a JSON object: {map_path}")

    merged: Dict[str, set[str]] = {}
    for raw_cat, raw_apis in obj.items():
        cat = canonical_category_name(raw_cat)
        if not cat:
            continue
        if isinstance(raw_apis, str):
            vals = [x.strip() for x in raw_apis.split(";") if x.strip()]
        elif isinstance(raw_apis, (list, tuple, set)):
            vals = [str(x).strip() for x in raw_apis if str(x).strip()]
        else:
            vals = []
        bucket = merged.setdefault(cat, set())
        bucket.update(vals)

    return {cat: sorted(vals) for cat, vals in merged.items()}


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {e}") from e
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def stable_id(prefix: str, *parts: Any, length: int = 10) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length].upper()
    return f"{prefix}-{digest}"


def canonical_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, dict):
        return " ".join(canonical_text(v) for v in x.values())
    if isinstance(x, (list, tuple, set)):
        return " ".join(canonical_text(v) for v in x)
    return str(x)


def tokens(x: Any) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_.-]+", canonical_text(x).lower())


def _clean_list(x: Any, limit: Optional[int] = None) -> List[Any]:
    if x is None:
        return []
    vals = x if isinstance(x, list) else [x]
    out, seen = [], set()
    for v in vals:
        key = json.dumps(v, ensure_ascii=False, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
        if limit is not None and len(out) >= limit:
            break
    return out


def _category(row: Mapping[str, Any]) -> str:
    return canonical_category_name(row.get("Category") or row.get("product_category") or "")


def _cutoff(row: Mapping[str, Any]) -> Optional[str]:
    c = row.get("analysis_cutoff")
    if c:
        return str(c)
    de = row.get("dynamic_evolution")
    if isinstance(de, dict) and de.get("analysis_cutoff"):
        return str(de.get("analysis_cutoff"))
    dp = row.get("dynamic_profile")
    if isinstance(dp, dict) and dp.get("latest_valid_month_used"):
        return str(dp.get("latest_valid_month_used"))
    return None


def build_profile_index(rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, Optional[str]], Dict[str, Any]]:
    idx: Dict[Tuple[str, Optional[str]], Dict[str, Any]] = {}
    for row in rows:
        cat = _category(row)
        if not cat:
            continue
        idx[(cat, _cutoff(row))] = row
    return idx


def resolve_profile_row(
    index: Dict[Tuple[str, Optional[str]], Dict[str, Any]],
    category: str,
    analysis_cutoff: Optional[str] = None,
) -> Dict[str, Any]:
    category = canonical_category_name(category)
    cutoff = None if analysis_cutoff is None else str(analysis_cutoff)
    if (category, cutoff) in index:
        return index[(category, cutoff)]
    candidates = [(c, r) for (cat, c), r in index.items() if cat == category]
    if not candidates:
        raise KeyError(f"No profile for category={category!r}")
    if cutoff is not None:
        raise KeyError(f"No profile for category={category!r}, analysis_cutoff={cutoff!r}")
    # Latest dated snapshot if cutoff was not requested; otherwise undated singleton.
    dated = [(c, r) for c, r in candidates if c is not None]
    if dated:
        dated.sort(key=lambda x: str(x[0]))
        return dated[-1][1]
    if len(candidates) == 1:
        return candidates[0][1]
    raise KeyError(f"Ambiguous undated profiles for category={category!r}")


def _is_general_dimension(
    value: Any,
    general_dimension_names: Optional[Sequence[str]] = None,
) -> bool:
    names = (
        GENERAL_DIMENSION_NAMES
        if general_dimension_names is None
        else {str(x).strip().lower() for x in general_dimension_names}
    )
    return str(value or "").strip().lower() in names


def _observed_card_id(
    category: str,
    static_row: Mapping[str, Any],
    card: Mapping[str, Any],
) -> str:
    """Stable dimension-level observed-evidence ID shared by catalog/detail views."""
    return stable_id(
        "OBS",
        canonical_category_name(category),
        static_row.get("analysis_cutoff"),
        str(card.get("dimension") or "").strip(),
        "category_dimension",
    )


def compact_observed_evidence(
    category: str,
    static_row: Dict[str, Any],
    max_cards: int = 6,
    statements_per_card: int = 2,
) -> List[Dict[str, Any]]:
    """Detailed top-K observed cards used for evidence-rich diagnosis.

    The current evidence builder uses Category x Dimension cards. Failure type
    and scenario are descriptive sub-signals rather than grouping keys; keep
    those fields intact here instead of reading the obsolete scalar schema.
    """
    cards = (
        static_row.get("evidence_cards")
        or static_row.get("evidence_card_summary")
        or []
    )
    out: List[Dict[str, Any]] = []
    for card in cards[: max(0, int(max_cards))]:
        if not isinstance(card, dict):
            continue
        dimension = str(card.get("dimension") or "").strip()
        if not dimension or _is_general_dimension(dimension):
            continue
        sev = card.get("severity_proxy") if isinstance(card.get("severity_proxy"), dict) else {}
        out.append({
            "evidence_id": _observed_card_id(category, static_row, card),
            "source_type": "aggregated_review_evidence",
            "product_category": canonical_category_name(category),
            "dimension": dimension,
            # Audit-only metadata. build_llm_case_view() intentionally drops it.
            "evidence_selection": copy.deepcopy(
                card.get("evidence_selection")
                if isinstance(card.get("evidence_selection"), dict)
                else {}
            ),
            "top_failure_types": _clean_list(card.get("top_failure_types"), 3),
            "top_scenarios": _clean_list(card.get("top_scenarios"), 3),
            "severity": {
                k: sev.get(k)
                for k in [
                    "n_stmt",
                    "n_review",
                    "mean_sentiment",
                    "recent_count",
                    "recent_review_count",
                    "share_in_product",
                    "review_share_in_product",
                ]
                if k in sev
            },
            "representative_statements": _clean_list(
                card.get("representative_statements") or card.get("top_statements"),
                statements_per_card,
            ),
            "epistemic_status": "observed_review_evidence",
        })
    return out


def compact_observed_dimension_catalog(
    category: str,
    static_row: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Compact all-dimension index used by the A-stage dimension anchor.

    This avoids the top-K censoring problem without sending every detailed
    review statement to the LLM. General is excluded by design.
    """
    cards = static_row.get("evidence_card_pool") or static_row.get("evidence_cards") or []
    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for card in cards:
        if not isinstance(card, dict):
            continue
        dimension = str(card.get("dimension") or "").strip()
        if not dimension or _is_general_dimension(dimension) or dimension in seen:
            continue
        seen.add(dimension)
        sev = card.get("severity_proxy") if isinstance(card.get("severity_proxy"), dict) else {}
        selection = card.get("evidence_selection") if isinstance(card.get("evidence_selection"), dict) else {}
        rows.append({
            "evidence_id": _observed_card_id(category, static_row, card),
            "product_category": canonical_category_name(category),
            "dimension": dimension,
            "n_review": sev.get("n_review"),
            "mean_sentiment": sev.get("mean_sentiment"),
            "recent_review_count": sev.get("recent_review_count"),
            "review_share_in_product": sev.get("review_share_in_product"),
            "evidence_utility": selection.get("evidence_utility"),
            "epistemic_status": "observed_review_evidence",
        })
    return rows

def _relation_profile_for(
    relation_rows: Sequence[Dict[str, Any]],
    category: str,
    analysis_cutoff: Optional[str],
) -> Optional[Dict[str, Any]]:
    category = canonical_category_name(category)
    exact = []
    undated = []
    for row in relation_rows:
        if _category(row) != category:
            continue
        c = _cutoff(row)
        if analysis_cutoff is not None and c == str(analysis_cutoff):
            exact.append(row)
        elif c is None:
            undated.append(row)
    if exact:
        return exact[0]
    if analysis_cutoff is None:
        candidates = [r for r in relation_rows if _category(r) == category]
        if candidates:
            candidates.sort(key=lambda r: str(_cutoff(r) or ""))
            return candidates[-1]
    # Undated fallback is allowed only for explicitly time-invariant relation files.
    return undated[0] if undated else None


def compact_static_relations(
    category: str,
    relation_rows: Sequence[Dict[str, Any]],
    analysis_cutoff: Optional[str],
    max_relations: int = 5,
) -> List[Dict[str, Any]]:
    profile = _relation_profile_for(relation_rows, category, analysis_cutoff)
    if not profile:
        return []
    cards = profile.get("relation_cards") or []
    cards = sorted(cards, key=lambda x: float(x.get("overall_relation_strength") or 0.0), reverse=True)
    return [dict(x) for x in cards[:max_relations] if isinstance(x, dict)]


def _filtered_dynamic_dimension_signals(
    dynamic_row: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """Return non-General per-dimension dynamic signals with backward fallback."""
    de = dynamic_row.get("dynamic_evolution") if isinstance(dynamic_row.get("dynamic_evolution"), dict) else {}
    signals = de.get("dimension_signals")
    if not isinstance(signals, list) or not signals:
        state = de.get("state_signals") if isinstance(de.get("state_signals"), dict) else {}
        signals = state.get("top_predicted_risk_dimensions") or []
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for raw in signals:
        if not isinstance(raw, dict):
            continue
        dim = str(raw.get("dimension") or "").strip()
        if not dim or _is_general_dimension(dim) or dim in seen:
            continue
        seen.add(dim)
        obj = dict(raw)
        obj["dimension"] = dim
        out.append(obj)
    return out


def compact_dynamic_facts(category: str, dynamic_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Compact Condition-B facts while excluding General at the evidence layer."""
    de = dynamic_row.get("dynamic_evolution") if isinstance(dynamic_row.get("dynamic_evolution"), dict) else {}
    cutoff = dynamic_row.get("analysis_cutoff") or de.get("analysis_cutoff")
    target_month = dynamic_row.get("target_month") or de.get("target_month")
    state = de.get("state_signals") if isinstance(de.get("state_signals"), dict) else {}
    evolution = de.get("evolution_signals") if isinstance(de.get("evolution_signals"), dict) else {}
    driver = de.get("driver_signals") if isinstance(de.get("driver_signals"), dict) else {}
    network = de.get("network_position_signals") if isinstance(de.get("network_position_signals"), dict) else {}

    signals = _filtered_dynamic_dimension_signals(dynamic_row)
    top_signals = signals[:5]
    priority = [str(x.get("dimension")) for x in top_signals if x.get("dimension")]

    facts = [
        {
            "fact_id": stable_id("MF", category, cutoff, target_month, "risk_state"),
            "fact_type": "predicted_risk_state",
            "analysis_cutoff": cutoff,
            "target_month": target_month,
            "predicted_next_risk_score": state.get("predicted_next_risk_score"),
            "predicted_next_risk_level": state.get("predicted_next_risk_level"),
            "max_dimension_risk_score": state.get("max_dimension_risk_score"),
            "top_predicted_risk_dimensions": top_signals,
            "dynamic_dimension_priority": priority,
            "general_dimension_excluded": True,
            "epistemic_status": "model_inferred",
        },
        {
            "fact_id": stable_id("MF", category, cutoff, target_month, "evolution"),
            "fact_type": "evolution_persistence",
            "persistence_score": evolution.get("persistence_score"),
            "persistence": evolution.get("persistence"),
            "epistemic_status": "model_inferred",
        },
        {
            "fact_id": stable_id("MF", category, cutoff, target_month, "drivers"),
            "fact_type": "dynamic_network_drivers",
            "beta_api": driver.get("beta_api"),
            "beta_coo": driver.get("beta_coo"),
            "beta_user": driver.get("beta_user"),
            "dominant_driver": driver.get("dominant_driver"),
            "epistemic_status": "model_inferred",
        },
        {
            "fact_id": stable_id("MF", category, cutoff, target_month, "network_position"),
            "fact_type": "dynamic_network_position",
            "spillover_output_score": network.get("spillover_output_score"),
            "spillover_output_level": network.get("spillover_output_level"),
            "spillover_exposure_score": network.get("spillover_exposure_score"),
            "spillover_exposure_level": network.get("spillover_exposure_level"),
            "direction_note": network.get("direction_note"),
            "epistemic_status": "model_inferred",
        },
    ]
    return facts


def compact_anchor_dynamic_facts(
    category: str,
    dynamic_row: Dict[str, Any],
    anchor_dimensions: Sequence[str],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Condition B on A-stage dimensions instead of presenting free candidates.

    Returns (facts, missing_anchor_dimensions). Older dynamic exports that only
    contain top_predicted_risk_dimensions remain usable; anchors outside that
    historical top list are reported as missing rather than hallucinated.
    """
    anchors = [
        str(x).strip() for x in anchor_dimensions
        if str(x).strip() and not _is_general_dimension(x)
    ]
    anchors = list(dict.fromkeys(anchors))
    signal_by_dim = {
        str(x.get("dimension")): x
        for x in _filtered_dynamic_dimension_signals(dynamic_row)
        if x.get("dimension")
    }
    de = dynamic_row.get("dynamic_evolution") if isinstance(dynamic_row.get("dynamic_evolution"), dict) else {}
    cutoff = dynamic_row.get("analysis_cutoff") or de.get("analysis_cutoff")
    target_month = dynamic_row.get("target_month") or de.get("target_month")
    evolution = de.get("evolution_signals") if isinstance(de.get("evolution_signals"), dict) else {}
    driver = de.get("driver_signals") if isinstance(de.get("driver_signals"), dict) else {}
    network = de.get("network_position_signals") if isinstance(de.get("network_position_signals"), dict) else {}

    facts: List[Dict[str, Any]] = []
    missing: List[str] = []
    for dim in anchors:
        signal = signal_by_dim.get(dim)
        if signal is None:
            missing.append(dim)
            continue
        facts.append({
            "fact_id": stable_id("MF", category, cutoff, target_month, "anchor_dimension", dim),
            "fact_type": "anchored_dimension_forecast",
            "analysis_cutoff": cutoff,
            "target_month": target_month,
            "dimension": dim,
            **{
                k: signal.get(k)
                for k in [
                    "predicted_next_risk",
                    "max_predicted_next_risk",
                    "predicted_risk_change",
                    "historical_risk_zscore",
                    "spillover_output",
                    "spillover_exposure",
                    "rho",
                    "beta_api",
                    "beta_coo",
                    "beta_user",
                    "dominant_driver",
                ]
                if k in signal
            },
            "epistemic_status": "model_inferred_dimension_conditioned",
        })

    if facts:
        facts.append({
            "fact_id": stable_id("MF", category, cutoff, target_month, "anchor_product_context"),
            "fact_type": "anchored_product_dynamic_context",
            "analysis_cutoff": cutoff,
            "target_month": target_month,
            "persistence_score": evolution.get("persistence_score"),
            "persistence": evolution.get("persistence"),
            "beta_api": driver.get("beta_api"),
            "beta_coo": driver.get("beta_coo"),
            "beta_user": driver.get("beta_user"),
            "dominant_driver": driver.get("dominant_driver"),
            "spillover_output_score": network.get("spillover_output_score"),
            "spillover_exposure_score": network.get("spillover_exposure_score"),
            "scope_note": "Product-level context; not a new risk dimension.",
            "epistemic_status": "model_inferred_product_context",
        })

    return facts, missing

def _published_not_after(card: Dict[str, Any], cutoff: Optional[str]) -> bool:
    if not cutoff:
        return True
    raw = card.get("published_date") or card.get("publication_date") or card.get("date")
    if not raw:
        return True  # unknown publication date is retained but remains non-probative knowledge
    try:
        return pd.Timestamp(raw) <= pd.Period(str(cutoff), freq="M").end_time
    except Exception:
        return True


def compact_api_card(card: Dict[str, Any]) -> Dict[str, Any]:
    api_name = card.get("api_name")
    if isinstance(api_name, dict):
        api_name = api_name.get("value")
    api_name = str(api_name or "")
    return {
        "knowledge_id": stable_id("KB", api_name),
        "knowledge_type": "api_documentation_card",
        "api_name": api_name,
        "functionality": card.get("functionality"),
        "device_context": card.get("device_context"),
        "dependencies": card.get("dependencies_or_related_APIs") or card.get("dependencies") or [],
        "design_constraints": card.get("design_constraints") or [],
        "common_root_causes": card.get("common_root_causes") or card.get("common_root_causes (structured)") or [],
        "diagnostic_keywords": card.get("diagnostic_keywords") or card.get("keywords") or [],
        "notes_for_root_cause_mapping": card.get("notes_for_root_cause_mapping"),
        "epistemic_status": "candidate_mechanism_knowledge_not_implementation_proof",
    }


def retrieve_api_evidence_static_only(
    category: str,
    static_row: Dict[str, Any],
    static_relations: Sequence[Dict[str, Any]],
    api_cards: Sequence[Dict[str, Any]],
    analysis_cutoff: Optional[str] = None,
    max_cards: int = 3,
    category_api_map: Optional[Mapping[str, Sequence[str]]] = None,
    category_api_map_path: Optional[str | Path] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve API knowledge using a two-stage rule:

    1) HARD FILTER (preferred): explicit category->API capability mapping derived
       from MAP_21 / T_cate_interfaces.
    2) RANKING: static issue semantics + light CATEGORY_API_HINTS bonus.

    If no explicit mapping is available for the category, the function falls
    back to the old hint/lexical-overlap eligibility rule.  Dynamic-model fields
    are intentionally excluded so KB_A == KB_B == KB_C in the ABC ablation.
    """
    category = canonical_category_name(category)

    # Load/canonicalize the explicit capability map.
    if category_api_map is None:
        category_api_map = load_category_api_map(category_api_map_path)
    else:
        normalized_map: Dict[str, List[str]] = {}
        for raw_cat, raw_apis in category_api_map.items():
            cat = canonical_category_name(raw_cat)
            vals = list(raw_apis) if isinstance(raw_apis, (list, tuple, set)) else [raw_apis]
            normalized_map.setdefault(cat, []).extend(str(x).strip() for x in vals if str(x).strip())
        category_api_map = normalized_map

    candidate_api_names = {
        normalize_api_name(x)
        for x in (category_api_map.get(category, []) if category_api_map else [])
        if normalize_api_name(x)
    }
    hard_filter_active = bool(candidate_api_names)

    # Static-only query: identical across A/B/C.
    query = canonical_text({
        "category": category,
        "dimensions": static_row.get("dominant_dimensions"),
        "failure_types": static_row.get("dominant_failure_types"),
        "scenarios": static_row.get("high_risk_scenarios"),
        "observed_cards": static_row.get("evidence_cards"),
        "static_relations": static_relations,
    }).lower()
    q_tokens = set(tokens(query))
    hints = CATEGORY_API_HINTS.get(category, [])

    # Extra terms are derived only from the static issue profile.
    issue_terms = set(tokens([
        static_row.get("dominant_dimensions"),
        static_row.get("dominant_failure_types"),
        static_row.get("high_risk_scenarios"),
    ]))

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for card in api_cards:
        if not _published_not_after(card, analysis_cutoff):
            continue

        api_name = _extract_api_name(card)
        api_key = normalize_api_name(api_name)
        if not api_key:
            continue

        # Old structured method becomes the preferred hard compatibility filter.
        if hard_filter_active and api_key not in candidate_api_names:
            continue

        api_text = canonical_text(card).lower()
        api_tokens = set(tokens(api_text))
        overlap = len(q_tokens.intersection(api_tokens))

        # Semantic/static-issue relevance.
        issue_overlap = len(issue_terms.intersection(api_tokens))
        bonus = 1.5 * float(issue_overlap)

        # Hints are secondary ranking signals only when hard filtering is active.
        hint_hits = 0
        api_name_l = api_name.lower()
        device_context_l = canonical_text(card.get("device_context")).lower()
        for hint in hints:
            if hint in api_name_l:
                bonus += 4.0 if hard_filter_active else 12.0
                hint_hits += 1
            elif hint in device_context_l:
                bonus += 2.0 if hard_filter_active else 6.0
                hint_hits += 1
            elif hint in api_text:
                bonus += 1.0 if hard_filter_active else 2.5
                hint_hits += 1

        # Under hard filtering every compatible API may compete; without a map,
        # preserve the conservative legacy eligibility rule.
        if hard_filter_active or hint_hits > 0 or overlap >= 18:
            scored.append((float(overlap) + bonus, card))

    scored.sort(key=lambda x: x[0], reverse=True)

    out: List[Dict[str, Any]] = []
    seen = set()
    for score, card in scored:
        compact = compact_api_card(card)
        key = normalize_api_name(compact.get("api_name"))
        if not key or key in seen:
            continue
        compact["retrieval_score"] = round(float(score), 3)
        compact["retrieval_basis"] = (
            "category_capability_filter+static_issue_relevance"
            if hard_filter_active
            else "category_hint_fallback+static_issue_relevance"
        )
        compact["category_match"] = category
        out.append(compact)
        seen.add(key)
        if len(out) >= max_cards:
            break

    return out

def _chain_involves_category(row: Dict[str, Any], category: str) -> bool:
    canonical = canonical_category_name(category)

    for k in ["source_category", "target_category"]:
        if canonical_category_name(row.get(k)) == canonical:
            return True

    for e in row.get("edges") or []:
        if not isinstance(e, dict):
            continue
        if (
            canonical_category_name(e.get("from_category")) == canonical
            or canonical_category_name(e.get("to_category")) == canonical
        ):
            return True

    # Fallback for old chain files that only contain path_text. Exact aliases
    # are used rather than fuzzy product-name matching.
    path_text = str(row.get("path_text") or "")
    return any(
        alias and alias in path_text
        for alias in category_aliases(canonical)
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if pd.isna(out):
        return float(default)
    return out


def _risk_chain_is_propagation(row: Mapping[str, Any]) -> bool:
    """Infer whether a risk-chain record contains an actual inter-node path.

    New files should carry ``is_propagation_chain`` and ``path_length``.
    The fallbacks keep older exports usable without treating the string
    ``"False"`` as truthy.
    """
    explicit = row.get("is_propagation_chain")

    if isinstance(explicit, bool):
        return explicit

    if isinstance(explicit, (int, float)) and not isinstance(explicit, bool):
        return bool(explicit)

    if isinstance(explicit, str):
        value = explicit.strip().lower()
        if value in {"true", "1", "yes", "y"}:
            return True
        if value in {"false", "0", "no", "n", ""}:
            return False

    chain_kind = str(
        row.get("chain_kind")
        or row.get("chain_type")
        or ""
    ).strip().lower()

    if chain_kind in {
        "propagation",
        "propagation_chain",
        "risk_propagation",
    }:
        return True

    if chain_kind in {
        "standalone",
        "standalone_risk",
        "self",
        "self_chain",
    }:
        return False

    try:
        return int(row.get("path_length") or 0) > 0
    except Exception:
        pass

    node_path = row.get("node_path")
    if isinstance(node_path, list):
        return len(node_path) > 1

    edges = row.get("edges")
    if isinstance(edges, list):
        return any(isinstance(e, dict) for e in edges)

    return False


def _chain_dimensions(row: Mapping[str, Any]) -> Set[str]:
    """Extract dimension labels conservatively from new and legacy chain records."""
    dims: Set[str] = set()
    for key in ("dimension", "source_dimension", "target_dimension"):
        val = str(row.get(key) or "").strip()
        if val:
            dims.add(val)
    for edge in row.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        for key in ("dimension", "from_dimension", "to_dimension"):
            val = str(edge.get(key) or "").strip()
            if val:
                dims.add(val)
    # Legacy path_text often uses Category::Dimension -> Category::Dimension.
    path_text = str(row.get("path_text") or "")
    for match in re.findall(r"::([^>]+?)(?:\s*->|$)", path_text):
        val = str(match).strip()
        if val:
            dims.add(val)
    return dims


def _chain_contains_general(row: Mapping[str, Any]) -> bool:
    return any(_is_general_dimension(x) for x in _chain_dimensions(row))


def load_optional_risk_chains(
    path: Optional[str | Path],
    category: str,
    target_month: Optional[str] = None,
    max_chains: int = 4,
    max_standalone: int = 1,
    *,
    exclude_general_dimension: bool = True,
    allowed_dimensions: Optional[Sequence[str]] = None,
    selection_mode: str = "dimension_diverse",
) -> List[Dict[str, Any]]:
    """Load compact risk-chain context with optional anchor conditioning.

    ``global_topk`` reproduces the legacy reserved-quota retrieval rule.

    ``dimension_diverse`` treats ``max_standalone`` as a true upper bound,
    maximizes marginal dimension coverage first, prefers propagation chains
    when coverage gain is tied, and then uses chain/source risk scores as
    strength tie-breakers.

    Condition C should pass ``allowed_dimensions=None``. Hybrid H may pass
    final anchor dimensions. No gold labels or LLM calls are used here.
    """
    if not path:
        return []
    path = Path(path)
    if not path.exists():
        return []

    max_chains = max(0, int(max_chains))
    if max_chains == 0:
        return []
    max_standalone = max(0, min(int(max_standalone), max_chains))

    selection_mode = str(selection_mode or "dimension_diverse").strip().lower()
    if selection_mode not in {"global_topk", "dimension_diverse"}:
        raise ValueError(
            "selection_mode must be one of: 'global_topk', 'dimension_diverse'"
        )

    allowed: Optional[Set[str]] = None
    if allowed_dimensions is not None:
        allowed = {
            str(x).strip()
            for x in allowed_dimensions
            if str(x).strip() and not _is_general_dimension(x)
        }
        if not allowed:
            return []

    matched: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in load_jsonl(path):
        if target_month is not None and str(row.get("target_month")) != str(target_month):
            continue
        if not _chain_involves_category(row, category):
            continue

        dims = {str(x).strip() for x in _chain_dimensions(row) if str(x).strip()}
        if exclude_general_dimension and any(_is_general_dimension(x) for x in dims):
            continue
        if allowed is not None and not (dims & allowed):
            continue

        obj = dict(row)
        obj["chain_id"] = str(
            obj.get("chain_id")
            or obj.get("source_chain_id")
            or stable_id("CHN", obj)
        )
        if obj["chain_id"] in seen_ids:
            continue
        seen_ids.add(obj["chain_id"])

        is_prop = _risk_chain_is_propagation(obj)
        obj["is_propagation_chain"] = bool(is_prop)
        obj["chain_kind"] = "propagation" if is_prop else "standalone"
        matched_dims = dims & allowed if allowed is not None else dims
        obj["matched_dimensions"] = sorted(matched_dims)
        obj.setdefault("epistemic_status", "model_inferred_risk_chain_not_causal")
        matched.append(obj)

    if not matched:
        return []

    def _rank_key(row: Dict[str, Any]) -> Tuple[float, float, str]:
        return (
            _safe_float(row.get("chain_score"), 0.0),
            _safe_float(row.get("source_risk_score"), 0.0),
            str(row.get("chain_id") or ""),
        )

    def _coverage_dims(row: Mapping[str, Any]) -> Set[str]:
        return {
            str(x).strip()
            for x in (row.get("matched_dimensions") or [])
            if str(x).strip() and not _is_general_dimension(x)
        }

    propagation = sorted(
        [r for r in matched if r.get("is_propagation_chain")],
        key=_rank_key,
        reverse=True,
    )
    standalone = sorted(
        [r for r in matched if not r.get("is_propagation_chain")],
        key=_rank_key,
        reverse=True,
    )

    if selection_mode == "global_topk":
        # Preserve legacy behavior exactly for reproducibility.
        prop_quota = max_chains - max_standalone
        selected_prop = propagation[:prop_quota]
        selected_standalone = standalone[:max_standalone]
        selected: List[Dict[str, Any]] = [*selected_prop, *selected_standalone]

        remaining = max_chains - len(selected)
        if remaining > 0:
            selected.extend(
                propagation[len(selected_prop): len(selected_prop) + remaining]
            )
        remaining = max_chains - len(selected)
        if remaining > 0:
            selected.extend(
                standalone[len(selected_standalone): len(selected_standalone) + remaining]
            )

        selected.sort(
            key=lambda r: (1 if r.get("is_propagation_chain") else 0, *_rank_key(r)),
            reverse=True,
        )
        selected = selected[:max_chains]

        cumulative_dims: set[str] = set()
        for prompt_rank, row in enumerate(selected, start=1):
            dims = _coverage_dims(row)
            new_dims = dims - cumulative_dims
            cumulative_dims.update(dims)
            row["retrieval_selection"] = {
                "selection_mode": "global_topk",
                "retrieval_rank": int(prompt_rank),
                "selection_order": int(prompt_rank),
                "selection_stage": "global_score",
                "matched_dimensions": sorted(dims),
                "new_dimensions_when_selected": sorted(new_dims),
                "new_dimensions_in_prompt_order": sorted(new_dims),
                "cumulative_dimension_coverage": sorted(cumulative_dims),
            }
        return selected

    # Diversified mode: max_standalone is a cap, not a reserved slot.
    selected: List[Dict[str, Any]] = []
    covered_dims: set[str] = set()
    remaining_pool: List[Dict[str, Any]] = list(matched)
    standalone_used = 0
    selection_meta: Dict[str, Dict[str, Any]] = {}

    while len(selected) < max_chains and remaining_pool:
        best_idx: Optional[int] = None
        best_key: Optional[Tuple[Any, ...]] = None
        best_new_dims: set[str] = set()

        for idx, row in enumerate(remaining_pool):
            is_prop = bool(row.get("is_propagation_chain"))
            if not is_prop and standalone_used >= max_standalone:
                continue

            dims = _coverage_dims(row)
            new_dims = dims - covered_dims
            candidate_key: Tuple[Any, ...] = (
                len(new_dims),
                1 if is_prop else 0,
                *_rank_key(row),
            )
            if best_key is None or candidate_key > best_key:
                best_idx = idx
                best_key = candidate_key
                best_new_dims = new_dims

        if best_idx is None:
            break

        row = remaining_pool.pop(best_idx)
        if not row.get("is_propagation_chain"):
            standalone_used += 1

        selected.append(row)
        covered_dims.update(best_new_dims)
        cid = str(row.get("chain_id") or "")
        selection_meta[cid] = {
            "selection_order": int(len(selected)),
            "selection_stage": "dimension_coverage" if best_new_dims else "score_fill",
            "new_dimensions_when_selected": sorted(best_new_dims),
            "matched_dimensions": sorted(_coverage_dims(row)),
        }

    # Stable presentation order; selection_order still records the greedy path.
    selected.sort(
        key=lambda r: (1 if r.get("is_propagation_chain") else 0, *_rank_key(r)),
        reverse=True,
    )
    selected = selected[:max_chains]

    cumulative_dims: set[str] = set()
    for prompt_rank, row in enumerate(selected, start=1):
        cid = str(row.get("chain_id") or "")
        dims = _coverage_dims(row)
        new_prompt_dims = dims - cumulative_dims
        cumulative_dims.update(dims)
        meta = selection_meta.get(cid, {})
        row["retrieval_selection"] = {
            "selection_mode": "dimension_diverse",
            "retrieval_rank": int(prompt_rank),
            "selection_order": meta.get("selection_order"),
            "selection_stage": meta.get("selection_stage", "score_fill"),
            "matched_dimensions": sorted(dims),
            "new_dimensions_when_selected": meta.get("new_dimensions_when_selected", []),
            "new_dimensions_in_prompt_order": sorted(new_prompt_dims),
            "cumulative_dimension_coverage": sorted(cumulative_dims),
            "max_standalone_cap": int(max_standalone),
        }

    return selected

def _common_case_meta(category: str, analysis_cutoff: Optional[str], target_month: Optional[str]) -> Dict[str, Any]:
    category = canonical_category_name(category)
    return {
        "case_id": stable_id("CASE", category, analysis_cutoff, target_month, "abc_v1"),
        "product_category": category,
        "analysis_cutoff": analysis_cutoff,
        "target_month": target_month,
        "case_version": "abc_information_ablation_v1",
    }


def build_all_condition_packages(
    *,
    category: str,
    static_row: Dict[str, Any],
    dynamic_row: Dict[str, Any],
    api_cards: Sequence[Dict[str, Any]],
    static_relation_rows: Sequence[Dict[str, Any]] = (),
    risk_chain_path: Optional[str | Path] = None,
    analysis_cutoff: Optional[str] = None,
    target_month: Optional[str] = None,
    max_observed_cards: int = 6,
    statements_per_card: int = 2,
    max_relations: int = 5,
    max_api_cards: int = 3,
    max_chains: int = 4,
    max_standalone_chains: int = 1,
    chain_selection_mode: str = "dimension_diverse",
    # Backward-compatible alias; if supplied, it wins.
    selection_mode: Optional[str] = None,
    category_api_map_path: Optional[str | Path] = None,
) -> Dict[str, Dict[str, Any]]:
    category = canonical_category_name(category)
    analysis_cutoff = analysis_cutoff or _cutoff(static_row) or _cutoff(dynamic_row)
    target_month = target_month or dynamic_row.get("target_month")
    if selection_mode is not None:
        chain_selection_mode = str(selection_mode)

    observed = compact_observed_evidence(
        category, static_row,
        max_cards=max_observed_cards,
        statements_per_card=statements_per_card,
    )
    dimension_catalog = compact_observed_dimension_catalog(
        category, static_row
    )
    relations = compact_static_relations(
        category, static_relation_rows, analysis_cutoff, max_relations=max_relations
    )
    kb = retrieve_api_evidence_static_only(
        category=category,
        static_row=static_row,
        static_relations=relations,
        api_cards=api_cards,
        analysis_cutoff=analysis_cutoff,
        max_cards=max_api_cards,
        category_api_map_path=category_api_map_path,
    )
    model_facts = compact_dynamic_facts(category, dynamic_row)
    
    chains = load_optional_risk_chains(
        risk_chain_path,
        category,
        target_month=target_month,
        max_chains=max_chains,
        max_standalone=max_standalone_chains,
        exclude_general_dimension=True,
        # Condition C remains unconditioned by H anchors.
        allowed_dimensions=None,
        selection_mode=chain_selection_mode,
    )

    shared = {
        "case_meta": _common_case_meta(category, analysis_cutoff, target_month),
        "observed_dimension_catalog": dimension_catalog,
        "observed_evidence": observed,
        "ecosystem_relations": relations,
        "api_knowledge": kb,
        "retrieval_config": {
            "max_observed_cards": int(max_observed_cards),
            "statements_per_card": int(statements_per_card),
            "max_relations": int(max_relations),
            "max_api_cards": int(max_api_cards),
            "max_chains": int(max_chains),
            "max_standalone_chains": int(max_standalone_chains),
            "chain_selection_mode": str(chain_selection_mode),
        },
        "source_notes": [
            "OBS: observed dimension catalog plus detailed aggregated review evidence.",
            "REL: static ecosystem associations; not causal propagation.",
            "KB: candidate mechanism knowledge; not proof of implementation failure.",
            "MF/CHN, when supplied, are model-inferred and not causal proof.",
        ],
    }

    A = copy.deepcopy(shared)
    A["model_facts"] = []
    A["risk_chains"] = []

    B = copy.deepcopy(shared)
    B["model_facts"] = copy.deepcopy(model_facts)
    B["risk_chains"] = []

    C = copy.deepcopy(shared)
    C["model_facts"] = copy.deepcopy(model_facts)
    C["risk_chains"] = copy.deepcopy(chains)

    return {CONDITION_A: A, CONDITION_B: B, CONDITION_C: C}



def select_b_promoted_dimensions(
    *,
    core_dimensions: Sequence[str],
    observed_dimension_catalog: Sequence[Dict[str, Any]],
    dynamic_row: Dict[str, Any],
    max_promotions: int = 1,
    max_dynamic_rank: int = 3,
) -> Dict[str, Any]:
    """Select a tightly controlled set of B-promoted dimensions.

    Experimental contract
    ---------------------
    - ``max_promotions=0`` reproduces the original hard-anchor Hybrid.
    - The current B-promotion treatment allows at most one promoted dimension.
    - A promoted dimension must already exist in the observed dimension catalog.
    - B may promote an observed-but-noncore dimension only when its dynamic
      ``risk_rank`` is within ``max_dynamic_rank``.

    The function is deterministic and does not use an LLM.
    """
    max_promotions = int(max_promotions)
    max_dynamic_rank = int(max_dynamic_rank)
    if max_promotions not in {0, 1}:
        raise ValueError(
            "max_promotions must be 0 or 1 for the current controlled-promotion experiment"
        )
    if max_dynamic_rank < 1:
        raise ValueError("max_dynamic_rank must be >= 1")

    core = list(dict.fromkeys(
        str(x).strip()
        for x in core_dimensions
        if str(x).strip()
    ))

    eligible = {
        str(x.get("dimension")).strip()
        for x in observed_dimension_catalog
        if isinstance(x, dict)
        and str(x.get("dimension") or "").strip()
        and str(x.get("dimension")).strip().lower() != "general"
    }

    invalid_core = [dim for dim in core if dim not in eligible]
    if invalid_core:
        raise ValueError(
            "core_dimensions contains dimensions absent from observed_dimension_catalog: "
            f"{invalid_core}"
        )

    signals = _filtered_dynamic_dimension_signals(
        dynamic_row
    )

    candidates = []

    for sig in signals:
        dim = str(
            sig.get("dimension") or ""
        ).strip()

        if not dim:
            continue

        # Must already exist in observed evidence.
        if dim not in eligible:
            continue

        # Already selected by A -> no promotion needed.
        if dim in core:
            continue

        rank = sig.get("risk_rank")

        try:
            rank = int(rank)
        except Exception:
            continue

        if rank < 1 or rank > max_dynamic_rank:
            continue

        candidates.append({
            "dimension": dim,
            "dynamic_rank": rank,
            "predicted_next_risk":
                sig.get("predicted_next_risk"),
            "historical_risk_zscore":
                sig.get("historical_risk_zscore"),
            "predicted_risk_change":
                sig.get("predicted_risk_change"),
        })

    candidates.sort(
        key=lambda x: (
            x["dynamic_rank"],
            -_safe_float(x.get("predicted_next_risk"), 0.0),
            x["dimension"],
        )
    )

    promoted = [
        x["dimension"] for x in candidates[:max_promotions]
    ]

    final_dimensions = list(
        dict.fromkeys(core + promoted)
    )

    return {
        "core_dimensions": core,
        "eligible_dimensions": sorted(eligible),
        "promotion_candidates": candidates,
        "promoted_dimensions": promoted,
        "final_dimensions": final_dimensions,
        "promotion_rule": {
            "max_promotions": max_promotions,
            "max_dynamic_rank": max_dynamic_rank,
            "must_be_observed_dimension": True,
        },
    }
        
        
def build_hybrid_case_package(
    *,
    base_c_package: Dict[str, Any],
    dynamic_row: Dict[str, Any],
    risk_chain_path: Optional[str | Path],
    anchor_dimensions: Sequence[str],
    max_chains: int = 4,
    max_standalone_chains: int = 1,
    max_promotions: int = 1,
    max_dynamic_rank: int = 3,
    chain_selection_mode: str = "dimension_diverse",
    # Backward-compatible alias; if supplied, it wins.
    selection_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the promotion-aware Hybrid package.

    Pipeline
    --------
    A observed evidence -> core anchor
    -> deterministic controlled B promotion
    -> final anchor = core + promoted
    -> B facts conditioned on the final anchor
    -> C chains retrieved only when they match the final anchor.

    ``max_promotions=0`` reproduces the original hard-anchor Hybrid. A missing
    matched chain remains a valid outcome: it means propagation is unsupported.
    """
    if not isinstance(base_c_package, dict):
        raise TypeError("base_c_package must be a dict")

    if selection_mode is not None:
        chain_selection_mode = str(selection_mode)

    max_promotions = int(max_promotions)
    max_dynamic_rank = int(max_dynamic_rank)
    if max_promotions not in {0, 1}:
        raise ValueError(
            "max_promotions must be 0 or 1 for the current controlled-promotion experiment"
        )
    if max_dynamic_rank < 1:
        raise ValueError("max_dynamic_rank must be >= 1")
    meta = base_c_package.get("case_meta") if isinstance(base_c_package.get("case_meta"), dict) else {}
    category = canonical_category_name(meta.get("product_category"))
    target_month = meta.get("target_month")
    anchors = [
        str(x).strip()
        for x in anchor_dimensions
        if str(x).strip() and not _is_general_dimension(x)
    ]
    core_anchors = list(dict.fromkeys(anchors))
    
    promotion = select_b_promoted_dimensions(
        core_dimensions=core_anchors,
        observed_dimension_catalog=base_c_package.get("observed_dimension_catalog", []),
        dynamic_row=dynamic_row,
        max_promotions=max_promotions,
        max_dynamic_rank=max_dynamic_rank,
    )
    final_anchors = promotion["final_dimensions"]

    H = copy.deepcopy(base_c_package)
    H["case_meta"] = dict(meta)
    H["case_meta"]["condition"] = CONDITION_H
    H["case_meta"]["pipeline_mode"] = (
        "A_core_hard_anchor_BC_conditioned"
        if max_promotions == 0
        else "A_core_anchor_B_controlled_promotion_C_conditioned"
    )
    
    H["dimension_anchor"] = {
        "case_id": meta.get("case_id"),
        "product_category": category,
        "source": (
            "A_core_hard_anchor"
            if max_promotions == 0
            else "A_core_plus_controlled_B_promotion"
        ),
        "core_risk_dimensions": list(core_anchors),
        "promoted_risk_dimensions": list(promotion["promoted_dimensions"]),
    
        # This is the ONLY dimension set that downstream H diagnosis is allowed to use.
        "risk_dimensions": list(final_anchors),
        "promotion_rule": dict(promotion["promotion_rule"]),
    }

    facts, missing_dynamic = (
        compact_anchor_dynamic_facts(
            category=category,
            dynamic_row=dynamic_row,
            anchor_dimensions=final_anchors,
        )
    )
    
    chains = load_optional_risk_chains(
        risk_chain_path,
        category,
        target_month=target_month,
        max_chains=max_chains,
        max_standalone=max_standalone_chains,
        exclude_general_dimension=True,
        allowed_dimensions=final_anchors,
        selection_mode=chain_selection_mode,
    )
    
    H["model_facts"] = facts
    H["risk_chains"] = chains
    matched_chain_dims: set[str] = set()
    for ch in chains:
        matched_chain_dims.update(_chain_dimensions(ch) & set(final_anchors))    
    H["hybrid_routing"] = {
        "core_anchor_dimensions":core_anchors,
        "eligible_observed_dimensions":promotion["eligible_dimensions"],
        "promoted_dimensions":promotion["promoted_dimensions"],
        "final_anchor_dimensions":final_anchors,
        "promotion_candidates":promotion["promotion_candidates"],
        "dynamic_matched_dimensions": sorted(set(final_anchors) - set(missing_dynamic)),
        "dynamic_missing_dimensions": missing_dynamic,
        "chain_matched_dimensions": sorted(matched_chain_dims),
        "n_anchor_dimensions": len(final_anchors),
        "n_anchor_dynamic_matched": len(set(final_anchors) - set(missing_dynamic)),
        "n_anchor_chains": len(chains),
        "n_unique_chain_dimensions": len(matched_chain_dims),
        "anchor_chain_dimension_coverage": (
            len(matched_chain_dims) / len(final_anchors) if final_anchors else None
        ),
        "chain_selection_mode": str(chain_selection_mode),
        "general_excluded": True,
        "promotion_mode": ("none" if max_promotions == 0 else "b_controlled"),
        "max_promotions": max_promotions,
        "max_dynamic_rank": max_dynamic_rank,
    }
            
    return H


def build_case_package(
    *,
    category: str,
    static_row: Dict[str, Any],
    dynamic_row: Dict[str, Any],
    api_cards: Sequence[Dict[str, Any]],
    condition: str = CONDITION_C,
    static_relation_rows: Sequence[Dict[str, Any]] = (),
    risk_chain_path: Optional[str | Path] = None,
    analysis_cutoff: Optional[str] = None,
    target_month: Optional[str] = None,
    max_observed_cards: int = 6,
    statements_per_card: int = 2,
    max_relations: int = 5,
    max_api_cards: int = 3,
    max_chains: int = 4,
    max_standalone_chains: int = 1,
    chain_selection_mode: str = "dimension_diverse",
    # Backward-compatible alias; if supplied, it wins.
    selection_mode: Optional[str] = None,
    category_api_map_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    if selection_mode is not None:
        chain_selection_mode = str(selection_mode)

    packages = build_all_condition_packages(
        category=category,
        static_row=static_row,
        dynamic_row=dynamic_row,
        api_cards=api_cards,
        static_relation_rows=static_relation_rows,
        risk_chain_path=risk_chain_path,
        analysis_cutoff=analysis_cutoff,
        target_month=target_month,
        max_observed_cards=max_observed_cards,
        statements_per_card=statements_per_card,
        max_relations=max_relations,
        max_api_cards=max_api_cards,
        max_chains=max_chains,
        max_standalone_chains=max_standalone_chains,
        chain_selection_mode=chain_selection_mode,
        category_api_map_path=category_api_map_path,
    )
    if condition not in packages:
        raise ValueError(f"Unknown condition={condition!r}; choose from {list(ALL_CONDITIONS)}")
    return packages[condition]


def all_reference_ids(case_package: Dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for x in case_package.get("observed_dimension_catalog", []):
        if isinstance(x, dict) and x.get("evidence_id"):
            ids.add(str(x["evidence_id"]))
    for x in case_package.get("observed_evidence", []):
        if x.get("evidence_id"): ids.add(str(x["evidence_id"]))
    for x in case_package.get("model_facts", []):
        if x.get("fact_id"): ids.add(str(x["fact_id"]))
    for x in case_package.get("ecosystem_relations", []):
        if x.get("relation_id"): ids.add(str(x["relation_id"]))
    for x in case_package.get("risk_chains", []):
        if x.get("chain_id"): ids.add(str(x["chain_id"]))
    for x in case_package.get("api_knowledge", []):
        if x.get("knowledge_id"): ids.add(str(x["knowledge_id"]))
    return ids


# ============================================================
# LLM-only compact views
# ============================================================

def _round_for_llm(value: Any, digits: int = 4) -> Any:
    """Round numeric values for compact LLM payloads without altering audit data."""
    if isinstance(value, bool) or value is None:
        return value
    try:
        return round(float(value), digits)
    except Exception:
        return value


def compact_relation_for_llm(row: Dict[str, Any]) -> Dict[str, Any]:
    """Minimal relation view preserving ID, counterpart, strength, and channel pattern."""
    channels = row.get("channel_strengths") if isinstance(row.get("channel_strengths"), dict) else {}
    channel_view: Dict[str, Any] = {}
    for name in ("api", "coo", "user"):
        obj = channels.get(name)
        if not isinstance(obj, dict):
            continue
        value = obj.get("undirected_strength")
        if value is None:
            value = obj.get("focal_to_other")
        if value is not None:
            channel_view[name] = _round_for_llm(value)

    out = {
        "relation_id": row.get("relation_id"),
        "related_product": row.get("related_product"),
        "overall_relation_strength": _round_for_llm(row.get("overall_relation_strength")),
    }
    if channel_view:
        out["channel_strengths"] = channel_view
    return {k: v for k, v in out.items() if v not in (None, "", [], {})}


def compact_api_knowledge_for_llm(
    row: Dict[str, Any],
    *,
    max_dependencies: int = 3,
    max_constraints: int = 3,
    max_root_causes: int = 3,
    max_keywords: int = 5,
) -> Dict[str, Any]:
    """Compact a selected API card for mechanism reasoning.

    The full API card remains in the research case package. This view removes
    repeated metadata and long-tail documentation details while preserving the
    knowledge ID required by diagnosis validation.
    """
    deps: List[Any] = []
    for dep in (row.get("dependencies") or [])[: max(0, int(max_dependencies))]:
        if isinstance(dep, dict):
            item = {
                "api_name": dep.get("api_name"),
                "relation_type": dep.get("relation_type"),
            }
            item = {k: v for k, v in item.items() if v not in (None, "")}
            if item:
                deps.append(item)
        else:
            text = str(dep).strip()
            if text:
                deps.append(text)

    out = {
        "knowledge_id": row.get("knowledge_id"),
        "api_name": row.get("api_name"),
        "functionality": row.get("functionality"),
        "dependencies": deps,
        "design_constraints": _clean_list(row.get("design_constraints"), max_constraints),
        "common_root_causes": _clean_list(row.get("common_root_causes"), max_root_causes),
        "diagnostic_keywords": _clean_list(row.get("diagnostic_keywords"), max_keywords),
    }
    return {k: v for k, v in out.items() if v not in (None, "", [], {})}


def compact_model_fact_for_llm(row: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only decision-relevant fields from one model fact."""
    fact_type = str(row.get("fact_type") or "").strip()
    out: Dict[str, Any] = {
        "fact_id": row.get("fact_id"),
        "fact_type": fact_type,
    }

    if fact_type == "predicted_risk_state":
        out.update({
            "predicted_next_risk_score": _round_for_llm(row.get("predicted_next_risk_score")),
            "predicted_next_risk_level": row.get("predicted_next_risk_level"),
        })
        dims = []
        for sig in (row.get("top_predicted_risk_dimensions") or [])[:5]:
            if not isinstance(sig, dict):
                continue
            d = {
                "dimension": sig.get("dimension"),
                "risk_rank": sig.get("risk_rank"),
                "predicted_next_risk": _round_for_llm(sig.get("predicted_next_risk")),
                "predicted_risk_change": _round_for_llm(sig.get("predicted_risk_change")),
                "dominant_driver": sig.get("dominant_driver"),
            }
            dims.append({k: v for k, v in d.items() if v not in (None, "")})
        if dims:
            out["top_predicted_risk_dimensions"] = dims

    elif fact_type == "evolution_persistence":
        out.update({
            "persistence_score": _round_for_llm(row.get("persistence_score")),
            "persistence": row.get("persistence"),
        })

    elif fact_type == "dynamic_network_drivers":
        out.update({
            "dominant_driver": row.get("dominant_driver"),
            "beta_api": _round_for_llm(row.get("beta_api")),
            "beta_coo": _round_for_llm(row.get("beta_coo")),
            "beta_user": _round_for_llm(row.get("beta_user")),
        })

    elif fact_type == "dynamic_network_position":
        out.update({
            "spillover_output_score": _round_for_llm(row.get("spillover_output_score")),
            "spillover_output_level": row.get("spillover_output_level"),
            "spillover_exposure_score": _round_for_llm(row.get("spillover_exposure_score")),
            "spillover_exposure_level": row.get("spillover_exposure_level"),
        })

    elif fact_type == "anchored_dimension_forecast":
        out.update({
            "dimension": row.get("dimension"),
            "predicted_next_risk": _round_for_llm(row.get("predicted_next_risk")),
            "predicted_risk_change": _round_for_llm(row.get("predicted_risk_change")),
            "historical_risk_zscore": _round_for_llm(row.get("historical_risk_zscore")),
            "dominant_driver": row.get("dominant_driver"),
            "spillover_output": _round_for_llm(row.get("spillover_output")),
            "spillover_exposure": _round_for_llm(row.get("spillover_exposure")),
        })

    elif fact_type == "anchored_product_dynamic_context":
        out.update({
            "persistence_score": _round_for_llm(row.get("persistence_score")),
            "persistence": row.get("persistence"),
            "dominant_driver": row.get("dominant_driver"),
            "spillover_output_score": _round_for_llm(row.get("spillover_output_score")),
            "spillover_exposure_score": _round_for_llm(row.get("spillover_exposure_score")),
        })

    else:
        # Conservative fallback for future fact types.
        for key in (
            "dimension", "risk_rank", "predicted_next_risk", "predicted_risk_change",
            "persistence", "dominant_driver", "spillover_output_score",
            "spillover_exposure_score",
        ):
            if key in row:
                out[key] = _round_for_llm(row.get(key))

    return {k: v for k, v in out.items() if v not in (None, "", [], {})}


def compact_risk_chain_for_llm(row: Dict[str, Any], max_edges: int = 3) -> Dict[str, Any]:
    """Compact a risk-chain record while retaining traceable path structure."""
    out: Dict[str, Any] = {
        "chain_id": row.get("chain_id"),
        "chain_kind": row.get("chain_kind"),
        "is_propagation_chain": row.get("is_propagation_chain"),
        "matched_dimensions": row.get("matched_dimensions") or [],
        "chain_score": _round_for_llm(row.get("chain_score")),
        "source_risk_score": _round_for_llm(row.get("source_risk_score")),
        "source_category": row.get("source_category"),
        "source_dimension": row.get("source_dimension"),
        "target_category": row.get("target_category"),
        "target_dimension": row.get("target_dimension"),
    }

    edges: List[Dict[str, Any]] = []
    keep_edge_keys = (
        "from_category", "from_dimension", "to_category", "to_dimension",
        "dimension", "relation_type", "edge_type", "channel", "weight", "strength",
    )
    for edge in (row.get("edges") or [])[: max(0, int(max_edges))]:
        if not isinstance(edge, dict):
            continue
        compact_edge: Dict[str, Any] = {}
        for key in keep_edge_keys:
            if key not in edge:
                continue
            value = edge.get(key)
            if key in {"weight", "strength"}:
                value = _round_for_llm(value)
            if value not in (None, "", [], {}):
                compact_edge[key] = value
        if compact_edge:
            edges.append(compact_edge)
    if edges:
        out["edges"] = edges
    elif row.get("path_text"):
        out["path_text"] = str(row.get("path_text"))[:400]

    return {k: v for k, v in out.items() if v not in (None, "", [], {})}


def _compact_observed_catalog_row_for_llm(row: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "evidence_id": row.get("evidence_id"),
        "dimension": row.get("dimension"),
        "n_review": row.get("n_review"),
        "mean_sentiment": _round_for_llm(row.get("mean_sentiment")),
        "recent_review_count": row.get("recent_review_count"),
        "review_share_in_product": _round_for_llm(row.get("review_share_in_product")),
    }
    return {k: v for k, v in out.items() if v not in (None, "", [], {})}


def _compact_observed_evidence_row_for_llm(
    row: Dict[str, Any],
    *,
    max_statements: int = 2,
) -> Dict[str, Any]:
    severity = row.get("severity") if isinstance(row.get("severity"), dict) else {}

    failures: List[Any] = []
    for item in (row.get("top_failure_types") or [])[:2]:
        if isinstance(item, dict):
            obj = {
                "failure_type": item.get("failure_type"),
                "count": item.get("count"),
            }
            failures.append({k: v for k, v in obj.items() if v not in (None, "")})
        else:
            text = str(item).strip()
            if text:
                failures.append(text)

    scenarios: List[Any] = []
    for item in (row.get("top_scenarios") or [])[:2]:
        if isinstance(item, dict):
            obj = {
                "scenario": item.get("scenario"),
                "count": item.get("count"),
            }
            scenarios.append({k: v for k, v in obj.items() if v not in (None, "")})
        else:
            text = str(item).strip()
            if text:
                scenarios.append(text)

    out = {
        "evidence_id": row.get("evidence_id"),
        "dimension": row.get("dimension"),
        "top_failure_types": failures,
        "top_scenarios": scenarios,
        "severity": {
            k: (_round_for_llm(severity.get(k)) if k == "mean_sentiment" else severity.get(k))
            for k in ("n_review", "mean_sentiment", "recent_review_count")
            if severity.get(k) is not None
        },
        "representative_statements": _clean_list(
            row.get("representative_statements"),
            max(0, int(max_statements)),
        ),
    }
    return {k: v for k, v in out.items() if v not in (None, "", [], {})}


def build_llm_case_view(
    case_package: Dict[str, Any],
    *,
    required_dimensions: Optional[Sequence[str]] = None,
    max_statements: int = 2,
) -> Dict[str, Any]:
    """Create a compact LLM-only view while preserving the full audit package.

    A/B/C/H treatment membership is unchanged: the function only removes
    redundant fields inside each evidence object. All evidence/reference IDs
    required by the validator are preserved.
    """
    if not isinstance(case_package, dict):
        raise TypeError("case_package must be a dict")

    required: Optional[Set[str]] = None
    if required_dimensions is not None:
        required = {
            str(x).strip()
            for x in required_dimensions
            if str(x).strip() and not _is_general_dimension(x)
        }

    def _dim_ok(row: Dict[str, Any]) -> bool:
        if required is None:
            return True
        return str(row.get("dimension") or "").strip() in required

    meta = case_package.get("case_meta") if isinstance(case_package.get("case_meta"), dict) else {}
    out: Dict[str, Any] = {
        "case_meta": {
            k: meta.get(k)
            for k in ("case_id", "product_category", "analysis_cutoff", "target_month")
            if k in meta
        },
        "observed_dimension_catalog": [
            _compact_observed_catalog_row_for_llm(x)
            for x in (case_package.get("observed_dimension_catalog") or [])
            if isinstance(x, dict) and _dim_ok(x)
        ],
        "observed_evidence": [
            _compact_observed_evidence_row_for_llm(x, max_statements=max_statements)
            for x in (case_package.get("observed_evidence") or [])
            if isinstance(x, dict) and _dim_ok(x)
        ],
        "ecosystem_relations": [
            compact_relation_for_llm(x)
            for x in (case_package.get("ecosystem_relations") or [])
            if isinstance(x, dict)
        ],
        "api_knowledge": [
            compact_api_knowledge_for_llm(x)
            for x in (case_package.get("api_knowledge") or [])
            if isinstance(x, dict)
        ],
        "model_facts": [
            compact_model_fact_for_llm(x)
            for x in (case_package.get("model_facts") or [])
            if isinstance(x, dict)
        ],
        "risk_chains": [
            compact_risk_chain_for_llm(x)
            for x in (case_package.get("risk_chains") or [])
            if isinstance(x, dict)
        ],
    }
    return out


__all__ = [
    "CONDITION_A", "CONDITION_B", "CONDITION_C", "CONDITION_H",
    "ALL_CONDITIONS", "ALL_EXPERIMENT_CONDITIONS",
    "load_jsonl", "save_json", "append_jsonl", "stable_id", "canonical_text",
    "canonical_category_name", "load_category_api_map",
    "build_profile_index", "resolve_profile_row",
    "compact_observed_evidence", "compact_observed_dimension_catalog",
    "compact_dynamic_facts", "compact_anchor_dynamic_facts",
    "build_all_condition_packages", "build_hybrid_case_package",
    "build_case_package", "all_reference_ids", "load_optional_risk_chains",
    "retrieve_api_evidence_static_only", "select_b_promoted_dimensions",
    "build_llm_case_view", "compact_relation_for_llm",
    "compact_api_knowledge_for_llm", "compact_model_fact_for_llm",
    "compact_risk_chain_for_llm",
]
