from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


_SAFE_KEY_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_workflow_id(
    *,
    product_category: str,
    analysis_cutoff: str,
    target_month: str,
) -> str:
    """Create a deterministic filesystem-safe ID for one product-month workflow."""
    category = str(product_category or "").strip()
    cutoff = str(analysis_cutoff or "").strip()
    target = str(target_month or "").strip()
    if not category or not cutoff or not target:
        raise ValueError(
            "product_category, analysis_cutoff, and target_month are required"
        )

    raw = f"{category}|{cutoff}|{target}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    # Keep file names portable. Non-ASCII categories remain uniquely represented
    # by the digest even when the readable slug falls back to "product".
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", category).strip("-") or "product"
    return f"H-{cutoff}-{target}-{slug[:40]}-{digest}"


def _validate_storage_key(value: str, *, field: str) -> str:
    key = str(value or "").strip()
    if not key:
        raise ValueError(f"{field} must be non-empty")
    if key in {".", ".."} or not _SAFE_KEY_RE.fullmatch(key):
        raise ValueError(
            f"{field} contains unsafe path characters: {key!r}"
        )
    return key


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, datetime):
        return value.isoformat()
    # numpy/pandas scalar values can enter ecosystem selection metadata via
    # DataFrame rows. Convert scalar-like objects without importing numpy/pandas.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            scalar = item()
        except Exception:
            scalar = None
        if scalar is not None and scalar is not value:
            return scalar
    # pandas Period/Timestamp and similar metadata types are stable as strings.
    module = type(value).__module__
    if module.startswith("pandas"):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _atomic_write_json(target: Path, payload: Dict[str, Any]) -> Path:
    """Write JSON through a unique same-directory temp file, then os.replace()."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        # os.replace is atomic when source/target are on the same filesystem.
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return target


def save_workflow(record: Dict[str, Any], root: str | Path) -> Path:
    """Atomically persist one product workflow.

    This file is the authoritative durable state. UI/session state should be
    treated only as a cache. The input record is not mutated.
    """
    if not isinstance(record, dict):
        raise TypeError("workflow record must be a dict")
    workflow_id = _validate_storage_key(
        record.get("workflow_id"),
        field="workflow_id",
    )

    root_path = Path(root)
    target = root_path / f"{workflow_id}.json"
    payload = dict(record)
    payload["updated_at"] = utc_now_iso()
    return _atomic_write_json(target, payload)


def save_named_artifact(name: str, record: Dict[str, Any], root: str | Path) -> Path:
    """Atomically persist a non-product artifact such as ecosystem synthesis."""
    if not isinstance(record, dict):
        raise TypeError("artifact record must be a dict")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(name or "")).strip("-")
    safe = _validate_storage_key(safe, field="artifact name")

    root_path = Path(root)
    target = root_path / f"{safe}.json"
    payload = dict(record)
    payload["updated_at"] = utc_now_iso()
    return _atomic_write_json(target, payload)


def load_workflow(workflow_id: str, root: str | Path) -> Dict[str, Any]:
    safe_id = _validate_storage_key(workflow_id, field="workflow_id")
    path = Path(root) / f"{safe_id}.json"
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Workflow file must contain one JSON object: {path}")
    return obj


def list_workflows(root: str | Path) -> List[Dict[str, Any]]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for path in sorted(root_path.glob("H-*.json")):
        obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            raise ValueError(f"Workflow file must contain one JSON object: {path}")
        rows.append(obj)
    return rows


__all__ = [
    "utc_now_iso",
    "make_workflow_id",
    "save_workflow",
    "save_named_artifact",
    "load_workflow",
    "list_workflows",
]
