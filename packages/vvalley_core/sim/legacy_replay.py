"""Adapter utilities for replaying legacy generative_agents simulations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import json


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_COMPRESSED_STORAGE = (
    WORKSPACE_ROOT
    / "generative_agents"
    / "environment"
    / "frontend_server"
    / "compressed_storage"
)


def _resolve_root(root: Optional[Path] = None) -> Path:
    return (root or DEFAULT_COMPRESSED_STORAGE).resolve()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def list_legacy_simulations(root: Optional[Path] = None) -> list[dict[str, Any]]:
    storage_root = _resolve_root(root)
    if not storage_root.exists():
        return []

    out: list[dict[str, Any]] = []
    for sim_dir in sorted(storage_root.iterdir()):
        if not sim_dir.is_dir():
            continue
        meta_path = sim_dir / "meta.json"
        movement_path = sim_dir / "master_movement.json"
        if not meta_path.exists() or not movement_path.exists():
            continue

        try:
            meta = _load_json(meta_path)
            movement = _load_json(movement_path)
        except Exception:
            continue

        step_count = int(meta.get("step") or len(movement))
        out.append(
            {
                "simulation_id": sim_dir.name,
                "maze_name": meta.get("maze_name"),
                "curr_time": meta.get("curr_time"),
                "sec_per_step": int(meta.get("sec_per_step") or 10),
                "step_count": step_count,
                "persona_names": list(meta.get("persona_names") or []),
                "persona_count": len(list(meta.get("persona_names") or [])),
            }
        )
    return out


def replay_events(
    *,
    simulation_id: str,
    start_step: int = 0,
    max_steps: int = 120,
    root: Optional[Path] = None,
) -> dict[str, Any]:
    storage_root = _resolve_root(root)
    sim_dir = (storage_root / simulation_id).resolve()
    if not sim_dir.exists() or not sim_dir.is_dir():
        raise FileNotFoundError(f"Legacy simulation not found: {simulation_id}")

    meta_path = sim_dir / "meta.json"
    movement_path = sim_dir / "master_movement.json"
    if not meta_path.exists() or not movement_path.exists():
        raise FileNotFoundError(
            f"Legacy simulation missing meta/movement files: {simulation_id}"
        )

    meta = _load_json(meta_path)
    movement_by_step: dict[str, Any] = _load_json(movement_path)

    max_steps = max(1, int(max_steps))
    start = max(0, int(start_step))
    end = start + max_steps - 1

    events: list[dict[str, Any]] = []
    for step in range(start, end + 1):
        frame = movement_by_step.get(str(step), {})
        if not isinstance(frame, dict):
            continue

        for persona_name, data in frame.items():
            move = data.get("movement") if isinstance(data, dict) else None
            x = int(move[0]) if isinstance(move, list) and len(move) >= 1 else None
            y = int(move[1]) if isinstance(move, list) and len(move) >= 2 else None
            events.append(
                {
                    "step": step,
                    "persona": str(persona_name),
                    "x": x,
                    "y": y,
                    "pronunciatio": data.get("pronunciatio") if isinstance(data, dict) else None,
                    "description": data.get("description") if isinstance(data, dict) else None,
                    "chat": data.get("chat") if isinstance(data, dict) else None,
                }
            )

    return {
        "simulation_id": simulation_id,
        "maze_name": meta.get("maze_name"),
        "sec_per_step": int(meta.get("sec_per_step") or 10),
        "persona_names": list(meta.get("persona_names") or []),
        "start_step": start,
        "end_step": end,
        "max_steps": max_steps,
        "event_count": len(events),
        "events": events,
    }

