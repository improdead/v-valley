"""Scenario matchmaking and lifecycle orchestration."""

from __future__ import annotations

from datetime import datetime, timezone
import random
from typing import Any

from packages.vvalley_core.sim.scenarios.cards import (
    arrange_reveal_order,
    choose_pass_cards,
    evaluate_best_five,
    hand_category_label,
    make_shuffled_deck,
)
from packages.vvalley_core.sim.scenarios.manager import TownScenarioManager
from packages.vvalley_core.sim.scenarios.rating import compute_pairwise_elo_updates

from ..storage.agents import list_active_agents_in_town
from ..storage import scenarios as scenario_store
from ..storage.interaction_hub import create_inbox_item


ACTIVE_MATCH_STATUSES = {"forming", "warmup", "active"}
RESOLVED_MATCH_STATUSES = {"resolved", "cancelled", "expired"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _rng(seed: str) -> random.Random:
    return random.Random(seed)


def _phase_sequence(scenario: dict[str, Any]) -> list[str]:
    rules = scenario.get("rules_json") or {}
    phases = rules.get("phase_sequence") if isinstance(rules, dict) else None
    if isinstance(phases, list):
        out = [str(value).strip() for value in phases if str(value).strip()]
        if out:
            return out
    if str(scenario.get("scenario_key") or "").startswith("werewolf"):
        return ["night", "day"]
    return ["deal", "pass1", "pass2", "pass3", "discard", "reveal1", "reveal2", "reveal3", "reveal4", "reveal5", "showdown"]


def _arena_id(scenario: dict[str, Any]) -> str:
    rules = scenario.get("rules_json") or {}
    if isinstance(rules, dict) and str(rules.get("arena_id") or "").strip():
        return str(rules.get("arena_id")).strip()
    key = str(scenario.get("scenario_key") or "").lower()
    if "werewolf" in key:
        return "park_circle"
    return "hobbs_cafe_table"


def _werewolf_role_distribution(player_count: int) -> dict[str, int]:
    if player_count <= 6:
        return {"werewolf": 2, "seer": 1, "villager": max(0, player_count - 3)}
    if player_count <= 8:
        return {"werewolf": 2, "seer": 1, "doctor": 1, "villager": max(0, player_count - 4)}
    if player_count == 9:
        return {"werewolf": 3, "seer": 1, "doctor": 1, "villager": 4}
    return {"werewolf": 3, "seer": 1, "doctor": 1, "hunter": 1, "villager": max(0, player_count - 6)}


def _assign_werewolf_roles(*, agent_ids: list[str], match_id: str) -> dict[str, str]:
    ids = list(agent_ids)
    rng = _rng(f"roles:{match_id}:{','.join(sorted(ids))}")
    rng.shuffle(ids)
    distribution = _werewolf_role_distribution(len(ids))
    roles: list[str] = []
    for role, count in distribution.items():
        roles.extend([role] * max(0, int(count)))
    if len(roles) < len(ids):
        roles.extend(["villager"] * (len(ids) - len(roles)))
    role_map: dict[str, str] = {}
    for idx, agent_id in enumerate(ids):
        role_map[agent_id] = roles[idx] if idx < len(roles) else "villager"
    return role_map


def _is_werewolf_role(role: str | None) -> bool:
    value = str(role or "").strip().lower()
    return value in {"werewolf", "alpha_werewolf"}


def _ranking_score(*, agent_id: str, winners: set[str], losers: set[str], draws: bool) -> float:
    if draws:
        return 0.5
    if agent_id in winners:
        return 1.0
    if agent_id in losers:
        return 0.0
    return 0.5


def _compute_rating_updates(
    *,
    scenario_key: str,
    participant_ids: list[str],
    winners: set[str],
    draws: bool,
) -> dict[str, dict[str, int]]:
    ratings: dict[str, dict[str, Any]] = {
        agent_id: scenario_store.get_rating(agent_id=agent_id, scenario_key=scenario_key)
        for agent_id in participant_ids
    }
    return compute_pairwise_elo_updates(
        ratings=ratings,
        participant_ids=participant_ids,
        winners=winners,
        draws=draws,
    )


def _ensure_wallet_for_buy_in(*, agent_id: str, buy_in: int) -> dict[str, Any]:
    wallet = scenario_store.get_wallet(agent_id=agent_id, create=True)
    balance = int(wallet.get("balance") or 0)
    if balance >= int(buy_in):
        return wallet

    # Safety net stipend once per UTC day.
    if balance < 50:
        last_stipend = str(wallet.get("last_stipend") or "").strip()
        today = datetime.now(timezone.utc).date().isoformat()
        if not last_stipend.startswith(today):
            wallet = scenario_store.update_wallet(
                agent_id=agent_id,
                delta=100,
                stipend_iso=f"{today}T00:00:00Z",
            )
    return wallet


def join_queue(
    *,
    scenario_key: str,
    town_id: str,
    agent_id: str,
    current_step: int | None = None,
) -> dict[str, Any]:
    scenario = scenario_store.get_scenario(scenario_key=scenario_key)
    if not scenario or not bool(scenario.get("enabled", False)):
        raise ValueError(f"Scenario not found or disabled: {scenario_key}")

    active_match = scenario_store.get_agent_active_match(agent_id=agent_id)
    if active_match is not None:
        raise RuntimeError("Agent is already in an active match")

    existing = scenario_store.list_queue(
        scenario_key=scenario_key,
        town_id=town_id,
        status="queued",
    )
    for item in existing:
        if str(item.get("agent_id")) == str(agent_id):
            return {
                "queued": True,
                "already_queued": True,
                "entry": item,
                "formed_matches": [],
            }

    buy_in = max(0, int(scenario.get("buy_in") or 0))
    if buy_in > 0:
        wallet = _ensure_wallet_for_buy_in(agent_id=agent_id, buy_in=buy_in)
        if int(wallet.get("balance") or 0) < buy_in:
            raise RuntimeError("Insufficient wallet balance for buy-in")

    rating = scenario_store.get_rating(agent_id=agent_id, scenario_key=scenario_key)
    entry = scenario_store.enqueue(
        scenario_key=scenario_key,
        agent_id=agent_id,
        town_id=town_id,
        rating_snapshot=int(rating.get("rating") or 1500),
    )
    resolved_step = max(0, int(current_step)) if current_step is not None else 0
    formed = form_matches(
        town_id=town_id,
        scenario_key=scenario_key,
        current_step=resolved_step,
    )
    return {
        "queued": True,
        "already_queued": False,
        "entry": entry,
        "formed_matches": formed,
    }


def leave_queue(*, scenario_key: str, agent_id: str) -> dict[str, Any]:
    queued = scenario_store.list_queue(scenario_key=scenario_key, status="queued")
    updated = 0
    for entry in queued:
        if str(entry.get("agent_id")) != str(agent_id):
            continue
        scenario_store.update_queue_status(entry_id=str(entry["entry_id"]), status="cancelled", match_id=None)
        updated += 1
    return {"left": bool(updated), "entries_updated": int(updated)}


def _collect_town_member_map(town_id: str) -> dict[str, dict[str, Any]]:
    members = list_active_agents_in_town(town_id)
    out: dict[str, dict[str, Any]] = {}
    for member in members:
        aid = str(member.get("agent_id") or "").strip()
        if not aid:
            continue
        out[aid] = member
    return out


def _scenario_display_name(scenario: dict[str, Any]) -> str:
    return str(scenario.get("name") or scenario.get("scenario_key") or "Scenario")


def _notify_match_formed(
    *,
    scenario: dict[str, Any],
    match: dict[str, Any],
    participants: list[str],
    current_step: int,
) -> None:
    scenario_name = _scenario_display_name(scenario)
    for aid in participants:
        create_inbox_item(
            agent_id=str(aid),
            kind="scenario_match_found",
            summary=f"Match found: {scenario_name}",
            payload={
                "match_id": str(match.get("match_id") or ""),
                "town_id": str(match.get("town_id") or ""),
                "scenario_key": str(match.get("scenario_key") or ""),
                "scenario_name": scenario_name,
                "arena_id": str(match.get("arena_id") or ""),
                "step": int(current_step),
            },
            dedupe_key=f"scenario_match_found:{match.get('match_id')}:{aid}",
        )


def _notify_match_resolved(
    *,
    match: dict[str, Any],
    scenario: dict[str, Any],
    participants: list[dict[str, Any]],
    result_json: dict[str, Any],
) -> None:
    scenario_name = _scenario_display_name(scenario)
    winner_set = {str(aid) for aid in (result_json.get("winners") or [])}
    rating_map = result_json.get("ratings") or {}
    payouts = result_json.get("wallet_payouts") or {}
    reason = str(result_json.get("reason") or "Match resolved")
    for participant in participants:
        aid = str(participant.get("agent_id") or "")
        if not aid:
            continue
        rating = rating_map.get(aid) if isinstance(rating_map, dict) else None
        rating_delta = int((rating or {}).get("delta") or 0)
        payout = int((payouts or {}).get(aid) or 0)
        won = aid in winner_set
        create_inbox_item(
            agent_id=aid,
            kind="scenario_match_resolved",
            summary=f"{scenario_name} {'win' if won else 'result'}: {reason}",
            payload={
                "match_id": str(match.get("match_id") or ""),
                "town_id": str(match.get("town_id") or ""),
                "scenario_key": str(match.get("scenario_key") or ""),
                "scenario_name": scenario_name,
                "won": bool(won),
                "rating_delta": int(rating_delta),
                "wallet_payout": int(payout),
                "reason": reason,
            },
            dedupe_key=f"scenario_match_resolved:{match.get('match_id')}:{aid}",
        )


def _record_scenario_memory(
    *,
    agent_id: str,
    match_id: str,
    kind: str,
    summary: str,
    payload: dict[str, Any] | None = None,
    dedupe_suffix: str | None = None,
) -> None:
    dedupe = f"scenario_memory:{match_id}:{agent_id}:{kind}:{dedupe_suffix or ''}"
    create_inbox_item(
        agent_id=str(agent_id),
        kind="scenario_memory_note",
        summary=str(summary)[:240],
        payload={
            "match_id": str(match_id),
            "kind": str(kind),
            **(payload or {}),
        },
        dedupe_key=dedupe,
    )


def _planner_call(planner: Any, method_name: str, context: dict[str, Any]) -> dict[str, Any] | None:
    if planner is None or not hasattr(planner, method_name):
        return None
    try:
        result = getattr(planner, method_name)(context)
    except Exception:
        return None
    if not isinstance(result, dict):
        return None
    return result


def _status_is_active_participant(status: str | None) -> bool:
    value = str(status or "").strip().lower()
    return value not in {"done", "eliminated", "forfeit", "cancelled"}


def _anaconda_resolve_side_pots(
    *,
    contributions: dict[str, int],
    score_map: dict[str, tuple[int, tuple[int, ...]]],
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    remaining = {
        str(aid): max(0, int(value))
        for aid, value in contributions.items()
        if str(aid) and int(value) > 0
    }
    payouts: dict[str, int] = {aid: 0 for aid in contributions}
    side_pots: list[dict[str, Any]] = []

    while True:
        positive = [value for value in remaining.values() if value > 0]
        if not positive:
            break
        level = min(positive)
        contributors = sorted([aid for aid, value in remaining.items() if value > 0])
        pot_value = int(level) * len(contributors)
        eligible = [aid for aid in contributors if aid in score_map]
        if not eligible:
            eligible = sorted(score_map.keys())
        if not eligible:
            break
        best = max(score_map[aid] for aid in eligible)
        winners = sorted([aid for aid in eligible if score_map.get(aid) == best])
        each = pot_value // max(1, len(winners))
        remainder = pot_value - (each * len(winners))
        for idx, aid in enumerate(winners):
            payouts[aid] = int(payouts.get(aid, 0)) + each + (1 if idx < remainder else 0)
        side_pots.append(
            {
                "amount": int(pot_value),
                "contributors": contributors,
                "eligible": eligible,
                "winners": winners,
            }
        )
        for aid in contributors:
            remaining[aid] = max(0, int(remaining.get(aid, 0)) - int(level))

    return payouts, side_pots


def get_town_agent_scenario_statuses(*, town_id: str) -> dict[str, str]:
    """Return agent->status markers used to pause normal sim behavior during matches."""
    out: dict[str, str] = {}
    matches = scenario_store.list_active_matches(town_id=town_id)
    for match in matches:
        phase = str(match.get("phase") or "").strip().lower()
        status = str(match.get("status") or "").strip().lower()
        scenario_status = "scenario_warmup" if status == "warmup" or phase == "warmup" else "scenario_active"
        participants = scenario_store.list_match_participants(match_id=str(match.get("match_id") or ""))
        for participant in participants:
            aid = str(participant.get("agent_id") or "").strip()
            pstatus = str(participant.get("status") or "").strip().lower()
            if pstatus in {"forfeit", "cancelled", "done"}:
                continue
            if aid:
                out[aid] = scenario_status
    return out


def _alive_ids_from_state(*, state: dict[str, Any], participants: list[dict[str, Any]]) -> list[str]:
    alive = [str(aid) for aid in (state.get("alive_agent_ids") or []) if str(aid)]
    if alive:
        return alive
    out: list[str] = []
    for item in participants:
        aid = str(item.get("agent_id") or "")
        if not aid:
            continue
        status = str(item.get("status") or "").strip().lower()
        if status not in {"eliminated", "done", "forfeit", "cancelled", "folded"}:
            out.append(aid)
    return out


def _eliminate_agent(
    *,
    match_id: str,
    alive: list[str],
    dead: list[str],
    agent_id: str,
    by_status: str = "eliminated",
) -> None:
    aid = str(agent_id or "").strip()
    if not aid:
        return
    if aid in alive:
        alive.remove(aid)
    if aid not in dead:
        dead.append(aid)
    scenario_store.update_match_participant(
        match_id=match_id,
        agent_id=aid,
        updates={"status": by_status},
    )


def _werewolf_vote_target(
    *,
    voter_id: str,
    role_map: dict[str, str],
    alive: list[str],
    suspicion: dict[str, int],
    rng_seed: str,
) -> str | None:
    candidates = [aid for aid in alive if aid != voter_id]
    if not candidates:
        return None
    voter_role = str(role_map.get(voter_id) or "")
    weighted: list[tuple[int, str]] = []
    for aid in candidates:
        score = int(suspicion.get(aid, 0))
        if _is_werewolf_role(voter_role):
            score += 6 if not _is_werewolf_role(role_map.get(aid)) else -8
        else:
            score += 3 if _is_werewolf_role(role_map.get(aid)) else 0
        weighted.append((score, aid))
    weighted.sort(key=lambda item: (-item[0], item[1]))
    top_score = weighted[0][0]
    tied = [aid for score, aid in weighted if score == top_score]
    return _rng(rng_seed).choice(tied)


def _anaconda_active_ids(state: dict[str, Any], participants: list[dict[str, Any]]) -> list[str]:
    folded = {str(aid) for aid in (state.get("folded_agent_ids") or []) if str(aid)}
    out: list[str] = []
    for item in participants:
        aid = str(item.get("agent_id") or "")
        status = str(item.get("status") or "").strip().lower()
        if aid and aid not in folded and status not in {"folded", "forfeit"} and _status_is_active_participant(status):
            out.append(aid)
    return out


def _anaconda_seat_order(state: dict[str, Any], participants: list[dict[str, Any]]) -> list[str]:
    seat_order = [str(aid) for aid in (state.get("seat_order") or []) if str(aid)]
    if seat_order:
        return seat_order
    return [str(item.get("agent_id") or "") for item in participants if str(item.get("agent_id") or "")]


def _anaconda_remove_cards(hand: list[str], cards: list[str]) -> list[str]:
    remaining = list(hand)
    for card in cards:
        token = str(card).upper()
        if token in remaining:
            remaining.remove(token)
    return remaining


def _anaconda_apply_bets(
    *,
    match_id: str,
    state: dict[str, Any],
    participants: list[dict[str, Any]],
    phase: str,
    current_step: int,
    cognition_planner: Any = None,
) -> dict[str, Any]:
    chips = {str(k): int(v) for k, v in (state.get("chips") or {}).items() if str(k)}
    folded = {str(aid) for aid in (state.get("folded_agent_ids") or []) if str(aid)}
    all_in = {str(aid) for aid in (state.get("all_in_agent_ids") or []) if str(aid)}
    contributions = {
        str(k): int(v)
        for k, v in (state.get("contributions") or {}).items()
        if str(k)
    }
    revealed = state.get("revealed_cards") or {}
    pot = int(state.get("pot") or 0)
    per_agent_bets: dict[str, int] = {}
    new_folds: list[str] = []
    new_all_in: list[str] = []

    for item in participants:
        aid = str(item.get("agent_id") or "")
        if not aid or aid in folded:
            continue
        if aid in all_in:
            per_agent_bets[aid] = 0
            continue
        chips.setdefault(aid, 0)
        contributions.setdefault(aid, 0)
        visible = len(revealed.get(aid) or [])
        stack = int(chips.get(aid) or 0)
        if stack <= 0:
            all_in.add(aid)
            new_all_in.append(aid)
            per_agent_bets[aid] = 0
            continue

        rng = _rng(f"bet:{phase}:{aid}:{current_step}")
        base = 2 + visible * 2
        volatility = rng.randint(0, 6)
        bet = min(stack, max(0, base + volatility))

        planner_decision = _planner_call(
            cognition_planner,
            "anaconda_choose_bet",
            {
                "town_id": str(state.get("town_id") or ""),
                "agent_id": aid,
                "match_id": str(match_id),
                "phase": str(phase),
                "step": int(current_step),
                "stack": int(stack),
                "visible_cards": int(visible),
                "pot": int(pot),
                "folded_agent_ids": sorted(folded),
                "all_in_agent_ids": sorted(all_in),
                "revealed_cards": revealed,
            },
        )
        if planner_decision:
            planner_fold = bool(planner_decision.get("fold"))
            if planner_fold:
                folded.add(aid)
                new_folds.append(aid)
                per_agent_bets[aid] = 0
                continue
            try:
                planned_bet = int(planner_decision.get("bet") or bet)
            except Exception:
                planned_bet = bet
            if bool(planner_decision.get("all_in")):
                bet = stack
            else:
                bet = min(stack, max(0, planned_bet))

        # Folding pressure rises after multiple reveal rounds with short stack.
        if visible >= 3 and stack <= max(6, bet) and rng.randint(0, 99) < 35:
            folded.add(aid)
            new_folds.append(aid)
            per_agent_bets[aid] = 0
            continue

        chips[aid] = max(0, stack - bet)
        pot += bet
        contributions[aid] = int(contributions.get(aid, 0)) + int(bet)
        per_agent_bets[aid] = bet
        if chips[aid] <= 0 and aid not in all_in:
            all_in.add(aid)
            new_all_in.append(aid)
            scenario_store.update_match_participant(
                match_id=str(match_id),
                agent_id=aid,
                updates={"status": "all_in"},
            )

    state["chips"] = chips
    state["pot"] = pot
    state["contributions"] = contributions
    state["folded_agent_ids"] = sorted(folded)
    state["all_in_agent_ids"] = sorted(all_in)
    return {
        "bets": per_agent_bets,
        "folded": new_folds,
        "all_in": new_all_in,
        "contributions": contributions,
        "pot": pot,
    }


def form_matches(*, town_id: str, scenario_key: str | None = None, current_step: int = 0) -> list[dict[str, Any]]:
    scenarios = (
        [scenario_store.get_scenario(scenario_key=scenario_key)]
        if scenario_key
        else scenario_store.list_scenarios(enabled_only=True)
    )
    formed: list[dict[str, Any]] = []
    member_map = _collect_town_member_map(town_id)

    for scenario in scenarios:
        if not scenario:
            continue
        key = str(scenario.get("scenario_key") or "").strip()
        if not key:
            continue
        min_players = max(2, int(scenario.get("min_players") or 2))
        max_players = max(min_players, int(scenario.get("max_players") or min_players))
        queued = scenario_store.list_queue(scenario_key=key, town_id=town_id, status="queued")
        if not queued:
            continue

        # Keep only members still in this town and not in active matches.
        eligible: list[dict[str, Any]] = []
        for entry in queued:
            aid = str(entry.get("agent_id") or "")
            if aid not in member_map:
                continue
            if scenario_store.get_agent_active_match(agent_id=aid) is not None:
                continue
            eligible.append(entry)
        if len(eligible) < min_players:
            continue

        # For buy-in scenarios, filter by balance.
        buy_in = max(0, int(scenario.get("buy_in") or 0))
        if buy_in > 0:
            buy_in_eligible: list[dict[str, Any]] = []
            for entry in eligible:
                wallet = _ensure_wallet_for_buy_in(agent_id=str(entry["agent_id"]), buy_in=buy_in)
                if int(wallet.get("balance") or 0) >= buy_in:
                    buy_in_eligible.append(entry)
            eligible = buy_in_eligible
        if len(eligible) < min_players:
            continue

        selected = eligible[: max_players]
        if len(selected) < min_players:
            continue

        agent_ids = [str(item["agent_id"]) for item in selected]
        role_map: dict[str, str] = {}
        state_json: dict[str, Any]
        key_lower = key.lower()
        if "werewolf" in key_lower:
            role_map = _assign_werewolf_roles(agent_ids=agent_ids, match_id=f"{key}:{','.join(agent_ids)}")
            werewolves = [aid for aid, role in role_map.items() if _is_werewolf_role(role)]
            state_json = {
                "alive_agent_ids": list(agent_ids),
                "dead_agent_ids": [],
                "role_map": role_map,
                "werewolf_agent_ids": werewolves,
                "seer_results": {},
                "last_votes": {},
                "phase_sequence": _phase_sequence(scenario),
                "phase_index": 0,
                "max_rounds": int((scenario.get("rules_json") or {}).get("max_rounds") or 6),
            }
        else:
            state_json = {
                "phase_sequence": _phase_sequence(scenario),
                "phase_index": 0,
                "buy_in": buy_in,
                "pot": 0,
                "chips": {aid: int(buy_in) for aid in agent_ids},
                "contributions": {aid: 0 for aid in agent_ids},
                "seat_order": list(agent_ids),
                "deck": [],
                "hands": {},
                "arranged_hands": {},
                "revealed_cards": {aid: [] for aid in agent_ids},
                "folded_agent_ids": [],
                "all_in_agent_ids": [],
                "showdown": {},
                "seed": f"{key}:{','.join(sorted(agent_ids))}",
                "town_id": str(town_id),
            }

        match = scenario_store.create_match(
            scenario_key=key,
            town_id=town_id,
            arena_id=_arena_id(scenario),
            status="warmup",
            phase="warmup",
            phase_step=0,
            round_number=0,
            created_step=int(current_step),
            warmup_start_step=int(current_step),
            start_step=None,
            state_json=state_json,
        )

        participants: list[dict[str, Any]] = []
        for aid in agent_ids:
            rating = scenario_store.get_rating(agent_id=aid, scenario_key=key)
            participants.append(
                {
                    "match_id": match["match_id"],
                    "agent_id": aid,
                    "role": role_map.get(aid),
                    "status": "alive" if "werewolf" in key_lower else "assigned",
                    "score": None,
                    "chips_start": buy_in if buy_in > 0 else None,
                    "chips_end": None,
                    "rating_before": int(rating.get("rating") or 1500),
                    "rating_after": None,
                    "rating_delta": None,
                }
            )
            if buy_in > 0:
                scenario_store.update_wallet(agent_id=aid, delta=-buy_in)

        scenario_store.insert_match_participants(participants=participants)
        for entry in selected:
            scenario_store.update_queue_status(
                entry_id=str(entry["entry_id"]),
                status="matched",
                match_id=str(match["match_id"]),
            )

        scenario_store.insert_match_event(
            match_id=str(match["match_id"]),
            step=int(current_step),
            phase="warmup",
            event_type="match_formed",
            agent_id=None,
            target_id=None,
            data_json={
                "scenario_key": key,
                "participants": agent_ids,
                "arena_id": match.get("arena_id"),
            },
        )
        _notify_match_formed(
            scenario=scenario,
            match=match,
            participants=agent_ids,
            current_step=int(current_step),
        )
        if "werewolf" in key_lower:
            for aid in agent_ids:
                role = str(role_map.get(aid) or "villager")
                _record_scenario_memory(
                    agent_id=aid,
                    match_id=str(match["match_id"]),
                    kind="werewolf_role_assignment",
                    summary=f"You are assigned role: {role}.",
                    payload={"role": role, "scenario_key": key},
                    dedupe_suffix="role",
                )
        elif buy_in > 0:
            for aid in agent_ids:
                _record_scenario_memory(
                    agent_id=aid,
                    match_id=str(match["match_id"]),
                    kind="anaconda_buy_in",
                    summary=f"Joined {key} with {buy_in} chips.",
                    payload={"buy_in": int(buy_in), "scenario_key": key},
                    dedupe_suffix="buyin",
                )
        formed.append(match)

    return formed


def _resolve_werewolf(*, alive: list[str], role_map: dict[str, str], round_number: int, max_rounds: int) -> tuple[bool, list[str], str]:
    alive_wolves = [aid for aid in alive if _is_werewolf_role(role_map.get(aid))]
    alive_villagers = [aid for aid in alive if not _is_werewolf_role(role_map.get(aid))]
    if not alive_wolves:
        return (True, alive_villagers, "All werewolves eliminated")
    if len(alive_wolves) >= len(alive_villagers):
        return (True, alive_wolves, "Werewolves reached parity")
    if round_number >= max_rounds:
        if len(alive_wolves) > len(alive_villagers):
            return (True, alive_wolves, "Max rounds reached")
        if len(alive_villagers) > len(alive_wolves):
            return (True, alive_villagers, "Max rounds reached")
        return (True, [], "Max rounds draw")
    return (False, [], "")


def _finalize_match(
    *,
    match: dict[str, Any],
    scenario: dict[str, Any],
    participants: list[dict[str, Any]],
    winners: list[str],
    current_step: int,
    reason: str,
    state_json: dict[str, Any],
) -> dict[str, Any]:
    participant_ids = [str(item["agent_id"]) for item in participants]
    winner_set = {str(aid) for aid in winners}
    draws = len(winner_set) == 0

    rating_updates = _compute_rating_updates(
        scenario_key=str(match["scenario_key"]),
        participant_ids=participant_ids,
        winners=winner_set,
        draws=draws,
    )

    buy_in = max(0, int(scenario.get("buy_in") or 0))
    wallet_payouts: dict[str, int] = {}
    if buy_in > 0 and participant_ids:
        chips_state = {
            str(aid): int(value)
            for aid, value in (state_json.get("chips") or {}).items()
            if str(aid)
        }
        if chips_state:
            for aid in participant_ids:
                payout = max(0, int(chips_state.get(aid, 0)))
                wallet_payouts[aid] = payout
        else:
            pot = buy_in * len(participant_ids)
            receivers = list(winner_set) if winner_set else list(participant_ids)
            each = pot // max(1, len(receivers))
            remainder = pot - (each * len(receivers))
            for idx, aid in enumerate(receivers):
                payout = each + (1 if idx < remainder else 0)
                wallet_payouts[aid] = payout
        for aid in participant_ids:
            payout = int(wallet_payouts.get(aid, 0))
            if payout:
                scenario_store.update_wallet(agent_id=aid, delta=payout)

    for item in participants:
        aid = str(item["agent_id"])
        update = rating_updates[aid]
        scenario_store.upsert_rating(
            agent_id=aid,
            scenario_key=str(match["scenario_key"]),
            rating=int(update["after"]),
            games_played=int(update["games_played"]),
            wins=int(update["wins"]),
            losses=int(update["losses"]),
            draws=int(update["draws"]),
            peak_rating=int(update["peak_rating"]),
        )
        score = _ranking_score(
            agent_id=aid,
            winners=winner_set,
            losers=set(participant_ids) - winner_set,
            draws=draws,
        )
        chips_start = int(item.get("chips_start") or buy_in or 0) if buy_in > 0 else None
        chips_end = None
        if buy_in > 0:
            chips_end = int(wallet_payouts.get(aid, 0))

        scenario_store.update_match_participant(
            match_id=str(match["match_id"]),
            agent_id=aid,
            updates={
                "status": "done",
                "score": score,
                "chips_start": chips_start,
                "chips_end": chips_end,
                "rating_before": int(update["before"]),
                "rating_after": int(update["after"]),
                "rating_delta": int(update["delta"]),
            },
        )

    result_json = {
        "winners": list(winner_set),
        "reason": reason,
        "ratings": {
            aid: {
                "before": int(values["before"]),
                "after": int(values["after"]),
                "delta": int(values["delta"]),
            }
            for aid, values in rating_updates.items()
        },
        "wallet_payouts": wallet_payouts,
        "showdown": dict(state_json.get("showdown") or {}),
        "resolved_at": _utc_now_iso(),
    }

    updated = scenario_store.update_match(
        match_id=str(match["match_id"]),
        status="resolved",
        phase="resolved",
        end_step=int(current_step),
        state_json=state_json,
        result_json=result_json,
    )
    scenario_store.insert_match_event(
        match_id=str(match["match_id"]),
        step=int(current_step),
        phase="resolved",
        event_type="match_resolved",
        agent_id=None,
        target_id=None,
        data_json={
            "winners": list(winner_set),
            "reason": reason,
        },
    )
    _notify_match_resolved(
        match=match,
        scenario=scenario,
        participants=participants,
        result_json=result_json,
    )
    scenario_name = _scenario_display_name(scenario)
    for item in participants:
        aid = str(item.get("agent_id") or "")
        if not aid:
            continue
        won = aid in winner_set
        _record_scenario_memory(
            agent_id=aid,
            match_id=str(match.get("match_id") or ""),
            kind="match_summary",
            summary=f"{scenario_name}: {'victory' if won else 'result'} ({reason})",
            payload={
                "scenario_key": str(match.get("scenario_key") or ""),
                "won": bool(won),
                "reason": reason,
                "rating_delta": int((result_json.get("ratings") or {}).get(aid, {}).get("delta") or 0),
            },
            dedupe_suffix="resolved",
        )
    return updated or match


def _advance_werewolf_match(
    *,
    match: dict[str, Any],
    scenario: dict[str, Any],
    current_step: int,
    cognition_planner: Any = None,
) -> dict[str, Any]:
    participants = scenario_store.list_match_participants(match_id=str(match["match_id"]))
    if not participants:
        return match

    state = dict(match.get("state_json") or {})
    role_map = dict(state.get("role_map") or {})
    alive = _alive_ids_from_state(state=state, participants=participants)
    dead = [str(aid) for aid in (state.get("dead_agent_ids") or []) if str(aid)]
    seer_results = {
        str(seer_id): {
            str(target_id): str(value)
            for target_id, value in (targets or {}).items()
            if str(target_id)
        }
        for seer_id, targets in (state.get("seer_results") or {}).items()
        if str(seer_id)
    }
    round_number = int(match.get("round_number") or 0)
    phase = str(match.get("phase") or "night")

    if phase == "night":
        wolves = [aid for aid in alive if _is_werewolf_role(role_map.get(aid))]
        villagers = [aid for aid in alive if not _is_werewolf_role(role_map.get(aid))]
        doctors = [aid for aid in alive if str(role_map.get(aid) or "") == "doctor"]
        seers = [aid for aid in alive if str(role_map.get(aid) or "") == "seer"]
        last_votes = {
            str(voter): str(target)
            for voter, target in (state.get("last_votes") or {}).items()
            if str(voter) and str(target)
        }
        suspicion: dict[str, int] = {}
        for target in last_votes.values():
            suspicion[target] = suspicion.get(target, 0) + 1

        target: str | None = None
        if wolves and villagers:
            planner_target = _planner_call(
                cognition_planner,
                "werewolf_choose_target",
                {
                    "town_id": str(match.get("town_id") or ""),
                    "agent_id": wolves[0],
                    "match_id": str(match["match_id"]),
                    "phase": "night",
                    "step": int(current_step),
                    "round_number": int(round_number),
                    "alive_agent_ids": list(alive),
                    "candidate_agent_ids": list(villagers),
                    "role_map": role_map,
                    "suspicion": suspicion,
                },
            )
            planner_target_id = str((planner_target or {}).get("target_agent_id") or "").strip()
            if planner_target_id in villagers:
                target = planner_target_id
            else:
                ranked = sorted(
                    villagers,
                    key=lambda aid: (
                        -int(suspicion.get(aid, 0)),
                        _rng(f"night-target:{match['match_id']}:{aid}:{current_step}").random(),
                    ),
                )
                target = ranked[0] if ranked else None

        protected: str | None = None
        if doctors:
            protect_candidates = sorted(alive, key=lambda aid: (-int(suspicion.get(aid, 0)), aid))
            planner_protect = _planner_call(
                cognition_planner,
                "werewolf_choose_protect",
                {
                    "town_id": str(match.get("town_id") or ""),
                    "agent_id": doctors[0],
                    "match_id": str(match["match_id"]),
                    "phase": "night",
                    "step": int(current_step),
                    "round_number": int(round_number),
                    "alive_agent_ids": list(alive),
                    "candidate_agent_ids": protect_candidates,
                    "suspicion": suspicion,
                },
            )
            planner_protect_id = str((planner_protect or {}).get("target_agent_id") or "").strip()
            if planner_protect_id in protect_candidates:
                protected = planner_protect_id
            else:
                protected = _rng(f"night-protect:{match['match_id']}:{round_number}:{current_step}").choice(protect_candidates)
            scenario_store.insert_match_event(
                match_id=str(match["match_id"]),
                step=int(current_step),
                phase="night",
                event_type="protect",
                agent_id=doctors[0],
                target_id=None,
                data_json={"doctor": doctors[0]},
            )

        for seer_id in seers:
            known = seer_results.get(seer_id, {})
            inspect_candidates = [aid for aid in alive if aid != seer_id and aid not in known]
            if not inspect_candidates:
                inspect_candidates = [aid for aid in alive if aid != seer_id]
            if not inspect_candidates:
                continue
            planner_inspect = _planner_call(
                cognition_planner,
                "werewolf_choose_inspect",
                {
                    "town_id": str(match.get("town_id") or ""),
                    "agent_id": seer_id,
                    "match_id": str(match["match_id"]),
                    "phase": "night",
                    "step": int(current_step),
                    "round_number": int(round_number),
                    "alive_agent_ids": list(alive),
                    "candidate_agent_ids": inspect_candidates,
                    "known_results": known,
                },
            )
            inspected = str((planner_inspect or {}).get("target_agent_id") or "").strip()
            if inspected not in inspect_candidates:
                inspected = _rng(f"night-inspect:{match['match_id']}:{seer_id}:{round_number}:{current_step}").choice(inspect_candidates)
            alignment = "werewolf" if _is_werewolf_role(role_map.get(inspected)) else "village"
            known[inspected] = alignment
            seer_results[seer_id] = known
            scenario_store.insert_match_event(
                match_id=str(match["match_id"]),
                step=int(current_step),
                phase="night",
                event_type="inspect",
                agent_id=seer_id,
                target_id=inspected,
                data_json={"alignment": alignment},
            )
            _record_scenario_memory(
                agent_id=seer_id,
                match_id=str(match["match_id"]),
                kind="werewolf_seer_result",
                summary=f"Inspection result: {inspected} looks {alignment}.",
                payload={"target_agent_id": inspected, "alignment": alignment, "round": int(round_number)},
                dedupe_suffix=f"inspect:{round_number}:{seer_id}",
            )

        if target:
            if protected and target == protected:
                scenario_store.insert_match_event(
                    match_id=str(match["match_id"]),
                    step=int(current_step),
                    phase="night",
                    event_type="saved",
                    agent_id=None,
                    target_id=target,
                    data_json={"target": target},
                )
                _record_scenario_memory(
                    agent_id=target,
                    match_id=str(match["match_id"]),
                    kind="werewolf_saved",
                    summary="You survived the night due to protection.",
                    payload={"round": int(round_number)},
                    dedupe_suffix=f"saved:{round_number}",
                )
            else:
                _eliminate_agent(
                    match_id=str(match["match_id"]),
                    alive=alive,
                    dead=dead,
                    agent_id=target,
                )
                scenario_store.insert_match_event(
                    match_id=str(match["match_id"]),
                    step=int(current_step),
                    phase="night",
                    event_type="kill",
                    agent_id=wolves[0] if wolves else None,
                    target_id=target,
                    data_json={"by": wolves, "target": target},
                )
                _record_scenario_memory(
                    agent_id=target,
                    match_id=str(match["match_id"]),
                    kind="werewolf_eliminated_night",
                    summary="You were eliminated during the night.",
                    payload={"round": int(round_number)},
                    dedupe_suffix=f"night_elim:{round_number}",
                )
                if str(role_map.get(target) or "") == "hunter":
                    retaliation_targets = [aid for aid in alive if aid != target]
                    if retaliation_targets:
                        wolf_targets = [aid for aid in retaliation_targets if _is_werewolf_role(role_map.get(aid))]
                        chosen_pool = wolf_targets if wolf_targets else retaliation_targets
                        retaliation = _rng(
                            f"hunter-retaliate:{match['match_id']}:{target}:{round_number}:{current_step}"
                        ).choice(chosen_pool)
                        _eliminate_agent(
                            match_id=str(match["match_id"]),
                            alive=alive,
                            dead=dead,
                            agent_id=retaliation,
                        )
                        scenario_store.insert_match_event(
                            match_id=str(match["match_id"]),
                            step=int(current_step),
                            phase="night",
                            event_type="hunter_shot",
                            agent_id=target,
                            target_id=retaliation,
                            data_json={"retaliation_target": retaliation},
                        )
        next_phase = "day"
        next_round = round_number
        state["last_night"] = {
            "round": int(round_number),
            "target": target,
            "protected": protected,
        }
    else:
        suspicion: dict[str, int] = {}
        for known in seer_results.values():
            for target_id, alignment in known.items():
                if alignment == "werewolf":
                    suspicion[target_id] = suspicion.get(target_id, 0) + 4
                else:
                    suspicion[target_id] = suspicion.get(target_id, 0) - 1
        previous_votes = {
            str(voter): str(target)
            for voter, target in (state.get("last_votes") or {}).items()
            if str(voter) and str(target)
        }
        for target in previous_votes.values():
            suspicion[target] = suspicion.get(target, 0) + 1

        for speaker in alive:
            chosen_target = _werewolf_vote_target(
                voter_id=speaker,
                role_map=role_map,
                alive=alive,
                suspicion=suspicion,
                rng_seed=f"speech-target:{match['match_id']}:{speaker}:{round_number}:{current_step}",
            )
            planner_speech = _planner_call(
                cognition_planner,
                "werewolf_generate_speech",
                {
                    "town_id": str(match.get("town_id") or ""),
                    "agent_id": speaker,
                    "match_id": str(match["match_id"]),
                    "phase": "day",
                    "step": int(current_step),
                    "round_number": int(round_number),
                    "alive_agent_ids": list(alive),
                    "role": str(role_map.get(speaker) or ""),
                    "suggested_target": chosen_target,
                    "suspicion": suspicion,
                },
            )
            line = str((planner_speech or {}).get("line") or "").strip()
            planner_target = str((planner_speech or {}).get("target_agent_id") or "").strip()
            if planner_target in alive and planner_target != speaker:
                chosen_target = planner_target
            if chosen_target:
                scenario_store.insert_match_event(
                    match_id=str(match["match_id"]),
                    step=int(current_step),
                    phase="day",
                    event_type="speech",
                    agent_id=speaker,
                    target_id=chosen_target,
                    data_json={"line": line or f"I suspect {chosen_target}."},
                )

        votes: dict[str, str] = {}
        for voter in alive:
            target = _werewolf_vote_target(
                voter_id=voter,
                role_map=role_map,
                alive=alive,
                suspicion=suspicion,
                rng_seed=f"day-vote:{match['match_id']}:{voter}:{round_number}:{current_step}",
            )
            planner_vote = _planner_call(
                cognition_planner,
                "werewolf_choose_vote",
                {
                    "town_id": str(match.get("town_id") or ""),
                    "agent_id": voter,
                    "match_id": str(match["match_id"]),
                    "phase": "day_vote",
                    "step": int(current_step),
                    "round_number": int(round_number),
                    "alive_agent_ids": [aid for aid in alive if aid != voter],
                    "role": str(role_map.get(voter) or ""),
                    "suspicion": suspicion,
                },
            )
            planner_target = str((planner_vote or {}).get("target_agent_id") or "").strip()
            if planner_target in alive and planner_target != voter:
                target = planner_target
            if target:
                votes[voter] = target

        tally: dict[str, int] = {}
        for target in votes.values():
            tally[target] = tally.get(target, 0) + 1

        eliminated: str | None = None
        if tally:
            top_count = max(tally.values())
            top_targets = sorted([aid for aid, count in tally.items() if count == top_count])
            majority = (len(alive) // 2) + 1
            if top_count >= majority and len(top_targets) == 1:
                eliminated = top_targets[0]

        scenario_store.insert_match_event(
            match_id=str(match["match_id"]),
            step=int(current_step),
            phase="day",
            event_type="vote",
            agent_id=None,
            target_id=eliminated,
            data_json={
                "votes": votes,
                "tally": tally,
                "majority": (len(alive) // 2) + 1,
                "eliminated": eliminated,
            },
        )

        if eliminated:
            _eliminate_agent(
                match_id=str(match["match_id"]),
                alive=alive,
                dead=dead,
                agent_id=eliminated,
            )
            scenario_store.insert_match_event(
                match_id=str(match["match_id"]),
                step=int(current_step),
                phase="day",
                event_type="eliminate",
                agent_id=None,
                target_id=eliminated,
                data_json={"eliminated": eliminated},
            )
            _record_scenario_memory(
                agent_id=eliminated,
                match_id=str(match["match_id"]),
                kind="werewolf_eliminated_day",
                summary="You were voted out during the day.",
                payload={"round": int(round_number)},
                dedupe_suffix=f"day_elim:{round_number}",
            )
            if str(role_map.get(eliminated) or "") == "hunter":
                retaliation_targets = [aid for aid in alive if aid != eliminated]
                if retaliation_targets:
                    wolf_targets = [aid for aid in retaliation_targets if _is_werewolf_role(role_map.get(aid))]
                    chosen_pool = wolf_targets if wolf_targets else retaliation_targets
                    retaliation = _rng(
                        f"hunter-day-retaliate:{match['match_id']}:{eliminated}:{round_number}:{current_step}"
                    ).choice(chosen_pool)
                    _eliminate_agent(
                        match_id=str(match["match_id"]),
                        alive=alive,
                        dead=dead,
                        agent_id=retaliation,
                    )
                    scenario_store.insert_match_event(
                        match_id=str(match["match_id"]),
                        step=int(current_step),
                        phase="day",
                        event_type="hunter_shot",
                        agent_id=eliminated,
                        target_id=retaliation,
                        data_json={"retaliation_target": retaliation},
                    )
        next_phase = "night"
        next_round = round_number + 1
        state["last_votes"] = votes

    state["alive_agent_ids"] = list(alive)
    state["dead_agent_ids"] = list(dead)
    state["seer_results"] = seer_results
    max_rounds = int((scenario.get("rules_json") or {}).get("max_rounds") or 6)
    resolved, winners, reason = _resolve_werewolf(
        alive=alive,
        role_map=role_map,
        round_number=next_round,
        max_rounds=max_rounds,
    )
    if resolved:
        return _finalize_match(
            match=match,
            scenario=scenario,
            participants=participants,
            winners=winners,
            current_step=current_step,
            reason=reason,
            state_json=state,
        )

    updated = scenario_store.update_match(
        match_id=str(match["match_id"]),
        phase=next_phase,
        round_number=int(next_round),
        phase_step=int(match.get("phase_step") or 0) + 1,
        state_json=state,
    )
    scenario_store.insert_match_event(
        match_id=str(match["match_id"]),
        step=int(current_step),
        phase=next_phase,
        event_type="phase_change",
        agent_id=None,
        target_id=None,
        data_json={"phase": next_phase, "round": next_round, "alive": alive},
    )
    return updated or match


def _advance_anaconda_match(
    *,
    match: dict[str, Any],
    scenario: dict[str, Any],
    current_step: int,
    cognition_planner: Any = None,
) -> dict[str, Any]:
    participants = scenario_store.list_match_participants(match_id=str(match["match_id"]))
    if not participants:
        return match

    state = dict(match.get("state_json") or {})
    phases = _phase_sequence(scenario)
    phase_index = int(state.get("phase_index") or 0)
    if phase_index < 0:
        phase_index = 0
    if phase_index >= len(phases):
        phase_index = len(phases) - 1
    phase = phases[phase_index]

    seat_order = _anaconda_seat_order(state, participants)
    chips = {str(k): int(v) for k, v in (state.get("chips") or {}).items() if str(k)}
    contributions = {
        str(k): int(v) for k, v in (state.get("contributions") or {}).items() if str(k)
    }
    for participant in participants:
        aid = str(participant.get("agent_id") or "")
        if aid and aid not in chips:
            chips[aid] = int(state.get("buy_in") or scenario.get("buy_in") or 0)
        if aid and aid not in contributions:
            contributions[aid] = 0

    hands = {
        str(aid): [str(card).upper() for card in (cards or [])]
        for aid, cards in (state.get("hands") or {}).items()
        if str(aid)
    }
    arranged_hands = {
        str(aid): [str(card).upper() for card in (cards or [])]
        for aid, cards in (state.get("arranged_hands") or {}).items()
        if str(aid)
    }
    revealed_cards = {
        str(aid): [str(card).upper() for card in (cards or [])]
        for aid, cards in (state.get("revealed_cards") or {}).items()
        if str(aid)
    }
    folded = {str(aid) for aid in (state.get("folded_agent_ids") or []) if str(aid)}
    all_in = {str(aid) for aid in (state.get("all_in_agent_ids") or []) if str(aid)}
    deck = [str(card).upper() for card in (state.get("deck") or [])]
    pot = int(state.get("pot") or 0)

    if phase == "deal":
        seed = f"{state.get('seed') or match['match_id']}:deal:{current_step}"
        deck = make_shuffled_deck(seed=seed)
        hands = {aid: [] for aid in seat_order}
        for _ in range(7):
            for aid in seat_order:
                if deck:
                    hands[aid].append(deck.pop())
        arranged_hands = {}
        revealed_cards = {aid: [] for aid in seat_order}
        all_in = set()
        contributions = {aid: 0 for aid in seat_order}
        scenario_store.insert_match_event(
            match_id=str(match["match_id"]),
            step=int(current_step),
            phase=phase,
            event_type="deal",
            agent_id=None,
            target_id=None,
            data_json={"players": len(seat_order), "cards_each": 7},
        )
    elif phase in {"pass1", "pass2", "pass3"}:
        pass_count = 3 if phase == "pass1" else (2 if phase == "pass2" else 1)
        direction = "left" if phase in {"pass1", "pass3"} else "right"
        active_order = [aid for aid in seat_order if aid not in folded]
        outgoing: dict[str, list[str]] = {}
        for aid in active_order:
            current_hand = list(hands.get(aid, []))
            chosen = choose_pass_cards(current_hand, count=pass_count)
            planner_pick = _planner_call(
                cognition_planner,
                "anaconda_choose_pass_cards",
                {
                    "town_id": str(match.get("town_id") or ""),
                    "agent_id": aid,
                    "match_id": str(match["match_id"]),
                    "phase": str(phase),
                    "step": int(current_step),
                    "count": int(pass_count),
                    "hand": list(current_hand),
                },
            )
            planned_cards = [str(card).upper() for card in ((planner_pick or {}).get("cards") or []) if str(card)]
            if len(planned_cards) == pass_count and all(card in current_hand for card in planned_cards):
                chosen = planned_cards
            outgoing[aid] = chosen
            hands[aid] = _anaconda_remove_cards(current_hand, chosen)

        for idx, aid in enumerate(active_order):
            recipient_idx = (idx + 1) % len(active_order) if direction == "left" else (idx - 1) % len(active_order)
            recipient = active_order[recipient_idx]
            hands.setdefault(recipient, [])
            hands[recipient].extend(outgoing.get(aid, []))

        scenario_store.insert_match_event(
            match_id=str(match["match_id"]),
            step=int(current_step),
            phase=phase,
            event_type="pass",
            agent_id=None,
            target_id=None,
            data_json={"direction": direction, "count": pass_count},
        )
    elif phase == "discard":
        arranged_hands = {}
        showdown_meta: dict[str, Any] = {}
        for aid in seat_order:
            hand = list(hands.get(aid, []))
            if len(hand) < 5 or aid in folded:
                continue
            category, tiebreak, best_cards = evaluate_best_five(hand)
            arranged = arrange_reveal_order(list(best_cards))
            arranged_hands[aid] = arranged
            showdown_meta[aid] = {
                "category": int(category),
                "category_label": hand_category_label(int(category)),
                "tiebreak": list(tiebreak),
            }
        state["showdown"] = showdown_meta
        scenario_store.insert_match_event(
            match_id=str(match["match_id"]),
            step=int(current_step),
            phase=phase,
            event_type="discard",
            agent_id=None,
            target_id=None,
            data_json={"players_ready": len(arranged_hands)},
        )
    elif phase.startswith("reveal"):
        try:
            reveal_idx = max(0, int(phase.replace("reveal", "")) - 1)
        except Exception:
            reveal_idx = 0
        reveal_payload: dict[str, str] = {}
        for aid in seat_order:
            if aid in folded:
                continue
            arranged = arranged_hands.get(aid, [])
            if reveal_idx >= len(arranged):
                continue
            visible = list(revealed_cards.get(aid, []))
            if len(visible) <= reveal_idx:
                visible.append(arranged[reveal_idx])
                revealed_cards[aid] = visible
            reveal_payload[aid] = arranged[reveal_idx]
        scenario_store.insert_match_event(
            match_id=str(match["match_id"]),
            step=int(current_step),
            phase=phase,
            event_type="reveal",
            agent_id=None,
            target_id=None,
            data_json={"cards": reveal_payload},
        )

    state["chips"] = chips
    state["hands"] = hands
    state["arranged_hands"] = arranged_hands
    state["revealed_cards"] = revealed_cards
    state["deck"] = deck
    state["pot"] = pot
    state["contributions"] = contributions
    state["folded_agent_ids"] = sorted(folded)
    state["all_in_agent_ids"] = sorted(all_in)

    if phase != "showdown":
        bet_result = _anaconda_apply_bets(
            match_id=str(match["match_id"]),
            state=state,
            participants=participants,
            phase=phase,
            current_step=current_step,
            cognition_planner=cognition_planner,
        )
        for folded_agent in bet_result.get("folded") or []:
            scenario_store.update_match_participant(
                match_id=str(match["match_id"]),
                agent_id=str(folded_agent),
                updates={"status": "folded"},
            )
        for all_in_agent in bet_result.get("all_in") or []:
            scenario_store.update_match_participant(
                match_id=str(match["match_id"]),
                agent_id=str(all_in_agent),
                updates={"status": "all_in"},
            )
        scenario_store.insert_match_event(
            match_id=str(match["match_id"]),
            step=int(current_step),
            phase=phase,
            event_type="bet",
            agent_id=None,
            target_id=None,
            data_json={
                "bets": bet_result.get("bets") or {},
                "folded": bet_result.get("folded") or [],
                "all_in": bet_result.get("all_in") or [],
                "pot": int(state.get("pot") or 0),
            },
        )

        active_after_bets = _anaconda_active_ids(state, participants)
        if len(active_after_bets) <= 1 and active_after_bets:
            winner = active_after_bets[0]
            chips = {str(k): int(v) for k, v in (state.get("chips") or {}).items() if str(k)}
            chips[winner] = int(chips.get(winner, 0)) + int(state.get("pot") or 0)
            state["chips"] = chips
            state["pot"] = 0
            state["phase_index"] = phase_index
            state["showdown"] = {
                "winners": [winner],
                "reason": "All opponents folded",
            }
            return _finalize_match(
                match=match,
                scenario=scenario,
                participants=participants,
                winners=[winner],
                current_step=current_step,
                reason="All opponents folded",
                state_json=state,
            )

    if phase == "showdown":
        contenders = _anaconda_active_ids(state, participants)
        if not contenders:
            contenders = [str(item.get("agent_id") or "") for item in participants if str(item.get("agent_id") or "")]
        showdown_rows: dict[str, Any] = {}
        best_score: tuple[int, tuple[int, ...]] | None = None
        winners: list[str] = []
        score_map: dict[str, tuple[int, tuple[int, ...]]] = {}
        for aid in contenders:
            cards = list(arranged_hands.get(aid, [])) or list(hands.get(aid, []))
            if len(cards) < 5:
                continue
            category, tiebreak, best_cards = evaluate_best_five(cards)
            score = (int(category), tuple(int(v) for v in tiebreak))
            score_map[aid] = score
            showdown_rows[aid] = {
                "cards": list(best_cards),
                "category": int(category),
                "category_label": hand_category_label(int(category)),
                "tiebreak": list(tiebreak),
            }
            if best_score is None or score > best_score:
                best_score = score
                winners = [aid]
            elif score == best_score:
                winners.append(aid)

        pot = int(state.get("pot") or 0)
        chips = {str(k): int(v) for k, v in (state.get("chips") or {}).items() if str(k)}
        payout_map: dict[str, int]
        side_pots: list[dict[str, Any]]
        contributions = {
            str(aid): int(value)
            for aid, value in (state.get("contributions") or {}).items()
            if str(aid)
        }
        if contributions and score_map:
            payout_map, side_pots = _anaconda_resolve_side_pots(
                contributions=contributions,
                score_map=score_map,
            )
        else:
            payout_map = {}
            side_pots = []
            if winners and pot > 0:
                each = pot // max(1, len(winners))
                remainder = pot - (each * len(winners))
                for idx, aid in enumerate(sorted(winners)):
                    payout_map[aid] = each + (1 if idx < remainder else 0)

        if payout_map:
            distributed = sum(int(value) for value in payout_map.values())
            if distributed != pot and winners:
                delta = int(pot) - int(distributed)
                payout_map[sorted(winners)[0]] = int(payout_map.get(sorted(winners)[0], 0)) + delta
        for aid, payout in payout_map.items():
            chips[aid] = int(chips.get(aid, 0)) + int(payout)

        state["chips"] = chips
        state["pot"] = 0
        state["showdown"] = {
            "rows": showdown_rows,
            "winners": winners,
            "side_pots": side_pots,
            "payouts": payout_map,
        }
        scenario_store.insert_match_event(
            match_id=str(match["match_id"]),
            step=int(current_step),
            phase=phase,
            event_type="showdown",
            agent_id=None,
            target_id=None,
            data_json={
                "winners": winners,
                "hands": showdown_rows,
                "side_pots": side_pots,
                "payouts": payout_map,
            },
        )
        state["phase_index"] = phase_index
        for item in participants:
            aid = str(item.get("agent_id") or "")
            if not aid:
                continue
            payout = int(payout_map.get(aid, 0))
            label = str((showdown_rows.get(aid) or {}).get("category_label") or "no showdown")
            _record_scenario_memory(
                agent_id=aid,
                match_id=str(match["match_id"]),
                kind="anaconda_showdown",
                summary=f"Showdown: {label}, payout {payout} chips.",
                payload={"category_label": label, "payout": payout},
                dedupe_suffix=f"showdown:{current_step}:{aid}",
            )
        return _finalize_match(
            match=match,
            scenario=scenario,
            participants=participants,
            winners=winners,
            current_step=current_step,
            reason="Showdown complete",
            state_json=state,
        )

    next_phase_index = min(len(phases) - 1, phase_index + 1)
    next_phase = phases[next_phase_index]
    state["phase_index"] = next_phase_index
    updated = scenario_store.update_match(
        match_id=str(match["match_id"]),
        phase=next_phase,
        phase_step=int(match.get("phase_step") or 0) + 1,
        round_number=int(match.get("round_number") or 0) + 1,
        state_json=state,
    )
    return updated or match


def _cancel_match_without_ratings(
    *,
    match: dict[str, Any],
    scenario: dict[str, Any],
    participants: list[dict[str, Any]],
    current_step: int,
    reason: str,
) -> dict[str, Any]:
    state = dict(match.get("state_json") or {})
    buy_in = max(0, int(scenario.get("buy_in") or 0))
    wallet_payouts: dict[str, int] = {}
    if buy_in > 0 and participants:
        chips_state = {
            str(aid): int(value)
            for aid, value in (state.get("chips") or {}).items()
            if str(aid)
        }
        for participant in participants:
            aid = str(participant.get("agent_id") or "")
            if not aid:
                continue
            payout = max(0, int(chips_state.get(aid, buy_in)))
            wallet_payouts[aid] = payout
            if payout:
                scenario_store.update_wallet(agent_id=aid, delta=payout)

    for participant in participants:
        aid = str(participant.get("agent_id") or "")
        if not aid:
            continue
        update_payload: dict[str, Any] = {"status": "cancelled"}
        if buy_in > 0:
            update_payload["chips_end"] = int(wallet_payouts.get(aid, 0))
        scenario_store.update_match_participant(
            match_id=str(match.get("match_id") or ""),
            agent_id=aid,
            updates=update_payload,
        )

    result_json = {
        "winners": [],
        "reason": str(reason),
        "cancelled": True,
        "wallet_payouts": wallet_payouts,
        "resolved_at": _utc_now_iso(),
    }
    updated = scenario_store.update_match(
        match_id=str(match.get("match_id") or ""),
        status="cancelled",
        phase="cancelled",
        end_step=int(current_step),
        state_json=state,
        result_json=result_json,
    )
    scenario_store.insert_match_event(
        match_id=str(match.get("match_id") or ""),
        step=int(current_step),
        phase="cancelled",
        event_type="match_cancelled",
        agent_id=None,
        target_id=None,
        data_json={"reason": str(reason)},
    )
    _notify_match_resolved(
        match=match,
        scenario=scenario,
        participants=participants,
        result_json=result_json,
    )
    for participant in participants:
        aid = str(participant.get("agent_id") or "")
        if not aid:
            continue
        _record_scenario_memory(
            agent_id=aid,
            match_id=str(match.get("match_id") or ""),
            kind="match_cancelled",
            summary=f"Match cancelled: {reason}",
            payload={"reason": str(reason)},
            dedupe_suffix="cancelled",
        )
    return updated or match


def _forfeit_from_match(
    *,
    match: dict[str, Any],
    scenario: dict[str, Any],
    participants: list[dict[str, Any]],
    agent_id: str,
    current_step: int,
    reason: str,
) -> dict[str, Any]:
    match_id = str(match.get("match_id") or "")
    forfeiter = str(agent_id or "").strip()
    if not forfeiter:
        raise ValueError("Missing forfeit agent id")
    ids = {str(item.get("agent_id") or "") for item in participants}
    if forfeiter not in ids:
        raise PermissionError("Agent is not a participant in this match")

    scenario_store.update_match_participant(
        match_id=match_id,
        agent_id=forfeiter,
        updates={"status": "forfeit"},
    )
    scenario_store.insert_match_event(
        match_id=match_id,
        step=int(current_step),
        phase=str(match.get("phase") or ""),
        event_type="forfeit",
        agent_id=forfeiter,
        target_id=None,
        data_json={"reason": str(reason)},
    )
    _record_scenario_memory(
        agent_id=forfeiter,
        match_id=match_id,
        kind="match_forfeit",
        summary=f"You forfeited the match ({reason}).",
        payload={"reason": str(reason)},
        dedupe_suffix=f"forfeit:{current_step}",
    )

    status = str(match.get("status") or "").strip().lower()
    state = dict(match.get("state_json") or {})
    min_players = max(2, int(scenario.get("min_players") or 2))

    # Warmup/forming forfeit can invalidate the match before it starts.
    if status in {"forming", "warmup"}:
        if "alive_agent_ids" in state:
            state["alive_agent_ids"] = [aid for aid in (state.get("alive_agent_ids") or []) if str(aid) != forfeiter]
        if "seat_order" in state:
            state["seat_order"] = [aid for aid in (state.get("seat_order") or []) if str(aid) != forfeiter]
        remaining = [aid for aid in ids if aid != forfeiter]
        if len(remaining) < min_players:
            return _cancel_match_without_ratings(
                match=match,
                scenario=scenario,
                participants=participants,
                current_step=current_step,
                reason="Insufficient players after forfeit",
            )
        updated = scenario_store.update_match(
            match_id=match_id,
            state_json=state,
        )
        return updated or match

    key = str(match.get("scenario_key") or "").lower()
    if "werewolf" in key:
        alive = _alive_ids_from_state(state=state, participants=participants)
        dead = [str(aid) for aid in (state.get("dead_agent_ids") or []) if str(aid)]
        _eliminate_agent(
            match_id=match_id,
            alive=alive,
            dead=dead,
            agent_id=forfeiter,
            by_status="forfeit",
        )
        state["alive_agent_ids"] = alive
        state["dead_agent_ids"] = dead
        max_rounds = int((scenario.get("rules_json") or {}).get("max_rounds") or 6)
        resolved, winners, resolved_reason = _resolve_werewolf(
            alive=alive,
            role_map=dict(state.get("role_map") or {}),
            round_number=int(match.get("round_number") or 0),
            max_rounds=max_rounds,
        )
        if resolved:
            return _finalize_match(
                match=match,
                scenario=scenario,
                participants=participants,
                winners=winners,
                current_step=current_step,
                reason=resolved_reason or "Forfeit ended match",
                state_json=state,
            )
        updated = scenario_store.update_match(match_id=match_id, state_json=state)
        return updated or match

    folded = {str(aid) for aid in (state.get("folded_agent_ids") or []) if str(aid)}
    folded.add(forfeiter)
    state["folded_agent_ids"] = sorted(folded)
    active_after = _anaconda_active_ids(state, participants)
    if len(active_after) <= 1 and active_after:
        winner = active_after[0]
        chips = {str(k): int(v) for k, v in (state.get("chips") or {}).items() if str(k)}
        chips[winner] = int(chips.get(winner, 0)) + int(state.get("pot") or 0)
        state["chips"] = chips
        state["pot"] = 0
        state["showdown"] = {"winners": [winner], "reason": "All opponents folded/forfeit"}
        return _finalize_match(
            match=match,
            scenario=scenario,
            participants=participants,
            winners=[winner],
            current_step=current_step,
            reason="All opponents folded/forfeit",
            state_json=state,
        )
    updated = scenario_store.update_match(match_id=match_id, state_json=state)
    return updated or match


def cancel_match(
    *,
    match_id: str,
    requested_by_agent_id: str | None,
    current_step: int = 0,
    reason: str = "Cancelled by participant",
) -> dict[str, Any]:
    match = scenario_store.get_match(match_id=str(match_id))
    if not match:
        raise ValueError(f"Match not found: {match_id}")
    status = str(match.get("status") or "").strip().lower()
    if status in RESOLVED_MATCH_STATUSES:
        return match
    if status == "active":
        raise RuntimeError("Active matches cannot be cancelled directly; use forfeit")

    participants = scenario_store.list_match_participants(match_id=str(match_id))
    participant_ids = {str(item.get("agent_id") or "") for item in participants}
    requested = str(requested_by_agent_id or "").strip()
    if requested and requested not in participant_ids:
        raise PermissionError("Only match participants can cancel this match")

    scenario = scenario_store.get_scenario(scenario_key=str(match.get("scenario_key") or ""))
    if not scenario:
        raise ValueError("Scenario definition missing")
    return _cancel_match_without_ratings(
        match=match,
        scenario=scenario,
        participants=participants,
        current_step=int(current_step),
        reason=str(reason),
    )


def forfeit_match(
    *,
    match_id: str,
    agent_id: str,
    current_step: int = 0,
    reason: str = "Player forfeited",
) -> dict[str, Any]:
    match = scenario_store.get_match(match_id=str(match_id))
    if not match:
        raise ValueError(f"Match not found: {match_id}")
    status = str(match.get("status") or "").strip().lower()
    if status in RESOLVED_MATCH_STATUSES:
        return match

    scenario = scenario_store.get_scenario(scenario_key=str(match.get("scenario_key") or ""))
    if not scenario:
        raise ValueError("Scenario definition missing")
    participants = scenario_store.list_match_participants(match_id=str(match_id))
    return _forfeit_from_match(
        match=match,
        scenario=scenario,
        participants=participants,
        agent_id=str(agent_id),
        current_step=int(current_step),
        reason=str(reason),
    )


def _auto_forfeit_departed_participants(
    *,
    match: dict[str, Any],
    scenario: dict[str, Any],
    current_step: int,
) -> dict[str, Any]:
    town_id = str(match.get("town_id") or "")
    if not town_id:
        return match
    member_map = _collect_town_member_map(town_id)
    participants = scenario_store.list_match_participants(match_id=str(match.get("match_id") or ""))
    updated_match = match
    for participant in participants:
        aid = str(participant.get("agent_id") or "")
        if not aid:
            continue
        status = str(participant.get("status") or "").strip().lower()
        if not _status_is_active_participant(status):
            continue
        if aid in member_map:
            continue
        updated_match = _forfeit_from_match(
            match=updated_match,
            scenario=scenario,
            participants=scenario_store.list_match_participants(match_id=str(match.get("match_id") or "")),
            agent_id=aid,
            current_step=int(current_step),
            reason="Agent left town",
        )
        if str(updated_match.get("status") or "").strip().lower() in RESOLVED_MATCH_STATUSES:
            return updated_match
    return updated_match


def _advance_one_match(
    *,
    match: dict[str, Any],
    scenario: dict[str, Any],
    current_step: int,
    cognition_planner: Any = None,
) -> dict[str, Any]:
    status = str(match.get("status") or "").strip().lower()
    if status in RESOLVED_MATCH_STATUSES:
        return match

    # Agents may leave the town while a match is running; treat that as forfeit.
    match = _auto_forfeit_departed_participants(
        match=match,
        scenario=scenario,
        current_step=int(current_step),
    )
    status = str(match.get("status") or "").strip().lower()
    if status in RESOLVED_MATCH_STATUSES:
        return match

    if status == "forming":
        updated = scenario_store.update_match(
            match_id=str(match["match_id"]),
            status="warmup",
            phase="warmup",
            warmup_start_step=int(current_step),
        )
        return updated or match

    if status == "warmup":
        warmup_steps = max(0, int(scenario.get("warmup_steps") or 0))
        warmup_start = match.get("warmup_start_step")
        if warmup_start is None:
            warmup_start = int(current_step)
        if int(current_step) - int(warmup_start) < warmup_steps:
            return match

        phases = _phase_sequence(scenario)
        start_phase = phases[0] if phases else "active"
        updated = scenario_store.update_match(
            match_id=str(match["match_id"]),
            status="active",
            phase=start_phase,
            phase_step=0,
            round_number=1,
            start_step=int(current_step),
            state_json={**dict(match.get("state_json") or {}), "phase_index": 0},
        )
        scenario_store.insert_match_event(
            match_id=str(match["match_id"]),
            step=int(current_step),
            phase=start_phase,
            event_type="match_started",
            agent_id=None,
            target_id=None,
            data_json={"phase": start_phase},
        )
        return updated or match

    # status == active
    max_duration = max(1, int(scenario.get("max_duration_steps") or 30))
    start_step = int(match.get("start_step") or current_step)
    if int(current_step) - start_step >= max_duration:
        participants = scenario_store.list_match_participants(match_id=str(match["match_id"]))
        winners: list[str] = []
        if participants:
            ordered = [str(item["agent_id"]) for item in participants]
            _rng(f"timeout:{match['match_id']}:{current_step}").shuffle(ordered)
            winners = ordered[:1]
        return _finalize_match(
            match=match,
            scenario=scenario,
            participants=participants,
            winners=winners,
            current_step=current_step,
            reason="Duration limit reached",
            state_json=dict(match.get("state_json") or {}),
        )

    key = str(match.get("scenario_key") or "").lower()
    if "werewolf" in key:
        return _advance_werewolf_match(
            match=match,
            scenario=scenario,
            current_step=current_step,
            cognition_planner=cognition_planner,
        )
    return _advance_anaconda_match(
        match=match,
        scenario=scenario,
        current_step=current_step,
        cognition_planner=cognition_planner,
    )


def advance_matches(*, town_id: str, current_step: int, cognition_planner: Any = None) -> list[dict[str, Any]]:
    active = scenario_store.list_active_matches(town_id=town_id)
    if not active:
        return []

    out: list[dict[str, Any]] = []
    for match in active:
        scenario = scenario_store.get_scenario(scenario_key=str(match.get("scenario_key") or ""))
        if not scenario:
            continue
        updated = _advance_one_match(
            match=match,
            scenario=scenario,
            current_step=int(current_step),
            cognition_planner=cognition_planner,
        )
        out.append(updated)
    return out


_SCENARIO_MANAGER = TownScenarioManager(
    form_matches_fn=lambda town_id, step: form_matches(town_id=town_id, current_step=step),
    advance_matches_fn=lambda town_id, step: advance_matches(town_id=town_id, current_step=step),
    list_active_matches_fn=lambda town_id: list_active_matches_for_town(town_id=town_id),
    scenario_statuses_fn=lambda town_id: get_town_agent_scenario_statuses(town_id=town_id),
)


def get_town_scenario_manager() -> TownScenarioManager:
    return _SCENARIO_MANAGER


def get_agent_queue_status(*, agent_id: str) -> list[dict[str, Any]]:
    return scenario_store.list_agent_queue(agent_id=agent_id)


def get_agent_active_match(*, agent_id: str) -> dict[str, Any] | None:
    return scenario_store.get_agent_active_match(agent_id=agent_id)


def list_active_matches_for_town(*, town_id: str) -> list[dict[str, Any]]:
    matches = scenario_store.list_active_matches(town_id=town_id)
    out: list[dict[str, Any]] = []
    for match in matches:
        participants = scenario_store.list_match_participants(match_id=str(match["match_id"]))
        scenario = scenario_store.get_scenario(scenario_key=str(match.get("scenario_key") or ""))
        out.append(
            {
                "match_id": match["match_id"],
                "scenario_key": match.get("scenario_key"),
                "status": match.get("status"),
                "phase": match.get("phase"),
                "round_number": int(match.get("round_number") or 0),
                "participant_count": len(participants),
                "town_id": match.get("town_id"),
                "arena_id": match.get("arena_id"),
                "created_step": int(match.get("created_step") or 0),
                "buy_in": int((scenario or {}).get("buy_in") or 0),
                "pot": int((match.get("state_json") or {}).get("pot") or 0),
            }
        )
    return out


def get_match_payload(*, match_id: str) -> dict[str, Any] | None:
    match = scenario_store.get_match(match_id=match_id)
    if match is None:
        return None
    participants = scenario_store.list_match_participants(match_id=match_id)
    events = list(reversed(scenario_store.list_match_events(match_id=match_id, limit=200)))
    scenario = scenario_store.get_scenario(scenario_key=str(match.get("scenario_key") or ""))
    return {
        "match": match,
        "scenario": scenario,
        "participants": participants,
        "events": events,
    }


def get_spectator_payload(*, match_id: str) -> dict[str, Any] | None:
    payload = get_match_payload(match_id=match_id)
    if payload is None:
        return None

    match = dict(payload["match"])
    scenario = payload["scenario"] or {}
    participants = list(payload["participants"])
    events = list(payload["events"])

    state = dict(match.get("state_json") or {})
    chips_map = {str(aid): int(value) for aid, value in (state.get("chips") or {}).items() if str(aid)}
    revealed_cards = {
        str(aid): [str(card) for card in (cards or [])]
        for aid, cards in (state.get("revealed_cards") or {}).items()
        if str(aid)
    }
    public_state = {
        "match_id": match.get("match_id"),
        "scenario_key": match.get("scenario_key"),
        "status": match.get("status"),
        "phase": match.get("phase"),
        "round_number": int(match.get("round_number") or 0),
        "town_id": match.get("town_id"),
        "arena_id": match.get("arena_id"),
        "participant_count": len(participants),
        "buy_in": int(scenario.get("buy_in") or 0),
        "pot": int(state.get("pot") or 0),
        "alive_agent_ids": [str(v) for v in (state.get("alive_agent_ids") or []) if str(v)],
        "folded_agent_ids": [str(v) for v in (state.get("folded_agent_ids") or []) if str(v)],
        "all_in_agent_ids": [str(v) for v in (state.get("all_in_agent_ids") or []) if str(v)],
        "revealed_cards": revealed_cards,
        "participants": [
            {
                "agent_id": str(item.get("agent_id") or ""),
                "status": str(item.get("status") or ""),
                "score": item.get("score"),
                "chips": chips_map.get(str(item.get("agent_id") or ""), 0),
                "chips_end": item.get("chips_end"),
                "rating_delta": item.get("rating_delta"),
                "role": item.get("role") if str(match.get("status")) == "resolved" else None,
            }
            for item in participants
        ],
        "showdown": dict(state.get("showdown") or {}) if str(match.get("status")) == "resolved" else {},
        "result": match.get("result_json") if str(match.get("status")) == "resolved" else {},
    }
    return {
        "match_id": match_id,
        "public_state": public_state,
        "recent_events": events[-80:],
    }
