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


def _scenario_rules(scenario: dict[str, Any] | None) -> dict[str, Any]:
    rules = (scenario or {}).get("rules_json") or {}
    if isinstance(rules, dict):
        return rules
    return {}


def _scenario_engine_key(scenario: dict[str, Any] | None, match: dict[str, Any] | None = None) -> str:
    rules = _scenario_rules(scenario)
    configured = str(rules.get("engine") or "").strip().lower()
    if configured in {"werewolf", "anaconda", "blackjack", "holdem"}:
        return configured
    key = str((scenario or {}).get("scenario_key") or (match or {}).get("scenario_key") or "").strip().lower()
    if "werewolf" in key:
        return "werewolf"
    if "blackjack" in key:
        return "blackjack"
    if "holdem" in key or "hold'em" in key:
        return "holdem"
    if "anaconda" in key:
        return "anaconda"
    return "anaconda"


def _scenario_ui_group(scenario: dict[str, Any] | None) -> str:
    rules = _scenario_rules(scenario)
    value = str(rules.get("ui_group") or "").strip().lower()
    if value:
        return value
    engine = _scenario_engine_key(scenario, None)
    if engine == "werewolf":
        return "social"
    return "casino"


def _scenario_spectator_kind(scenario: dict[str, Any] | None, match: dict[str, Any] | None = None) -> str:
    rules = _scenario_rules(scenario)
    value = str(rules.get("spectator_kind") or "").strip().lower()
    if value:
        return value
    return _scenario_engine_key(scenario, match)


def _phase_sequence(scenario: dict[str, Any]) -> list[str]:
    rules = _scenario_rules(scenario)
    phases = rules.get("phase_sequence") if isinstance(rules, dict) else None
    if isinstance(phases, list):
        out = [str(value).strip() for value in phases if str(value).strip()]
        if out:
            return out
    engine = _scenario_engine_key(scenario, None)
    if engine == "werewolf":
        return ["night", "day"]
    if engine == "blackjack":
        return ["deal", "player_turns", "dealer_turn", "settle"]
    if engine == "holdem":
        return ["preflop", "flop", "turn", "river", "showdown"]
    return ["deal", "pass1", "pass2", "pass3", "discard", "reveal1", "reveal2", "reveal3", "reveal4", "reveal5", "showdown"]


def _arena_id(scenario: dict[str, Any]) -> str:
    rules = _scenario_rules(scenario)
    if isinstance(rules, dict) and str(rules.get("arena_id") or "").strip():
        return str(rules.get("arena_id")).strip()
    engine = _scenario_engine_key(scenario, None)
    if engine == "werewolf":
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


def _participant_ids(participants: list[dict[str, Any]]) -> list[str]:
    return [str(item.get("agent_id") or "") for item in participants if str(item.get("agent_id") or "")]


def _blackjack_rank_value(card: str) -> int:
    token = str(card or "").strip().upper()
    rank = token[0] if token else ""
    if rank == "A":
        return 1
    if rank in {"T", "J", "Q", "K"}:
        return 10
    if rank.isdigit():
        try:
            return int(rank)
        except Exception:
            return 0
    return 0


def _blackjack_hand_state(cards: list[str]) -> dict[str, Any]:
    values = [_blackjack_rank_value(card) for card in cards]
    base_total = sum(max(0, int(value)) for value in values)
    aces = sum(1 for card in cards if str(card or "").strip().upper().startswith("A"))
    total = int(base_total)
    soft = False
    while aces > 0 and total + 10 <= 21:
        total += 10
        aces -= 1
        soft = True
    return {
        "total": int(total),
        "soft": bool(soft),
        "blackjack": len(cards) == 2 and int(total) == 21,
        "busted": int(total) > 21,
    }


def _blackjack_active_player_ids(state: dict[str, Any], participants: list[dict[str, Any]]) -> list[str]:
    chips = {str(k): int(v) for k, v in (state.get("chips") or {}).items() if str(k)}
    out: list[str] = []
    for item in participants:
        aid = str(item.get("agent_id") or "")
        if not aid:
            continue
        status = str(item.get("status") or "").strip().lower()
        if status in {"forfeit", "cancelled", "done", "eliminated"}:
            continue
        if int(chips.get(aid, 0)) > 0:
            out.append(aid)
    return out


def _holdem_active_ids(state: dict[str, Any], participants: list[dict[str, Any]]) -> list[str]:
    folded = {str(aid) for aid in (state.get("folded_agent_ids") or []) if str(aid)}
    out: list[str] = []
    for item in participants:
        aid = str(item.get("agent_id") or "")
        if not aid:
            continue
        status = str(item.get("status") or "").strip().lower()
        if aid in folded or status in {"forfeit", "cancelled", "done", "eliminated", "folded"}:
            continue
        out.append(aid)
    return out


def _holdem_street_index(phase: str) -> int:
    normalized = str(phase or "").strip().lower()
    if normalized == "preflop":
        return 0
    if normalized == "flop":
        return 1
    if normalized == "turn":
        return 2
    if normalized == "river":
        return 3
    return 3


def _holdem_next_seat(seat_order: list[str], current_aid: str, offset: int = 1) -> str | None:
    ids = [str(aid) for aid in seat_order if str(aid)]
    if not ids:
        return None
    current = str(current_aid or "")
    if current not in ids:
        return ids[0]
    idx = ids.index(current)
    return ids[(idx + max(1, int(offset))) % len(ids)]


def _holdem_visible_hole_cards(state: dict[str, Any], *, reveal: bool) -> dict[str, list[str]]:
    raw = state.get("hole_cards") or {}
    out: dict[str, list[str]] = {}
    for aid, cards in raw.items():
        key = str(aid)
        current = [str(card).upper() for card in (cards or []) if str(card)]
        if reveal:
            out[key] = current
        else:
            out[key] = ["??" for _ in current]
    return out


def _next_engine_phase(
    *,
    state: dict[str, Any],
    scenario: dict[str, Any],
    default_phase: str,
) -> tuple[list[str], int, str]:
    phases = _phase_sequence(scenario)
    if not phases:
        phases = [default_phase]
    phase_index = int(state.get("phase_index") or 0)
    phase_index = max(0, min(len(phases) - 1, phase_index))
    phase = str(phases[phase_index] or default_phase)
    return phases, phase_index, phase


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
        engine = _scenario_engine_key(scenario, None)
        rules = _scenario_rules(scenario)
        if engine == "werewolf":
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
                "max_rounds": int(rules.get("max_rounds") or 6),
            }
        elif engine == "anaconda":
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
        elif engine == "blackjack":
            state_json = {
                "phase_sequence": _phase_sequence(scenario),
                "phase_index": 0,
                "buy_in": buy_in,
                "pot": 0,
                "chips": {aid: int(buy_in) for aid in agent_ids},
                "seat_order": list(agent_ids),
                "deck": [],
                "dealer_cards": [],
                "dealer_cards_public": [],
                "player_hands": {aid: [] for aid in agent_ids},
                "player_bets": {aid: 0 for aid in agent_ids},
                "player_actions": {},
                "hand_results": [],
                "current_hand_index": 1,
                "max_hands": max(1, int(rules.get("max_hands") or 8)),
                "min_bet": max(1, int(rules.get("min_bet") or 5)),
                "max_bet": max(1, int(rules.get("max_bet") or 25)),
                "dealer_soft17_stand": bool(rules.get("dealer_soft17_stand", True)),
                "allow_double": bool(rules.get("allow_double", True)),
                "seed": f"{key}:{','.join(sorted(agent_ids))}",
                "town_id": str(town_id),
                "showdown": {},
            }
        else:
            bet_units_raw = rules.get("bet_units")
            if isinstance(bet_units_raw, list):
                bet_units = [max(1, int(value)) for value in bet_units_raw[:4] if str(value).strip()]
            else:
                bet_units = []
            if len(bet_units) < 4:
                bet_units = [10, 10, 20, 20]
            small_blind = max(1, int(rules.get("small_blind") or 5))
            big_blind = max(small_blind, int(rules.get("big_blind") or (small_blind * 2)))
            state_json = {
                "phase_sequence": _phase_sequence(scenario),
                "phase_index": 0,
                "buy_in": buy_in,
                "pot": 0,
                "chips": {aid: int(buy_in) for aid in agent_ids},
                "contributions": {aid: 0 for aid in agent_ids},
                "seat_order": list(agent_ids),
                "deck": [],
                "hole_cards": {aid: [] for aid in agent_ids},
                "board_cards": [],
                "folded_agent_ids": [],
                "all_in_agent_ids": [],
                "street_commitments": {aid: 0 for aid in agent_ids},
                "current_bet": 0,
                "street_raise_count": 0,
                "small_blind": int(small_blind),
                "big_blind": int(big_blind),
                "bet_units": bet_units,
                "max_raises_per_street": max(1, int(rules.get("max_raises_per_street") or 3)),
                "hands_per_match": max(1, int(rules.get("hands_per_match") or 1)),
                "hand_number": 1,
                "button_agent_id": None,
                "small_blind_agent_id": None,
                "big_blind_agent_id": None,
                "seed": f"{key}:{','.join(sorted(agent_ids))}",
                "town_id": str(town_id),
                "showdown": {},
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
                    "status": "alive" if engine == "werewolf" else "assigned",
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
        if engine == "werewolf":
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
                    kind="scenario_buy_in",
                    summary=f"Joined {key} with buy-in {buy_in}.",
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


def _advance_blackjack_match(
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
    phases, phase_index, phase = _next_engine_phase(
        state=state,
        scenario=scenario,
        default_phase="deal",
    )
    seat_order = [str(aid) for aid in (state.get("seat_order") or []) if str(aid)]
    if not seat_order:
        seat_order = _participant_ids(participants)

    chips = {str(k): int(v) for k, v in (state.get("chips") or {}).items() if str(k)}
    for aid in _participant_ids(participants):
        chips.setdefault(aid, int(state.get("buy_in") or scenario.get("buy_in") or 0))
    deck = [str(card).upper() for card in (state.get("deck") or [])]
    player_hands = {
        str(aid): [str(card).upper() for card in (cards or [])]
        for aid, cards in (state.get("player_hands") or {}).items()
        if str(aid)
    }
    player_bets = {str(aid): int(value) for aid, value in (state.get("player_bets") or {}).items() if str(aid)}
    player_actions = {
        str(aid): [str(action) for action in (actions or [])]
        for aid, actions in (state.get("player_actions") or {}).items()
        if str(aid)
    }
    dealer_cards = [str(card).upper() for card in (state.get("dealer_cards") or []) if str(card)]
    dealer_cards_public = [str(card).upper() for card in (state.get("dealer_cards_public") or []) if str(card)]
    hand_results = list(state.get("hand_results") or [])
    pot = int(state.get("pot") or 0)
    max_hands = max(1, int(state.get("max_hands") or _scenario_rules(scenario).get("max_hands") or 8))
    min_bet = max(1, int(state.get("min_bet") or _scenario_rules(scenario).get("min_bet") or 5))
    max_bet = max(min_bet, int(state.get("max_bet") or _scenario_rules(scenario).get("max_bet") or 25))
    allow_double = bool(state.get("allow_double", _scenario_rules(scenario).get("allow_double", True)))
    dealer_soft17_stand = bool(
        state.get("dealer_soft17_stand", _scenario_rules(scenario).get("dealer_soft17_stand", True))
    )
    current_hand_index = max(1, int(state.get("current_hand_index") or 1))

    active_ids = _blackjack_active_player_ids(state, participants)
    if phase == "deal" and len(active_ids) <= 1:
        survivors = active_ids or sorted(chips.keys(), key=lambda aid: (-int(chips.get(aid, 0)), aid))[:1]
        reason = "Single bankroll survivor" if survivors else "No eligible players"
        state["chips"] = chips
        state["showdown"] = {
            "winners": list(survivors),
            "reason": reason,
            "chips": chips,
            "hands_played": int(max(0, current_hand_index - 1)),
        }
        return _finalize_match(
            match=match,
            scenario=scenario,
            participants=participants,
            winners=list(survivors),
            current_step=current_step,
            reason=reason,
            state_json=state,
        )

    if phase == "deal":
        seed = f"{state.get('seed') or match['match_id']}:blackjack:{current_step}:{current_hand_index}"
        cards_needed = (len(active_ids) * 3) + 4
        if len(deck) < cards_needed:
            deck = make_shuffled_deck(seed=seed)

        dealer_cards = []
        dealer_cards_public = []
        player_hands = {aid: [] for aid in seat_order}
        player_bets = {aid: 0 for aid in seat_order}
        player_actions = {aid: [] for aid in seat_order}
        pot = 0

        for aid in active_ids:
            stack = max(0, int(chips.get(aid, 0)))
            rng = _rng(f"blackjack-bet:{match['match_id']}:{aid}:{current_step}:{current_hand_index}")
            baseline = min_bet + rng.randint(0, max(0, max_bet - min_bet))
            default_bet = min(stack, max(min_bet if stack >= min_bet else stack, baseline))
            decision = _planner_call(
                cognition_planner,
                "blackjack_choose_bet",
                {
                    "town_id": str(match.get("town_id") or ""),
                    "agent_id": aid,
                    "match_id": str(match["match_id"]),
                    "phase": "deal",
                    "step": int(current_step),
                    "hand_index": int(current_hand_index),
                    "max_hands": int(max_hands),
                    "stack": int(stack),
                    "min_bet": int(min_bet),
                    "max_bet": int(max_bet),
                    "active_player_count": int(len(active_ids)),
                },
            )
            try:
                planned_bet = int((decision or {}).get("bet") or default_bet)
            except Exception:
                planned_bet = default_bet
            if stack <= 0:
                bet = 0
            elif stack < min_bet:
                bet = stack
            else:
                bet = max(min_bet, min(stack, min(max_bet, planned_bet)))
            chips[aid] = max(0, stack - bet)
            player_bets[aid] = int(bet)
            pot += int(bet)

        for _ in range(2):
            for aid in active_ids:
                if deck:
                    player_hands.setdefault(aid, []).append(deck.pop())
            if deck:
                dealer_cards.append(deck.pop())
        if dealer_cards:
            dealer_cards_public = [dealer_cards[0]]
            if len(dealer_cards) > 1:
                dealer_cards_public.append("??")

        scenario_store.insert_match_event(
            match_id=str(match["match_id"]),
            step=int(current_step),
            phase=phase,
            event_type="blackjack_deal",
            agent_id=None,
            target_id=None,
            data_json={
                "hand_index": int(current_hand_index),
                "bets": player_bets,
                "dealer_upcard": dealer_cards[0] if dealer_cards else None,
                "active_players": active_ids,
            },
        )
    elif phase == "player_turns":
        for aid in active_ids:
            bet = int(player_bets.get(aid, 0))
            if bet <= 0:
                continue
            hand = list(player_hands.get(aid, []))
            actions: list[str] = []
            doubled = False
            for _ in range(8):
                hand_state = _blackjack_hand_state(hand)
                if hand_state["busted"] or hand_state["total"] >= 21:
                    break
                stack = max(0, int(chips.get(aid, 0)))
                can_double = bool(allow_double and not doubled and len(hand) == 2 and stack >= bet and bet > 0)
                default_action = "hit" if int(hand_state["total"]) <= 14 else "stand"
                decision = _planner_call(
                    cognition_planner,
                    "blackjack_choose_action",
                    {
                        "town_id": str(match.get("town_id") or ""),
                        "agent_id": aid,
                        "match_id": str(match["match_id"]),
                        "phase": "player_turns",
                        "step": int(current_step),
                        "hand_index": int(current_hand_index),
                        "cards": list(hand),
                        "total": int(hand_state["total"]),
                        "soft": bool(hand_state["soft"]),
                        "dealer_upcard": dealer_cards[0] if dealer_cards else None,
                        "stack": int(stack),
                        "bet": int(bet),
                        "can_double": bool(can_double),
                        "allowed_actions": ["hit", "stand", "double"] if can_double else ["hit", "stand"],
                    },
                )
                action = str((decision or {}).get("action") or default_action).strip().lower()
                if action not in {"hit", "stand", "double"}:
                    action = default_action
                if action == "double":
                    if not can_double:
                        action = "stand" if int(hand_state["total"]) >= 16 else "hit"
                    else:
                        extra = min(stack, bet)
                        chips[aid] = max(0, stack - extra)
                        bet += int(extra)
                        player_bets[aid] = int(bet)
                        pot += int(extra)
                        doubled = True
                        if deck:
                            hand.append(deck.pop())
                        actions.append("double")
                        break
                if action == "hit":
                    if deck:
                        hand.append(deck.pop())
                    actions.append("hit")
                    continue
                actions.append("stand")
                break
            player_hands[aid] = hand
            player_actions[aid] = actions

        scenario_store.insert_match_event(
            match_id=str(match["match_id"]),
            step=int(current_step),
            phase=phase,
            event_type="blackjack_player_turns",
            agent_id=None,
            target_id=None,
            data_json={
                "actions": player_actions,
                "totals": {aid: _blackjack_hand_state(cards).get("total") for aid, cards in player_hands.items()},
            },
        )
    elif phase == "dealer_turn":
        if len(dealer_cards) < 2 and deck:
            dealer_cards.append(deck.pop())
        while True:
            dealer_state = _blackjack_hand_state(dealer_cards)
            total = int(dealer_state["total"])
            if total > 21:
                break
            if total < 17:
                if not deck:
                    break
                dealer_cards.append(deck.pop())
                continue
            if total == 17 and bool(dealer_state["soft"]) and not dealer_soft17_stand:
                if not deck:
                    break
                dealer_cards.append(deck.pop())
                continue
            break
        dealer_cards_public = list(dealer_cards)
        scenario_store.insert_match_event(
            match_id=str(match["match_id"]),
            step=int(current_step),
            phase=phase,
            event_type="blackjack_dealer_turn",
            agent_id=None,
            target_id=None,
            data_json={"dealer_cards": dealer_cards_public, "dealer_total": _blackjack_hand_state(dealer_cards)["total"]},
        )
    elif phase == "settle":
        dealer_state = _blackjack_hand_state(dealer_cards)
        payout_map: dict[str, int] = {}
        outcome_rows: dict[str, Any] = {}
        for aid in seat_order:
            bet = max(0, int(player_bets.get(aid, 0)))
            if bet <= 0:
                continue
            hand = list(player_hands.get(aid, []))
            hand_state = _blackjack_hand_state(hand)
            payout = 0
            outcome = "lose"
            if hand_state["busted"]:
                outcome = "bust"
            elif dealer_state["busted"]:
                payout = bet * 2
                outcome = "win_dealer_bust"
            elif int(hand_state["total"]) > int(dealer_state["total"]):
                payout = bet * 2
                outcome = "win"
            elif int(hand_state["total"]) == int(dealer_state["total"]):
                payout = bet
                outcome = "push"
            if payout:
                chips[aid] = int(chips.get(aid, 0)) + int(payout)
            payout_map[aid] = int(payout)
            outcome_rows[aid] = {
                "cards": hand,
                "total": int(hand_state["total"]),
                "outcome": outcome,
                "bet": int(bet),
                "payout": int(payout),
            }
            if int(chips.get(aid, 0)) <= 0:
                scenario_store.update_match_participant(
                    match_id=str(match["match_id"]),
                    agent_id=aid,
                    updates={"status": "eliminated"},
                )

        pot = 0
        hand_results.append(
            {
                "hand_index": int(current_hand_index),
                "dealer_cards": list(dealer_cards),
                "dealer_total": int(dealer_state["total"]),
                "rows": outcome_rows,
                "payouts": payout_map,
            }
        )
        survivors = [aid for aid in seat_order if int(chips.get(aid, 0)) > 0]
        should_finish = int(current_hand_index) >= int(max_hands) or len(survivors) <= 1
        state["chips"] = chips
        state["pot"] = 0
        state["player_hands"] = player_hands
        state["player_bets"] = player_bets
        state["player_actions"] = player_actions
        state["dealer_cards"] = dealer_cards
        state["dealer_cards_public"] = list(dealer_cards)
        state["hand_results"] = hand_results[-24:]
        state["current_hand_index"] = int(current_hand_index)
        state["showdown"] = {
            "hands_played": int(current_hand_index),
            "dealer_cards": list(dealer_cards),
            "dealer_total": int(dealer_state["total"]),
            "rows": outcome_rows,
            "payouts": payout_map,
            "chips": chips,
        }
        scenario_store.insert_match_event(
            match_id=str(match["match_id"]),
            step=int(current_step),
            phase=phase,
            event_type="blackjack_settle",
            agent_id=None,
            target_id=None,
            data_json={
                "hand_index": int(current_hand_index),
                "dealer_total": int(dealer_state["total"]),
                "payouts": payout_map,
            },
        )
        if should_finish:
            top_stack = max([int(chips.get(aid, 0)) for aid in seat_order], default=0)
            winners = sorted([aid for aid in seat_order if int(chips.get(aid, 0)) == int(top_stack)])
            reason = "Maximum hands completed" if int(current_hand_index) >= int(max_hands) else "Single bankroll survivor"
            state["showdown"]["winners"] = winners
            state["showdown"]["reason"] = reason
            return _finalize_match(
                match=match,
                scenario=scenario,
                participants=participants,
                winners=winners,
                current_step=current_step,
                reason=reason,
                state_json=state,
            )
        current_hand_index += 1

    state["phase_sequence"] = phases
    state["phase_index"] = phase_index
    state["seat_order"] = seat_order
    state["chips"] = chips
    state["deck"] = deck
    state["player_hands"] = player_hands
    state["player_bets"] = player_bets
    state["player_actions"] = player_actions
    state["dealer_cards"] = dealer_cards
    state["dealer_cards_public"] = dealer_cards_public
    state["pot"] = int(pot)
    state["current_hand_index"] = int(current_hand_index)
    state["max_hands"] = int(max_hands)
    state["min_bet"] = int(min_bet)
    state["max_bet"] = int(max_bet)
    state["allow_double"] = bool(allow_double)
    state["dealer_soft17_stand"] = bool(dealer_soft17_stand)
    state["hand_results"] = hand_results[-24:]

    next_phase_index = phase_index + 1
    if next_phase_index >= len(phases):
        next_phase_index = 0
    next_phase = phases[next_phase_index]
    state["phase_index"] = next_phase_index
    updated = scenario_store.update_match(
        match_id=str(match["match_id"]),
        phase=str(next_phase),
        phase_step=int(match.get("phase_step") or 0) + 1,
        round_number=int(current_hand_index),
        state_json=state,
    )
    scenario_store.insert_match_event(
        match_id=str(match["match_id"]),
        step=int(current_step),
        phase=str(next_phase),
        event_type="phase_change",
        agent_id=None,
        target_id=None,
        data_json={"phase": str(next_phase), "hand_index": int(current_hand_index)},
    )
    return updated or match


def _advance_holdem_match(
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
    phases, phase_index, phase = _next_engine_phase(
        state=state,
        scenario=scenario,
        default_phase="preflop",
    )
    seat_order = [str(aid) for aid in (state.get("seat_order") or []) if str(aid)]
    if not seat_order:
        seat_order = _participant_ids(participants)
    chips = {str(k): int(v) for k, v in (state.get("chips") or {}).items() if str(k)}
    for aid in _participant_ids(participants):
        chips.setdefault(aid, int(state.get("buy_in") or scenario.get("buy_in") or 0))
    contributions = {str(k): int(v) for k, v in (state.get("contributions") or {}).items() if str(k)}
    for aid in seat_order:
        contributions.setdefault(aid, 0)

    deck = [str(card).upper() for card in (state.get("deck") or []) if str(card)]
    hole_cards = {
        str(aid): [str(card).upper() for card in (cards or [])]
        for aid, cards in (state.get("hole_cards") or {}).items()
        if str(aid)
    }
    board_cards = [str(card).upper() for card in (state.get("board_cards") or []) if str(card)]
    folded = {str(aid) for aid in (state.get("folded_agent_ids") or []) if str(aid)}
    all_in = {str(aid) for aid in (state.get("all_in_agent_ids") or []) if str(aid)}
    street_commitments = {str(k): int(v) for k, v in (state.get("street_commitments") or {}).items() if str(k)}
    for aid in seat_order:
        street_commitments.setdefault(aid, 0)

    pot = int(state.get("pot") or 0)
    small_blind = max(1, int(state.get("small_blind") or _scenario_rules(scenario).get("small_blind") or 5))
    big_blind = max(small_blind, int(state.get("big_blind") or _scenario_rules(scenario).get("big_blind") or 10))
    bet_units_raw = state.get("bet_units") or _scenario_rules(scenario).get("bet_units") or [10, 10, 20, 20]
    bet_units = [max(1, int(value)) for value in (bet_units_raw if isinstance(bet_units_raw, list) else [10, 10, 20, 20])]
    while len(bet_units) < 4:
        bet_units.append(bet_units[-1] if bet_units else 10)
    max_raises = max(1, int(state.get("max_raises_per_street") or _scenario_rules(scenario).get("max_raises_per_street") or 3))
    button_agent_id = str(state.get("button_agent_id") or "")
    small_blind_agent_id = str(state.get("small_blind_agent_id") or "")
    big_blind_agent_id = str(state.get("big_blind_agent_id") or "")
    current_bet = int(state.get("current_bet") or 0)
    street_raise_count = int(state.get("street_raise_count") or 0)
    hand_number = max(1, int(state.get("hand_number") or 1))

    # Initialize a new hand once at preflop.
    if phase == "preflop" and not any(hole_cards.get(aid) for aid in seat_order):
        seed = f"{state.get('seed') or match['match_id']}:holdem:{current_step}:{hand_number}"
        deck = make_shuffled_deck(seed=seed)
        folded = set()
        all_in = set()
        board_cards = []
        pot = 0
        contributions = {aid: 0 for aid in seat_order}
        street_commitments = {aid: 0 for aid in seat_order}
        hole_cards = {aid: [] for aid in seat_order}

        if not button_agent_id or button_agent_id not in seat_order:
            button_agent_id = seat_order[0] if seat_order else ""
        small_blind_agent_id = _holdem_next_seat(seat_order, button_agent_id, 1) or button_agent_id
        big_blind_agent_id = _holdem_next_seat(seat_order, small_blind_agent_id, 1) or small_blind_agent_id

        def _post_blind(aid: str, blind_amount: int) -> int:
            if not aid:
                return 0
            stack = max(0, int(chips.get(aid, 0)))
            paid = min(stack, max(0, int(blind_amount)))
            chips[aid] = max(0, stack - paid)
            contributions[aid] = int(contributions.get(aid, 0)) + int(paid)
            street_commitments[aid] = int(street_commitments.get(aid, 0)) + int(paid)
            if chips[aid] <= 0 and paid > 0:
                all_in.add(aid)
                scenario_store.update_match_participant(
                    match_id=str(match["match_id"]),
                    agent_id=aid,
                    updates={"status": "all_in"},
                )
            return int(paid)

        sb_paid = _post_blind(small_blind_agent_id, small_blind)
        bb_paid = _post_blind(big_blind_agent_id, big_blind)
        pot = sb_paid + bb_paid
        current_bet = max(sb_paid, bb_paid)
        street_raise_count = 0

        for _ in range(2):
            for aid in seat_order:
                if deck:
                    hole_cards.setdefault(aid, []).append(deck.pop())

        scenario_store.insert_match_event(
            match_id=str(match["match_id"]),
            step=int(current_step),
            phase=phase,
            event_type="holdem_deal",
            agent_id=None,
            target_id=None,
            data_json={
                "button": button_agent_id,
                "small_blind": small_blind_agent_id,
                "big_blind": big_blind_agent_id,
                "small_blind_paid": int(sb_paid),
                "big_blind_paid": int(bb_paid),
            },
        )

    active_ids = _holdem_active_ids(
        {
            **state,
            "folded_agent_ids": sorted(folded),
        },
        participants,
    )
    if len(active_ids) <= 1 and active_ids:
        winner = active_ids[0]
        chips[winner] = int(chips.get(winner, 0)) + int(pot)
        pot = 0
        state["chips"] = chips
        state["showdown"] = {"winners": [winner], "reason": "All opponents folded"}
        return _finalize_match(
            match=match,
            scenario=scenario,
            participants=participants,
            winners=[winner],
            current_step=current_step,
            reason="All opponents folded",
            state_json=state,
        )

    if phase in {"preflop", "flop", "turn", "river"}:
        street_idx = _holdem_street_index(phase)
        bet_unit = bet_units[min(street_idx, len(bet_units) - 1)]

        if phase == "flop" and len(board_cards) < 3:
            while len(board_cards) < 3 and deck:
                board_cards.append(deck.pop())
        elif phase == "turn" and len(board_cards) < 4 and deck:
            board_cards.append(deck.pop())
        elif phase == "river" and len(board_cards) < 5 and deck:
            board_cards.append(deck.pop())

        if phase == "preflop":
            first_actor = _holdem_next_seat(seat_order, big_blind_agent_id, 1) or (seat_order[0] if seat_order else "")
        else:
            first_actor = _holdem_next_seat(seat_order, button_agent_id, 1) or (seat_order[0] if seat_order else "")
            current_bet = 0
            street_raise_count = 0
            street_commitments = {aid: 0 for aid in seat_order}

        if first_actor in seat_order:
            start_idx = seat_order.index(first_actor)
            action_order = seat_order[start_idx:] + seat_order[:start_idx]
        else:
            action_order = list(seat_order)

        action_rows: list[dict[str, Any]] = []
        for aid in action_order:
            if aid in folded:
                continue
            if aid not in active_ids:
                continue
            stack = max(0, int(chips.get(aid, 0)))
            to_call = max(0, int(current_bet) - int(street_commitments.get(aid, 0)))
            if stack <= 0:
                all_in.add(aid)
                continue
            can_check = to_call == 0
            can_raise = bool(street_raise_count < max_raises and stack > to_call)
            rng = _rng(f"holdem-action:{match['match_id']}:{phase}:{aid}:{current_step}")
            default_action = "check"
            if to_call > 0:
                default_action = "call"
                if to_call > max(1, stack // 2) and rng.randint(0, 99) < 35:
                    default_action = "fold"
                elif can_raise and rng.randint(0, 99) < 18:
                    default_action = "raise"
            elif can_raise and rng.randint(0, 99) < 20:
                default_action = "raise"

            decision = _planner_call(
                cognition_planner,
                "holdem_choose_action",
                {
                    "town_id": str(match.get("town_id") or ""),
                    "agent_id": aid,
                    "match_id": str(match["match_id"]),
                    "phase": str(phase),
                    "step": int(current_step),
                    "stack": int(stack),
                    "to_call": int(to_call),
                    "current_bet": int(current_bet),
                    "street_commitment": int(street_commitments.get(aid, 0)),
                    "pot": int(pot),
                    "board_cards": list(board_cards),
                    "hole_cards": list(hole_cards.get(aid) or []),
                    "allowed_actions": [
                        "fold",
                        *(["check"] if can_check else []),
                        *(["call"] if to_call > 0 else []),
                        *(["raise"] if can_raise else []),
                    ],
                    "bet_unit": int(bet_unit),
                    "street_raise_count": int(street_raise_count),
                    "max_raises_per_street": int(max_raises),
                },
            )
            action = str((decision or {}).get("action") or default_action).strip().lower()
            if action not in {"fold", "check", "call", "raise"}:
                action = default_action
            if action == "check" and not can_check:
                action = "call"
            if action == "call" and to_call == 0:
                action = "check"
            if action == "raise" and not can_raise:
                action = "call" if to_call > 0 else "check"

            paid = 0
            if action == "fold":
                folded.add(aid)
                scenario_store.update_match_participant(
                    match_id=str(match["match_id"]),
                    agent_id=aid,
                    updates={"status": "folded"},
                )
            elif action == "call":
                paid = min(stack, to_call)
                chips[aid] = max(0, stack - paid)
                contributions[aid] = int(contributions.get(aid, 0)) + int(paid)
                street_commitments[aid] = int(street_commitments.get(aid, 0)) + int(paid)
                pot += int(paid)
            elif action == "raise":
                raise_to = int(current_bet) + int(bet_unit)
                needed = max(0, raise_to - int(street_commitments.get(aid, 0)))
                paid = min(stack, needed)
                chips[aid] = max(0, stack - paid)
                contributions[aid] = int(contributions.get(aid, 0)) + int(paid)
                street_commitments[aid] = int(street_commitments.get(aid, 0)) + int(paid)
                pot += int(paid)
                if int(street_commitments.get(aid, 0)) >= raise_to:
                    current_bet = raise_to
                    street_raise_count += 1
                elif int(street_commitments.get(aid, 0)) > int(current_bet):
                    current_bet = int(street_commitments.get(aid, 0))
            if chips.get(aid, 0) <= 0 and aid not in folded:
                all_in.add(aid)
                scenario_store.update_match_participant(
                    match_id=str(match["match_id"]),
                    agent_id=aid,
                    updates={"status": "all_in"},
                )
            action_rows.append(
                {
                    "agent_id": aid,
                    "action": action,
                    "to_call": int(to_call),
                    "paid": int(paid),
                    "current_bet": int(current_bet),
                    "stack": int(chips.get(aid, 0)),
                }
            )

        state_probe = {
            **state,
            "folded_agent_ids": sorted(folded),
        }
        active_after = _holdem_active_ids(state_probe, participants)
        if len(active_after) <= 1 and active_after:
            winner = active_after[0]
            chips[winner] = int(chips.get(winner, 0)) + int(pot)
            pot = 0
            state["chips"] = chips
            state["showdown"] = {"winners": [winner], "reason": "All opponents folded"}
            return _finalize_match(
                match=match,
                scenario=scenario,
                participants=participants,
                winners=[winner],
                current_step=current_step,
                reason="All opponents folded",
                state_json=state,
            )

        scenario_store.insert_match_event(
            match_id=str(match["match_id"]),
            step=int(current_step),
            phase=phase,
            event_type="holdem_betting_round",
            agent_id=None,
            target_id=None,
            data_json={
                "phase": str(phase),
                "actions": action_rows,
                "board_cards": list(board_cards),
                "pot": int(pot),
                "current_bet": int(current_bet),
            },
        )

    if phase == "showdown":
        contenders = _holdem_active_ids(
            {
                **state,
                "folded_agent_ids": sorted(folded),
            },
            participants,
        )
        if not contenders:
            contenders = [aid for aid in seat_order if aid not in folded]
        showdown_rows: dict[str, Any] = {}
        winners: list[str] = []
        best_score: tuple[int, tuple[int, ...]] | None = None
        score_map: dict[str, tuple[int, tuple[int, ...]]] = {}
        for aid in contenders:
            cards = list(hole_cards.get(aid) or []) + list(board_cards)
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
                "hole_cards": list(hole_cards.get(aid) or []),
            }
            if best_score is None or score > best_score:
                best_score = score
                winners = [aid]
            elif score == best_score:
                winners.append(aid)

        payout_map: dict[str, int]
        side_pots: list[dict[str, Any]]
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
                anchor = sorted(winners)[0]
                payout_map[anchor] = int(payout_map.get(anchor, 0)) + delta
        for aid, payout in payout_map.items():
            chips[aid] = int(chips.get(aid, 0)) + int(payout)
        pot = 0

        state["showdown"] = {
            "rows": showdown_rows,
            "winners": winners,
            "board_cards": list(board_cards),
            "payouts": payout_map,
            "side_pots": side_pots,
        }
        scenario_store.insert_match_event(
            match_id=str(match["match_id"]),
            step=int(current_step),
            phase=phase,
            event_type="holdem_showdown",
            agent_id=None,
            target_id=None,
            data_json={
                "winners": winners,
                "board_cards": list(board_cards),
                "hands": showdown_rows,
                "payouts": payout_map,
                "side_pots": side_pots,
            },
        )
        state["chips"] = chips
        state["pot"] = 0
        state["contributions"] = contributions
        state["hole_cards"] = hole_cards
        state["board_cards"] = board_cards
        state["folded_agent_ids"] = sorted(folded)
        state["all_in_agent_ids"] = sorted(all_in)
        state["phase_index"] = phase_index
        return _finalize_match(
            match=match,
            scenario=scenario,
            participants=participants,
            winners=winners,
            current_step=current_step,
            reason="Hold'em showdown complete",
            state_json=state,
        )

    state["phase_sequence"] = phases
    state["phase_index"] = phase_index
    state["seat_order"] = seat_order
    state["chips"] = chips
    state["contributions"] = contributions
    state["deck"] = deck
    state["hole_cards"] = hole_cards
    state["board_cards"] = board_cards
    state["folded_agent_ids"] = sorted(folded)
    state["all_in_agent_ids"] = sorted(all_in)
    state["street_commitments"] = street_commitments
    state["current_bet"] = int(current_bet)
    state["street_raise_count"] = int(street_raise_count)
    state["small_blind"] = int(small_blind)
    state["big_blind"] = int(big_blind)
    state["bet_units"] = bet_units
    state["max_raises_per_street"] = int(max_raises)
    state["pot"] = int(pot)
    state["button_agent_id"] = button_agent_id
    state["small_blind_agent_id"] = small_blind_agent_id
    state["big_blind_agent_id"] = big_blind_agent_id
    state["hand_number"] = int(hand_number)

    next_phase_index = min(len(phases) - 1, phase_index + 1)
    next_phase = phases[next_phase_index]
    state["phase_index"] = next_phase_index
    # Reset street-level commitments between betting streets.
    if next_phase in {"flop", "turn", "river", "showdown"}:
        state["street_commitments"] = {aid: 0 for aid in seat_order}
        state["current_bet"] = 0
        state["street_raise_count"] = 0

    updated = scenario_store.update_match(
        match_id=str(match["match_id"]),
        phase=str(next_phase),
        phase_step=int(match.get("phase_step") or 0) + 1,
        round_number=int(match.get("round_number") or 0) + 1,
        state_json=state,
    )
    scenario_store.insert_match_event(
        match_id=str(match["match_id"]),
        step=int(current_step),
        phase=str(next_phase),
        event_type="phase_change",
        agent_id=None,
        target_id=None,
        data_json={"phase": str(next_phase), "board_cards": list(board_cards), "pot": int(pot)},
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

    engine = _scenario_engine_key(scenario, match)
    if engine == "werewolf":
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
        max_rounds = int(_scenario_rules(scenario).get("max_rounds") or 6)
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

    if engine == "blackjack":
        chips = {str(k): int(v) for k, v in (state.get("chips") or {}).items() if str(k)}
        player_bets = {str(k): int(v) for k, v in (state.get("player_bets") or {}).items() if str(k)}
        player_bets[forfeiter] = 0
        state["player_bets"] = player_bets
        active_after = [
            aid
            for aid in sorted(chips.keys())
            if aid != forfeiter and int(chips.get(aid, 0)) > 0
        ]
        if len(active_after) <= 1 and active_after:
            winner = active_after[0]
            chips[winner] = int(chips.get(winner, 0)) + int(state.get("pot") or 0)
            state["chips"] = chips
            state["pot"] = 0
            state["showdown"] = {"winners": [winner], "reason": "All opponents forfeited"}
            return _finalize_match(
                match=match,
                scenario=scenario,
                participants=participants,
                winners=[winner],
                current_step=current_step,
                reason="All opponents forfeited",
                state_json=state,
            )
        state["chips"] = chips
        updated = scenario_store.update_match(match_id=match_id, state_json=state)
        return updated or match

    folded = {str(aid) for aid in (state.get("folded_agent_ids") or []) if str(aid)}
    folded.add(forfeiter)
    state["folded_agent_ids"] = sorted(folded)
    if engine == "holdem":
        active_after = _holdem_active_ids(state, participants)
    else:
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

    engine = _scenario_engine_key(scenario, match)
    if engine == "werewolf":
        return _advance_werewolf_match(
            match=match,
            scenario=scenario,
            current_step=current_step,
            cognition_planner=cognition_planner,
        )
    if engine == "anaconda":
        return _advance_anaconda_match(
            match=match,
            scenario=scenario,
            current_step=current_step,
            cognition_planner=cognition_planner,
        )
    if engine == "blackjack":
        return _advance_blackjack_match(
            match=match,
            scenario=scenario,
            current_step=current_step,
            cognition_planner=cognition_planner,
        )
    return _advance_holdem_match(
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
        state = match.get("state_json") or {}
        engine = _scenario_engine_key(scenario, match)
        out.append(
            {
                "match_id": match["match_id"],
                "scenario_key": match.get("scenario_key"),
                "scenario_name": _scenario_display_name(scenario or {}),
                "scenario_category": str((scenario or {}).get("category") or ""),
                "scenario_kind": engine,
                "ui_group": _scenario_ui_group(scenario),
                "status": match.get("status"),
                "phase": match.get("phase"),
                "round_number": int(match.get("round_number") or 0),
                "participant_count": len(participants),
                "town_id": match.get("town_id"),
                "arena_id": match.get("arena_id"),
                "created_step": int(match.get("created_step") or 0),
                "buy_in": int((scenario or {}).get("buy_in") or 0),
                "pot": int(state.get("pot") or 0),
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
    scenario_kind = _scenario_spectator_kind(scenario, match)
    resolved = str(match.get("status") or "").strip().lower() == "resolved"
    dealer_cards = [str(card).upper() for card in (state.get("dealer_cards") or []) if str(card)]
    dealer_cards_public = [str(card).upper() for card in (state.get("dealer_cards_public") or []) if str(card)]
    if resolved:
        dealer_public_view = dealer_cards
    elif str(match.get("phase") or "").strip().lower() in {"dealer_turn", "settle"} and dealer_cards:
        dealer_public_view = dealer_cards
    else:
        dealer_public_view = dealer_cards_public
    holdem_hole_cards = _holdem_visible_hole_cards(state, reveal=resolved or str(match.get("phase") or "").strip().lower() == "showdown")

    public_state = {
        "match_id": match.get("match_id"),
        "scenario_key": match.get("scenario_key"),
        "scenario_name": _scenario_display_name(scenario),
        "scenario_kind": scenario_kind,
        "scenario_category": str(scenario.get("category") or ""),
        "ui_group": _scenario_ui_group(scenario),
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
        "showdown": dict(state.get("showdown") or {}) if resolved else {},
        "result": match.get("result_json") if resolved else {},
        "blackjack": {
            "current_hand_index": int(state.get("current_hand_index") or 1),
            "max_hands": int(state.get("max_hands") or 0),
            "dealer_cards": dealer_public_view,
            "dealer_upcard": dealer_public_view[0] if dealer_public_view else None,
            "player_hands": {
                str(aid): [str(card).upper() for card in (cards or [])]
                for aid, cards in (state.get("player_hands") or {}).items()
                if str(aid)
            },
            "player_bets": {
                str(aid): int(value)
                for aid, value in (state.get("player_bets") or {}).items()
                if str(aid)
            },
            "hand_results": list(state.get("hand_results") or [])[-8:],
        }
        if scenario_kind == "blackjack"
        else {},
        "holdem": {
            "board_cards": [str(card).upper() for card in (state.get("board_cards") or []) if str(card)],
            "hole_cards": holdem_hole_cards,
            "button_agent_id": str(state.get("button_agent_id") or ""),
            "small_blind_agent_id": str(state.get("small_blind_agent_id") or ""),
            "big_blind_agent_id": str(state.get("big_blind_agent_id") or ""),
            "street_raise_count": int(state.get("street_raise_count") or 0),
            "max_raises_per_street": int(state.get("max_raises_per_street") or 0),
            "current_bet": int(state.get("current_bet") or 0),
            "pot_contributions": {
                str(aid): int(value)
                for aid, value in (state.get("contributions") or {}).items()
                if str(aid)
            },
        }
        if scenario_kind == "holdem"
        else {},
    }
    return {
        "match_id": match_id,
        "public_state": public_state,
        "recent_events": events[-80:],
    }
