"""Scenario runtime manager abstractions used by API and scheduler layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


MatchListFn = Callable[[str], list[dict[str, Any]]]
MatchStepFn = Callable[[str, int], list[dict[str, Any]]]
StatusMapFn = Callable[[str], dict[str, str]]


@dataclass(frozen=True)
class ScenarioRuntimeState:
    """Runtime snapshot for one town at a specific simulation step."""

    town_id: str
    step: int
    formed_matches: list[dict[str, Any]]
    advanced_matches: list[dict[str, Any]]
    active_matches: list[dict[str, Any]]
    scenario_agent_statuses: dict[str, str]

    @property
    def formed_count(self) -> int:
        return len(self.formed_matches)

    @property
    def advanced_count(self) -> int:
        return len(self.advanced_matches)


class TownScenarioManager:
    """Orchestrates scenario lifecycle hooks around town simulation ticks."""

    def __init__(
        self,
        *,
        form_matches_fn: MatchStepFn,
        advance_matches_fn: MatchStepFn,
        list_active_matches_fn: MatchListFn,
        scenario_statuses_fn: StatusMapFn,
    ) -> None:
        self._form_matches = form_matches_fn
        self._advance_matches = advance_matches_fn
        self._list_active_matches = list_active_matches_fn
        self._scenario_statuses = scenario_statuses_fn

    def statuses_for_town(self, *, town_id: str) -> dict[str, str]:
        return dict(self._scenario_statuses(str(town_id)))

    def active_for_town(self, *, town_id: str) -> list[dict[str, Any]]:
        return list(self._list_active_matches(str(town_id)))

    def snapshot(self, *, town_id: str, current_step: int) -> ScenarioRuntimeState:
        return ScenarioRuntimeState(
            town_id=str(town_id),
            step=int(current_step),
            formed_matches=[],
            advanced_matches=[],
            active_matches=self.active_for_town(town_id=str(town_id)),
            scenario_agent_statuses=self.statuses_for_town(town_id=str(town_id)),
        )

    def tick(self, *, town_id: str, current_step: int) -> ScenarioRuntimeState:
        town = str(town_id)
        step = int(current_step)
        formed = list(self._form_matches(town, step))
        advanced = list(self._advance_matches(town, step))
        return ScenarioRuntimeState(
            town_id=town,
            step=step,
            formed_matches=formed,
            advanced_matches=advanced,
            active_matches=self.active_for_town(town_id=town),
            scenario_agent_statuses=self.statuses_for_town(town_id=town),
        )
