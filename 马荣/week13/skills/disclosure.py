from __future__ import annotations

from collections import defaultdict
from typing import Any

from skill_harness.models import DisclosureLevel

from .loader import load_instructions, load_overview, load_resource
from .registry import SkillRegistry

_LEVEL_ORDER = {
    DisclosureLevel.CATALOG: 0,
    DisclosureLevel.OVERVIEW: 1,
    DisclosureLevel.INSTRUCTIONS: 2,
    DisclosureLevel.RESOURCE: 3,
}


class DisclosureError(RuntimeError):
    pass


class DisclosureController:
    """Enforces monotonic, just-in-time disclosure for every run and skill."""

    def __init__(self, registry: SkillRegistry, resource_max_bytes: int) -> None:
        self.registry = registry
        self.resource_max_bytes = resource_max_bytes
        self._levels: dict[str, dict[str, DisclosureLevel]] = defaultdict(dict)

    def current_level(self, run_id: str, skill_id: str) -> DisclosureLevel:
        return self._levels[run_id].get(skill_id, DisclosureLevel.CATALOG)

    def open(
        self,
        run_id: str,
        skill_id: str,
        level: DisclosureLevel,
        resource_path: str | None = None,
    ) -> Any:
        current = self.current_level(run_id, skill_id)
        requested_order = _LEVEL_ORDER[level]
        current_order = _LEVEL_ORDER[current]
        if requested_order > current_order + 1:
            raise DisclosureError(
                f"cannot jump from {current.value} to {level.value}; disclose the next level first"
            )

        skill = self.registry.get(skill_id)
        if level == DisclosureLevel.CATALOG:
            return next(card for card in self.registry.cards() if card.id == skill_id)
        if level == DisclosureLevel.OVERVIEW:
            result = load_overview(skill)
        elif level == DisclosureLevel.INSTRUCTIONS:
            result = load_instructions(skill)
        elif level == DisclosureLevel.RESOURCE:
            if current_order < _LEVEL_ORDER[DisclosureLevel.INSTRUCTIONS]:
                raise DisclosureError("instructions must be disclosed before resources")
            if not resource_path:
                raise DisclosureError("resource_path is required for resource disclosure")
            result = load_resource(skill, resource_path, self.resource_max_bytes)
        else:  # pragma: no cover - StrEnum validation prevents this
            raise DisclosureError(f"unsupported disclosure level: {level}")

        if requested_order > current_order:
            self._levels[run_id][skill_id] = level
        return result
