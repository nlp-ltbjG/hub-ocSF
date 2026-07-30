from __future__ import annotations

import re
import tomllib
from pathlib import Path

from .models import InstalledSkill, SkillCard, SkillManifest


class SkillRegistryError(RuntimeError):
    pass


class SkillRegistry:
    def __init__(self, roots: list[Path]) -> None:
        self.roots = [root.resolve() for root in roots]
        self._skills: dict[str, InstalledSkill] = {}

    def scan(self) -> list[InstalledSkill]:
        discovered: dict[str, InstalledSkill] = {}
        for root in self.roots:
            if not root.exists():
                continue
            for manifest_path in sorted(root.glob("*/skill.toml")):
                with manifest_path.open("rb") as handle:
                    manifest = SkillManifest.model_validate(tomllib.load(handle))
                skill_root = manifest_path.parent.resolve()
                instructions_path = skill_root / "SKILL.md"
                if not instructions_path.is_file():
                    raise SkillRegistryError(f"{manifest_path} is missing SKILL.md")
                if manifest.id in discovered:
                    previous = discovered[manifest.id].root
                    raise SkillRegistryError(
                        f"duplicate skill id '{manifest.id}' in {previous} and {skill_root}"
                    )
                discovered[manifest.id] = InstalledSkill(
                    manifest=manifest,
                    root=skill_root,
                    instructions_path=instructions_path,
                )
        self._skills = discovered
        return list(discovered.values())

    def _ensure_scanned(self) -> None:
        if not self._skills:
            self.scan()

    def get(self, skill_id: str) -> InstalledSkill:
        self._ensure_scanned()
        try:
            return self._skills[skill_id]
        except KeyError as exc:
            raise SkillRegistryError(f"unknown skill: {skill_id}") from exc

    def cards(self) -> list[SkillCard]:
        self._ensure_scanned()
        return [
            SkillCard(
                id=skill.manifest.id,
                name=skill.manifest.name,
                description=skill.manifest.description,
                triggers=skill.manifest.triggers,
                risk=skill.manifest.risk,
            )
            for skill in sorted(self._skills.values(), key=lambda item: item.manifest.id)
        ]

    def search(self, query: str, limit: int = 8) -> list[SkillCard]:
        cards = self.cards()
        normalized = query.casefold().strip()
        if not normalized:
            return cards[:limit]

        query_tokens = set(re.findall(r"[\w-]+", normalized))

        def score(card: SkillCard) -> tuple[int, str]:
            haystack = " ".join([card.id, card.name, card.description, *card.triggers]).casefold()
            haystack_tokens = set(re.findall(r"[\w-]+", haystack))
            value = len(query_tokens & haystack_tokens) * 2
            if normalized in haystack:
                value += 6
            value += sum(4 for trigger in card.triggers if trigger.casefold() in normalized)
            return value, card.id

        ranked = sorted(cards, key=lambda card: (-score(card)[0], score(card)[1]))
        matching = [card for card in ranked if score(card)[0] > 0]
        return (matching or ranked)[:limit]
