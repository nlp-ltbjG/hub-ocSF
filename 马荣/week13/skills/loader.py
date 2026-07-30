from __future__ import annotations

from pathlib import Path

from .models import (
    DisclosedSkill,
    InstalledSkill,
    SkillCard,
    SkillOverview,
    SkillResource,
)


class SkillResourceError(RuntimeError):
    pass


def _card(skill: InstalledSkill) -> SkillCard:
    manifest = skill.manifest
    return SkillCard(
        id=manifest.id,
        name=manifest.name,
        description=manifest.description,
        triggers=manifest.triggers,
        risk=manifest.risk,
    )


def list_resources(skill: InstalledSkill) -> list[str]:
    resources: list[str] = []
    for directory_name in ("references", "scripts", "assets"):
        directory = skill.root / directory_name
        if not directory.is_dir():
            continue
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            resources.append(path.relative_to(skill.root).as_posix())
    return resources


def load_overview(skill: InstalledSkill) -> SkillOverview:
    return SkillOverview(
        card=_card(skill),
        version=skill.manifest.version,
        executable=bool(skill.manifest.execution and skill.manifest.execution.command),
        resources=list_resources(skill),
    )


def load_instructions(skill: InstalledSkill) -> DisclosedSkill:
    return DisclosedSkill(
        overview=load_overview(skill),
        instructions=skill.instructions_path.read_text(encoding="utf-8"),
    )


def safe_resource_path(skill: InstalledSkill, relative_path: str) -> Path:
    candidate = (skill.root / relative_path).resolve()
    try:
        candidate.relative_to(skill.root)
    except ValueError as exc:
        raise SkillResourceError("resource path escapes the skill directory") from exc
    if candidate == skill.instructions_path:
        raise SkillResourceError("use instruction disclosure to read SKILL.md")
    if not candidate.is_file():
        raise SkillResourceError(f"resource does not exist: {relative_path}")
    return candidate


def load_resource(skill: InstalledSkill, relative_path: str, max_bytes: int) -> SkillResource:
    path = safe_resource_path(skill, relative_path)
    raw = path.read_bytes()
    if len(raw) > max_bytes:
        raise SkillResourceError(
            f"resource is {len(raw)} bytes, exceeding the {max_bytes}-byte limit"
        )
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SkillResourceError("only UTF-8 text resources can be disclosed") from exc
    return SkillResource(
        skill_id=skill.manifest.id,
        path=path.relative_to(skill.root).as_posix(),
        content=content,
        size_bytes=len(raw),
    )
