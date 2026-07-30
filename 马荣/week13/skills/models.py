from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from skill_harness.models import DisclosureLevel, RiskLevel


class SkillExecution(BaseModel):
    command: list[str] = Field(default_factory=list)
    timeout_seconds: float | None = Field(default=None, gt=0, le=3600)
    allowed_env: list[str] = Field(default_factory=list)

    @field_validator("command")
    @classmethod
    def command_must_not_be_empty_strings(cls, value: list[str]) -> list[str]:
        if any(not part.strip() for part in value):
            raise ValueError("execution command cannot contain empty arguments")
        return value


class SkillManifest(BaseModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_-]{1,63}$")
    name: str
    version: str = "0.1.0"
    description: str
    triggers: list[str] = Field(default_factory=list)
    risk: RiskLevel = RiskLevel.READ
    execution: SkillExecution | None = None


class InstalledSkill(BaseModel):
    manifest: SkillManifest
    root: Path
    instructions_path: Path

    model_config = {"arbitrary_types_allowed": True}


class SkillCard(BaseModel):
    id: str
    name: str
    description: str
    triggers: list[str]
    risk: RiskLevel
    disclosure: DisclosureLevel = DisclosureLevel.CATALOG


class SkillOverview(BaseModel):
    card: SkillCard
    version: str
    executable: bool
    resources: list[str]
    disclosure: DisclosureLevel = DisclosureLevel.OVERVIEW


class DisclosedSkill(BaseModel):
    overview: SkillOverview
    instructions: str
    disclosure: DisclosureLevel = DisclosureLevel.INSTRUCTIONS


class SkillResource(BaseModel):
    skill_id: str
    path: str
    content: str
    size_bytes: int
    disclosure: DisclosureLevel = DisclosureLevel.RESOURCE
