from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class RiskLevel(StrEnum):
    READ = "read"
    WRITE = "write"
    DESTRUCTIVE = "destructive"


class DisclosureLevel(StrEnum):
    CATALOG = "catalog"
    OVERVIEW = "overview"
    INSTRUCTIONS = "instructions"
    RESOURCE = "resource"


class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    risk: RiskLevel = RiskLevel.READ
    source: str = "builtin"


class ToolCall(BaseModel):
    call_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    ok: bool
    content: Any = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def as_model_output(self) -> str:
        return self.model_dump_json(exclude_none=True)


class RunContext(BaseModel):
    run_id: str
    workspace: Path
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}
