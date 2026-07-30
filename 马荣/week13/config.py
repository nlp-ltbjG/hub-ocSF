from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class AgentConfig(BaseModel):
    provider: Literal["openai"] = "openai"
    model: str = "gpt-5.6"
    max_steps: int = Field(default=12, ge=1, le=100)
    max_tool_calls: int = Field(default=24, ge=0, le=500)


class SkillsConfig(BaseModel):
    roots: list[Path] = Field(default_factory=lambda: [Path("skills")])
    catalog_limit: int = Field(default=8, ge=1, le=100)
    resource_max_bytes: int = Field(default=256 * 1024, ge=1024)


class MemoryConfig(BaseModel):
    database: Path = Path(".harness/harness.db")
    retrieval_limit: int = Field(default=6, ge=0, le=100)
    auto_record_runs: bool = True


class PolicyConfig(BaseModel):
    allow_read: bool = True
    allow_write: bool = False
    allow_destructive: bool = False
    tool_timeout_seconds: float = Field(default=30.0, gt=0, le=3600)
    max_tool_output_bytes: int = Field(default=256 * 1024, ge=1024)


class MCPServerConfig(BaseModel):
    transport: Literal["stdio", "streamable_http"]
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    url: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)
    inherit_env: list[str] = Field(default_factory=list)
    default_risk: Literal["read", "write", "destructive"] = "write"
    enabled: bool = True
    timeout_seconds: float = Field(default=30.0, gt=0, le=3600)

    @model_validator(mode="after")
    def validate_transport_fields(self) -> MCPServerConfig:
        if self.transport == "stdio" and not self.command:
            raise ValueError("stdio MCP servers require 'command'")
        if self.transport == "streamable_http" and not self.url:
            raise ValueError("streamable_http MCP servers require 'url'")
        return self


class HarnessConfig(BaseModel):
    workspace: Path
    agent: AgentConfig = Field(default_factory=AgentConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)

    def resolve_paths(self) -> HarnessConfig:
        workspace = self.workspace.resolve()
        self.workspace = workspace
        self.skills.roots = [
            path.resolve() if path.is_absolute() else (workspace / path).resolve()
            for path in self.skills.roots
        ]
        database = self.memory.database
        self.memory.database = (
            database.resolve() if database.is_absolute() else (workspace / database).resolve()
        )
        return self


def load_config(path: Path | str = "harness.toml") -> HarnessConfig:
    config_path = Path(path).resolve()
    raw: dict[str, object] = {}
    if config_path.exists():
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
    raw["workspace"] = str(config_path.parent)

    env_model = os.getenv("SKILL_HARNESS_MODEL")
    if env_model:
        raw.setdefault("agent", {})
        assert isinstance(raw["agent"], dict)
        raw["agent"]["model"] = env_model

    return HarnessConfig.model_validate(raw).resolve_paths()
