from __future__ import annotations

import asyncio
from typing import Any

from pydantic import ValidationError

from skill_harness.models import RunContext, ToolResult, ToolSpec
from skill_harness.policy import PolicyDenied, PolicyEngine

from .protocol import BaseTool


class ToolRegistryError(RuntimeError):
    pass


class ToolRegistry:
    def __init__(
        self,
        policy: PolicyEngine,
        timeout_seconds: float,
        max_output_bytes: int,
    ) -> None:
        self.policy = policy
        self.timeout_seconds = timeout_seconds
        self.max_output_bytes = max_output_bytes
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            raise ToolRegistryError(f"duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolRegistryError(f"unknown tool: {name}") from exc

    def contains(self, name: str) -> bool:
        return name in self._tools

    def specs(self) -> list[ToolSpec]:
        return [self._tools[name].spec for name in sorted(self._tools)]

    async def call(
        self,
        name: str,
        arguments: dict[str, Any],
        context: RunContext,
    ) -> ToolResult:
        try:
            tool = self.get(name)
            self.policy.check(tool.risk_for(arguments), name)
            validated = tool.validate(arguments)
            result = await asyncio.wait_for(
                tool.execute(validated, context), timeout=self.timeout_seconds
            )
        except (ToolRegistryError, PolicyDenied, ValidationError, TimeoutError) as exc:
            return ToolResult(ok=False, error=str(exc))
        except Exception as exc:
            return ToolResult(ok=False, error=f"{type(exc).__name__}: {exc}")

        encoded = result.as_model_output().encode("utf-8")
        if len(encoded) > self.max_output_bytes:
            return ToolResult(
                ok=False,
                error=f"tool output exceeded the {self.max_output_bytes}-byte limit",
            )
        return result
