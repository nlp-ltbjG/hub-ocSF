from __future__ import annotations

import re
from typing import Any

from jsonschema import Draft202012Validator
from pydantic import BaseModel, RootModel

from skill_harness.models import RiskLevel, RunContext, ToolResult, ToolSpec
from skill_harness.tools import BaseTool

from .manager import MCPManager, MCPToolInfo


class MCPArguments(RootModel[dict[str, Any]]):
    pass


def proxy_tool_name(server_name: str, tool_name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]", "_", f"mcp_{server_name}_{tool_name}")
    return normalized[:64]


class MCPProxyTool(BaseTool):
    arguments_model = MCPArguments

    def __init__(
        self,
        manager: MCPManager,
        info: MCPToolInfo,
        risk: RiskLevel,
    ) -> None:
        self.manager = manager
        self.info = info
        self.name = proxy_tool_name(info.server, info.name)
        self.description = (
            f"MCP tool '{info.name}' from server '{info.server}'. {info.description}"
        ).strip()
        self.risk = risk
        self.source = f"mcp:{info.server}"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            input_schema=self.info.input_schema,
            risk=self.risk,
            source=self.source,
        )

    def validate(self, arguments: dict[str, Any]) -> BaseModel:
        Draft202012Validator(self.info.input_schema).validate(arguments)
        return MCPArguments(arguments)

    async def execute(self, arguments: MCPArguments, context: RunContext) -> ToolResult:
        content = await self.manager.call_tool(self.info.server, self.info.name, arguments.root)
        return ToolResult(ok=True, content=content)
