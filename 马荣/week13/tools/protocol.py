from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from skill_harness.models import RiskLevel, RunContext, ToolResult, ToolSpec


class EmptyArguments(BaseModel):
    pass


class BaseTool(ABC):
    name: str
    description: str
    risk: RiskLevel = RiskLevel.READ
    source: str = "builtin"
    arguments_model: type[BaseModel] = EmptyArguments

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            input_schema=self.arguments_model.model_json_schema(),
            risk=self.risk,
            source=self.source,
        )

    def risk_for(self, arguments: dict[str, Any]) -> RiskLevel:
        return self.risk

    def validate(self, arguments: dict[str, Any]) -> BaseModel:
        return self.arguments_model.model_validate(arguments)

    @abstractmethod
    async def execute(self, arguments: BaseModel, context: RunContext) -> ToolResult:
        raise NotImplementedError
