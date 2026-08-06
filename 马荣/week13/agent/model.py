from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from skill_harness.models import ToolCall, ToolSpec


class ModelRequest(BaseModel):
    instructions: str
    input_items: list[dict[str, Any]]
    tools: list[ToolSpec] = Field(default_factory=list)


class ModelTurn(BaseModel):
    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    output_items: list[dict[str, Any]] = Field(default_factory=list)
    response_id: str | None = None


class ModelAdapter(ABC):
    @abstractmethod
    async def respond(self, request: ModelRequest) -> ModelTurn:
        raise NotImplementedError


class OpenAIResponsesModel(ModelAdapter):
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model
        options: dict[str, Any] = {}
        if api_key:
            options["api_key"] = api_key
        if base_url:
            options["base_url"] = base_url
        self.client = AsyncOpenAI(**options)

    async def respond(self, request: ModelRequest) -> ModelTurn:
        tools = [
            {
                "type": "function",
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.input_schema,
                "strict": False,
            }
            for spec in request.tools
        ]
        response = await self.client.responses.create(
            model=self.model,
            instructions=request.instructions,
            input=request.input_items,
            tools=tools,
        )

        output_items: list[dict[str, Any]] = []
        tool_calls: list[ToolCall] = []
        for item in response.output:
            dumped = item.model_dump(mode="json", exclude_none=True)
            output_items.append(dumped)
            if item.type != "function_call":
                continue
            try:
                arguments = json.loads(item.arguments)
            except json.JSONDecodeError:
                arguments = {"_invalid_json": item.arguments}
            tool_calls.append(
                ToolCall(
                    call_id=item.call_id,
                    name=item.name,
                    arguments=arguments,
                )
            )

        return ModelTurn(
            text=response.output_text or "",
            tool_calls=tool_calls,
            output_items=output_items,
            response_id=response.id,
        )
