from __future__ import annotations

import asyncio
import os
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx2
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from pydantic import BaseModel, Field

from skill_harness.config import MCPServerConfig


class MCPConnectionError(RuntimeError):
    pass


class MCPToolInfo(BaseModel):
    server: str
    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)


_ENV_REFERENCE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


class MCPManager:
    def __init__(self, servers: dict[str, MCPServerConfig]) -> None:
        self.servers = servers

    def configured_servers(self) -> list[str]:
        return sorted(name for name, config in self.servers.items() if config.enabled)

    def _config(self, server_name: str) -> MCPServerConfig:
        try:
            config = self.servers[server_name]
        except KeyError as exc:
            raise MCPConnectionError(f"unknown MCP server: {server_name}") from exc
        if not config.enabled:
            raise MCPConnectionError(f"MCP server is disabled: {server_name}")
        return config

    @staticmethod
    def _expand_mapping(values: dict[str, str]) -> dict[str, str]:
        expanded: dict[str, str] = {}
        for key, value in values.items():
            match = _ENV_REFERENCE.match(value)
            if match:
                variable = match.group(1)
                if variable not in os.environ:
                    raise MCPConnectionError(
                        f"environment variable '{variable}' required by MCP configuration is missing"
                    )
                expanded[key] = os.environ[variable]
            else:
                expanded[key] = value
        return expanded

    @staticmethod
    def _stdio_environment(config: MCPServerConfig) -> dict[str, str]:
        inherited_names = {
            "PATH",
            "PATHEXT",
            "SYSTEMROOT",
            "WINDIR",
            "TEMP",
            "TMP",
        } | {name.upper() for name in config.inherit_env}
        environment = {
            name: value for name, value in os.environ.items() if name.upper() in inherited_names
        }
        environment.update(MCPManager._expand_mapping(config.env))
        return environment

    @asynccontextmanager
    async def session(self, server_name: str) -> AsyncIterator[ClientSession]:
        config = self._config(server_name)
        try:
            if config.transport == "stdio":
                parameters = StdioServerParameters(
                    command=config.command or "",
                    args=config.args,
                    env=self._stdio_environment(config),
                )
                async with stdio_client(parameters) as (read_stream, write_stream):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        yield session
            else:
                headers = self._expand_mapping(config.headers)
                async with httpx2.AsyncClient(headers=headers) as http_client:
                    async with streamable_http_client(
                        config.url or "", http_client=http_client
                    ) as (read_stream, write_stream, _):
                        async with ClientSession(read_stream, write_stream) as session:
                            await session.initialize()
                            yield session
        except MCPConnectionError:
            raise
        except Exception as exc:
            raise MCPConnectionError(
                f"failed to use MCP server '{server_name}': {type(exc).__name__}: {exc}"
            ) from exc

    async def ping(self, server_name: str) -> None:
        timeout = self._config(server_name).timeout_seconds
        async with asyncio.timeout(timeout):
            async with self.session(server_name) as session:
                await session.send_ping()

    async def list_tools(self, server_name: str) -> list[MCPToolInfo]:
        timeout = self._config(server_name).timeout_seconds
        async with asyncio.timeout(timeout):
            async with self.session(server_name) as session:
                result = await session.list_tools()
        return [
            MCPToolInfo(
                server=server_name,
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.input_schema,
            )
            for tool in result.tools
        ]

    async def call_tool(
        self, server_name: str, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        timeout = self._config(server_name).timeout_seconds
        async with asyncio.timeout(timeout):
            async with self.session(server_name) as session:
                result = await session.call_tool(tool_name, arguments)
        if hasattr(result, "model_dump"):
            return result.model_dump(mode="json", exclude_none=True)
        return {"content": str(result)}
