from __future__ import annotations

from dataclasses import dataclass

from skill_harness.agent import AgentRuntime, ModelAdapter, OpenAIResponsesModel
from skill_harness.config import HarnessConfig
from skill_harness.mcp import MCPManager, MCPProxyTool
from skill_harness.mcp.tools import proxy_tool_name
from skill_harness.memory import MemoryStore
from skill_harness.models import RiskLevel
from skill_harness.policy import PolicyEngine
from skill_harness.skills import DisclosureController, SkillExecutor, SkillRegistry
from skill_harness.tools import ToolRegistry
from skill_harness.tools.builtin import builtin_tools
from skill_harness.tracing import TraceStore


@dataclass
class Harness:
    config: HarnessConfig
    skills: SkillRegistry
    disclosure: DisclosureController
    executor: SkillExecutor
    memory: MemoryStore
    traces: TraceStore
    mcp: MCPManager
    tools: ToolRegistry

    async def load_mcp_tools(self) -> dict[str, str]:
        failures: dict[str, str] = {}
        for server_name in self.mcp.configured_servers():
            try:
                infos = await self.mcp.list_tools(server_name)
                risk = RiskLevel(self.config.mcp_servers[server_name].default_risk)
                for info in infos:
                    if not self.tools.contains(proxy_tool_name(info.server, info.name)):
                        self.tools.register(MCPProxyTool(self.mcp, info, risk))
            except Exception as exc:
                failures[server_name] = str(exc)
        return failures

    def agent(self, model: ModelAdapter | None = None) -> AgentRuntime:
        selected_model = model or OpenAIResponsesModel(self.config.agent.model)
        return AgentRuntime(
            config=self.config,
            model=selected_model,
            skills=self.skills,
            tools=self.tools,
            memory=self.memory,
            traces=self.traces,
        )


def build_harness(config: HarnessConfig) -> Harness:
    skills = SkillRegistry(config.skills.roots)
    skills.scan()
    disclosure = DisclosureController(skills, config.skills.resource_max_bytes)
    executor = SkillExecutor(
        skills,
        default_timeout_seconds=config.policy.tool_timeout_seconds,
        max_output_bytes=config.policy.max_tool_output_bytes,
    )
    memory = MemoryStore(config.memory.database)
    traces = TraceStore(config.memory.database)
    policy = PolicyEngine(config.policy)
    tools = ToolRegistry(
        policy=policy,
        timeout_seconds=config.policy.tool_timeout_seconds,
        max_output_bytes=config.policy.max_tool_output_bytes,
    )
    for tool in builtin_tools(skills, disclosure, executor, memory):
        tools.register(tool)
    return Harness(
        config=config,
        skills=skills,
        disclosure=disclosure,
        executor=executor,
        memory=memory,
        traces=traces,
        mcp=MCPManager(config.mcp_servers),
        tools=tools,
    )
