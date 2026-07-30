from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Any

import typer

from skill_harness.config import load_config
from skill_harness.models import DisclosureLevel, RunContext
from skill_harness.runtime import Harness, build_harness

app = typer.Typer(
    name="skill-harness",
    help="渐进式披露 Skill、调用工具与 MCP，并持久化记忆。",
    no_args_is_help=True,
)
skills_app = typer.Typer(help="发现、检查和执行 Skill。", no_args_is_help=True)
mcp_app = typer.Typer(help="检查和调用 MCP Server。", no_args_is_help=True)
memory_app = typer.Typer(help="查看和管理持久记忆。", no_args_is_help=True)
trace_app = typer.Typer(help="查看 Agent 运行轨迹。", no_args_is_help=True)
tools_app = typer.Typer(help="查看和调用统一工具。", no_args_is_help=True)
app.add_typer(skills_app, name="skills")
app.add_typer(mcp_app, name="mcp")
app.add_typer(memory_app, name="memory")
app.add_typer(trace_app, name="trace")
app.add_typer(tools_app, name="tools")


def _harness(config: Path) -> Harness:
    return build_harness(load_config(config))


def _json(value: Any) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    typer.echo(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _parse_object(value: str) -> dict[str, Any]:
    if value == "-":
        value = typer.get_text_stream("stdin").read()
    elif value.startswith("@"):
        input_path = Path(value[1:]).resolve()
        if not input_path.is_file():
            raise typer.BadParameter(f"输入文件不存在: {input_path}")
        value = input_path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"无效 JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise typer.BadParameter("输入必须是 JSON 对象")
    return parsed


@app.command()
def run(
    prompt: str = typer.Argument(..., help="发送给 Agent 的任务。"),
    config: Path = typer.Option(Path("harness.toml"), "--config", "-c"),
    no_mcp: bool = typer.Option(False, "--no-mcp", help="本次运行不连接 MCP。"),
) -> None:
    """运行一次完整 Agent 循环。"""

    async def _run() -> None:
        harness = _harness(config)
        if not no_mcp:
            failures = await harness.load_mcp_tools()
            for server, error in failures.items():
                typer.echo(f"[MCP:{server}] {error}", err=True)
        if not os.getenv("OPENAI_API_KEY"):
            raise typer.BadParameter("缺少 OPENAI_API_KEY；Skill、Memory、MCP 子命令仍可离线使用")
        result = await harness.agent().run(prompt)
        typer.echo(result.output)
        typer.echo(f"\nrun_id={result.run_id}", err=True)

    asyncio.run(_run())


@app.command()
def doctor(
    config: Path = typer.Option(Path("harness.toml"), "--config", "-c"),
    check_mcp: bool = typer.Option(False, "--check-mcp"),
) -> None:
    """验证配置、Skill、数据库和可选 MCP 连接。"""

    async def _doctor() -> None:
        harness = _harness(config)
        report: dict[str, Any] = {
            "workspace": str(harness.config.workspace),
            "database": str(harness.config.memory.database),
            "skills": [card.model_dump(mode="json") for card in harness.skills.cards()],
            "tools": [spec.name for spec in harness.tools.specs()],
            "mcp": {},
        }
        if check_mcp:
            for server in harness.mcp.configured_servers():
                try:
                    await harness.mcp.ping(server)
                    report["mcp"][server] = "ok"
                except Exception as exc:
                    report["mcp"][server] = str(exc)
        _json(report)

    asyncio.run(_doctor())


@skills_app.command("list")
def skills_list(
    query: str = typer.Option("", "--query", "-q"),
    config: Path = typer.Option(Path("harness.toml"), "--config", "-c"),
) -> None:
    harness = _harness(config)
    _json([card.model_dump(mode="json") for card in harness.skills.search(query, 100)])


@skills_app.command("inspect")
def skills_inspect(
    skill_id: str,
    level: DisclosureLevel = typer.Option(DisclosureLevel.OVERVIEW, "--level", "-l"),
    resource: str | None = typer.Option(None, "--resource", "-r"),
    config: Path = typer.Option(Path("harness.toml"), "--config", "-c"),
) -> None:
    harness = _harness(config)
    run_id = f"cli:{uuid.uuid4()}"
    result: Any = harness.disclosure.open(run_id, skill_id, DisclosureLevel.OVERVIEW)
    if level in {DisclosureLevel.INSTRUCTIONS, DisclosureLevel.RESOURCE}:
        result = harness.disclosure.open(run_id, skill_id, DisclosureLevel.INSTRUCTIONS)
    if level == DisclosureLevel.RESOURCE:
        if not resource:
            raise typer.BadParameter("--resource is required for resource level")
        result = harness.disclosure.open(
            run_id,
            skill_id,
            DisclosureLevel.RESOURCE,
            resource_path=resource,
        )
    _json(result)


@skills_app.command("execute")
def skills_execute(
    skill_id: str,
    input_json: str = typer.Option(
        "{}",
        "--input",
        "-i",
        help="JSON 对象；使用 @path 从 UTF-8 文件读取，或使用 - 从 stdin 读取。",
    ),
    config: Path = typer.Option(Path("harness.toml"), "--config", "-c"),
) -> None:
    async def _execute() -> None:
        harness = _harness(config)
        run_id = f"cli:{uuid.uuid4()}"
        result = await harness.tools.call(
            "skill_execute",
            {"skill_id": skill_id, "input": _parse_object(input_json)},
            RunContext(run_id=run_id, workspace=harness.config.workspace),
        )
        _json(result)
        if not result.ok:
            raise typer.Exit(1)

    asyncio.run(_execute())


@mcp_app.command("list")
def mcp_list(
    config: Path = typer.Option(Path("harness.toml"), "--config", "-c"),
) -> None:
    harness = _harness(config)
    result: dict[str, Any] = {}
    for name in harness.mcp.configured_servers():
        server = harness.config.mcp_servers[name]
        dumped = server.model_dump(mode="json", exclude={"env", "headers"})
        dumped["env_keys"] = sorted(server.env)
        dumped["header_keys"] = sorted(server.headers)
        result[name] = dumped
    _json(result)


@mcp_app.command("test")
def mcp_test(
    server: str,
    config: Path = typer.Option(Path("harness.toml"), "--config", "-c"),
) -> None:
    async def _test() -> None:
        harness = _harness(config)
        await harness.mcp.ping(server)
        typer.echo("ok")

    asyncio.run(_test())


@mcp_app.command("tools")
def mcp_tools(
    server: str,
    config: Path = typer.Option(Path("harness.toml"), "--config", "-c"),
) -> None:
    async def _tools() -> None:
        harness = _harness(config)
        infos = await harness.mcp.list_tools(server)
        _json([info.model_dump(mode="json") for info in infos])

    asyncio.run(_tools())


@mcp_app.command("call")
def mcp_call(
    server: str,
    tool: str,
    input_json: str = typer.Option(
        "{}",
        "--input",
        "-i",
        help="JSON 对象；使用 @path 从 UTF-8 文件读取，或使用 - 从 stdin 读取。",
    ),
    config: Path = typer.Option(Path("harness.toml"), "--config", "-c"),
) -> None:
    async def _call() -> None:
        harness = _harness(config)
        _json(await harness.mcp.call_tool(server, tool, _parse_object(input_json)))

    asyncio.run(_call())


@memory_app.command("list")
def memory_list(
    limit: int = typer.Option(50, "--limit", min=1, max=1000),
    config: Path = typer.Option(Path("harness.toml"), "--config", "-c"),
) -> None:
    harness = _harness(config)
    _json([record.model_dump(mode="json") for record in harness.memory.list(limit)])


@memory_app.command("search")
def memory_search(
    query: str,
    limit: int = typer.Option(6, "--limit", min=1, max=100),
    config: Path = typer.Option(Path("harness.toml"), "--config", "-c"),
) -> None:
    harness = _harness(config)
    _json([record.model_dump(mode="json") for record in harness.memory.search(query, limit=limit)])


@memory_app.command("remember")
def memory_remember(
    content: str,
    kind: str = typer.Option("semantic", "--kind"),
    scope: str = typer.Option("global", "--scope"),
    config: Path = typer.Option(Path("harness.toml"), "--config", "-c"),
) -> None:
    harness = _harness(config)
    if kind not in {"working", "episodic", "semantic", "procedural"}:
        raise typer.BadParameter("kind 必须是 working/episodic/semantic/procedural")
    record = harness.memory.remember(content, kind=kind, scope=scope, source="cli")  # type: ignore[arg-type]
    _json(record)


@memory_app.command("forget")
def memory_forget(
    memory_id: str,
    config: Path = typer.Option(Path("harness.toml"), "--config", "-c"),
) -> None:
    harness = _harness(config)
    if not harness.memory.forget(memory_id):
        raise typer.BadParameter(f"没有找到记忆: {memory_id}")
    typer.echo("forgotten")


@memory_app.command("clear")
def memory_clear(
    scope: str | None = typer.Option(None, "--scope"),
    yes: bool = typer.Option(False, "--yes", help="确认删除。"),
    config: Path = typer.Option(Path("harness.toml"), "--config", "-c"),
) -> None:
    if not yes:
        raise typer.BadParameter("该操作会删除记忆；请传入 --yes")
    harness = _harness(config)
    typer.echo(f"deleted={harness.memory.clear(scope)}")


@trace_app.command("list")
def trace_list(
    limit: int = typer.Option(20, "--limit", min=1, max=1000),
    config: Path = typer.Option(Path("harness.toml"), "--config", "-c"),
) -> None:
    harness = _harness(config)
    _json([trace.model_dump(mode="json") for trace in harness.traces.list(limit)])


@trace_app.command("show")
def trace_show(
    run_id: str,
    config: Path = typer.Option(Path("harness.toml"), "--config", "-c"),
) -> None:
    harness = _harness(config)
    trace = harness.traces.get(run_id)
    if trace is None:
        raise typer.BadParameter(f"没有找到运行记录: {run_id}")
    _json(trace)


@tools_app.command("list")
def tools_list(
    include_mcp: bool = typer.Option(False, "--include-mcp"),
    config: Path = typer.Option(Path("harness.toml"), "--config", "-c"),
) -> None:
    async def _list() -> None:
        harness = _harness(config)
        if include_mcp:
            await harness.load_mcp_tools()
        _json([spec.model_dump(mode="json") for spec in harness.tools.specs()])

    asyncio.run(_list())


@tools_app.command("call")
def tools_call(
    name: str,
    input_json: str = typer.Option(
        "{}",
        "--input",
        "-i",
        help="JSON 对象；使用 @path 从 UTF-8 文件读取，或使用 - 从 stdin 读取。",
    ),
    include_mcp: bool = typer.Option(False, "--include-mcp"),
    config: Path = typer.Option(Path("harness.toml"), "--config", "-c"),
) -> None:
    async def _call() -> None:
        harness = _harness(config)
        if include_mcp:
            await harness.load_mcp_tools()
        result = await harness.tools.call(
            name,
            _parse_object(input_json),
            RunContext(
                run_id=f"cli:{uuid.uuid4()}",
                workspace=harness.config.workspace,
            ),
        )
        _json(result)
        if not result.ok:
            raise typer.Exit(1)

    asyncio.run(_call())
