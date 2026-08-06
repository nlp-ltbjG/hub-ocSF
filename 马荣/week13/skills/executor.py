from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from skill_harness.models import ToolResult

from .loader import safe_resource_path
from .registry import SkillRegistry


class SkillExecutionError(RuntimeError):
    pass


class SkillExecutor:
    def __init__(
        self,
        registry: SkillRegistry,
        default_timeout_seconds: float,
        max_output_bytes: int,
    ) -> None:
        self.registry = registry
        self.default_timeout_seconds = default_timeout_seconds
        self.max_output_bytes = max_output_bytes

    async def execute(self, skill_id: str, arguments: dict[str, Any]) -> ToolResult:
        skill = self.registry.get(skill_id)
        execution = skill.manifest.execution
        if execution is None or not execution.command:
            raise SkillExecutionError(f"skill '{skill_id}' has no executable entrypoint")

        command = self._resolve_command(skill.root, execution.command)
        env = self._build_environment(execution.allowed_env)
        timeout = execution.timeout_seconds or self.default_timeout_seconds
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=skill.root,
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        payload = json.dumps(arguments, ensure_ascii=False).encode("utf-8")
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(payload), timeout)
        except TimeoutError:
            process.kill()
            await process.wait()
            raise SkillExecutionError(
                f"skill '{skill_id}' exceeded its {timeout:g}-second timeout"
            ) from None

        if len(stdout) + len(stderr) > self.max_output_bytes:
            raise SkillExecutionError(
                f"skill output exceeded the {self.max_output_bytes}-byte limit"
            )
        stdout_text = stdout.decode("utf-8", errors="replace").strip()
        stderr_text = stderr.decode("utf-8", errors="replace").strip()
        if process.returncode != 0:
            return ToolResult(
                ok=False,
                error=stderr_text or f"skill exited with code {process.returncode}",
                metadata={"exit_code": process.returncode},
            )

        try:
            content: Any = json.loads(stdout_text) if stdout_text else None
        except json.JSONDecodeError:
            content = stdout_text
        return ToolResult(
            ok=True,
            content=content,
            metadata={"exit_code": process.returncode, "stderr": stderr_text or None},
        )

    @staticmethod
    def _build_environment(allowed_env: list[str]) -> dict[str, str]:
        base_names = {
            "PATH",
            "PATHEXT",
            "SYSTEMROOT",
            "WINDIR",
            "TEMP",
            "TMP",
            "LANG",
            "LC_ALL",
        }
        selected = base_names | {name.upper() for name in allowed_env}
        env = {name: value for name, value in os.environ.items() if name.upper() in selected}
        env["PYTHONIOENCODING"] = "utf-8"
        return env

    @staticmethod
    def _resolve_command(skill_root: Path, command: list[str]) -> list[str]:
        resolved = [sys.executable if part == "{python}" else part for part in command]
        executable = Path(command[0])
        if command[0] != "{python}" and (
            executable.is_absolute() or executable.parent != Path(".")
        ):
            raise SkillExecutionError(
                "the executable must be {python} or a command name resolved from PATH"
            )
        if command[0] == "{python}":
            if len(command) < 2:
                raise SkillExecutionError("{python} entrypoints require a script path")
            if Path(command[1]).is_absolute():
                raise SkillExecutionError("skill script paths must be relative")
            script = safe_resource_path(
                type(
                    "_SkillRoot",
                    (),
                    {"root": skill_root, "instructions_path": skill_root / "SKILL.md"},
                )(),
                command[1],
            )
            resolved[1] = str(script)
        return resolved
