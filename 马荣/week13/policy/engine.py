from __future__ import annotations

from skill_harness.config import PolicyConfig
from skill_harness.models import RiskLevel


class PolicyDenied(PermissionError):
    pass


class PolicyEngine:
    def __init__(self, config: PolicyConfig) -> None:
        self.config = config

    def check(self, risk: RiskLevel, action: str) -> None:
        allowed = {
            RiskLevel.READ: self.config.allow_read,
            RiskLevel.WRITE: self.config.allow_write,
            RiskLevel.DESTRUCTIVE: self.config.allow_destructive,
        }[risk]
        if not allowed:
            raise PolicyDenied(
                f"policy blocks {risk.value} action '{action}'; enable it in harness.toml"
            )
