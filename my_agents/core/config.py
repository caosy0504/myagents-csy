"""Central configuration shared by all Agent implementations."""

import os
from typing import Any, Literal

from pydantic import BaseModel, Field


def _optional_int(name: str) -> int | None:
    value = os.getenv(name, "").strip()
    return int(value) if value else None


class Config(BaseModel):
    """Common runtime policy for Agents."""

    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    max_history_length: int = Field(default=20, ge=0)
    context_max_tokens: int = Field(default=4096, gt=0)
    history_summary_enabled: bool = True
    history_keep_recent_messages: int = Field(default=8, ge=2)
    history_summary_max_tokens: int = Field(default=400, gt=0)
    tool_call_mode: Literal["auto", "native", "text"] = "auto"
    max_steps: int = Field(default=5, ge=1)
    reflection_iterations: int = Field(default=1, ge=0)
    debug: bool = False

    @classmethod
    def from_env(cls) -> "Config":
        """Build configuration from environment variables."""
        return cls(
            temperature=float(os.getenv("AGENT_TEMPERATURE", "0.3")),
            max_tokens=_optional_int("AGENT_MAX_TOKENS"),
            max_history_length=int(os.getenv("AGENT_MAX_HISTORY", "20")),
            context_max_tokens=int(os.getenv("AGENT_CONTEXT_MAX_TOKENS", "4096")),
            history_summary_enabled=os.getenv("AGENT_HISTORY_SUMMARY", "true").lower() == "true",
            history_keep_recent_messages=int(os.getenv("AGENT_HISTORY_KEEP_RECENT", "8")),
            tool_call_mode=os.getenv("AGENT_TOOL_CALL_MODE", "auto"),
            max_steps=int(os.getenv("AGENT_MAX_STEPS", "5")),
            reflection_iterations=int(os.getenv("AGENT_REFLECTION_ITERATIONS", "1")),
            debug=os.getenv("AGENT_DEBUG", "false").lower() == "true",
        )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
