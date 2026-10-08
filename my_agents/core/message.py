"""Unified message contract used by every Agent."""

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


MessageRole = Literal["system", "user", "assistant", "tool"]


class Message(BaseModel):
    """One framework message with API-compatible role and content fields."""

    content: str
    role: MessageRole
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_api_dict(self) -> dict[str, str]:
        """Return the subset accepted by OpenAI-compatible chat APIs."""
        return {"role": self.role, "content": self.content}

    def __str__(self) -> str:
        return f"[{self.role}] {self.content}"
