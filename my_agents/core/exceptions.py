"""Framework-specific exceptions."""


class MyAgentsError(Exception):
    """Base exception for the framework."""


class ConfigurationError(MyAgentsError):
    """Raised when required configuration is missing or invalid."""


class LLMError(MyAgentsError):
    """Raised when an LLM request fails."""


class ToolError(MyAgentsError):
    """Raised when a tool cannot complete its operation."""
