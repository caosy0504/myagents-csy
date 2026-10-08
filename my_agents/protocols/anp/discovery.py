"""Chapter 10's local ANP-style discovery exercise, not an ANP wire implementation."""

from dataclasses import dataclass, field
from time import monotonic
from urllib.parse import urlparse


@dataclass(frozen=True)
class AgentService:
    service_id: str
    name: str
    service_type: str
    capabilities: tuple[str, ...]
    endpoint: str
    load: float = 0.0
    cost: float = 0.0
    ttl_seconds: float = 300.0
    registered_at: float = field(default_factory=monotonic)

    def __post_init__(self) -> None:
        parsed = urlparse(self.endpoint)
        if not self.service_id or parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("A service needs an ID and an HTTP(S) endpoint")
        if parsed.scheme == "http" and parsed.hostname not in {
            "localhost", "127.0.0.1", "::1"
        }:
            raise ValueError("Remote services require HTTPS")
        if not isinstance(self.capabilities, tuple) or not all(
            isinstance(value, str) and value for value in self.capabilities
        ):
            raise ValueError("Capabilities must be a tuple of names")
        if not 0 <= self.load <= 1 or self.cost < 0 or self.ttl_seconds <= 0:
            raise ValueError("Invalid load, cost, or TTL")

    def active(self) -> bool:
        return monotonic() - self.registered_at < self.ttl_seconds


class ANPDiscovery:
    """Register local service descriptions, filter by capability, route by load."""

    def __init__(self) -> None:
        self._services: dict[str, AgentService] = {}

    def register(self, service: AgentService) -> None:
        self._services[service.service_id] = service

    def unregister(self, service_id: str) -> bool:
        return self._services.pop(service_id, None) is not None

    def discover(
        self, *, service_type: str | None = None, capability: str | None = None
    ) -> list[AgentService]:
        return sorted(
            (
                service for service in self._services.values()
                if service.active()
                and (service_type is None or service.service_type == service_type)
                and (capability is None or capability in service.capabilities)
            ),
            key=lambda service: (service.load, service.cost, service.service_id),
        )

    def choose(self, capability: str, *, service_type: str | None = None) -> AgentService:
        matches = self.discover(service_type=service_type, capability=capability)
        if not matches:
            raise LookupError(f"No active service provides {capability}")
        return matches[0]


class ANPNetwork:
    """Small local graph for inspecting connections between registered services."""

    def __init__(self, network_id: str) -> None:
        self.network_id = network_id
        self.nodes: dict[str, str] = {}
        self.edges: set[frozenset[str]] = set()

    def add_node(self, service_id: str, endpoint: str) -> None:
        self.nodes[service_id] = endpoint

    def connect_nodes(self, left: str, right: str) -> None:
        if left not in self.nodes or right not in self.nodes:
            raise KeyError("Both services must be registered before connecting")
        if left != right:
            self.edges.add(frozenset((left, right)))

    def get_network_stats(self) -> dict[str, int | str]:
        return {
            "network_id": self.network_id,
            "total_nodes": len(self.nodes),
            "total_connections": len(self.edges),
        }
