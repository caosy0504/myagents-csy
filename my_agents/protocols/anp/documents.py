"""Read the public ANP-08 discovery index without claiming DID authentication."""

import json
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx


@dataclass(frozen=True)
class AgentDescriptionReference:
    name: str
    url: str


class ANPDocumentDiscovery:
    """Read a trusted domain's .well-known index and its description documents."""

    def __init__(
        self,
        allowed_domains: set[str],
        *,
        timeout: float = 10.0,
        max_pages: int = 5,
        max_items: int = 100,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if not allowed_domains or timeout <= 0 or max_pages <= 0 or max_items <= 0:
            raise ValueError("ANP discovery needs allowed domains and positive limits")
        self.allowed_domains = {domain.lower() for domain in allowed_domains}
        self.timeout = timeout
        self.max_pages = max_pages
        self.max_items = max_items
        self.http_client = http_client

    def _check_url(self, url: str, domain: str) -> None:
        parsed = urlparse(url)
        host = urlparse("http://" + domain).hostname
        local = host in {"localhost", "127.0.0.1", "::1"}
        if parsed.netloc != domain or parsed.scheme != ("http" if local else "https"):
            raise ValueError("ANP discovery URL left the allowed origin")
        if parsed.username or parsed.password or parsed.fragment:
            raise ValueError("ANP discovery URL contains unsupported URL components")

    async def _get_json(self, client: httpx.AsyncClient, url: str, domain: str) -> dict:
        self._check_url(url, domain)
        content = bytearray()
        async with client.stream(
            "GET", url, timeout=self.timeout, follow_redirects=False
        ) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                content.extend(chunk)
                if len(content) > 1_000_000:
                    raise ValueError("ANP document exceeds 1 MB")
        document = json.loads(content)
        if not isinstance(document, dict):
            raise ValueError("ANP discovery document must be a JSON object")
        return document

    async def discover(self, domain: str) -> list[AgentDescriptionReference]:
        domain = domain.lower().strip()
        if domain not in self.allowed_domains:
            raise ValueError("ANP domain is not allowed")
        host = urlparse("http://" + domain).hostname
        scheme = "http" if host in {"localhost", "127.0.0.1", "::1"} else "https"
        url = f"{scheme}://{domain}/.well-known/agent-descriptions"
        owns_client = self.http_client is None
        client = self.http_client or httpx.AsyncClient()
        references = []
        visited = set()
        try:
            for _ in range(self.max_pages):
                if url in visited:
                    raise ValueError("ANP discovery pagination loop")
                visited.add(url)
                document = await self._get_json(client, url, domain)
                if document.get("@type") not in {"CollectionPage", "https://schema.org/CollectionPage"}:
                    raise ValueError("ANP index is not a CollectionPage")
                if "@context" not in document:
                    raise ValueError("ANP index has no JSON-LD context")
                items = document.get("items")
                if not isinstance(items, list):
                    raise ValueError("ANP index items must be an array")
                for item in items:
                    if (
                        not isinstance(item, dict)
                        or not isinstance(item.get("name"), str)
                        or not str(item.get("@type", "")).endswith("AgentDescription")
                    ):
                        raise ValueError("ANP index contains an invalid item")
                    description_url = item.get("@id")
                    if not isinstance(description_url, str):
                        raise ValueError("ANP description URL is missing")
                    self._check_url(description_url, domain)
                    references.append(AgentDescriptionReference(item["name"], description_url))
                    if len(references) > self.max_items:
                        raise ValueError("ANP discovery exceeded its item limit")
                next_url = document.get("next")
                if next_url is None:
                    return references
                if not isinstance(next_url, str):
                    raise ValueError("ANP next page URL is invalid")
                self._check_url(next_url, domain)
                url = next_url
            raise ValueError("ANP discovery exceeded its page limit")
        finally:
            if owns_client:
                await client.aclose()

    async def fetch_description(self, reference: AgentDescriptionReference) -> dict:
        domain = urlparse(reference.url).netloc
        if domain not in self.allowed_domains:
            raise ValueError("ANP description domain is not allowed")
        owns_client = self.http_client is None
        client = self.http_client or httpx.AsyncClient()
        try:
            document = await self._get_json(client, reference.url, domain)
            item_types = document.get("@type")
            names = item_types if isinstance(item_types, list) else [item_types]
            if not any(str(name).endswith("AgentDescription") for name in names):
                raise ValueError("ANP document is not an AgentDescription")
            if document.get("@id") not in {None, reference.url}:
                raise ValueError("ANP description ID does not match its URL")
            return document
        finally:
            if owns_client:
                await client.aclose()
