"""MCP v2 client with scoped catalog caching and optional connection reuse."""

import asyncio
import hashlib
from concurrent.futures import Future
from collections.abc import Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from threading import Event, RLock
from typing import Any
from urllib.parse import urlparse


class MCPClient:
    """Connect by server object, stdio command, or Streamable HTTP URL."""

    def __init__(
        self,
        source: Any,
        *,
        timeout: float = 30.0,
        bearer_token: str | None = None,
        allow_insecure_http: bool = False,
    ) -> None:
        if isinstance(source, Sequence) and not isinstance(source, (str, bytes)) and not source:
            raise ValueError("MCP stdio command cannot be empty")
        if timeout <= 0:
            raise ValueError("MCP timeout must be positive")
        if bearer_token and not isinstance(source, str):
            raise ValueError("Bearer authentication requires an HTTP MCP server")
        if isinstance(source, str):
            parsed = urlparse(source)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError("MCP URL must be HTTP(S)")
            if parsed.scheme == "http" and parsed.hostname not in {
                "localhost", "127.0.0.1", "::1"
            } and not allow_insecure_http:
                raise ValueError("Remote MCP servers require HTTPS")
        self.source = source
        self.timeout = timeout
        self.bearer_token = bearer_token
        self._session = None
        self._stack: AsyncExitStack | None = None
        self._portal_manager = None
        self._portal = None
        self._worker = None
        self._worker_queue = None
        self._worker_error: BaseException | None = None
        self._sync_lock = RLock()
        try:
            from mcp.client.caching import CacheConfig, InMemoryResponseCacheStore
        except ImportError as exc:
            raise RuntimeError("MCP requires: pip install 'my-agents-csy[protocol-mcp]'") from exc
        principal = (
            hashlib.sha256(bearer_token.encode()).hexdigest()
            if bearer_token else "local-single-user"
        )
        if isinstance(source, str):
            target_id = source
        elif isinstance(source, Sequence):
            target_id = "stdio:" + hashlib.sha256(
                "\0".join(source).encode()
            ).hexdigest()
        else:
            target_id = f"in-process:{id(source)}"
        # The SDK honors ttlMs/cacheScope. It never shares private entries between principals.
        self._cache = CacheConfig(
            store=InMemoryResponseCacheStore(), partition=principal,
            target_id=target_id, share_public=False
        )

    def _target(self) -> Any:
        from mcp import StdioServerParameters

        if isinstance(self.source, Sequence) and not isinstance(self.source, (str, bytes)):
            if not all(isinstance(part, str) for part in self.source):
                raise ValueError("MCP command must contain only strings")
            return StdioServerParameters(command=self.source[0], args=list(self.source[1:]))
        return self.source

    async def _enter_client(self, stack: AsyncExitStack):
        from mcp import Client

        target = self._target()
        if self.bearer_token:
            import httpx2
            from mcp.client.streamable_http import streamable_http_client

            http = await stack.enter_async_context(httpx2.AsyncClient(
                headers={"Authorization": f"Bearer {self.bearer_token}"},
                timeout=httpx2.Timeout(self.timeout),
            ))
            target = streamable_http_client(target, http_client=http)
        return await stack.enter_async_context(Client(
            target, read_timeout_seconds=self.timeout, cache=self._cache
        ))

    async def __aenter__(self):
        if self._session is not None:
            raise RuntimeError("MCP client is already connected")
        stack = AsyncExitStack()
        await stack.__aenter__()
        try:
            self._session = await self._enter_client(stack)
        except BaseException:
            await stack.aclose()
            raise
        self._stack = stack
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        if self._stack is not None:
            stack, self._stack = self._stack, None
            self._session = None
            await stack.__aexit__(exc_type, exc_value, traceback)

    @asynccontextmanager
    async def _connection(self):
        if self._session is not None:
            yield self._session
        else:
            async with AsyncExitStack() as stack:
                yield await self._enter_client(stack)

    async def list_tools(self, *, refresh: bool = False) -> list[dict[str, Any]]:
        tools = []
        cursor = None
        async with self._connection() as client:
            while True:
                result = await client.list_tools(
                    cursor=cursor, cache_mode="refresh" if refresh else "use"
                )
                tools.extend(
                    {
                        "name": tool.name,
                        "description": tool.description or "",
                        "input_schema": tool.input_schema,
                    }
                    for tool in result.tools
                )
                cursor = result.next_cursor
                if not cursor:
                    return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        async with self._connection() as client:
            result = await client.call_tool(
                name, arguments, read_timeout_seconds=self.timeout
            )
        if result.is_error:
            detail = "\n".join(
                getattr(item, "text", "") for item in result.content
                if getattr(item, "type", "") == "text"
            )
            raise RuntimeError(detail or f"MCP tool {name} failed")
        if result.structured_content is not None:
            return result.structured_content
        texts = [
            getattr(item, "text", "") for item in result.content
            if getattr(item, "type", "") == "text"
        ]
        if texts:
            return "\n".join(texts)
        # Preserve references and media type without dumping binary data into the prompt.
        return {
            "content": [
                {
                    "type": getattr(item, "type", "unknown"),
                    "mime_type": getattr(item, "mime_type", None),
                    "uri": getattr(item, "uri", None),
                    "data_omitted": getattr(item, "type", "") in {"image", "audio"},
                }
                for item in result.content
            ]
        }

    async def _sync_worker(self, ready: Event) -> None:
        """One task owns the SDK context manager for its entire lifetime."""
        try:
            async with self:
                self._worker_queue = asyncio.Queue()
                ready.set()
                while True:
                    action, arguments, future = await self._worker_queue.get()
                    if action == "close":
                        if not future.done():
                            future.set_result(None)
                        break
                    try:
                        if action == "list":
                            value = await self.list_tools(refresh=arguments[0])
                        else:
                            value = await self.call_tool(arguments[0], arguments[1])
                        if not future.done():
                            future.set_result(value)
                    except BaseException as exc:
                        if not future.done():
                            future.set_exception(exc)
        except BaseException as exc:
            self._worker_error = exc
            ready.set()
            raise
        finally:
            self._worker_queue = None

    def open_sync(self) -> None:
        """Keep one SDK connection alive for synchronous Agents."""
        from anyio.from_thread import start_blocking_portal

        with self._sync_lock:
            if self._portal is not None:
                return
            self._worker_error = None
            manager = start_blocking_portal(name="my-agents-mcp")
            portal = manager.__enter__()
            ready = Event()
            worker = None
            try:
                worker = portal.start_task_soon(self._sync_worker, ready)
                if not ready.wait(self.timeout):
                    raise TimeoutError("MCP connection did not become ready")
                if self._worker_error is not None:
                    raise RuntimeError("MCP connection failed") from self._worker_error
            except BaseException:
                if worker is not None:
                    worker.cancel()
                manager.__exit__(None, None, None)
                raise
            self._portal_manager = manager
            self._portal = portal
            self._worker = worker

    def _request_sync(self, action: str, *arguments):
        if self._portal is None or self._worker_queue is None:
            raise RuntimeError("MCP client is not open")
        future: Future = Future()
        self._portal.call(self._worker_queue.put, (action, arguments, future))
        return future.result(timeout=self.timeout + 5)

    def close_sync(self) -> None:
        with self._sync_lock:
            if self._portal is None:
                return
            manager, worker = self._portal_manager, self._worker
            try:
                self._request_sync("close")
                worker.result(timeout=self.timeout + 5)
            finally:
                self._portal = self._portal_manager = self._worker = None
                manager.__exit__(None, None, None)

    def list_tools_sync(self, *, refresh: bool = False) -> list[dict[str, Any]]:
        if self._portal is not None:
            return self._request_sync("list", refresh)
        return _run_standalone(self.list_tools(refresh=refresh))

    def call_tool_sync(self, name: str, arguments: dict[str, Any]) -> Any:
        if self._portal is not None:
            return self._request_sync("call", name, arguments)
        return _run_standalone(self.call_tool(name, arguments))


def _run_standalone(coroutine):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    coroutine.close()
    raise RuntimeError("Use async MCPClient methods inside an event loop")
