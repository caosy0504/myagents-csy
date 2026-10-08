"""Expose and call MyAgents through the official A2A Python SDK v1."""

import asyncio
import hmac
from collections.abc import Callable, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

from ...core.agent import Agent
from .sessions import AgentSessionStore


_LOOPBACK = {"localhost", "127.0.0.1", "::1"}


def _check_url(base_url: str, allow_insecure_http: bool) -> None:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("A2A URL must be HTTP(S)")
    if parsed.scheme == "http" and parsed.hostname not in _LOOPBACK and not allow_insecure_http:
        raise ValueError("Remote A2A servers require HTTPS")


class _BearerGate:
    """Authenticate JSON-RPC requests before the A2A SDK builds a user context."""

    def __init__(self, app, tokens: Mapping[str, str] | None) -> None:
        self.app = app
        if tokens is not None and not isinstance(tokens, Mapping):
            raise ValueError("A2A auth_tokens must map principals to tokens")
        self.tokens = dict(tokens or {})
        if any(
            not isinstance(principal, str) or not principal
            or not isinstance(token, str) or not token
            for principal, token in self.tokens.items()
        ):
            raise ValueError("A2A bearer principals and tokens must be nonempty")

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope["path"].startswith("/a2a/"):
            await self.app(scope, receive, send)
            return
        from starlette.authentication import AuthCredentials, SimpleUser
        from starlette.responses import JSONResponse

        principal = "local"
        if self.tokens:
            headers = dict(scope.get("headers", []))
            header = headers.get(b"authorization", b"").decode("utf-8", "replace")
            supplied = header[7:] if header.lower().startswith("bearer ") else ""
            principal = next((
                name for name, token in self.tokens.items()
                if hmac.compare_digest(supplied, token)
            ), "")
            if not principal:
                response = JSONResponse({"error": "unauthorized"}, status_code=401)
                await response(scope, receive, send)
                return
        scope["user"] = SimpleUser(principal)
        scope["auth"] = AuthCredentials(["authenticated"])
        await self.app(scope, receive, send)


def create_a2a_app(
    agent_factory: Callable[[], Agent],
    *,
    name: str,
    description: str,
    base_url: str,
    session_db_path: str | Path | None = None,
    task_db_path: str | Path | None = None,
    auth_tokens: Mapping[str, str] | None = None,
    allow_insecure_http: bool = False,
):
    """Build an ASGI app with identity-scoped conversation and task storage."""
    _check_url(base_url, allow_insecure_http)
    if urlparse(base_url).hostname not in _LOOPBACK and not auth_tokens:
        raise ValueError("A public A2A server requires auth_tokens")
    try:
        from a2a.server.agent_execution import AgentExecutor
        from a2a.server.request_handlers import DefaultRequestHandler
        from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
        from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
        from a2a.types import (
            AgentCapabilities, AgentCard, AgentInterface, AgentSkill,
            HTTPAuthSecurityScheme, Part, SecurityRequirement, SecurityScheme,
            StringList, Task, TaskState, TaskStatus,
        )
        from starlette.applications import Starlette
    except ImportError as exc:
        raise RuntimeError("A2A requires: pip install 'my-agents-csy[protocol-a2a]'") from exc

    sessions = AgentSessionStore(session_db_path)
    engine = None
    if task_db_path is not None:
        try:
            from sqlalchemy.ext.asyncio import create_async_engine
            from a2a.server.tasks import DatabaseTaskStore
        except ImportError as exc:
            raise RuntimeError(
                "Persistent A2A tasks require: pip install 'my-agents-csy[protocol-a2a]'"
            ) from exc
        task_path = Path(task_db_path).expanduser().resolve()
        task_path.parent.mkdir(parents=True, exist_ok=True)
        engine = create_async_engine(f"sqlite+aiosqlite:///{task_path}")
        task_store = DatabaseTaskStore(engine)
    else:
        task_store = InMemoryTaskStore()

    locks: dict[tuple[str, str], asyncio.Lock] = {}

    class MyAgentExecutor(AgentExecutor):
        @staticmethod
        def run_agent(owner: str, context_id: str, query: str) -> str:
            agent = agent_factory()
            state = sessions.load(owner, context_id)
            if state is not None:
                agent.load_session_state(state)
            answer = agent.run(query)
            sessions.save(owner, context_id, agent.export_session_state())
            return answer

        async def execute(self, context, event_queue) -> None:
            if not context.task_id or not context.context_id or not context.message:
                raise ValueError("A2A task context is incomplete")
            await event_queue.enqueue_event(Task(
                id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
                history=[context.message],
            ))
            updater = TaskUpdater(
                event_queue=event_queue,
                task_id=context.task_id,
                context_id=context.context_id,
            )
            await updater.start_work()
            query = context.get_user_input()
            owner = context.call_context.user.user_name
            key = (owner, context.context_id)
            lock = locks.setdefault(key, asyncio.Lock())
            try:
                async with lock:
                    answer = await asyncio.to_thread(
                        self.run_agent, owner, context.context_id, query
                    )
            except Exception:
                await updater.failed()
                return
            await updater.add_artifact(parts=[Part(text=answer)], name="answer", last_chunk=True)
            await updater.complete()

        async def cancel(self, context, event_queue) -> None:
            raise NotImplementedError("This synchronous Agent does not expose cancelable tasks")

    rpc_url = base_url.rstrip("/") + "/a2a/jsonrpc"
    card = AgentCard(
        name=name,
        description=description,
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        default_input_modes=["text"],
        default_output_modes=["text"],
        skills=[
            AgentSkill(
                id="answer", name="answer", description=description,
                tags=["question-answering"], examples=["请回答这个问题"],
                input_modes=["text"], output_modes=["text"],
            )
        ],
        supported_interfaces=[
            AgentInterface(url=rpc_url, protocol_binding="JSONRPC", protocol_version="1.0")
        ],
        security_schemes={
            "bearer": SecurityScheme(
                http_auth_security_scheme=HTTPAuthSecurityScheme(scheme="bearer")
            )
        } if auth_tokens else {},
        security_requirements=[
            SecurityRequirement(schemes={"bearer": StringList(list=[])})
        ] if auth_tokens else [],
    )
    handler = DefaultRequestHandler(
        agent_executor=MyAgentExecutor(), task_store=task_store, agent_card=card
    )
    routes = create_agent_card_routes(card)
    routes.extend(create_jsonrpc_routes(handler, rpc_url="/a2a/jsonrpc"))

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            if engine is not None:
                await engine.dispose()

    return _BearerGate(Starlette(routes=routes, lifespan=lifespan), auth_tokens)


class A2AClient:
    """Send a text request to an A2A v1 agent and collect its response."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 30.0,
        http_client=None,
        bearer_token: str | None = None,
        allow_insecure_http: bool = False,
    ) -> None:
        _check_url(base_url, allow_insecure_http)
        if bearer_token and http_client is not None:
            raise ValueError("Configure bearer headers on a caller-owned HTTP client")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.http_client = http_client
        self.bearer_token = bearer_token
        self.context_id: str | None = None

    def reset_context(self) -> None:
        self.context_id = None

    def _validate_card(self, card) -> None:
        expected = urlparse(self.base_url)
        interfaces = [
            interface for interface in card.supported_interfaces
            if interface.protocol_binding == "JSONRPC"
        ]
        if not interfaces:
            raise ValueError("A2A Agent Card has no JSON-RPC interface")
        for interface in interfaces:
            endpoint = urlparse(interface.url)
            if (endpoint.scheme, endpoint.netloc) != (
                expected.scheme, expected.netloc
            ):
                raise ValueError("A2A Agent Card points to another origin")

    async def send(self, text: str) -> str:
        try:
            import httpx
            from a2a.client import A2ACardResolver, ClientConfig, create_client
            from a2a.helpers import get_artifact_text, get_message_text
            from a2a.types import Message, Part, Role, SendMessageRequest, TaskState
        except ImportError as exc:
            raise RuntimeError("A2A requires: pip install 'my-agents-csy[protocol-a2a]'") from exc
        owns_client = self.http_client is None
        http_client = self.http_client or httpx.AsyncClient(
            timeout=self.timeout,
            headers=(
                {"Authorization": f"Bearer {self.bearer_token}"}
                if self.bearer_token else None
            ),
        )
        sdk_client = None
        try:
            card = await A2ACardResolver(http_client, self.base_url).get_agent_card()
            self._validate_card(card)
            sdk_client = await create_client(
                card,
                client_config=ClientConfig(
                    httpx_client=http_client,
                    supported_protocol_bindings=["JSONRPC"],
                    streaming=False,
                ),
            )
            fields = {
                "role": Role.ROLE_USER, "message_id": str(uuid4()),
                "parts": [Part(text=text)],
            }
            if self.context_id:
                fields["context_id"] = self.context_id
            message = Message(**fields)
            output = []
            async for event in sdk_client.send_message(SendMessageRequest(message=message)):
                if event.HasField("message"):
                    if event.message.context_id:
                        self.context_id = event.message.context_id
                    output.append(get_message_text(event.message))
                elif event.HasField("task"):
                    if event.task.context_id:
                        self.context_id = event.task.context_id
                    output.extend(
                        get_artifact_text(artifact) for artifact in event.task.artifacts
                    )
                    if event.task.status.state in {
                        TaskState.TASK_STATE_FAILED,
                        TaskState.TASK_STATE_REJECTED,
                        TaskState.TASK_STATE_CANCELED,
                    }:
                        raise RuntimeError(f"A2A task ended: {TaskState.Name(event.task.status.state)}")
                elif event.HasField("artifact_update"):
                    output.append(get_artifact_text(event.artifact_update.artifact))
                elif event.HasField("status_update") and event.status_update.status.state in {
                    TaskState.TASK_STATE_FAILED,
                    TaskState.TASK_STATE_REJECTED,
                    TaskState.TASK_STATE_CANCELED,
                }:
                    raise RuntimeError(
                        f"A2A task ended: {TaskState.Name(event.status_update.status.state)}"
                    )
            if not output:
                raise RuntimeError("A2A server returned no text message or artifact")
            return "\n".join(part for part in output if part)
        finally:
            # The SDK closes its httpx client; preserve caller-owned clients.
            if owns_client and sdk_client is not None:
                await sdk_client.close()
            elif owns_client:
                await http_client.aclose()
