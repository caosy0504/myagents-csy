"""Local-only web server for the simulated travel planner.

Run from the project root: .venv/bin/python -m examples.travel_demo.server
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from pydantic import ValidationError

from .data import CITIES
from .planner import PlanGenerationError, TravelRequest, build_plan_from_selection, create_plan
from .research import ResearchError, ResearchUnavailableError, tavily_available

INDEX = Path(__file__).with_name("index.html")
STATIC_FILES = {
    "/": (INDEX, "text/html; charset=utf-8"),
    "/style.css": (INDEX.with_name("style.css"), "text/css; charset=utf-8"),
    "/app.js": (INDEX.with_name("app.js"), "text/javascript; charset=utf-8"),
}


class Handler(BaseHTTPRequestHandler):
    def _json(self, status: int, value: dict) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in STATIC_FILES:
            path, content_type = STATIC_FILES[self.path]
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; object-src 'none'")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        elif self.path == "/api/cities":
            self._json(200, {"cities": list(CITIES), "tavily_available": tavily_available()})
        elif self.path == "/api/health":
            self._json(200, {
                "status": "ready", "source": "local mock data",
                "model": "Ollama qwen2.5:7b", "tavily_available": tavily_available(),
            })
        else:
            self._json(404, {"error": "未找到页面"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in {"/api/plan", "/api/rebuild"}:
            self._json(404, {"error": "未找到接口"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 10_000:
                self._json(413, {"error": "请求内容为空或过长"})
                return
            payload = json.loads(self.rfile.read(length))
            if self.path == "/api/plan":
                request = TravelRequest.model_validate(payload)
                plan = create_plan(request)
            else:
                if not isinstance(payload, dict):
                    raise ValueError("编辑请求必须是对象")
                request = TravelRequest.model_validate(payload.get("request"))
                days = payload.get("days")
                if not isinstance(days, list):
                    raise ValueError("缺少行程列表")
                tips = payload.get("tips", "")
                if not isinstance(tips, str):
                    raise ValueError("建议必须是文本")
                attempts = payload.get("model_attempts", 0)
                if not isinstance(attempts, int) or not 0 <= attempts <= 2:
                    raise ValueError("模型调用次数无效")
                research = payload.get("research")
                if request.web_research and research is None:
                    raise ValueError("联网行程缺少原始搜索资料")
                plan = build_plan_from_selection(
                    request, days, tips, model_attempts=attempts,
                    allow_empty_days=True, edited=True, research=research,
                )
        except ResearchUnavailableError as exc:
            self._json(400, {"error": str(exc)})
            return
        except ResearchError as exc:
            self._json(502, {"error": str(exc)})
            return
        except PlanGenerationError as exc:
            self._json(502, {"error": str(exc)})
            return
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            self._json(400, {"error": str(exc)})
            return
        except Exception as exc:
            self._json(502, {"error": f"Ollama 调用失败：{exc}"})
            return
        self._json(200, plan)


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    print("本地旅行助手：http://127.0.0.1:8765", flush=True)
    print("模拟数据 + 本机 Ollama；按 Ctrl+C 停止", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
